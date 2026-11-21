"""Run the paper's KSRA parameter grid using repository-relative paths."""
import argparse
import csv
import ctypes
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, replace

ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parent
DATASETS = REPOSITORY / "Datasets"
RESULTS = REPOSITORY / "Results"
VARIANTS = ("v1", "v2", "v3", "v4")
PAPER_DATASETS = ('audiology_variant1', 'chess_nowin_16_variant1', 'chess_nowin_34_variant1', 'lymphography', 'monks_0_4_variant1', 'mushroom_p_221_variant1', 'tic_tac_toe_negative_12_variant1', 'tic_tac_toe_negative_32_variant1', 'vote_republican_29_variant1', 'cardio', 'cardiotocography_2and3_33_variant1', 'ecoli', 'iris_Irisvirginica_11_variant1', 'wbc_malignant_39_variant1', 'wdbc_M_39_variant1', 'yeast_ERL_5_variant1', 'annealing_variant1', 'arrhythmia_variant1', 'bands_band_6_variant1', 'sick_sick_72_variant1')
LAMBDA_GRID = tuple(10.0 ** exponent for exponent in range(-8, 8))


# File handling
def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    for attempt in range(20):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.05 * (attempt + 1))


def core_module():
    name = "KSRA_public" if (ROOT / "KSRA_public.py").is_file() else "KSRA"
    return importlib.import_module(name)


def config_name(variant, lam):
    return f"{variant}_lam_{lam:.0e}"


def memory_info(pid=None):
    if os.name != 'nt':
        return {}
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [(x, ctypes.c_size_t) for x in ['PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage']]
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    h = kernel.OpenProcess(1024 | 16, False, os.getpid() if pid is None else pid)
    if not h:
        return {}
    counters = PMC()
    counters.cb = ctypes.sizeof(counters)
    try:
        if not psapi.GetProcessMemoryInfo(h, ctypes.byref(counters), counters.cb):
            return {}
        return {k: int(getattr(counters, k)) for k in ['WorkingSetSize', 'PeakWorkingSetSize', 'PagefileUsage', 'PeakPagefileUsage']}
    finally:
        kernel.CloseHandle(h)


# ROC-AUC and non-interpolated average precision
def metrics(y, s):
    """ROC-AUC and non-interpolated AP, ties grouped exactly, anomaly label 1."""
    import numpy as np
    from scipy.stats import rankdata
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    if y.shape != s.shape or not np.isfinite(s).all() or (not np.all(np.isin(y, [0, 1]))) or (len(np.unique(y)) != 2):
        raise ValueError('Need finite scores and both binary classes')
    pos = int(y.sum())
    neg = len(y) - pos
    auc = float((rankdata(s)[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))
    idx = np.argsort(-s, kind='stable')
    ys = y[idx]
    ss = s[idx]
    ends = np.r_[np.flatnonzero(ss[1:] != ss[:-1]), len(ss) - 1]
    tp = np.cumsum(ys)[ends]
    fp = ends + 1 - tp
    recall = tp / pos
    precision = tp / (tp + fp)
    ap = float(np.sum(np.diff(np.r_[0, recall]) * precision))
    return (dict(roc_auc=auc, average_precision=ap), dict(fpr=np.r_[0, fp / neg], tpr=np.r_[0, recall], thresholds=np.r_[np.inf, ss[ends]], precision=precision, recall=recall))


# Full-input benchmark loading
def load_dataset(name):
    import numpy as np
    from scipy.io import loadmat

    path = DATASETS / (name + ".mat")
    raw = np.asarray(loadmat(path, variable_names=["trandata"])["trandata"], dtype=float)
    if raw.ndim != 2 or raw.shape[0] < 2 or raw.shape[1] < 2 or not np.isfinite(raw).all():
        raise ValueError("Expected a finite sample-by-column trandata matrix")
    x, y = raw[:, :-1], raw[:, -1]
    if set(np.unique(y)) != {0.0, 1.0}:
        raise ValueError("The final column must contain both binary labels; anomaly=1")
    numeric = np.all(x <= 1 + 1e-9, axis=0)
    return x, y, numeric, digest(path)


# Solver execution
def run_worker(args):
    import numpy as np
    import scipy

    core = core_module()
    x, y, numeric, data_hash = load_dataset(args.worker[0])
    variant = args.worker[1]
    folder = args.output / args.worker[0]
    folder.mkdir(parents=True, exist_ok=True)
    profile = core.prepare_profiles(x, numeric_mask=numeric)
    config = core.SolverConfig(max_iter=20000, max_seconds=120.0, adaptive_rho=True)
    identity = dict(data_sha256=data_hash, core_sha256=digest(core.__file__),
                    runner_sha256=digest(__file__), solver_config=asdict(config),
                    restart=0.85, pr_tol=1e-12, threads=1)
    manifest = dict(dataset=args.worker[0], n=len(y), m=x.shape[1],
                    normal_count=int((y == 0).sum()), outlier_count=int((y == 1).sum()),
                    profile=profile.info, identity=identity)
    atomic_json(folder / "dataset.json", manifest)
    cache = {}
    for lam in args.lambdas:
        run = folder / config_name(variant, lam)
        run.mkdir(exist_ok=True)
        key = dict(**identity, variant=variant, lambda_reg=lam)
        path = run / "result.json"
        if path.exists():
            prior = json.loads(path.read_text(encoding="utf-8"))
            if prior["identity"] != key:
                raise ValueError(f"Saved configuration differs: {run}")
            if prior["status"] in ("success", "unconverged"):
                print("SAVED", args.worker[0], variant, lam, prior["status"], flush=True)
                continue
        row = dict(dataset=args.worker[0], identity=key, status="running", metrics=None,
                   environment=dict(python=platform.python_version(), numpy=np.__version__,
                                    scipy=scipy.__version__, platform=platform.platform()))
        atomic_json(path, row)
        history = []
        last_write = [0.0]

        def progress(info):
            now = time.monotonic()
            if now - last_write[0] >= 10:
                atomic_json(run / "progress.json", info)
                last_write[0] = now

        def diagnostics(state, info, current_history):
            atomic_json(run / "history.json", history + current_history)
            atomic_json(run / "progress.json", info)

        try:
            started = time.monotonic()
            state = None
            stages = []
            for limit in (2000, 5000, 20000):
                remaining = max(0.001, config.max_seconds - (time.monotonic() - started))
                stage_config = replace(config, max_iter=limit, max_seconds=remaining)
                result = core.solve_representation(
                    profile.K, lam, variant, stage_config, state=state,
                    checkpoint=diagnostics, callback=progress, cache=cache)
                stages.append(dict(iteration_limit=limit, **result.info))
                if result.info["converged"] or result.info["stop_reason"] != "iteration_limit":
                    break
                history += result.history
                state = result.state
            row.update(solver=result.info, stages=stages,
                       status="success" if result.info["converged"] else "unconverged")
            if result.info["converged"]:
                scores, walk, stationary = core.score_representation(result.Z, profile.weight)
                met, curves = metrics(y, scores)
                np.save(run / "scores.npy", scores)
                np.save(run / "stationary.npy", stationary)
                np.savez(run / "curves.npz", **curves)
                row.update(metrics=met, walk=walk)
                row["selected_configuration_seconds"] = (
                    profile.info["preprocessing_seconds"] +
                    result.info["total_solve_seconds"] + walk["score_seconds"])
            del result
        except Exception:
            row.update(status="failed", error=traceback.format_exc())
        atomic_json(path, row)
        print("DONE", args.worker[0], variant, lam, row["status"], flush=True)


# Batch execution and result selection
def summarize(args, names):
    selected = []
    counts = {}
    for name in names:
        for variant in args.variants:
            records = []
            for lam in args.lambdas:
                path = args.output / name / config_name(variant, lam) / "result.json"
                if path.exists():
                    row = json.loads(path.read_text(encoding="utf-8"))
                    records.append(row)
                    counts[row["status"]] = counts.get(row["status"], 0) + 1
            valid = [r for r in records if r["status"] == "success"]
            best = min(valid, key=lambda r: (-r["metrics"]["roc_auc"],
                                             r["identity"]["lambda_reg"])) if valid else None
            selected.append(dict(
                dataset=name, variant=variant, valid_configs=len(valid),
                expected_configs=len(args.lambdas), grid_complete=len(valid) == len(args.lambdas),
                selected_lambda=best["identity"]["lambda_reg"] if best else None,
                roc_auc=best["metrics"]["roc_auc"] if best else None,
                average_precision=best["metrics"]["average_precision"] if best else None,
                selected_configuration_seconds=best["selected_configuration_seconds"] if best else None,
                scores_path=(Path(name) / config_name(variant, best["identity"]["lambda_reg"]) /
                             "scores.npy").as_posix() if best else None))
    temporary = args.output / "summary.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    try:
        temporary.replace(args.output / "summary.csv")
    except PermissionError:
        print("Summary is open; latest table saved as summary.csv.tmp", flush=True)
    expected = len(names) * len(args.variants) * len(args.lambdas)
    atomic_json(args.output / "summary.json", dict(
        expected=expected, status=counts,
        finished=sum(counts.values()) == expected and counts.get("running", 0) == 0,
        all_converged=counts.get("success", 0) == expected))


def batch(args, names):
    for name in names:
        for variant in args.variants:
            command = [sys.executable, "-B", "-u", str(Path(__file__).resolve()),
                       "--output", str(args.output), "--worker", name, variant,
                       "--lambdas", *map(str, args.lambdas)]
            log_path = args.output / f"{name}_{variant}.log"
            with log_path.open("a", encoding="utf-8") as log:
                process = subprocess.Popen(
                    command, stdout=log, stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                try:
                    while process.poll() is None:
                        mem = memory_info(process.pid)
                        if mem.get("PagefileUsage", 0) > 4096 * 1024**2:
                            process.terminate()
                            raise RuntimeError(f"Worker exceeded 4096 MiB: {name}/{variant}")
                        time.sleep(0.5)
                    if process.returncode:
                        raise RuntimeError(f"Worker failed; see {log_path}")
                finally:
                    if process.poll() is None:
                        process.terminate()
                    process.wait()
            summarize(args, names)
            print("Completed", name, variant, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(PAPER_DATASETS))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--lambdas", nargs="+", type=float, default=list(LAMBDA_GRID))
    parser.add_argument("--output", type=Path, default=RESULTS)
    parser.add_argument("--worker", nargs=2, metavar=("DATASET", "VARIANT"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    import math

    names = args.datasets if not args.worker else [args.worker[0]]
    if (len(set(names)) != len(names) or
            any(Path(name).name != name or "/" in name or "\\" in name for name in names)):
        parser.error("Use unique dataset stems without paths")
    if len(set(args.variants)) != len(args.variants):
        parser.error("Use unique variants")
    if any(not math.isfinite(x) or x <= 0 for x in args.lambdas) or len(set(args.lambdas)) != len(args.lambdas):
        parser.error("Use unique positive finite lambda values")
    ids = [config_name("v2", x) for x in args.lambdas]
    if len(set(ids)) != len(ids):
        parser.error("Lambda values collide in output folder names")
    for name in names:
        if not (DATASETS / (name + ".mat")).is_file():
            parser.error(f"Missing dataset: {DATASETS / (name + '.mat')}")
    if args.worker:
        if args.worker[1] not in VARIANTS:
            parser.error("Invalid worker variant")
        run_worker(args)
    else:
        batch(args, names)


if __name__ == "__main__":
    main()
