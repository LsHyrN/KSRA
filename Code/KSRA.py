"""Kernelized self-representation anomaly detection (KSRA)."""

from dataclasses import dataclass, asdict

from time import perf_counter

import warnings

import numpy as np

from scipy.linalg import cho_factor, cho_solve, svd, eigh

from scipy.linalg.lapack import get_lapack_funcs

from scipy.spatial.distance import cdist

VERSION = '2026-11-06-row-absolute-v4-certified-output'

# Configuration and result types

@dataclass(frozen=True)
class SolverConfig:
    eps: float = 0.0001
    max_iter: int = 20000
    step_tol: float = 1e-07
    grad_tol: float = 1e-07
    rho: float = 1.0
    adaptive_rho: bool = False
    rho_update_interval: int = 25
    rho_adapt_until: int = 1000
    admm_atol: float = 1e-08
    admm_rtol: float = 1e-06
    max_seconds: float = 0.0
    checkpoint_every: int = 250
    diagnostic_every: int = 5

@dataclass
class SolverResult:
    Z: np.ndarray
    E: np.ndarray
    info: dict
    history: list
    state: dict

@dataclass
class Profile:
    K: np.ndarray
    weight: np.ndarray
    info: dict

class ConvergenceError(RuntimeError):
    pass

# Row norms and objectives

def row_norm(A):
    return np.sqrt(np.einsum('ij,ij->i', A, A))

def phi(A, eps=0.0):
    return float(np.hypot(row_norm(A), eps).sum())

def objective(K, Z, lam, variant, eps=0.0001, E=None):
    E = K - K @ Z if E is None else E
    loss = phi(E, eps) if variant in ('v1', 'v3') else float(np.sum(E * E))
    reg = phi(Z, eps) if variant == 'v1' else phi(Z) if variant == 'v4' else float(np.sum(Z * Z))
    value = loss + lam * reg
    offset = len(K) * eps * (1 + lam if variant == 'v1' else 1 if variant == 'v3' else 0)
    return dict(objective=float(value), centered_objective=float(value - offset), nonsmooth_objective=float(phi(E) + lam * phi(Z) if variant == 'v1' else phi(E) + lam * np.sum(Z * Z) if variant == 'v3' else value))

def smooth_gradient(K, Z, lam, variant, eps, E=None):
    E = K - K @ Z if E is None else E
    loss_grad = -K.T @ (E / np.hypot(row_norm(E), eps)[:, None]) if variant in ('v1', 'v3') else -2 * K.T @ E
    reg_grad = lam * Z / np.hypot(row_norm(Z), eps)[:, None] if variant == 'v1' else 2 * lam * Z
    grad = loss_grad + reg_grad
    scale = 1 + np.linalg.norm(loss_grad) + np.linalg.norm(reg_grad)
    return (grad, float(np.linalg.norm(grad) / scale))

def _validate_config(c):
    positive = ('eps', 'step_tol', 'grad_tol', 'rho', 'admm_atol', 'admm_rtol')
    if any((not np.isfinite(getattr(c, k)) or getattr(c, k) <= 0 for k in positive)):
        raise ValueError('Positive finite solver parameters required')
    if c.rho_update_interval < 1 or c.rho_adapt_until < 0:
        raise ValueError('Invalid finite rho adaptation schedule')
    if c.max_iter < 1 or c.checkpoint_every < 1 or c.diagnostic_every < 1 or (c.max_seconds < 0):
        raise ValueError('Invalid iteration/checkpoint/time budget')

# Data validation and hybrid similarity profiles

def prepare_profiles(data, sigma=None, numeric_mask=None, block_size=256):
    t = perf_counter()
    X = np.asarray(data, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < 1 or (not np.isfinite(X).all()):
        raise ValueError('Need finite 2D data with >=2 samples and >=1 feature')
    n, m = X.shape
    numeric = np.all(X <= 1 + 1e-09, axis=0) if numeric_mask is None else np.asarray(numeric_mask, dtype=bool)
    if numeric.shape != (m,):
        raise ValueError('numeric_mask must have one entry per feature')
    if block_size < 1:
        raise ValueError('block_size must be positive')
    D = np.zeros((n, n))
    if numeric.any():
        cdist(X[:, numeric], X[:, numeric], metric='euclidean', out=D)
    nominal = np.flatnonzero(~numeric)
    if len(nominal):
        for lo in range(0, n, block_size):
            hi = min(n, lo + block_size)
            count = np.zeros((hi - lo, n))
            for col in nominal:
                count += np.abs(X[lo:hi, col, None] - X[None, :, col]) > 1e-09
            D[lo:hi] += count / len(nominal)
    sigma0 = float((D.sum() - np.trace(D)) / (n * (n - 1)))
    if sigma0 <= 0 or not np.isfinite(sigma0):
        raise ValueError('Degenerate input: nonpositive mean off-diagonal distance')
    sigma = sigma0 if sigma is None else float(sigma)
    if sigma <= 0 or not np.isfinite(sigma):
        raise ValueError('sigma must be positive finite')
    D /= sigma
    np.square(D, out=D)
    D *= -0.5
    np.exp(D, out=D)
    K = D
    weight = 1 - np.cbrt(K.sum(axis=1) / n)
    if not np.isfinite(K).all() or weight.min() < -1e-12:
        raise FloatingPointError('Invalid profiles/weights')
    return Profile(K, weight, dict(n=n, m=m, sigma=sigma, sigma0=sigma0, numeric_columns=np.flatnonzero(numeric).tolist(), nominal_columns=nominal.tolist(), preprocessing_seconds=perf_counter() - t, block_size=block_size))

# Linear algebra

def _quadratic_solve(K, gl, reg, H):
    A = H.copy()
    A.flat[::len(K) + 1] += reg
    method = 'cholesky'
    try:
        factor = cho_factor(A, lower=True, check_finite=False)
        rcond, lapack_info = get_lapack_funcs('pocon', (A,))(factor[0], float(np.linalg.norm(A, 1)), uplo='L')
        if lapack_info != 0 or rcond < 1e-06:
            raise np.linalg.LinAlgError('Ill-conditioned normal equations')
        Z = cho_solve(factor, H, check_finite=False)
        back = float(np.linalg.norm(A @ Z - H) / (1 + np.linalg.norm(A) * np.linalg.norm(Z) + np.linalg.norm(H)))
        if not np.isfinite(Z).all() or back > 1e-10:
            raise np.linalg.LinAlgError('Poor backward error')
    except (np.linalg.LinAlgError, ValueError):
        method = 'svd_tikhonov_no_truncation'
        W = np.sqrt(gl)[:, None] * K
        B = W / np.sqrt(reg)[None, :]
        U, s, Vh = svd(B, full_matrices=False, check_finite=False, lapack_driver='gesdd')
        filt = (s / np.hypot(s, 1.0)) ** 2
        if float(filt.mean()) > 0.5:
            complement = (1 / np.hypot(s, 1.0)) ** 2
            Z = np.eye(len(K)) - Vh.T * complement @ Vh * np.sqrt(reg)[None, :] / np.sqrt(reg)[:, None]
            method = 'svd_tikhonov_complement_no_truncation'
        else:
            Z = Vh.T * filt @ Vh * np.sqrt(reg)[None, :] / np.sqrt(reg)[:, None]
        back = float(np.linalg.norm(A @ Z - H) / (1 + np.linalg.norm(A) * np.linalg.norm(Z) + np.linalg.norm(H)))
        if not np.isfinite(Z).all() or back > 1e-10:
            raise FloatingPointError('Surrogate solve failed')
    return (Z, method, back)

def _admm_setup(K, B, rho, cache):
    A = B.copy()
    A.flat[::len(K) + 1] += rho
    try:
        fac = cho_factor(A, lower=True, check_finite=False)
        rcond, code = get_lapack_funcs('pocon', (A,))(fac[0], float(np.linalg.norm(A, 1)), uplo='L')
        if code or rcond < 1e-08:
            raise np.linalg.LinAlgError('Ill-conditioned ADMM system')
        return dict(method='cholesky', factor=fac, rho=rho)
    except np.linalg.LinAlgError:
        if np.linalg.norm(K - K.T) / (1 + np.linalg.norm(K)) < 1e-14:
            if 'eig' not in cache:
                cache['eig'] = eigh(K, check_finite=False)
            e, V = cache['eig']
            singular = np.abs(e)
        else:
            if 'admm_svd' not in cache:
                _, singular, Vh = svd(K, full_matrices=False, check_finite=False)
                cache['admm_svd'] = (singular, Vh.T)
            singular, V = cache['admm_svd']
        den = np.hypot(np.sqrt(2.0) * singular, np.sqrt(rho))
        f = (np.sqrt(2.0) * singular / den) ** 2
        h = (np.sqrt(rho) / den) ** 2
        base = np.eye(len(K)) - V * h @ V.T if f.mean() > 0.5 else V * f @ V.T
        return dict(method='stable_spectral_affine', V=V, h=h, base=base, rho=rho)

def _admm_primal(linear, B, J, U):
    if linear['method'] == 'cholesky':
        return cho_solve(linear['factor'], B + linear['rho'] * (J - U), check_finite=False)
    V = linear['V']
    return linear['base'] + V @ (linear['h'][:, None] * (V.T @ (J - U)))

def _kkt4(K, J, lam):
    E = K - K @ J
    G = -2 * K.T @ E
    norms = row_norm(J)
    viol = row_norm(G)
    active = norms > 0
    viol[active] = row_norm(G[active] + lam * J[active] / norms[active, None])
    viol[~active] = np.maximum(0, viol[~active] - lam)
    return (E, float(viol.max()), float(viol.max() / (1 + row_norm(G).max() + lam)))

def _candidate4(K, Z, J, lam, grad_tol, consider_z):
    E, abs_j, kkt_j = _kkt4(K, J, lam)
    chosen = J
    key = 'J'
    absolute = abs_j
    scaled = kkt_j
    kkt_z = None
    if consider_z and kkt_j > grad_tol:
        Ez, abs_z, kkt_z = _kkt4(K, Z, lam)
        if kkt_z < kkt_j:
            chosen = Z
            E = Ez
            absolute = abs_z
            scaled = kkt_z
            key = 'Z'
    return (chosen, E, absolute, scaled, key, kkt_j, kkt_z)

# Self-representation

def solve_representation(K, lambda_reg, variant='v2', config=None, state=None, checkpoint=None, callback=None, cache=None):
    c = config or SolverConfig()
    _validate_config(c)
    K = np.asarray(K, dtype=np.float64)
    lam = float(lambda_reg)
    if K.ndim != 2 or K.shape[0] != K.shape[1] or (not np.isfinite(K).all()):
        raise ValueError('K must be a finite square matrix')
    if not np.isfinite(lam) or lam <= 0 or variant not in ('v1', 'v2', 'v3', 'v4'):
        raise ValueError('Need positive lambda and variant v1/v2/v3/v4')
    n = len(K)
    t0 = perf_counter()
    cache = {} if cache is None else cache
    if cache.get('_K') is not K:
        cache.clear()
        cache['_K'] = K
    hist = []
    resumed = state is not None
    start = int(state.get('iteration', 0)) if state else 0
    if variant != 'v2' and start >= c.max_iter:
        raise ValueError('Restart iteration already at requested maximum')
    elapsed_prior = float(state.get('elapsed_seconds', 0)) if state else 0.0
    Z = np.array(state['Z'], copy=True) if state else np.zeros_like(K)
    if Z.shape != K.shape or not np.isfinite(Z).all():
        raise ValueError('Invalid restart state')
    methods = {}
    maxback = 0.0
    maxinc = 0.0
    converged = False
    reason = 'iteration_limit'
    info = dict(variant=variant, lambda_reg=lam, config=asdict(c), resumed=resumed, initialization='zero', converged=False, stop_reason=reason)
    if variant == 'v2':
        asym = np.linalg.norm(K - K.T) / (1 + np.linalg.norm(K))
        if asym < 1e-14:
            if 'eig' not in cache:
                cache['eig'] = eigh(K, check_finite=False)
            e, V = cache['eig']
            f = (np.abs(e) / np.hypot(e, np.sqrt(lam))) ** 2
            Z = V * f @ V.T
            method = 'symmetric_spectral_ridge'
        else:
            U, s, Vh = svd(K, full_matrices=False, check_finite=False)
            f = (s / np.hypot(s, np.sqrt(lam))) ** 2
            Z = Vh.T * f @ Vh
            method = 'svd_ridge'
        E = K - K @ Z
        grad, gn = smooth_gradient(K, Z, lam, variant, c.eps, E)
        converged = gn <= c.grad_tol
        reason = 'stationary' if converged else 'numerical_residual'
        info.update(objective(K, Z, lam, variant, c.eps, E))
        info.update(iterations=1, scaled_gradient=gn, gradient_norm=float(np.linalg.norm(grad)), linear_method=method)
        final_state = {'Z': Z, 'iteration': 1}
    elif variant in ('v1', 'v3'):
        E = K - K @ Z
        previous = objective(K, Z, lam, variant, c.eps, E)['centered_objective']
        for it in range(start + 1, c.max_iter + 1):
            gl = 0.5 / np.hypot(row_norm(E), c.eps)
            H = K.T @ (gl[:, None] * K)
            reg = lam * 0.5 / np.hypot(row_norm(Z), c.eps) if variant == 'v1' else np.full(n, lam)
            new, method, back = _quadratic_solve(K, gl, reg, H)
            methods[method] = methods.get(method, 0) + 1
            maxback = max(maxback, back)
            step = float(np.linalg.norm(new - Z) / (1 + np.linalg.norm(Z)))
            Z = new
            E = K - K @ Z
            obj = objective(K, Z, lam, variant, c.eps, E)
            increase = obj['centered_objective'] - previous
            maxinc = max(maxinc, increase)
            monotone_limit = 1e-09 * (1 + abs(previous) + abs(obj['centered_objective']))
            relobj = abs(increase) / (1 + abs(previous))
            previous = obj['centered_objective']
            diagnostic = it == start + 1 or it % c.diagnostic_every == 0 or step < c.step_tol or (it == c.max_iter) or (increase > monotone_limit)
            if diagnostic:
                grad, gn = smooth_gradient(K, Z, lam, variant, c.eps, E)
                row = dict(iteration=it, **obj, relative_step=step, relative_objective_change=float(relobj), scaled_gradient=gn, gradient_norm=float(np.linalg.norm(grad)), backward_error=back)
                hist.append(row)
                info.update(row)
                converged = gn <= c.grad_tol and step <= c.step_tol
                if increase > monotone_limit:
                    reason = 'objective_increase'
                    converged = False
                    break
                if callback:
                    callback(dict(info))
                if converged:
                    reason = 'stationary'
                    break
            elapsed = perf_counter() - t0
            if checkpoint and it % c.checkpoint_every == 0:
                checkpoint(dict(Z=Z, iteration=it, elapsed_seconds=elapsed_prior + elapsed), dict(info), hist)
            if c.max_seconds and elapsed >= c.max_seconds:
                reason = 'time_limit'
                break
        else:
            it = c.max_iter
        grad, gn = smooth_gradient(K, Z, lam, variant, c.eps, E)
        info.update(objective(K, Z, lam, variant, c.eps, E))
        info.update(iterations=it, scaled_gradient=gn, gradient_norm=float(np.linalg.norm(grad)), relative_step=step, linear_methods=methods, max_backward_error=maxback, max_objective_increase=maxinc)
        final_state = {'Z': Z, 'iteration': it}
    else:
        if 'KTK' not in cache:
            cache['KTK'] = K.T @ K
        B = 2 * cache['KTK']
        rho = float(state.get('rho', c.rho)) if state else c.rho
        updates_total = int(state.get('rho_updates_total', 0)) if state else 0
        changes = []
        threshold = float(row_norm(B).max())
        info.update(zero_solution_threshold=threshold, rho_initial=rho, rho_policy='finite_residual_balance' if c.adaptive_rho else 'fixed')
        if lam >= threshold:
            Z = np.zeros_like(K)
            E = K.copy()
            converged = True
            reason = 'exact_zero_kkt_certificate'
            info.update(iterations=0, primal=0.0, dual=0.0, scaled_kkt=0.0, kkt_row_violation=0.0)
            info.update(objective(K, Z, lam, variant, c.eps, E))
            final_state = {'Z': Z, 'J': Z.copy(), 'U': Z.copy(), 'iteration': 0, 'rho': rho, 'rho_updates_total': updates_total}
        else:
            J = np.array(state.get('J', Z), copy=True) if state else Z.copy()
            U = np.array(state.get('U', np.zeros_like(K)), copy=True) if state else Z.copy()
            linear = _admm_setup(K, B, rho, cache)
            methods[linear['method']] = 0
            for it in range(start + 1, c.max_iter + 1):
                Z = _admm_primal(linear, B, J, U)
                methods[linear['method']] = methods.get(linear['method'], 0) + 1
                V = Z + U
                norms = row_norm(V)
                factors = np.zeros(n)
                positive = norms > lam / rho
                factors[positive] = 1 - lam / rho / norms[positive]
                new = factors[:, None] * V
                r = float(np.linalg.norm(Z - new))
                dual = float(rho * np.linalg.norm(new - J))
                U += Z - new
                J = new
                er = n * c.admm_atol + c.admm_rtol * max(np.linalg.norm(Z), np.linalg.norm(J))
                es = n * c.admm_atol + c.admm_rtol * np.linalg.norm(rho * U)
                residual_ok = r <= er and dual <= es
                if it == start + 1 or it % c.diagnostic_every == 0 or residual_ok or (it == c.max_iter):
                    output, E, kkt_abs, kkt, output_key, kkt_j, kkt_z = _candidate4(K, Z, J, lam, c.grad_tol, residual_ok)
                    row = dict(iteration=it, rho=rho, primal=r, dual=dual, primal_threshold=float(er), dual_threshold=float(es), kkt_row_violation=kkt_abs, scaled_kkt=kkt, output_variable=output_key, auxiliary_scaled_kkt=kkt_j, primal_scaled_kkt=kkt_z, **objective(K, output, lam, variant, c.eps, E))
                    hist.append(row)
                    info.update(row)
                    converged = residual_ok and kkt <= c.grad_tol
                    if callback:
                        callback(dict(info))
                    if converged:
                        reason = 'admm_residuals_and_kkt'
                        break
                if c.adaptive_rho and it <= c.rho_adapt_until and (it % c.rho_update_interval == 0):
                    target = rho
                    if r > 10 * dual:
                        target = min(100000000.0, 2 * rho)
                    elif dual > 10 * r:
                        target = max(1e-08, rho / 2)
                    if target != rho:
                        old = rho
                        U *= old / target
                        rho = target
                        updates_total += 1
                        changes.append(dict(iteration=it, old=old, new=rho, unscaled_dual_preserved=True))
                        linear = _admm_setup(K, B, rho, cache)
                if checkpoint and it % c.checkpoint_every == 0:
                    checkpoint(dict(Z=Z, J=J, U=U, iteration=it, rho=rho, rho_updates_total=updates_total, elapsed_seconds=elapsed_prior + perf_counter() - t0), dict(info), hist)
                if c.max_seconds and perf_counter() - t0 >= c.max_seconds:
                    reason = 'time_limit'
                    break
            else:
                it = c.max_iter
            output, E, kkt_abs, kkt, output_key, kkt_j, kkt_z = _candidate4(K, Z, J, lam, c.grad_tol, True)
            converged = bool(r <= er and dual <= es and (kkt <= c.grad_tol))
            if converged:
                reason = 'admm_residuals_and_kkt'
            info.update(kkt_row_violation=kkt_abs, scaled_kkt=kkt, primal_threshold=float(er), dual_threshold=float(es), auxiliary_scaled_kkt=kkt_j, primal_scaled_kkt=kkt_z)
            info.update(objective(K, output, lam, variant, c.eps, E))
            info.update(iterations=it, primal=r, dual=dual, z_j_gap=float(np.linalg.norm(Z - J)), output_variable=output_key)
            final_state = {'Z': Z, 'J': J, 'U': U, 'iteration': it, 'rho': rho, 'rho_updates_total': updates_total}
            Z = output
        info.update(rho_final=rho, rho_changes=changes, rho_updates_total=updates_total, linear_methods=methods)
    if not np.isfinite(Z).all():
        raise FloatingPointError('Nonfinite representation')
    info.update(converged=bool(converged), stop_reason=reason, solve_seconds=perf_counter() - t0, total_solve_seconds=elapsed_prior + perf_counter() - t0)
    final_state['elapsed_seconds'] = info['total_solve_seconds']
    if checkpoint:
        checkpoint(final_state, dict(info), hist)
    return SolverResult(Z, E, info, hist, final_state)

# Random walk and outlier scores

def transition_matrix(Z):
    Z = np.asarray(Z, dtype=float)
    if Z.ndim != 2 or Z.shape[0] != Z.shape[1] or (not len(Z)) or (not np.isfinite(Z).all()):
        raise ValueError('Finite nonempty square Z required')
    A = 0.5 * np.abs(Z) + 0.5 * np.abs(Z.T)
    maxima = A.max(axis=1)
    positive = maxima > 0
    A[positive] /= maxima[positive, None]
    deg = A.sum(axis=1)
    A[positive] /= deg[positive, None]
    A[~positive] = 1 / len(Z)
    err = float(np.max(np.abs(A.sum(axis=1) - 1)))
    if A.min() < 0 or err > 1e-12:
        raise FloatingPointError('Invalid stochastic matrix')
    return (A, dict(zero_degree_count=int((~positive).sum()), row_sum_error=err, min_transition=float(A.min())))

def normalize_structural(pi):
    values = -np.asarray(pi, dtype=float)
    span = float(np.ptp(values))
    floor = 64 * np.finfo(float).eps * max(float(np.max(np.abs(values))), 1 / len(values))
    if span <= floor:
        return (np.zeros_like(values), dict(structural_span=span, normalization_floor=floor, constant_structural=True))
    return ((values - values.min()) / span, dict(structural_span=span, normalization_floor=floor, constant_structural=False))

def score_representation(Z, weight, restart=0.85, pr_tol=1e-12, max_iter=1000):
    if not 0 < restart < 1 or not np.isfinite(pr_tol) or pr_tol <= 0 or (max_iter < 1):
        raise ValueError('Invalid random-walk settings')
    t = perf_counter()
    P, info = transition_matrix(Z)
    n = len(P)
    weight = np.asarray(weight, dtype=float)
    if weight.shape != (n,) or not np.isfinite(weight).all():
        raise ValueError('Invalid weights')
    pi = np.ones(n) / n
    for it in range(1, max_iter + 1):
        new = (1 - restart) * (pi @ P) + restart / n
        delta = float(np.linalg.norm(new - pi))
        pi = new
        if delta < pr_tol:
            break
    resid = float(np.linalg.norm(pi - ((1 - restart) * (pi @ P) + restart / n)))
    norm, ni = normalize_structural(pi)
    scores = norm * weight
    info.update(ni)
    info.update(restart=restart, pr_tol=pr_tol, pr_iterations=it, pr_converged=bool(delta < pr_tol), stationary_residual=resid, probability_mass_error=float(abs(pi.sum() - 1)), min_probability=float(pi.min()), score_min=float(scores.min()), score_max=float(scores.max()), score_seconds=perf_counter() - t)
    if not np.isfinite(scores).all() or not info['pr_converged'] or info['probability_mass_error'] > 1e-10:
        raise FloatingPointError('Invalid/nonconverged walk')
    return (scores, info, pi)

# Public interfaces

def KSRA(data, lambda_val, sigma=None, solver='v2', return_info=False, numeric_mask=None, restart=0.85, pr_tol=1e-12, block_size=256, **solver_kwargs):
    profile = prepare_profiles(data, sigma, numeric_mask, block_size)
    result = solve_representation(profile.K, lambda_val, solver, SolverConfig(**solver_kwargs))
    if not result.info['converged']:
        raise ConvergenceError(str(result.info))
    scores, walk, _ = score_representation(result.Z, profile.weight, restart, pr_tol)
    if return_info:
        return (scores, dict(profile=profile.info, solver=result.info, walk=walk))
    return scores

def KSR_v1(K, lambda_reg, return_info=False, **kwargs):
    return _core(K, lambda_reg, 'v1', return_info, kwargs)

def KSR_v2(K, lambda_reg, return_info=False, **kwargs):
    return _core(K, lambda_reg, 'v2', return_info, kwargs)

def KSR_v3(K, lambda_reg, return_info=False, **kwargs):
    return _core(K, lambda_reg, 'v3', return_info, kwargs)

def KSR_v4(K, lambda_reg, return_info=False, **kwargs):
    return _core(K, lambda_reg, 'v4', return_info, kwargs)

def _core(K, lam, variant, return_info, kwargs):
    if 'iter_num' in kwargs:
        kwargs['max_iter'] = kwargs.pop('iter_num')
    if 'tol' in kwargs:
        raise TypeError('Use explicit step_tol/grad_tol or admm_atol/admm_rtol, not ambiguous tol')
    result = solve_representation(K, lam, variant, SolverConfig(**kwargs))
    if return_info:
        return (result.Z, result.E, result.info)
    if not result.info['converged']:
        raise ConvergenceError(str(result.info))
    return (result.Z, result.E)
