import numpy as np
import torch
from scipy.io import loadmat
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.decomposition import KernelPCA
from sklearn.metrics import roc_auc_score


# -----------------Variant 1-----------------
def KSR_v1(K, lambda_reg, iter_num=50, tol=1e-6):
    """
    Core self-representation optimization using IRLS for l2,1 norm.
    Solves: min ||K - KZ||_21 + lambda*||Z||_21
    """
    n, m = K.shape

    Z = np.zeros((n, n))
    obj = []

    IRLS_STABILITY_M = 1e-4

    for iter_idx in range(iter_num):
        zc = np.sqrt(np.sum(Z * Z, axis=1, keepdims=True))

        Gr_inv = 2 * np.maximum(zc, IRLS_STABILITY_M)
        Gr = 1.0 / Gr_inv

        E = K - K @ Z

        ec = np.sqrt(np.sum(E * E, axis=0, keepdims=True))

        Gl_inv = 2 * np.maximum(ec, IRLS_STABILITY_M)
        Gl = 1.0 / Gl_inv.T

        G_R = np.diag(Gr.flatten())
        G_L = np.diag(Gl.flatten())

        KGKt = K @ G_L @ K.T

        G_R_inv_diag = 1.0 / Gr.flatten()
        G_R_inv = np.diag(G_R_inv_diag)

        A = G_R_inv @ KGKt + lambda_reg * np.eye(n)

        b = G_R_inv @ KGKt

        try:
            Z = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            Z = np.linalg.pinv(A) @ b

        E_obj = K - K @ Z
        ec_obj = np.sqrt(np.sum(E_obj * E_obj, axis=0))

        zc_obj = np.sqrt(np.sum(Z * Z, axis=1))

        current_obj = np.sum(ec_obj) + lambda_reg * np.sum(zc_obj)
        obj.append(current_obj)

        if iter_idx > 0 and np.abs(obj[-1] - obj[-2]) < tol:
            print(f'Converged at iteration = {iter_idx + 1}')
            break

    MAX_W_VALUE = 1e10

    Z[np.isinf(Z) | (Z > MAX_W_VALUE)] = 0.0
    Z[np.isnan(Z)] = 0.0

    E_final = K.T @ Z - K.T

    return Z, E_final

# -----------------Varian 2-----------------
def KSR_v2(K, lambda_reg):
    """
    Solve min ||K - KZ||_F^2 + lambda * ||Z||_F^2
    """
    n, m = K.shape
    KTK = K.T @ K
    I = np.eye(n)
    A = KTK + lambda_reg * I
    b = KTK

    try:
        Z = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        Z = np.linalg.pinv(A) @ b
    E = K.T @ Z - K.T
    return Z, E

# -----------------Variant 3-----------------
def KSR_v3(K, lambda_reg, iter_num=50, tol=1e-6, eps=1e-12):
    """
    Solve: min_Z ||K - KZ||_{2,1} + lambda * ||Z||_F^2
    Using IRLS on the l2,1 loss and closed-form solve for W each iteration.
    """
    n, m = K.shape

    Z = np.zeros((n, n))
    obj_vals = []

    for it in range(iter_num):
        E = K - K @ Z

        col_norms = np.sqrt(np.sum(E**2, axis=0, keepdims=False) + eps)
        g = 0.5 / col_norms
        G = np.diag(g)

        KGKt = K @ G @ K.T
        A = KGKt + lambda_reg * np.eye(n)
        B = KGKt

        try:
            Z_new = np.linalg.solve(A, B)
        except np.linalg.LinAlgError:
            Z_new = np.linalg.pinv(A) @ B

        diff = np.linalg.norm(Z_new - Z, ord='fro')
        denom = 1.0 + np.linalg.norm(Z, ord='fro')
        Z = Z_new

        E = K - K @ Z
        approx_loss = np.sum(np.sqrt(np.sum(E**2, axis=0) + eps))
        obj = approx_loss + lambda_reg * np.linalg.norm(Z, ord='fro')**2
        obj_vals.append(obj)

        if it > 0 and abs(obj_vals[-1] - obj_vals[-2]) < tol:
            print(f"Converged at iteration {it+1}")
            break

    E = K - K @ Z
    return Z, E

# -----------------Varian 4-----------------
def KSR_v4(K, lambda_reg, rho=1.0, iter_num=100, tol=1e-6, eps=1e-12):
    """
    Solve: min_Z ||K - KZ||_F^2 + lambda * ||Z||_{2,1}
    Using ADMM with splitting Z=J.

    Variables:
        Z: main variable
        J: prox variable
        U: scaled dual variable
    """
    n, m = K.shape

    Z = np.zeros((n, n))
    J = np.zeros((n, n))
    U = np.zeros((n, n))

    KTK = K.T @ K * 2
    I = np.eye(n)

    try:
        A = KTK + rho * I
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        A_inv = None

    for it in range(iter_num):
        # ------------------ W-update ------------------
        B = 2 * (K.T @ K) + rho * (J - U)

        if A_inv is not None:
            Z_new = A_inv @ B
        else:
            try:
                Z_new = np.linalg.solve(KTK + rho * I, B)
            except np.linalg.LinAlgError:
                Z_new = np.linalg.pinv(KTK + rho * I) @ B

        # ------------------ Z-update (row-wise shrinkage) ------------------
        V = Z_new + U
        J_new = np.zeros_like(J)

        tau = lambda_reg / rho
        for i in range(n):
            row = V[i, :]
            norm_row = np.linalg.norm(row, 2)
            if norm_row > tau:
                J_new[i, :] = (1 - tau / norm_row) * row
            else:
                J_new[i, :] = 0

        # ------------------ Dual update ------------------
        U_new = U + (Z_new - J_new)

        # ------------------ Convergence check ------------------
        r_norm = np.linalg.norm(Z_new - J_new, 'fro')
        s_norm = np.linalg.norm(rho * (J_new - J), 'fro')

        Z, J, U = Z_new, J_new, U_new

        if r_norm < tol and s_norm < tol:
            break

    E = K - K @ Z
    return Z, E


# ==================== Hamming  ====================
def pdist2_hamming_np(X, Y):
    """Hamming distance for nominal features (0/1 or categorical <=1)"""
    X = np.asarray(X)
    Y = np.asarray(Y)
    if X.size == 0 or Y.size == 0:
        return np.zeros((X.shape[0], Y.shape[0]))

    X_exp = X[:, np.newaxis, :]
    Y_exp = Y[np.newaxis, :, :]
    diff = np.abs(X_exp - Y_exp) > 1e-9
    hamming = diff.sum(axis=2) / X.shape[1]
    return hamming


# ==================== KSRA Wrapping ====================
def KSRA(data, lambda_val, sigma=None, solver='v2', **solver_kwargs):
    data = np.asarray(data, dtype=np.float64)
    n, m = data.shape

    ID = np.all(data <= 1 + 1e-9, axis=0)
    num_idx = np.where(ID)[0]
    nom_idx = np.where(~ID)[0]

    num_dis = np.zeros((n, n))
    if len(num_idx) > 0:
        num_data = data[:, num_idx]
        num_dis = cdist(num_data, num_data, metric='euclidean')

    nom_dis = np.zeros((n, n))
    if len(nom_idx) > 0:
        nom_data = data[:, nom_idx]
        nom_dis = pdist2_hamming_np(nom_data, nom_data)

    distMatrix = num_dis + nom_dis
    non_diag_mask = ~np.eye(n, dtype=bool)
    if sigma is None:
        sigma = np.mean(distMatrix[non_diag_mask])

    kernel_matrix = np.exp(-distMatrix**2 / (2 * sigma**2))

    row_sums_K = kernel_matrix.sum(axis=1)
    Weight = 1 - (row_sums_K / n) ** (1 / 3)
    Weight = Weight.reshape(-1, 1)

    # ----------------- Choose Solver -----------------
    if solver == 'v1':
        Z, E = KSR_v1(kernel_matrix, lambda_val, **solver_kwargs)
    elif solver == 'v2':
        Z, E = KSR_v2(kernel_matrix, lambda_val, **solver_kwargs)
    elif solver == 'v3':
        Z, E = KSR_v3(kernel_matrix, lambda_val, **solver_kwargs)
    elif solver == 'v4':
        Z, E = KSR_v4(kernel_matrix, lambda_val)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # ----------------- PageRank Propagation -----------------
    NUMERICAL_STABILITY_EPS = 1e-8

    Z_sym = (Z + Z.T) / 2
    row_sums = Z_sym.sum(axis=1, keepdims=True)

    row_sums_clipped = np.maximum(row_sums, NUMERICAL_STABILITY_EPS)
    P = Z_sym / row_sums_clipped

    d, tol_pr, max_iter = 0.85, 1e-6, 1000
    pi = np.ones(n) / n

    for i in range(max_iter):
        if not np.all(np.isfinite(pi)):
            break

        pi_new = d * (1 / n) + (1 - d) * (pi @ P)

        if not np.all(np.isfinite(pi_new)):
            break

        if np.linalg.norm(pi_new - pi, 2) < tol_pr:
            pi = pi_new
            break
        pi = pi_new

    out_scores = -pi
    min_val = out_scores.min()
    max_val = out_scores.max()

    range_val = np.maximum(max_val - min_val, NUMERICAL_STABILITY_EPS)
    normalized_scores = (out_scores - min_val) / range_val
    Fuse_out_scores = normalized_scores * Weight.ravel()

    return Fuse_out_scores