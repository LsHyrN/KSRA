import numpy as np
import torch
from scipy.io import loadmat
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.decomposition import KernelPCA
from sklearn.metrics import roc_auc_score


# -----------------Variant 1-----------------
def KSR_v1(X, lambda_reg, iter_num=50, tol=1e-6):
    """
    Core self-representation optimization (NumPy version) using IRLS for l2,1 norm.
    Solves: min ||X - XW||_21 + lambda*||W||_21

    Inputs:
        X (np.ndarray): n x m matrix (kernel matrix in KSRA framework).
        lambda_reg (float): Regularization parameter (lambda).
        iter_num (int): Maximum iterations.
        tol (float): Convergence tolerance.

    Outputs:
        W (np.ndarray): n x n self-representation matrix.
        E (np.ndarray): Residual matrix (X - XW)' (m x n).
    """
    n, m = X.shape

    W = np.zeros((n, n))
    obj = []

    IRLS_STABILITY_M = 1e-4

    for iter_idx in range(iter_num):
        wc = np.sqrt(np.sum(W * W, axis=1, keepdims=True))

        Gr_inv = 2 * np.maximum(wc, IRLS_STABILITY_M)
        Gr = 1.0 / Gr_inv

        E = X - X @ W

        ec = np.sqrt(np.sum(E * E, axis=0, keepdims=True))

        Gl_inv = 2 * np.maximum(ec, IRLS_STABILITY_M)
        Gl = 1.0 / Gl_inv.T

        G_R = np.diag(Gr.flatten())
        G_L = np.diag(Gl.flatten())

        XGXt = X @ G_L @ X.T

        G_R_inv_diag = 1.0 / Gr.flatten()
        G_R_inv = np.diag(G_R_inv_diag)

        A = G_R_inv @ XGXt + lambda_reg * np.eye(n)

        b = G_R_inv @ XGXt

        try:
            W = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            W = np.linalg.pinv(A) @ b

        E_obj = X - X @ W
        ec_obj = np.sqrt(np.sum(E_obj * E_obj, axis=0))

        wc_obj = np.sqrt(np.sum(W * W, axis=1))

        current_obj = np.sum(ec_obj) + lambda_reg * np.sum(wc_obj)
        obj.append(current_obj)

        if iter_idx > 0 and np.abs(obj[-1] - obj[-2]) < tol:
            print(f'Converged at iteration = {iter_idx + 1}')
            break

    MAX_W_VALUE = 1e10

    W[np.isinf(W) | (W > MAX_W_VALUE)] = 0.0
    W[np.isnan(W)] = 0.0

    E_final = X.T @ W - X.T

    return W, E_final

# -----------------Varian 2-----------------
def KSR_v2(X, lambda_reg):
    """
    Solve min ||X - XW||_F^2 + lambda * ||W||_F^2
    """
    n, m = X.shape
    XTX = X.T @ X
    I = np.eye(n)
    A = XTX + lambda_reg * I
    b = XTX

    try:
        W = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        W = np.linalg.pinv(A) @ b
    E = X.T @ W - X.T
    return W, E

# -----------------Variant 3-----------------
def KSR_v3(X, lambda_reg, iter_num=50, tol=1e-6, eps=1e-12):
    """
    Solve: min_W ||X - XW||_{2,1} + lambda * ||W||_F^2
    Using IRLS on the l2,1 loss and closed-form solve for W each iteration.

    Returns W (n x n) and residual E = X - XW (n x n)
    """
    n, m = X.shape

    W = np.zeros((n, n))
    obj_vals = []

    for it in range(iter_num):
        E = X - X @ W

        col_norms = np.sqrt(np.sum(E**2, axis=0, keepdims=False) + eps)
        g = 0.5 / col_norms
        G = np.diag(g)

        XGXt = X @ G @ X.T
        A = XGXt + lambda_reg * np.eye(n)
        B = XGXt

        try:
            W_new = np.linalg.solve(A, B)
        except np.linalg.LinAlgError:
            W_new = np.linalg.pinv(A) @ B

        diff = np.linalg.norm(W_new - W, ord='fro')
        denom = 1.0 + np.linalg.norm(W, ord='fro')
        W = W_new

        E = X - X @ W
        approx_loss = np.sum(np.sqrt(np.sum(E**2, axis=0) + eps))
        obj = approx_loss + lambda_reg * np.linalg.norm(W, ord='fro')**2
        obj_vals.append(obj)

        if it > 0 and abs(obj_vals[-1] - obj_vals[-2]) < tol:
            print(f"Converged at iteration {it+1}")
            break

    E = X - X @ W
    return W, E

# -----------------Varian 4-----------------
def KSR_v4(X, lambda_reg, rho=1.0, iter_num=100, tol=1e-6, eps=1e-12):
    """
    Solve: min_W ||X - XW||_F^2 + lambda * ||W||_{2,1}
    Using ADMM with splitting W=Z.

    Variables:
        W: main variable
        Z: prox variable (group-lasso style)
        U: scaled dual variable
    """
    n, m = X.shape

    W = np.zeros((n, n))
    Z = np.zeros((n, n))
    U = np.zeros((n, n))

    XTX = X.T @ X * 2
    I = np.eye(n)

    try:
        A = XTX + rho * I
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        A_inv = None

    for it in range(iter_num):
        # ------------------ W-update ------------------
        B = 2 * (X.T @ X) + rho * (Z - U)

        if A_inv is not None:
            W_new = A_inv @ B
        else:
            try:
                W_new = np.linalg.solve(XTX + rho * I, B)
            except np.linalg.LinAlgError:
                W_new = np.linalg.pinv(XTX + rho * I) @ B

        # ------------------ Z-update (row-wise shrinkage) ------------------
        Y = W_new + U
        Z_new = np.zeros_like(Z)

        tau = lambda_reg / rho
        for i in range(n):
            row = Y[i, :]
            norm_row = np.linalg.norm(row, 2)
            if norm_row > tau:
                Z_new[i, :] = (1 - tau / norm_row) * row
            else:
                Z_new[i, :] = 0

        # ------------------ Dual update ------------------
        U_new = U + (W_new - Z_new)

        # ------------------ Convergence check ------------------
        r_norm = np.linalg.norm(W_new - Z_new, 'fro')
        s_norm = np.linalg.norm(rho * (Z_new - Z), 'fro')

        W, Z, U = W_new, Z_new, U_new

        if r_norm < tol and s_norm < tol:
            break

    E = X - X @ W
    return W, E


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
        W, E = KSR_v1(kernel_matrix, lambda_val, **solver_kwargs)
    elif solver == 'v2':
        W, E = KSR_v2(kernel_matrix, lambda_val, **solver_kwargs)
    elif solver == 'v3':
        W, E = KSR_v3(kernel_matrix, lambda_val, **solver_kwargs)
    elif solver == 'v4':
        W, E = KSR_v4(kernel_matrix, lambda_val)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # ----------------- PageRank Propagation -----------------
    NUMERICAL_STABILITY_EPS = 1e-8

    W_sym = (W + W.T) / 2
    row_sums = W_sym.sum(axis=1, keepdims=True)

    row_sums_clipped = np.maximum(row_sums, NUMERICAL_STABILITY_EPS)
    P = W_sym / row_sums_clipped

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