import numpy as np

# ==========================================================
# COVARIÂNCIA ROBUSTA (Shrinkage + Clipping)
# ==========================================================

def robust_covariance(returns):

    cov = np.cov(returns.T)

    # Eigenvalue clipping
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 1e-6, None)

    cov_clipped = eigvecs @ np.diag(eigvals) @ eigvecs.T

    return cov_clipped


# ==========================================================
# RISK PARITY ESTRUTURAL
# ==========================================================

def risk_parity_weights(cov):

    n = cov.shape[0]
    w = np.ones(n) / n

    for _ in range(300):

        portfolio_var = w @ cov @ w
        marginal = cov @ w
        risk_contrib = w * marginal

        target = portfolio_var / n

        grad = risk_contrib - target

        w -= 0.01 * grad
        w = np.clip(w, 0, None)

        if w.sum() == 0:
            w = np.ones(n) / n
        else:
            w /= w.sum()

    return w