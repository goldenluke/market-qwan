import numpy as np


def eigenvalue_stress(returns_matrix):

    corr = np.corrcoef(returns_matrix.T)
    eigvals = np.linalg.eigvalsh(corr)

    lambda_max = np.max(eigvals)
    lambda_mean = np.mean(eigvals)

    stress_index = lambda_max / lambda_mean

    return {
        "lambda_max": lambda_max,
        "stress_index": stress_index
    }