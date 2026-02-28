import numpy as np


# ==========================================================
# MÉTRICAS INSTITUCIONAIS COMPLETAS
# ==========================================================

def compute_metrics(portfolio_returns, benchmark_returns=None):

    portfolio_returns = np.array(portfolio_returns)

    # ---------------------------
    # CAGR
    # ---------------------------
    cumulative = np.prod(1 + portfolio_returns)
    years = len(portfolio_returns) / 252
    cagr = cumulative ** (1 / years) - 1 if years > 0 else 0

    # ---------------------------
    # Sharpe
    # ---------------------------
    mean = np.mean(portfolio_returns) * 252
    std = np.std(portfolio_returns) * np.sqrt(252)
    sharpe = mean / (std + 1e-8)

    # ---------------------------
    # Sortino
    # ---------------------------
    downside = portfolio_returns[portfolio_returns < 0]
    downside_std = np.std(downside) * np.sqrt(252)
    sortino = mean / (downside_std + 1e-8)

    # ---------------------------
    # Max Drawdown
    # ---------------------------
    equity = np.cumprod(1 + portfolio_returns)
    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1
    max_dd = drawdown.min()

    # ---------------------------
    # Alpha (vs benchmark)
    # ---------------------------
    alpha = 0
    if benchmark_returns is not None:
        benchmark_returns = np.array(benchmark_returns)
        alpha = (
            np.mean(portfolio_returns - benchmark_returns) * 252
        )

    return {
        "cagr": float(cagr),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(max_dd),
        "alpha": float(alpha)
    }