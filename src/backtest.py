import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.critical.rolling_metrics import rolling_susceptibility
from src.critical.critical_overlay import apply_critical_overlay


# ============================================================
# MÉTRICAS
# ============================================================

def sharpe_ratio(returns, annualization=252):
    mean = np.nanmean(returns)
    std = np.nanstd(returns)
    if std == 0:
        return 0.0
    return (mean / std) * np.sqrt(annualization)


def max_drawdown(equity_curve):
    cumulative_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - cumulative_max) / cumulative_max
    return np.min(drawdown)


# ============================================================
# BACKTEST PRINCIPAL
# ============================================================

def run_backtest(
    returns,
    use_critical_overlay=True,
    overlay_mode="dynamic_percentile",
    window=252,
    percentile=80,
    hedge_exposure=0.5,
    plot=True
):

    returns = np.asarray(returns)

    # --------------------------------------------------------
    # EQUITY ORIGINAL
    # --------------------------------------------------------

    equity_original = np.cumprod(1 + returns)

    sharpe_original = sharpe_ratio(returns)
    mdd_original = max_drawdown(equity_original)

    # --------------------------------------------------------
    # CRITICAL OVERLAY
    # --------------------------------------------------------

    if use_critical_overlay:

        chi = rolling_susceptibility(returns, window=window)

        adjusted_returns, hedge = apply_critical_overlay(
            returns,
            chi,
            mode=overlay_mode,
            window=window,
            percentile=percentile,
            hedge_exposure=hedge_exposure
        )

        equity_overlay = np.cumprod(1 + adjusted_returns)

        sharpe_overlay = sharpe_ratio(adjusted_returns)
        mdd_overlay = max_drawdown(equity_overlay)

    else:
        equity_overlay = None
        sharpe_overlay = None
        mdd_overlay = None
        hedge = None

    # --------------------------------------------------------
    # RESULTADOS
    # --------------------------------------------------------

    results = {
        "sharpe_original": sharpe_original,
        "max_dd_original": mdd_original,
        "sharpe_overlay": sharpe_overlay,
        "max_dd_overlay": mdd_overlay,
    }

    # --------------------------------------------------------
    # PLOT
    # --------------------------------------------------------

    if plot:

        plt.figure(figsize=(10, 6))
        plt.plot(equity_original, label="Original")

        if use_critical_overlay:
            plt.plot(equity_overlay, label="Critical Overlay")

        plt.legend()
        plt.title("Equity Curve Comparison")
        plt.tight_layout()
        plt.show()

    return results


# ============================================================
# EXECUÇÃO DIRETA (TESTE)
# ============================================================

if __name__ == "__main__":

    # Simulação dummy
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.01, 2000)

    results = run_backtest(
        returns,
        use_critical_overlay=True,
        overlay_mode="dynamic_percentile",
        window=252,
        percentile=80,
        hedge_exposure=0.4
    )

    print("\n=== RESULTADOS ===")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")