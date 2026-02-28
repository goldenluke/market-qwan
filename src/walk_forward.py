import numpy as np
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
# OTIMIZAÇÃO DO THRESHOLD (NO TREINO)
# ============================================================

def optimize_threshold(train_returns, chi_train,
                       percentiles=[70, 75, 80, 85, 90],
                       hedge_exposure=0.5):

    best_sharpe = -np.inf
    best_percentile = percentiles[0]

    for p in percentiles:

        adjusted, _ = apply_critical_overlay(
            train_returns,
            chi_train,
            mode="threshold",
            threshold=np.nanpercentile(chi_train, p),
            hedge_exposure=hedge_exposure
        )

        sr = sharpe_ratio(adjusted)

        if sr > best_sharpe:
            best_sharpe = sr
            best_percentile = p

    return best_percentile


# ============================================================
# WALK FORWARD PRINCIPAL
# ============================================================

def run_walk_forward(
    returns,
    train_window=1000,
    test_window=250,
    chi_window=252,
    hedge_exposure=0.5,
    optimize=True,
    plot=True
):

    returns = np.asarray(returns)

    equity_base = []
    equity_overlay = []

    position_base = 1.0
    position_overlay = 1.0

    i = train_window

    while i + test_window <= len(returns):

        train = returns[i-train_window:i]
        test = returns[i:i+test_window]

        # ------------------------------------------
        # CALCULA χ APENAS NO TREINO
        # ------------------------------------------

        chi_train = rolling_susceptibility(train, window=chi_window)

        # ------------------------------------------
        # OTIMIZA THRESHOLD NO TREINO
        # ------------------------------------------

        if optimize:
            best_percentile = optimize_threshold(
                train,
                chi_train,
                hedge_exposure=hedge_exposure
            )
        else:
            best_percentile = 80

        # ------------------------------------------
        # APLICA NO TESTE
        # ------------------------------------------

        chi_test_full = rolling_susceptibility(
            returns[:i+test_window],
            window=chi_window
        )

        chi_test = chi_test_full[i:i+test_window]

        adjusted_test, hedge = apply_critical_overlay(
            test,
            chi_test,
            mode="threshold",
            threshold=np.nanpercentile(chi_train, best_percentile),
            hedge_exposure=hedge_exposure
        )

        # ------------------------------------------
        # ACUMULA EQUITY
        # ------------------------------------------

        for r_base, r_adj in zip(test, adjusted_test):

            position_base *= (1 + r_base)
            position_overlay *= (1 + r_adj)

            equity_base.append(position_base)
            equity_overlay.append(position_overlay)

        i += test_window

    equity_base = np.array(equity_base)
    equity_overlay = np.array(equity_overlay)

    # ------------------------------------------
    # MÉTRICAS
    # ------------------------------------------

    returns_base = np.diff(np.insert(equity_base, 0, 1)) / \
                   np.insert(equity_base, 0, 1)[:-1]

    returns_overlay = np.diff(np.insert(equity_overlay, 0, 1)) / \
                      np.insert(equity_overlay, 0, 1)[:-1]

    results = {
        "sharpe_base": sharpe_ratio(returns_base),
        "sharpe_overlay": sharpe_ratio(returns_overlay),
        "max_dd_base": max_drawdown(equity_base),
        "max_dd_overlay": max_drawdown(equity_overlay),
    }

    # ------------------------------------------
    # PLOT
    # ------------------------------------------

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(equity_base, label="Base Strategy")
        plt.plot(equity_overlay, label="Critical Overlay")
        plt.legend()
        plt.title("Walk-Forward Equity Comparison")
        plt.tight_layout()
        plt.show()

    return results


# ============================================================
# TESTE DIRETO
# ============================================================

if __name__ == "__main__":

    np.random.seed(42)

    # Simulação com clusters de volatilidade
    returns = np.random.normal(0.0005, 0.01, 4000)

    results = run_walk_forward(
        returns,
        train_window=1000,
        test_window=250,
        hedge_exposure=0.4
    )

    print("\n=== WALK FORWARD RESULTS ===")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")