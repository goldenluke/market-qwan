import numpy as np


# ============================================================
# 1️⃣ HEDGE SIMPLES POR THRESHOLD FIXO
# ============================================================

def critical_hedge_threshold(critical_series, threshold, 
                             hedge_exposure=0.5,
                             normal_exposure=1.0):
    """
    Reduz exposição quando criticalidade ultrapassa threshold.
    """

    critical_series = np.asarray(critical_series)
    hedge = np.full(len(critical_series), normal_exposure)

    mask = critical_series > threshold
    hedge[mask] = hedge_exposure

    return hedge


# ============================================================
# 2️⃣ HEDGE CONTÍNUO PROPORCIONAL
# ============================================================

def critical_hedge_continuous(critical_series,
                              min_exposure=0.3,
                              max_exposure=1.0):
    """
    Ajusta exposição continuamente de acordo com a intensidade
    da criticalidade.
    """

    critical_series = np.asarray(critical_series)

    # Normalização robusta
    cs = critical_series.copy()
    cs = (cs - np.nanmin(cs)) / (np.nanmax(cs) - np.nanmin(cs) + 1e-8)

    # Inverte: mais crítico → menos exposição
    exposure = max_exposure - cs * (max_exposure - min_exposure)

    # Garantir limites
    exposure = np.clip(exposure, min_exposure, max_exposure)

    return exposure


# ============================================================
# 3️⃣ THRESHOLD DINÂMICO (PERCENTIL ROLLING)
# ============================================================

def rolling_percentile_threshold(critical_series, window=252, percentile=80):
    """
    Calcula threshold adaptativo baseado em percentil rolling.
    """

    critical_series = np.asarray(critical_series)
    threshold_series = np.full(len(critical_series), np.nan)

    for i in range(window, len(critical_series)):
        window_data = critical_series[i-window:i]
        threshold_series[i] = np.nanpercentile(window_data, percentile)

    return threshold_series


def critical_hedge_dynamic_percentile(critical_series,
                                       window=252,
                                       percentile=80,
                                       hedge_exposure=0.5,
                                       normal_exposure=1.0):
    """
    Hedge baseado em threshold dinâmico rolling.
    """

    threshold_series = rolling_percentile_threshold(
        critical_series, window, percentile
    )

    hedge = np.full(len(critical_series), normal_exposure)

    for i in range(len(critical_series)):
        if not np.isnan(threshold_series[i]):
            if critical_series[i] > threshold_series[i]:
                hedge[i] = hedge_exposure

    return hedge


# ============================================================
# 4️⃣ FUNÇÃO PRINCIPAL PARA BACKTEST
# ============================================================

def apply_critical_overlay(returns,
                           critical_series,
                           mode="threshold",
                           **kwargs):
    """
    Aplica overlay crítico aos retornos.

    Modes:
        - "threshold"
        - "continuous"
        - "dynamic_percentile"
    """

    if mode == "threshold":
        hedge = critical_hedge_threshold(
            critical_series,
            threshold=kwargs.get("threshold", np.nanpercentile(critical_series, 80)),
            hedge_exposure=kwargs.get("hedge_exposure", 0.5),
            normal_exposure=kwargs.get("normal_exposure", 1.0)
        )

    elif mode == "continuous":
        hedge = critical_hedge_continuous(
            critical_series,
            min_exposure=kwargs.get("min_exposure", 0.3),
            max_exposure=kwargs.get("max_exposure", 1.0)
        )

    elif mode == "dynamic_percentile":
        hedge = critical_hedge_dynamic_percentile(
            critical_series,
            window=kwargs.get("window", 252),
            percentile=kwargs.get("percentile", 80),
            hedge_exposure=kwargs.get("hedge_exposure", 0.5),
            normal_exposure=kwargs.get("normal_exposure", 1.0)
        )

    else:
        raise ValueError("Modo inválido para critical overlay.")

    adjusted_returns = returns * hedge

    return adjusted_returns, hedge