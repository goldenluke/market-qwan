import numpy as np
from numpy.fft import fft2, fftshift
from scipy.stats import entropy
from scipy.signal import welch

# ============================================================
# MÉTRICAS BÁSICAS DE COMPLEXIDADE
# ============================================================

def shannon_entropy(series, bins=50):
    """
    Entropia de Shannon da distribuição de retornos.
    """
    hist, _ = np.histogram(series, bins=bins, density=True)
    hist = hist[hist > 0]
    return entropy(hist)


def variance_complexity(series):
    """
    Variância simples (proxy de energia estrutural).
    """
    return np.var(series)


def hurst_exponent(series):
    """
    Estimativa simples do expoente de Hurst.
    """
    lags = range(2, 50)
    tau = [np.std(np.subtract(series[lag:], series[:-lag])) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0]


# ============================================================
# SUSCEPTIBILIDADE (χ)
# ============================================================

def susceptibility(series):
    """
    χ = N ( <x²> - <x>² )
    Mede flutuação coletiva.
    """
    series = np.asarray(series)
    N = len(series)
    m = np.mean(series)
    return N * (np.mean(series**2) - m**2)


# ============================================================
# FUNÇÃO DE ESTRUTURA S(k)
# ============================================================

def structure_factor(series):
    """
    S(k) unidimensional via FFT.
    """
    series = series - np.mean(series)
    F = np.fft.fft(series)
    Sk = np.abs(F)**2 / len(series)
    return Sk[:len(series)//2]


# ============================================================
# ÍNDICE DE CRITICALIDADE DE MERCADO
# ============================================================

def market_criticality_index(series):
    """
    Combina múltiplas métricas estruturais.
    """

    series = np.asarray(series)

    H = hurst_exponent(series)
    chi = susceptibility(series)
    ent = shannon_entropy(series)

    # Normalização robusta
    chi_norm = np.log(chi + 1e-8)

    # Índice composto
    criticality = (
        0.4 * abs(H - 0.5) +
        0.4 * chi_norm +
        0.2 * ent
    )

    return {
        "criticality_index": criticality,
        "hurst": H,
        "susceptibility": chi,
        "entropy": ent,
    }


# ============================================================
# DETECÇÃO DE REGIME CRÍTICO
# ============================================================

def detect_critical_regime(series, threshold=2.0):
    """
    Detecta regime crítico com base no índice composto.
    """

    metrics = market_criticality_index(series)

    if metrics["criticality_index"] > threshold:
        regime = "critical"
    elif metrics["hurst"] > 0.6:
        regime = "trending"
    elif metrics["hurst"] < 0.4:
        regime = "mean_reverting"
    else:
        regime = "neutral"

    metrics["regime"] = regime

    return metrics