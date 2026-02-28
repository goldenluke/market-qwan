import numpy as np
import pandas as pd

from src.data_loader import DataLoader
from src.quant_fund_engine import QuantFundEngine


# ==========================================================
# MÉTRICAS INSTITUCIONAIS
# ==========================================================

def compute_metrics(equity_curve: pd.Series):

    returns = equity_curve.pct_change().dropna()

    sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252) + 1e-8)

    downside = returns[returns < 0]
    sortino = (returns.mean() * 252) / (
        downside.std() * np.sqrt(252) + 1e-8
    )

    rolling_max = equity_curve.cummax()
    drawdown = equity_curve / rolling_max - 1
    max_dd = drawdown.min()

    cagr = (equity_curve.iloc[-1]) ** (252 / len(equity_curve)) - 1

    vol = returns.std() * np.sqrt(252)

    return {
        "CAGR": round(cagr, 4),
        "Sharpe": round(sharpe, 4),
        "Sortino": round(sortino, 4),
        "Vol": round(vol, 4),
        "Max Drawdown": round(max_dd, 4)
    }


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("📥 Baixando dados...")

    tickers = ["SPY", "TLT", "GLD", "VNQ", "DBC"]

    loader = DataLoader(tickers=tickers, period="10y")
    returns = loader.load()

    print("Dados carregados.")
    print(returns.tail())

    print("\n🧠 Rodando engine institucional...")

    engine = QuantFundEngine(
        returns=returns,
        window=504,         # 2 anos rolling
        target_vol=0.15,
        cvar_limit=0.05
    )

    equity = engine.run()

    print("Engine concluída.")

    # ======================================================
    # BENCHMARKS
    # ======================================================

    print("\n📊 Calculando benchmarks...")

    # Equal Weight
    eq_weights = np.ones(len(tickers)) / len(tickers)
    eq_returns = returns.iloc[504:] @ eq_weights
    eq_equity = (1 + eq_returns).cumprod()

    # SPY benchmark
    spy_equity = (1 + returns["SPY"].iloc[504:]).cumprod()

    # ======================================================
    # MÉTRICAS
    # ======================================================

    print("\n===== 📈 QWAN ENGINE =====")
    qwan_metrics = compute_metrics(equity)
    for k, v in qwan_metrics.items():
        print(f"{k}: {v}")

    print("\n===== 📊 EQUAL WEIGHT =====")
    eq_metrics = compute_metrics(eq_equity)
    for k, v in eq_metrics.items():
        print(f"{k}: {v}")

    print("\n===== 📉 SPY =====")
    spy_metrics = compute_metrics(spy_equity)
    for k, v in spy_metrics.items():
        print(f"{k}: {v}")

    # ======================================================
    # EXPORT OPCIONAL
    # ======================================================

    equity.to_csv("qwan_equity.csv")
    print("\nEquity exportada para qwan_equity.csv")


# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":
    main()