import numpy as np
import pandas as pd


class Backtest:

    def __init__(
        self,
        returns,
        weights,
        transaction_cost=0.001,
        structural_stop=None
    ):

        self.returns = returns
        self.weights = weights
        self.transaction_cost = transaction_cost
        self.structural_stop = structural_stop

        self._run_backtest()

    # ==========================================================
    # EXECUÇÃO
    # ==========================================================

    def _run_backtest(self):

        # Portfolio returns
        port_returns = self.returns @ self.weights

        # Aplicar custo de transação (simples)
        port_returns = port_returns - self.transaction_cost

        # Equity curve
        self.equity_curve = (1 + port_returns).cumprod()

        # Stop estrutural (opcional)
        if self.structural_stop is not None:
            dd = self.equity_curve / self.equity_curve.cummax() - 1
            if dd.min() < self.structural_stop:
                self.equity_curve.loc[dd.idxmin():] = self.equity_curve.loc[dd.idxmin()]

        # Drawdown
        self.drawdown = self.equity_curve / self.equity_curve.cummax() - 1

        # Guardar retornos
        self.portfolio_returns = port_returns

    # ==========================================================
    # MÉTRICAS
    # ==========================================================

    def compute_metrics(self):

        r = self.portfolio_returns

        sharpe = np.mean(r) / (np.std(r) + 1e-12)
        sortino = np.mean(r) / (np.std(r[r < 0]) + 1e-12)
        max_dd = self.drawdown.min()
        cvar = np.mean(r[r <= np.percentile(r, 5)])

        return {
            "sharpe": sharpe,
            "sortino": sortino,
            "max_dd": max_dd,
            "cvar": cvar
        }

    # ==========================================================
    # RELATÓRIO (CLI)
    # ==========================================================

    def report(self):

        metrics = self.compute_metrics()

        print("\n===== PERFORMANCE =====")
        print("Sharpe:", metrics["sharpe"])
        print("Sortino:", metrics["sortino"])
        print("Max Drawdown:", metrics["max_dd"])
        print("CVaR:", metrics["cvar"])

        return metrics