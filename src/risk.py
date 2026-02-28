import numpy as np


class RiskMetrics:

    @staticmethod
    def sharpe(returns):
        return np.mean(returns) / (np.std(returns) + 1e-12) * np.sqrt(252)

    @staticmethod
    def sortino(returns):
        downside = returns[returns < 0]
        return np.mean(returns) / (np.std(downside) + 1e-12) * np.sqrt(252)

    @staticmethod
    def max_drawdown(equity):
        peak = np.maximum.accumulate(equity)
        return np.min(equity / peak - 1)

    @staticmethod
    def cvar(returns, alpha=0.05):
        var = np.quantile(returns, alpha)
        return np.mean(returns[returns <= var])