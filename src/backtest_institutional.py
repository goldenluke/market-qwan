import numpy as np
import pandas as pd


class InstitutionalBacktest:
    """
    Backtest institucional completo:

    ✔ Walk-forward rolling
    ✔ Benchmark integrado
    ✔ Vol targeting
    ✔ Stop estrutural
    ✔ Controle defensivo
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        model_class,
        lookback: int = 252,
        rebalance_frequency: int = 21,
        target_vol: float = 0.15,
        stop_drawdown: float = -0.25,
        benchmark_mode: str = "equal_weight"
    ):

        self.returns = returns.copy()
        self.model_class = model_class
        self.lookback = lookback
        self.rebalance_frequency = rebalance_frequency
        self.target_vol = target_vol
        self.stop_drawdown = stop_drawdown
        self.benchmark_mode = benchmark_mode

        self.portfolio_returns = None
        self.benchmark_returns = None
        self.equity_curve = None
        self.benchmark_equity = None
        self.drawdown = None

        self._run()

    # ==========================================================
    # WALK-FORWARD ROLLING
    # ==========================================================

    def _run(self):

        returns = self.returns
        n = len(returns)

        portfolio_returns = []
        benchmark_returns = []
        current_weights = None

        peak_equity = 1.0
        equity = 1.0

        for t in range(self.lookback, n):

            # ==========================================
            # WALK FORWARD TRAINING WINDOW
            # ==========================================

            train_slice = returns.iloc[t - self.lookback:t]

            # Rebalance
            if (t - self.lookback) % self.rebalance_frequency == 0:

                model = self.model_class(train_slice)

                try:
                    current_weights = model.get_probabilistic_weights()
                except:
                    current_weights = np.ones(train_slice.shape[1]) / train_slice.shape[1]

                current_weights = np.array(current_weights)

                if current_weights.sum() != 0:
                    current_weights = current_weights / np.sum(np.abs(current_weights))

            if current_weights is None:
                continue

            # ==========================================
            # DAILY RETURN
            # ==========================================

            daily_ret = np.dot(current_weights, returns.iloc[t])

            # ==========================================
            # VOL TARGETING
            # ==========================================

            recent_returns = np.array(portfolio_returns[-60:]) if len(portfolio_returns) > 60 else np.array(portfolio_returns)

            if len(recent_returns) > 10:
                realized_vol = np.std(recent_returns) * np.sqrt(252)
                if realized_vol > 0:
                    scaling = self.target_vol / realized_vol
                    daily_ret *= scaling

            # ==========================================
            # STOP ESTRUTURAL
            # ==========================================

            equity *= (1 + daily_ret)
            peak_equity = max(peak_equity, equity)
            dd = equity / peak_equity - 1

            if dd < self.stop_drawdown:
                daily_ret = 0  # vai para cash
                equity = peak_equity  # trava perda

            portfolio_returns.append(daily_ret)

            # ==========================================
            # BENCHMARK
            # ==========================================

            if self.benchmark_mode == "equal_weight":
                bench_ret = returns.iloc[t].mean()
            else:
                bench_ret = returns.iloc[t].mean()

            benchmark_returns.append(bench_ret)

        # ==============================================
        # SERIES
        # ==============================================

        index = returns.index[self.lookback:self.lookback + len(portfolio_returns)]

        self.portfolio_returns = pd.Series(portfolio_returns, index=index)
        self.benchmark_returns = pd.Series(benchmark_returns, index=index)

        self.equity_curve = (1 + self.portfolio_returns).cumprod()
        self.benchmark_equity = (1 + self.benchmark_returns).cumprod()

        self.drawdown = self.equity_curve / self.equity_curve.cummax() - 1

    # ==========================================================
    # MÉTRICAS
    # ==========================================================

    def compute_metrics(self):

        returns = self.portfolio_returns

        if len(returns) < 2:
            return {}

        equity = self.equity_curve

        total_return = equity.iloc[-1]
        years = len(returns) / 252

        cagr = total_return ** (1 / years) - 1 if years > 0 else 0

        mean_ret = returns.mean() * 252
        vol = returns.std() * np.sqrt(252)

        sharpe = mean_ret / vol if vol > 0 else 0

        downside = returns[returns < 0]
        sortino = mean_ret / (downside.std() * np.sqrt(252)) if len(downside) > 0 else 0

        max_dd = self.drawdown.min()

        alpha = (self.portfolio_returns - self.benchmark_returns).mean() * 252

        return {
            "cagr": cagr,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_dd": max_dd,
            "alpha": alpha
        }