import numpy as np
import pandas as pd


class DynamicBacktest:
    """
    Backtest walk-forward regime-aware institucional.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        model_class,
        rebalance_frequency: int = 21,
        target_vol: float = 0.15
    ):

        self.returns = returns.copy()
        self.model_class = model_class
        self.rebalance_frequency = rebalance_frequency
        self.target_vol = target_vol

        self.portfolio_returns = None
        self.equity_curve = None
        self.drawdown = None

        self._run_backtest()

    # ==========================================================
    # CORE ENGINE
    # ==========================================================

    def _run_backtest(self):

        returns = self.returns
        n = len(returns)

        portfolio_returns = []
        current_weights = None

        # Começamos após janela mínima
        start_index = 252 if n > 252 else int(n / 2)

        for t in range(start_index, n):

            # Rebalance
            if (t - start_index) % self.rebalance_frequency == 0:

                train_slice = returns.iloc[:t]

                if len(train_slice) < 60:
                    continue

                model = self.model_class(train_slice)

                try:
                    current_weights = model.get_probabilistic_weights()
                except:
                    current_weights = np.ones(train_slice.shape[1]) / train_slice.shape[1]

                current_weights = np.array(current_weights)

                # Normaliza pesos
                if current_weights.sum() != 0:
                    current_weights = current_weights / np.sum(np.abs(current_weights))

            if current_weights is None:
                continue

            daily_ret = np.dot(current_weights, returns.iloc[t])

            portfolio_returns.append(daily_ret)

        if len(portfolio_returns) == 0:
            self.portfolio_returns = pd.Series(dtype=float)
            self.equity_curve = pd.Series(dtype=float)
            self.drawdown = pd.Series(dtype=float)
            return

        portfolio_returns = pd.Series(portfolio_returns, index=returns.index[start_index:start_index + len(portfolio_returns)])

        # ======================================================
        # VOL TARGETING
        # ======================================================

        realized_vol = portfolio_returns.std() * np.sqrt(252)

        if realized_vol > 0:
            scaling = self.target_vol / realized_vol
            portfolio_returns = portfolio_returns * scaling

        # Limite extremo defensivo
        portfolio_returns = portfolio_returns.clip(-0.95, 5)

        # ======================================================
        # EQUITY
        # ======================================================

        equity = (1 + portfolio_returns).cumprod()

        # Proteção contra colapso numérico
        equity[equity <= 0] = np.nan
        equity = equity.ffill().fillna(1)

        drawdown = equity / equity.cummax() - 1

        self.portfolio_returns = portfolio_returns
        self.equity_curve = equity
        self.drawdown = drawdown

    # ==========================================================
    # MÉTRICAS
    # ==========================================================

    def compute_metrics(self):

        returns = self.portfolio_returns

        if returns is None or len(returns) < 2:
            return {
                "sharpe": 0,
                "sortino": 0,
                "max_dd": 0,
                "cagr": 0
            }

        equity = self.equity_curve

        # ==============================
        # CAGR
        # ==============================

        total_return = equity.iloc[-1]
        n_years = len(returns) / 252

        if n_years > 0:
            cagr = total_return ** (1 / n_years) - 1
        else:
            cagr = 0

        # ==============================
        # Sharpe
        # ==============================

        mean_ret = returns.mean() * 252
        vol = returns.std() * np.sqrt(252)

        sharpe = mean_ret / vol if vol != 0 else 0

        # ==============================
        # Sortino
        # ==============================

        downside = returns[returns < 0]

        if len(downside) > 0:
            downside_vol = downside.std() * np.sqrt(252)
            sortino = mean_ret / downside_vol if downside_vol != 0 else 0
        else:
            sortino = 0

        # ==============================
        # Max Drawdown
        # ==============================

        max_dd = self.drawdown.min()

        return {
            "sharpe": sharpe,
            "sortino": sortino,
            "max_dd": max_dd,
            "cagr": cagr
        }