import pandas as pd
import numpy as np


def regime_attribution(returns, regimes, portfolio_returns):
    df = pd.DataFrame({
        "ret": portfolio_returns,
        "regime": regimes[-len(portfolio_returns):]
    })

    attribution = df.groupby("regime")["ret"].agg(
        ["mean", "std", "count"]
    )

    attribution["annual_return"] = attribution["mean"] * 252
    attribution["annual_vol"] = attribution["std"] * np.sqrt(252)

    return attribution