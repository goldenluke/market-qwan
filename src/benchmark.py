import numpy as np


def equal_weight_backtest(returns):

    n_assets = returns.shape[1]
    w = np.ones(n_assets) / n_assets

    equity = (1 + returns @ w).cumprod()

    return equity


def single_asset_backtest(returns, asset_index=0):

    equity = (1 + returns.iloc[:, asset_index]).cumprod()

    return equity