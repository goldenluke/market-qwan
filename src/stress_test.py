import numpy as np


def stress_test(portfolio_returns, shock=-0.2):
    stressed = portfolio_returns.copy()
    stressed.iloc[0] += shock
    equity = (1 + stressed).cumprod()
    return equity