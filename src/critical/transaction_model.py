import numpy as np

def apply_transaction_costs(returns, exposure, cost_per_turnover=0.0005):
    """
    Aplica custo proporcional ao turnover.
    """

    exposure = np.asarray(exposure)
    returns = np.asarray(returns)

    turnover = np.abs(np.diff(np.insert(exposure, 0, exposure[0])))
    costs = turnover * cost_per_turnover

    adjusted_returns = returns - costs

    return adjusted_returns, turnover