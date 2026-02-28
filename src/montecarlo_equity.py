import numpy as np
import pandas as pd


def monte_carlo_equity(model, T=252, n_sim=1000):

    regimes = model.hidden_states
    transition = model.transition_matrix

    regime_means = {}
    regime_covs = {}

    for k in range(model.n_regimes):
        r = model.returns[regimes == k]
        regime_means[k] = r.mean().values
        regime_covs[k] = np.cov(r.values.T)

    paths = []

    for _ in range(n_sim):

        state = model.get_current_regime()
        equity = [1.0]

        for _ in range(T):

            mean = regime_means[state]
            cov = regime_covs[state]

            simulated_return = np.random.multivariate_normal(mean, cov)
            portfolio_return = np.dot(simulated_return, model.regime_weights[state])

            equity.append(equity[-1] * (1 + portfolio_return))

            state = np.random.choice(
                model.n_regimes,
                p=transition[state]
            )

        paths.append(equity)

    return pd.DataFrame(paths).T