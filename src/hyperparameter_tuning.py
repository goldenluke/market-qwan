import numpy as np


def tune_parameters(model_class, returns):

    best_score = -np.inf
    best_params = None

    for alpha in np.linspace(0.2, 1.0, 5):
        for beta in np.linspace(0.2, 1.0, 5):
            for gamma in np.linspace(0.2, 1.0, 5):

                model = model_class(
                    returns,
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma
                )

                weights = model.get_probabilistic_weights()
                portfolio = returns.values @ weights
                sharpe = np.mean(portfolio) / (np.std(portfolio) + 1e-12)

                if sharpe > best_score:
                    best_score = sharpe
                    best_params = (alpha, beta, gamma)

    return best_params