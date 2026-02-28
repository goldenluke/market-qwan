import numpy as np
from src.qwan_regime_model import RegimeAwareQWAN

def nested_alpha_gamma_search(train_data):

    alphas = [0.3, 0.6, 1.0]
    gammas = [0.3, 0.6, 1.0]

    best_score = -np.inf
    best_params = (0.6, 0.5)

    split = int(len(train_data) * 0.7)

    train = train_data[:split]
    val = train_data[split:]

    for a in alphas:
        for g in gammas:

            model = RegimeAwareQWAN(
                returns=train,
                alpha=a,
                gamma=g
            )

            w = model.get_current_weights()
            val_returns = val.values @ w

            sharpe = (
                np.mean(val_returns) * 252 /
                (np.std(val_returns) * np.sqrt(252) + 1e-8)
            )

            if sharpe > best_score:
                best_score = sharpe
                best_params = (a, g)

    return best_params