import pandas as pd


def walk_forward(model_class, returns, window=252, step=21):

    weights_history = []

    for start in range(0, len(returns) - window, step):

        train = returns.iloc[start:start + window]
        model = model_class(train)
        weights = model.get_probabilistic_weights()

        weights_history.append(weights)

    return pd.DataFrame(weights_history)