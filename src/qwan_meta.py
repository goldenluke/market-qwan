import numpy as np


class MetaAllocator:
    """
    Alocador de capital entre múltiplos engines.
    Usa Sharpe ratio como critério de alocação.
    """

    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon

    def allocate(self, engine_returns):
        """
        engine_returns: array shape (n_engines, T)
        """

        engine_returns = np.array(engine_returns)

        if engine_returns.ndim != 2:
            raise ValueError("engine_returns deve ser 2D (n_engines, T)")

        mean_returns = engine_returns.mean(axis=1)
        std_returns = engine_returns.std(axis=1) + self.epsilon

        sharpe = mean_returns / std_returns

        # evitar negativos extremos
        sharpe = np.clip(sharpe, 0, None)

        if sharpe.sum() == 0:
            weights = np.ones(len(sharpe)) / len(sharpe)
        else:
            weights = sharpe / sharpe.sum()

        return weights