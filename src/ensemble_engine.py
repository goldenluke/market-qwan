class EnsembleEngine:

    def __init__(self, engines):

        self.engines = engines

    def get_weights(self):

        weights = []

        for engine in self.engines:
            w = engine.get_current_weights()
            weights.append(w)

        weights = np.array(weights)

        return np.mean(weights, axis=0)