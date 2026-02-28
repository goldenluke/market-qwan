import numpy as np


class Autopoiesis:

    def closure_score(self, returns, weights):

        portfolio = returns.values @ weights
        stability = 1 / (1 + np.std(portfolio))

        return stability
    
class ComplexityAbsorption:

    def absorption(self, portfolio_returns, environment_vol):

        internal_complexity = np.std(portfolio_returns)
        external_complexity = environment_vol

        return internal_complexity / (1 + external_complexity)