import numpy as np


class ParetoQWAN:

    def pareto_score(self, entropy, coherence):

        return entropy / (1 + coherence)