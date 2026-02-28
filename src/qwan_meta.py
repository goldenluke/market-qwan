import numpy as np


class MetaStability:

    def is_metastable(self, phi_values):

        grad = np.gradient(phi_values)
        return np.abs(grad[-1]) < 1e-3