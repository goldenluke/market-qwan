import numpy as np


class QWANDynamics:

    def __init__(self, mobility=None):
        self.L = mobility

    def gradient_flow(self, grad_phi):

        if self.L is None:
            self.L = np.eye(len(grad_phi))

        return - self.L @ grad_phi

    def dissipative_lagrangian(self, x_dot, grad_phi):

        L_inv = np.linalg.inv(self.L)
        term = x_dot + self.L @ grad_phi

        return 0.5 * term.T @ L_inv @ term