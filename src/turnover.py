import numpy as np


def calculate_turnover(weight_history):
    turnover = 0
    for i in range(1, len(weight_history)):
        turnover += np.sum(np.abs(weight_history[i] - weight_history[i-1]))
    return turnover / len(weight_history)