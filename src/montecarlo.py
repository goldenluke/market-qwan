import numpy as np


def simulate_regime_paths(P, current_state, T):

    states = [current_state]

    for _ in range(T):
        next_state = np.random.choice(
            len(P),
            p=P[states[-1]]
        )
        states.append(next_state)

    return states