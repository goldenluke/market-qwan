import numpy as np

def block_bootstrap(returns, block_size=20, T=252):

    n = len(returns)
    blocks = []

    while len(blocks) * block_size < T:
        start = np.random.randint(0, n - block_size)
        blocks.append(returns[start:start+block_size])

    return np.concatenate(blocks)[:T]