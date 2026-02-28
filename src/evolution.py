# src/evolution.py

import numpy as np
from multiprocessing import Pool

def avaliar_modelo(args):
    model_class, returns, params = args
    model = model_class(returns, **params)
    w = model.optimize()
    r = returns.values @ w
    score = np.mean(r) / np.std(r)
    return score, params

class EvolutionEngine:

    def __init__(self, model_class, population):
        self.model_class = model_class
        self.population = population

    def evolve(self, returns):

        args = [(self.model_class, returns, p)
                for p in self.population]

        with Pool() as pool:
            results = pool.map(avaliar_modelo, args)

        results.sort(reverse=True, key=lambda x: x[0])
        return results[:len(results)//2]
    

from multiprocessing import Pool


def evolve_population(models):

    with Pool() as pool:
        results = pool.map(run_model, models)

    return results