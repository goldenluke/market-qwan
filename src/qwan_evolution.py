import numpy as np


class QWANEvolution:

    def replicate(self, population, fitness_scores):

        sorted_idx = np.argsort(fitness_scores)[::-1]
        survivors = population[sorted_idx[:len(population)//2]]

        new_population = survivors.copy()

        for w in survivors:
            mutation = w + np.random.normal(0, 0.02, len(w))
            mutation = np.clip(mutation, 0, 1)
            mutation /= np.sum(mutation)
            new_population = np.vstack([new_population, mutation])

        return new_population