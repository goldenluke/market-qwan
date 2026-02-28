import numpy as np


def phi_surface(
    model,
    grid_size=20,
    mode="posterior",  # "posterior", "single"
    regime_index=None
):

    alpha_vals = np.linspace(0.1, 2.0, grid_size)
    gamma_vals = np.linspace(0.1, 2.0, grid_size)

    Z = np.zeros((grid_size, grid_size))

    # Probabilidades atuais
    posterior = model.posterior_probs[-1]

    # Se modo regime único
    if mode == "single":
        if regime_index is None:
            regime_index = np.argmax(posterior)

    for i, a in enumerate(alpha_vals):
        for j, g in enumerate(gamma_vals):

            model.alpha = a
            model.gamma = g

            w = model.get_probabilistic_weights()

            if mode == "posterior":

                # Φ esperado ponderado por probabilidade
                phi_val = 0
                for k in range(model.n_regimes):
                    phi_val += posterior[k] * model.phi(w, k)

            elif mode == "single":

                phi_val = model.phi(w, regime_index)

            else:
                phi_val = model.phi(w, np.argmax(posterior))

            Z[i, j] = phi_val

    return alpha_vals, gamma_vals, Z