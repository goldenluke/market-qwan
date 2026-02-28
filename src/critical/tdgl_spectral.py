import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, fftfreq
from tqdm import tqdm


# ============================================================
# CONFIGURAÇÕES PADRÃO
# ============================================================

DEFAULT_CONFIG = {
    "N": 256,
    "DT": 0.5,
    "STEPS": 2000,
    "a0": 1.0,
    "b": 1.0,
    "kappa": 1.0,
    "Tc_guess": 1.0,
    "temperatures": np.linspace(0.8, 1.2, 15),
}


# ============================================================
# TDGL ESPECTRAL SEMI-IMPLÍCITO
# ============================================================

class TDGLSpectral:

    def __init__(self, T, config=DEFAULT_CONFIG):
        self.N = config["N"]
        self.DT = config["DT"]
        self.STEPS = config["STEPS"]
        self.a0 = config["a0"]
        self.b = config["b"]
        self.kappa = config["kappa"]
        self.Tc_guess = config["Tc_guess"]

        self.T = T
        self.a = self.a0 * (T - self.Tc_guess)
        self.x = np.random.normal(0, 0.1, (self.N, self.N))

        # Pré-calcula malha k²
        kx = fftfreq(self.N) * 2 * np.pi
        ky = fftfreq(self.N) * 2 * np.pi
        KX, KY = np.meshgrid(kx, ky)
        self.k2 = KX**2 + KY**2

    def simulate(self):

        for _ in range(self.STEPS):

            x3 = self.x**3

            xk = fft2(self.x)
            x3k = fft2(x3)

            noise = np.sqrt(2 * self.T * self.DT) * \
                    np.random.normal(size=(self.N, self.N))
            noisek = fft2(noise)

            denom = 1 + self.DT * (self.a + self.kappa * self.k2)

            xk_new = (xk - self.DT * self.b * x3k + noisek) / denom

            self.x = np.real(ifft2(xk_new))

        return self.x


# ============================================================
# FUNÇÃO DE ESTRUTURA S(k)
# ============================================================

def structure_factor(field):

    N = field.shape[0]

    field = field - np.mean(field)
    F = fft2(field)
    Sk = np.abs(F)**2 / (N * N)

    Sk = fftshift(Sk)

    center = N // 2
    y, x = np.indices((N, N))
    r = np.sqrt((x - center)**2 + (y - center)**2)
    r = r.astype(int)

    radial = np.bincount(r.ravel(), Sk.ravel())
    counts = np.bincount(r.ravel())
    radial /= counts

    return radial[:N // 2]


# ============================================================
# SUSCEPTIBILIDADE χ
# ============================================================

def susceptibility(field):

    N = field.shape[0]
    m = np.mean(field)

    return N * N * (np.mean(field**2) - m**2)


# ============================================================
# EXPERIMENTO COMPLETO
# ============================================================

def run_critical_experiment(config=DEFAULT_CONFIG):

    temperatures = config["temperatures"]

    chi_values = []
    Sk_list = []

    print("Simulando TDGL espectral...")

    for T in tqdm(temperatures):

        model = TDGLSpectral(T, config=config)
        field = model.simulate()

        chi = susceptibility(field)
        chi_values.append(chi)

        Sk = structure_factor(field)
        Sk_list.append(Sk)

    chi_values = np.array(chi_values)

    # --------------------------------------------------------
    # Determinação automática de Tc
    # --------------------------------------------------------

    Tc_index = np.argmax(chi_values)
    Tc_est = temperatures[Tc_index]

    print("\nTc estimado via susceptibilidade =", Tc_est)

    # --------------------------------------------------------
    # Expoente crítico γ
    # χ ~ |T - Tc|^{-γ}
    # --------------------------------------------------------

    reduced_temp = np.abs(temperatures - Tc_est)

    mask = reduced_temp > 0.02

    logt = np.log(reduced_temp[mask])
    logchi = np.log(chi_values[mask])

    coef = np.polyfit(logt, logchi, 1)
    gamma = -coef[0]

    print("Expoente crítico γ ≈", gamma)

    # --------------------------------------------------------
    # Plot Susceptibilidade
    # --------------------------------------------------------

    plt.figure()
    plt.plot(temperatures, chi_values, 'o-')
    plt.xlabel("T")
    plt.ylabel("Susceptibilidade χ")
    plt.title("Determinação de Tc")
    plt.tight_layout()
    plt.savefig("susceptibility.png", dpi=300)
    plt.close()

    # --------------------------------------------------------
    # Plot Estrutura no Crítico
    # --------------------------------------------------------

    Sk_crit = Sk_list[Tc_index]
    k_vals = np.arange(len(Sk_crit))

    plt.figure()
    plt.loglog(k_vals[1:], Sk_crit[1:], 'o-')
    plt.xlabel("k")
    plt.ylabel("S(k)")
    plt.title("Função de Estrutura no Crítico")
    plt.tight_layout()
    plt.savefig("structure_factor.png", dpi=300)
    plt.close()

    print("\nArquivos gerados:")
    print("- susceptibility.png")
    print("- structure_factor.png")

    return {
        "Tc": Tc_est,
        "gamma": gamma,
        "chi": chi_values,
        "temperatures": temperatures,
    }


# ============================================================
# EXECUÇÃO DIRETA
# ============================================================

if __name__ == "__main__":
    run_critical_experiment()