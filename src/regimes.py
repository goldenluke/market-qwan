# src/regimes.py

from sklearn.cluster import KMeans

class RegimeDetector:

    def __init__(self, n_regimes=3):
        self.n_regimes = n_regimes

    def detect(self, returns):

        vol = returns.rolling(20).std()

        # remover NaN
        valid = vol.dropna()

        model = KMeans(n_clusters=self.n_regimes, random_state=42)
        regimes = model.fit_predict(valid)

        # criar vetor completo alinhado
        full_regimes = returns.index.to_series().copy()
        full_regimes[:] = None
        full_regimes.loc[valid.index] = regimes

        # remover períodos sem regime
        return full_regimes.dropna().astype(int)