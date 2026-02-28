# src/regimes.py

from sklearn.cluster import KMeans

class RegimeDetector:

    def __init__(self, n_regimes=3):
        self.n_regimes = n_regimes

    def detect(self, returns):
        vol = returns.rolling(20).std().dropna()
        model = KMeans(n_clusters=self.n_regimes, random_state=42)
        regimes = model.fit_predict(vol)
        return regimes[-len(returns):]