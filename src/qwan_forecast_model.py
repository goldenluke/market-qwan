# src/qwan_forecast_model.py

import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
import pickle
import os

class QWANForecastModel:

    def __init__(self, horizon=10, input_size=60):
        self.horizon = horizon
        self.input_size = input_size

        self.model = NHITS(
            h=horizon,
            input_size=input_size,
            max_steps=20,
            accelerator="cpu",
            devices=1,
            logger=False,
            enable_progress_bar=False
        )

        self.nf = NeuralForecast(
            models=[self.model],
            freq="D"
        )

    def prepare_data(self, returns):
        market_returns = returns.mean(axis=1)

        df_nf = pd.DataFrame({
            "unique_id": "market",
            "ds": pd.to_datetime(market_returns.index),
            "y": market_returns.values
        }).sort_values("ds")

        return df_nf

    def fit(self, returns):
        df_nf = self.prepare_data(returns)
        self.nf.fit(df=df_nf)

    def predict(self):
        return self.nf.predict()

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)