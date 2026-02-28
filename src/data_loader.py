# src/data_loader.py

import yfinance as yf
import numpy as np


class DataLoader:

    def __init__(self, tickers, period="5y"):
        self.tickers = tickers
        self.period = period

    def load(self):
        data = yf.download(self.tickers, period=self.period)

        # Se vier MultiIndex
        if isinstance(data.columns, tuple) or len(data.columns.names) > 1:
            data = data["Close"]
        else:
            data = data["Close"]

        returns = np.log(data / data.shift(1)).dropna()
        return returns