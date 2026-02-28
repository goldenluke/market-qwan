from fastapi import FastAPI
import numpy as np
import pandas as pd

from src.data_loader import DataLoader
from src.qwan_regime_model import RegimeAwareQWAN

app = FastAPI()


@app.get("/weights")
def get_weights():

    loader = DataLoader(
        tickers=["AAPL","MSFT","GOOGL","AMZN","META"],
        period="5y"
    )

    returns = loader.load()

    model = RegimeAwareQWAN(returns)
    weights = model.get_probabilistic_weights()

    return {
        "weights": dict(zip(returns.columns, weights.tolist()))
    }