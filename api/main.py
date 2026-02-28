from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from src.data_loader import DataLoader
from src.qwan_regime_model import RegimeAwareQWAN


app = FastAPI(title="Market-QWAN API")


class ModelRequest(BaseModel):
    tickers: List[str]
    period: str = "5y"
    n_regimes: int = 3
    alpha: float = 0.6
    gamma: float = 0.5
    target_vol: float = 0.15
    cvar_limit: float = 0.03


@app.post("/optimize")
def optimize_portfolio(request: ModelRequest):

    loader = DataLoader(
        tickers=request.tickers,
        period=request.period
    )

    returns = loader.load()

    model = RegimeAwareQWAN(
        returns=returns,
        n_regimes=request.n_regimes,
        alpha=request.alpha,
        gamma=request.gamma,
        target_vol=request.target_vol,
        cvar_limit=request.cvar_limit
    )

    weights = model.get_probabilistic_weights()

    return {
        "tickers": request.tickers,
        "weights": weights.tolist(),
        "current_regime": model.get_current_regime(),
        "forecast_3_steps": model.forecast_regime_probability(3).tolist()
    }


@app.get("/")
def health_check():
    return {"status": "Market-QWAN API running"}