# train_forecast.py

import argparse
from xml.parsers.expat import model
import yfinance as yf
import pandas as pd
from src.qwan_forecast_model import QWANForecastModel


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--tickers", type=str, required=True)
    parser.add_argument("--input_size", type=int, required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    args = parser.parse_args()

    tickers = args.tickers.split(",")

    print("Baixando dados...")
    data = yf.download(tickers, period="5y")["Close"]
    returns = data.pct_change().dropna()

    print("Treinando modelo...")
    model = QWANForecastModel(
        horizon=args.horizon,
        input_size=args.input_size
    )

    model.fit(returns)

    model.save(f"models/{args.model_name}.pkl")

    print("Treino finalizado com sucesso.")


if __name__ == "__main__":
    main()