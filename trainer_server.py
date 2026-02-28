from flask import Flask, request, jsonify, render_template
import subprocess
import os

app = Flask(__name__)


# -------------------------
# PÁGINA PRINCIPAL
# -------------------------
@app.route("/")
def index():
    return render_template("trainer.html")


# -------------------------
# TREINAR MODELO
# -------------------------
@app.route("/train", methods=["POST"])
def train():

    data = request.json

    tickers = data["tickers"]
    input_size = data["input_size"]
    horizon = data["horizon"]
    model_name = data["model_name"]

    command = [
        "python",
        "train_forecast.py",
        "--tickers", tickers,
        "--input_size", str(input_size),
        "--horizon", str(horizon),
        "--model_name", model_name
    ]

    subprocess.Popen(command)

    return jsonify({"status": "started"})


# -------------------------
# LISTAR MODELOS
# -------------------------
@app.route("/models", methods=["GET"])
def list_models():

    if not os.path.exists("models"):
        return jsonify([])

    files = [
        f for f in os.listdir("models")
        if f.endswith(".pkl")
    ]

    return jsonify(files)


if __name__ == "__main__":
    app.run(port=5050, debug=True)