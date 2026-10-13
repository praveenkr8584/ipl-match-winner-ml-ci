from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

MODEL_PATH = Path("ipl_winner_model.pkl")
FEATURES = [
    "team1",
    "team2",
    "venue",
    "toss_decision",
    "toss_winner_is_team1",
    "year"
]


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "ipl_winner_model.pkl was not found. "
            "Run the training pipeline first."
        )
    return joblib.load(MODEL_PATH)


@app.get("/")
def health_check():
    return jsonify({
        "status": "ok",
        "service": "ipl-match-winner-prediction"
    })


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "JSON request body is required"}), 400

    missing_fields = [
        feature for feature in FEATURES
        if feature not in data
    ]
    if missing_fields:
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing_fields
        }), 400

    if data["team1"] == data["team2"]:
        return jsonify({"error": "team1 and team2 must be different"}), 400

    sample = pd.DataFrame([{
        "Team1": data["team1"],
        "Team2": data["team2"],
        "Venue": data["venue"],
        "TossDecision": data["toss_decision"],
        "toss_winner_is_team1": data["toss_winner_is_team1"],
        "year": data["year"]
    }])

    model = load_model()
    prediction_code = int(model.predict(sample)[0])
    team1_probability = float(model.predict_proba(sample)[0][1])
    winner = data["team1"] if prediction_code == 1 else data["team2"]

    return jsonify({
        "predicted_winner": winner,
        "prediction_code": prediction_code,
        "team1_win_probability": round(team1_probability, 4)
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
