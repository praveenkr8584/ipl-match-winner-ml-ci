import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_FILE = "IPL_Matches_2008_2022.csv"

TEAM_NAMES = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
}

CATEGORICAL = ["Team1", "Team2", "Venue", "TossDecision"]
NUMERIC = ["toss_winner_is_team1", "year"]


def create_dataset():
    data = pd.read_csv(DATA_FILE)

    # Remove matches that have no winner (no result / abandoned)
    data = data.dropna(subset=["WinningTeam"]).copy()

    # Standardise renamed teams
    for column in ["Team1", "Team2", "TossWinner", "WinningTeam"]:
        data[column] = data[column].replace(TEAM_NAMES)

    # Keep only the stadium name (remove the city part)
    data["Venue"] = data["Venue"].str.split(",").str[0].str.strip()

    data["year"] = pd.to_datetime(data["Date"]).dt.year
    data["toss_winner_is_team1"] = (data["TossWinner"] == data["Team1"]).astype(int)

    # Target: 1 = Team1 wins, 0 = Team2 wins
    data["team1_win"] = (data["WinningTeam"] == data["Team1"]).astype(int)

    return data[CATEGORICAL + NUMERIC + ["team1_win"]]


def train_model():
    print("Creating dataset...")
    data = create_dataset()
    data.to_csv("ipl_model_data.csv", index=False)
    print("Dataset created successfully.")
    print("Number of records:", len(data))

    X = data[CATEGORICAL + NUMERIC]
    y = data["team1_win"]

    # Time-based split: past seasons train, latest seasons test
    train_mask = data["year"] <= 2020
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[~train_mask], y[~train_mask]

    print("Training records:", len(X_train))
    print("Testing records :", len(X_test))

    preprocess = ColumnTransformer(
        [("categories", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)],
        remainder="passthrough",
    )
    model = Pipeline([
        ("preprocess", preprocess),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])

    print("Training model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    baseline = max(y_test.mean(), 1 - y_test.mean())
    matrix = confusion_matrix(y_test, predictions)

    print("\nModel Evaluation")
    print("----------------")
    print("Accuracy:", round(accuracy, 4))
    print("Baseline accuracy (always predict the majority class):", round(baseline, 4))
    print("\nConfusion Matrix:")
    print(matrix)

    joblib.dump(model, "ipl_winner_model.pkl")
    print("\nModel saved as ipl_winner_model.pkl")

    metrics = {
        "accuracy": float(accuracy),
        "baseline_accuracy": float(baseline),
        "training_records": len(X_train),
        "testing_records": len(X_test),
    }
    with open("metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)
    print("Metrics saved as metrics.json")

    return accuracy


if __name__ == "__main__":
    train_model()
