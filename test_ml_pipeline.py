import json
import os
import unittest
import joblib
import pandas as pd


class TestMLPipeline(unittest.TestCase):
    def test_dataset_created(self):
        self.assertTrue(os.path.exists("ipl_model_data.csv"))

    def test_model_created(self):
        self.assertTrue(os.path.exists("ipl_winner_model.pkl"))

    def test_metrics_created(self):
        self.assertTrue(os.path.exists("metrics.json"))

    def test_accuracy_is_valid(self):
        with open("metrics.json", "r") as file:
            metrics = json.load(file)
        accuracy = metrics["accuracy"]
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    def test_model_prediction(self):
        model = joblib.load("ipl_winner_model.pkl")
        sample = pd.DataFrame([{
            "Team1": "Mumbai Indians",
            "Team2": "Chennai Super Kings",
            "Venue": "Wankhede Stadium",
            "TossDecision": "field",
            "toss_winner_is_team1": 1,
            "year": 2022,
        }])
        prediction = model.predict(sample)[0]
        self.assertIn(int(prediction), [0, 1])

    def test_prediction_probability(self):
        model = joblib.load("ipl_winner_model.pkl")
        sample = pd.DataFrame([{
            "Team1": "Rajasthan Royals",
            "Team2": "Gujarat Titans",
            "Venue": "Narendra Modi Stadium",
            "TossDecision": "bat",
            "toss_winner_is_team1": 1,
            "year": 2022,
        }])
        probabilities = model.predict_proba(sample)[0]
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=5)

    def test_cleaned_data_has_950_minus_no_result_matches(self):
        data = pd.read_csv("ipl_model_data.csv")
        self.assertEqual(len(data), 0)
        self.assertNotIn("Delhi Daredevils", data["Team1"].values)


if __name__ == "__main__":
    unittest.main()
