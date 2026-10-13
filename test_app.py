import unittest
from app import app

MATCH = {
    "team1": "Mumbai Indians",
    "team2": "Chennai Super Kings",
    "venue": "Wankhede Stadium",
    "toss_decision": "field",
    "toss_winner_is_team1": 1,
    "year": 2022
}


class TestPredictionApplication(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    def test_health_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")

    def test_valid_prediction_response(self):
        response = self.client.post("/predict", json=MATCH)
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertIn(body["prediction_code"], [0, 1])
        self.assertIn(
            body["predicted_winner"],
            [MATCH["team1"], MATCH["team2"]]
        )

    def test_probability_is_valid(self):
        response = self.client.post("/predict", json=MATCH)
        probability = response.get_json()["team1_win_probability"]
        self.assertGreaterEqual(probability, 0)
        self.assertLessEqual(probability, 1)

    def test_missing_field_validation(self):
        response = self.client.post(
            "/predict",
            json={"team1": "Mumbai Indians", "team2": "Chennai Super Kings"}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("missing_fields", response.get_json())

    def test_same_team_validation(self):
        match = dict(MATCH, team2=MATCH["team1"])
        response = self.client.post("/predict", json=match)
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
