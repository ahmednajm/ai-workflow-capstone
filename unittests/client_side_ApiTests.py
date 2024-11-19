#!/usr/bin/env python3

import unittest
import json
from client_side.api import server

class TestApi(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = server.test_client()  # Create a test client for the Flask app
        cls.app.testing = True          # Enable testing mode

    def test_ping(self):
        """Test the /ping endpoint."""
        response = self.app.get('/ping')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['status'], 200)
        self.assertEqual(response.json['message'], 'Ping successful!')

    def test_train_invalid_model(self):
        """Test the /train endpoint with an invalid model name."""
        data = {
            "model": "invalid_model",
            "country": "United Kingdom",
            "mode": "prod"
        }
        response = self.app.post('/train', data=json.dumps(data), content_type='application/json')
        self.assertEqual(response.status_code, 400)
        self.assertIn("Model 'invalid_model' is not recognized.", response.json['error'])

    def test_train_success(self):
        """Test the /train endpoint with valid data."""
        data = {
            "model": "rf",
            "model_scaler": "ss",
            "country": "United Kingdom",
            "mode": "prod"
        }
        response = self.app.post('/train', data=json.dumps(data), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        self.assertIn("Training completed successfully", response.json['status'])

    def test_predict_missing_parameters(self):
        """Test the /predict endpoint with missing parameters."""
        data = {
            "country": "United Kingdom",
            "mode": "prod"
        }
        response = self.app.post('/predict', data=json.dumps(data), content_type='application/json')
        self.assertEqual(response.status_code, 400)
        self.assertIn("Missing required parameters", response.json['error'])

    def test_predict_success(self):
        """Test the /predict endpoint with valid data."""
        data = {
            "preprocessed_data_dir": "./data/preprocessed_data",
            "country": "World",
            "year": 2019,
            "month": 8,
            "day": 11,
            "mode": "prod"
        }
        response = self.app.post('/predict', data=json.dumps(data), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        self.assertIn("Prediction completed successfully", response.json['status'])

    def test_list_logs_success(self):
        """Test the /logs/ endpoint when log directory exists."""
        response = self.app.get('/logs/')
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.json, list)


    def test_logs_success(self):
        """Test the /logs/<filename> endpoint with a valid file."""
        # Assuming "prod-training-logfile.csv" exists in the LOG_DIR
        response = self.app.get('/logs/prod-training-logfile.csv')
        self.assertEqual(response.status_code, 200)
        self.assertIn("attachment", response.headers['Content-Disposition'])

if __name__ == '__main__':
    unittest.main(failfast=True)