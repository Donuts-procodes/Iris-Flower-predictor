import unittest
import json
import numpy as np
from app import app, dt_model, tf_model, iris_target_names
import pickle

class TestFlaskApp(unittest.TestCase):
    """Test cases for Flask ML app"""

    def setUp(self):
        """Set up test client before each test"""
        self.app = app
        self.client = self.app.test_client()
        self.app.config['TESTING'] = True

    def test_home_page_get(self):
        """Test home page loads correctly"""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Iris Flower Classifier', response.data)

    def test_home_page_contains_form(self):
        """Test home page contains form elements"""
        response = self.client.get('/')
        self.assertIn(b'sepal_length', response.data)
        self.assertIn(b'sepal_width', response.data)
        self.assertIn(b'petal_length', response.data)
        self.assertIn(b'petal_width', response.data)

    def test_prediction_setosa(self):
        """Test prediction for Setosa iris"""
        data = {
            'sepal_length': 5.1,
            'sepal_width': 3.5,
            'petal_length': 1.4,
            'petal_width': 0.2
        }
        response = self.client.post('/', data=data)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Prediction Results', response.data)
        # Check for lowercase version
        self.assertIn(b'setosa', response.data)

    def test_prediction_versicolor(self):
        """Test prediction for Versicolor iris"""
        data = {
            'sepal_length': 7.0,
            'sepal_width': 3.2,
            'petal_length': 4.7,
            'petal_width': 1.4
        }
        response = self.client.post('/', data=data)
        self.assertEqual(response.status_code, 200)
        # Check for lowercase version
        self.assertIn(b'versicolor', response.data)

    def test_prediction_virginica(self):
        """Test prediction for Virginica iris"""
        data = {
            'sepal_length': 6.3,
            'sepal_width': 3.3,
            'petal_length': 6.0,
            'petal_width': 2.5
        }
        response = self.client.post('/', data=data)
        self.assertEqual(response.status_code, 200)
        # Check for lowercase version
        self.assertIn(b'virginica', response.data)

    def test_missing_parameters(self):
        """Test with missing form parameters"""
        data = {
            'sepal_length': 5.1,
            'sepal_width': 3.5
            # Missing petal_length and petal_width
        }
        response = self.client.post('/', data=data)
        # Should still return 200 but might show error
        self.assertEqual(response.status_code, 200)

    def test_invalid_input_non_numeric(self):
        """Test with non-numeric input"""
        data = {
            'sepal_length': 'abc',
            'sepal_width': 3.5,
            'petal_length': 1.4,
            'petal_width': 0.2
        }
        response = self.client.post('/', data=data)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Error', response.data)

    def test_negative_values(self):
        """Test with negative values"""
        data = {
            'sepal_length': -5.1,
            'sepal_width': 3.5,
            'petal_length': 1.4,
            'petal_width': 0.2
        }
        response = self.client.post('/', data=data)
        self.assertEqual(response.status_code, 200)
        # App should still process (no validation against negatives)

    def test_very_large_values(self):
        """Test with very large values"""
        data = {
            'sepal_length': 999.9,
            'sepal_width': 999.9,
            'petal_length': 999.9,
            'petal_width': 999.9
        }
        response = self.client.post('/', data=data)
        self.assertEqual(response.status_code, 200)
        # Should not crash

    def test_zero_values(self):
        """Test with zero values"""
        data = {
            'sepal_length': 0,
            'sepal_width': 0,
            'petal_length': 0,
            'petal_width': 0
        }
        response = self.client.post('/', data=data)
        self.assertEqual(response.status_code, 200)

    def test_decimal_precision(self):
        """Test with high decimal precision"""
        data = {
            'sepal_length': 5.123456789,
            'sepal_width': 3.987654321,
            'petal_length': 1.555555555,
            'petal_width': 0.222222222
        }
        response = self.client.post('/', data=data)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Prediction Results', response.data)


class TestModelAccuracy(unittest.TestCase):
    """Test model accuracy and predictions"""

    def test_dt_model_exists(self):
        """Test Decision Tree model is loaded"""
        self.assertIsNotNone(dt_model)

    def test_tf_model_exists(self):
        """Test TensorFlow model is loaded"""
        self.assertIsNotNone(tf_model)

    def test_dt_prediction_shape(self):
        """Test Decision Tree returns correct prediction shape"""
        features = np.array([[5.1, 3.5, 1.4, 0.2]])
        prediction = dt_model.predict(features)
        self.assertEqual(len(prediction), 1)
        self.assertIn(prediction[0], [0, 1, 2])

    def test_tf_prediction_shape(self):
        """Test TensorFlow model returns correct prediction shape"""
        features = np.array([[5.1, 3.5, 1.4, 0.2]])
        prediction = tf_model.predict(features, verbose=0)
        self.assertEqual(prediction.shape[0], 1)
        self.assertEqual(prediction.shape[1], 3)

    def test_predictions_are_valid_classes(self):
        """Test predictions return valid class indices"""
        test_features = np.array([
            [5.1, 3.5, 1.4, 0.2],
            [7.0, 3.2, 4.7, 1.4],
            [6.3, 3.3, 6.0, 2.5]
        ])
        
        for features in test_features:
            dt_pred = dt_model.predict(features.reshape(1, -1))[0]
            tf_pred_probs = tf_model.predict(features.reshape(1, -1), verbose=0)
            tf_pred = np.argmax(tf_pred_probs, axis=1)[0]
            
            self.assertIn(dt_pred, [0, 1, 2])
            self.assertIn(tf_pred, [0, 1, 2])


if __name__ == '__main__':
    unittest.main()
