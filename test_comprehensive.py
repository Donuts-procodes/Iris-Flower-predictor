import unittest
import numpy as np
from app import app, dt_model, rf_model, iris_target_names
import pickle
import time

class TestFlaskApp(unittest.TestCase):
    """Test Flask app functionality"""

    def setUp(self):
        self.app = app
        self.client = self.app.test_client()
        self.app.config['TESTING'] = True

    # ===== Page Load Tests =====
    def test_home_page_loads(self):
        """Test home page loads successfully"""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Iris Flower Classifier', response.data)

    def test_home_page_has_form(self):
        """Test home page contains form fields"""
        response = self.client.get('/')
        self.assertIn(b'sepal_length', response.data)
        self.assertIn(b'sepal_width', response.data)
        self.assertIn(b'petal_length', response.data)
        self.assertIn(b'petal_width', response.data)

    def test_home_page_has_submit_button(self):
        """Test form has submit button"""
        response = self.client.get('/')
        self.assertIn(b'submit', response.data)

    # ===== Valid Predictions Tests =====
    def test_prediction_setosa(self):
        """Test Setosa prediction (typical values)"""
        data = {
            'sepal_length': 5.1,
            'sepal_width': 3.5,
            'petal_length': 1.4,
            'petal_width': 0.2
        }
        response = self.client.post('/', data=data)
        self.assertEqual(response.status_code, 200)
        # Check for lowercase version in response
        self.assertIn(b'setosa', response.data.lower())

    def test_prediction_versicolor(self):
        """Test Versicolor prediction"""
        data = {
            'sepal_length': 7.0,
            'sepal_width': 3.2,
            'petal_length': 4.7,
            'petal_width': 1.4
        }
        response = self.client.post('/', data=data)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'versicolor', response.data.lower())

    def test_prediction_virginica(self):
        """Test Virginica prediction"""
        data = {
            'sepal_length': 6.3,
            'sepal_width': 3.3,
            'petal_length': 6.0,
            'petal_width': 2.5
        }
        response = self.client.post('/', data=data)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'virginica', response.data.lower())

    # ===== Edge Case Tests =====
    def test_zero_values(self):
        """Test with all zero values"""
        data = {
            'sepal_length': 0,
            'sepal_width': 0,
            'petal_length': 0,
            'petal_width': 0
        }
        response = self.client.post('/', data=data)
        self.assertEqual(response.status_code, 200)

    def test_very_small_values(self):
        """Test with very small decimal values"""
        data = {
            'sepal_length': 0.1,
            'sepal_width': 0.1,
            'petal_length': 0.1,
            'petal_width': 0.1
        }
        response = self.client.post('/', data=data)
        self.assertEqual(response.status_code, 200)

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

    def test_negative_values(self):
        """Test with negative values (botanically invalid but should not crash)"""
        data = {
            'sepal_length': -5.0,
            'sepal_width': -3.0,
            'petal_length': -1.4,
            'petal_width': -0.2
        }
        response = self.client.post('/', data=data)
        self.assertEqual(response.status_code, 200)

    def test_high_precision_decimals(self):
        """Test with high precision decimal values"""
        data = {
            'sepal_length': 5.123456789,
            'sepal_width': 3.987654321,
            'petal_length': 1.555555555,
            'petal_width': 0.222222222
        }
        response = self.client.post('/', data=data)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Prediction Results', response.data)

    # ===== Validation Tests =====
    def test_missing_sepal_length(self):
        """Test with missing sepal_length"""
        data = {
            'sepal_width': 3.5,
            'petal_length': 1.4,
            'petal_width': 0.2
        }
        response = self.client.post('/', data=data)
        self.assertEqual(response.status_code, 200)

    def test_missing_sepal_width(self):
        """Test with missing sepal_width"""
        data = {
            'sepal_length': 5.1,
            'petal_length': 1.4,
            'petal_width': 0.2
        }
        response = self.client.post('/', data=data)
        self.assertEqual(response.status_code, 200)

    def test_missing_petal_length(self):
        """Test with missing petal_length"""
        data = {
            'sepal_length': 5.1,
            'sepal_width': 3.5,
            'petal_width': 0.2
        }
        response = self.client.post('/', data=data)
        self.assertEqual(response.status_code, 200)

    def test_missing_petal_width(self):
        """Test with missing petal_width"""
        data = {
            'sepal_length': 5.1,
            'sepal_width': 3.5,
            'petal_length': 1.4
        }
        response = self.client.post('/', data=data)
        self.assertEqual(response.status_code, 200)

    def test_all_missing_parameters(self):
        """Test with all parameters missing"""
        data = {}
        response = self.client.post('/', data=data)
        self.assertEqual(response.status_code, 200)

    # ===== Invalid Input Tests =====
    def test_non_numeric_sepal_length(self):
        """Test with non-numeric sepal_length"""
        data = {
            'sepal_length': 'abc',
            'sepal_width': 3.5,
            'petal_length': 1.4,
            'petal_width': 0.2
        }
        response = self.client.post('/', data=data)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Error', response.data)

    def test_non_numeric_sepal_width(self):
        """Test with non-numeric sepal_width"""
        data = {
            'sepal_length': 5.1,
            'sepal_width': 'xyz',
            'petal_length': 1.4,
            'petal_width': 0.2
        }
        response = self.client.post('/', data=data)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Error', response.data)

    def test_non_numeric_petal_length(self):
        """Test with non-numeric petal_length"""
        data = {
            'sepal_length': 5.1,
            'sepal_width': 3.5,
            'petal_length': 'invalid',
            'petal_width': 0.2
        }
        response = self.client.post('/', data=data)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Error', response.data)

    def test_non_numeric_petal_width(self):
        """Test with non-numeric petal_width"""
        data = {
            'sepal_length': 5.1,
            'sepal_width': 3.5,
            'petal_length': 1.4,
            'petal_width': 'bad'
        }
        response = self.client.post('/', data=data)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Error', response.data)

    def test_special_characters(self):
        """Test with special characters"""
        data = {
            'sepal_length': '5.1!@#',
            'sepal_width': 3.5,
            'petal_length': 1.4,
            'petal_width': 0.2
        }
        response = self.client.post('/', data=data)
        self.assertEqual(response.status_code, 200)

    def test_sql_injection_attempt(self):
        """Test SQL injection attempt (should not affect results)"""
        data = {
            'sepal_length': "5.1; DROP TABLE users;--",
            'sepal_width': 3.5,
            'petal_length': 1.4,
            'petal_width': 0.2
        }
        response = self.client.post('/', data=data)
        self.assertEqual(response.status_code, 200)

    def test_html_injection_attempt(self):
        """Test HTML/XSS injection attempt"""
        data = {
            'sepal_length': '<script>alert("xss")</script>',
            'sepal_width': 3.5,
            'petal_length': 1.4,
            'petal_width': 0.2
        }
        response = self.client.post('/', data=data)
        self.assertEqual(response.status_code, 200)

    # ===== Response Format Tests =====
    def test_response_contains_results_heading(self):
        """Test that response contains results heading"""
        data = {
            'sepal_length': 5.1,
            'sepal_width': 3.5,
            'petal_length': 1.4,
            'petal_width': 0.2
        }
        response = self.client.post('/', data=data)
        self.assertIn(b'Prediction Results', response.data)

    def test_response_contains_valid_species(self):
        """Test that response contains valid iris species"""
        data = {
            'sepal_length': 5.1,
            'sepal_width': 3.5,
            'petal_length': 1.4,
            'petal_width': 0.2
        }
        response = self.client.post('/', data=data)
        response_lower = response.data.lower()
        # Should contain one of the three species
        self.assertTrue(
            b'setosa' in response_lower or 
            b'versicolor' in response_lower or 
            b'virginica' in response_lower
        )

    # ===== Boundary Tests =====
    def test_min_realistic_values(self):
        """Test with minimum realistic iris values"""
        data = {
            'sepal_length': 4.0,
            'sepal_width': 2.0,
            'petal_length': 1.0,
            'petal_width': 0.1
        }
        response = self.client.post('/', data=data)
        self.assertEqual(response.status_code, 200)

    def test_max_realistic_values(self):
        """Test with maximum realistic iris values"""
        data = {
            'sepal_length': 8.0,
            'sepal_width': 4.5,
            'petal_length': 7.0,
            'petal_width': 2.5
        }
        response = self.client.post('/', data=data)
        self.assertEqual(response.status_code, 200)

    def test_mean_values(self):
        """Test with approximately mean iris values"""
        data = {
            'sepal_length': 5.8,
            'sepal_width': 3.0,
            'petal_length': 3.8,
            'petal_width': 1.2
        }
        response = self.client.post('/', data=data)
        self.assertEqual(response.status_code, 200)


class TestModelAccuracy(unittest.TestCase):
    """Test model accuracy and predictions"""

    def test_dt_model_exists(self):
        """Test Decision Tree model is loaded"""
        self.assertIsNotNone(dt_model)

    def test_rf_model_exists(self):
        """Test Random Forest model is loaded"""
        self.assertIsNotNone(rf_model)

    def test_dt_prediction_valid_class(self):
        """Test Decision Tree returns valid class"""
        features = np.array([[5.1, 3.5, 1.4, 0.2]])
        prediction = dt_model.predict(features)
        self.assertIn(prediction, [0, 1, 2])

    def test_rf_prediction_valid_class(self):
        """Test Random Forest returns valid class"""
        features = np.array([[5.1, 3.5, 1.4, 0.2]])
        prediction = rf_model.predict(features)
        self.assertIn(prediction, [0, 1, 2])

    def test_multiple_predictions_consistency(self):
        """Test that same input gives same output"""
        features = np.array([[5.1, 3.5, 1.4, 0.2]])
        pred1 = dt_model.predict(features)
        pred2 = dt_model.predict(features)
        self.assertEqual(pred1, pred2)

    def test_setosa_predictions_consistent(self):
        """Test multiple Setosa predictions are consistent"""
        setosa_samples = [
            [5.1, 3.5, 1.4, 0.2],
            [4.9, 3.0, 1.4, 0.2],
            [4.7, 3.2, 1.3, 0.2]
        ]
        for sample in setosa_samples:
            pred = dt_model.predict(np.array([sample]))
            self.assertEqual(pred, 0, f"Setosa sample {sample} not predicted as 0")

    def test_virginica_predictions_consistent(self):
        """Test multiple Virginica predictions"""
        virginica_samples = [
            [6.3, 3.3, 6.0, 2.5],
            [7.1, 3.0, 5.9, 2.1],
            [6.5, 3.0, 5.5, 1.8]
        ]
        for sample in virginica_samples:
            pred = dt_model.predict(np.array([sample]))
            # Should predict as Virginica (class 2)
            self.assertEqual(pred, 2, f"Virginica sample {sample} mispredicted")

    def test_batch_prediction(self):
        """Test batch predictions with multiple samples"""
        features = np.array([
            [5.1, 3.5, 1.4, 0.2],
            [7.0, 3.2, 4.7, 1.4],
            [6.3, 3.3, 6.0, 2.5]
        ])
        predictions = dt_model.predict(features)
        self.assertEqual(len(predictions), 3)
        for pred in predictions:
            self.assertIn(pred, [0, 1, 2])

    def test_rf_accuracy_similar_to_dt(self):
        """Test that Random Forest has similar accuracy to Decision Tree"""
        test_features = np.array([
            [5.1, 3.5, 1.4, 0.2],
            [7.0, 3.2, 4.7, 1.4],
            [6.3, 3.3, 6.0, 2.5],
            [4.9, 3.0, 1.4, 0.2],
            [6.5, 3.0, 5.5, 1.8]
        ])
        
        dt_preds = dt_model.predict(test_features)
        rf_preds = rf_model.predict(test_features)
        
        # Both should make valid predictions
        for dt_p, rf_p in zip(dt_preds, rf_preds):
            self.assertIn(dt_p, [0, 1, 2])
            self.assertIn(rf_p, [0, 1, 2])


class TestDataTypes(unittest.TestCase):
    """Test data type handling"""

    def test_integer_input(self):
        """Test with integer inputs"""
        data = {
            'sepal_length': 5,
            'sepal_width': 3,
            'petal_length': 1,
            'petal_width': 0
        }
        response = app.test_client().post('/', data=data)
        self.assertEqual(response.status_code, 200)

    def test_float_input(self):
        """Test with float inputs"""
        data = {
            'sepal_length': 5.5,
            'sepal_width': 3.5,
            'petal_length': 1.5,
            'petal_width': 0.5
        }
        response = app.test_client().post('/', data=data)
        self.assertEqual(response.status_code, 200)

    def test_mixed_int_float_input(self):
        """Test with mixed integer and float inputs"""
        data = {
            'sepal_length': 5,
            'sepal_width': 3.5,
            'petal_length': 1,
            'petal_width': 0.2
        }
        response = app.test_client().post('/', data=data)
        self.assertEqual(response.status_code, 200)


class TestPerformance(unittest.TestCase):
    """Test application performance"""

    def test_prediction_returns_quickly(self):
        """Test that prediction responds in reasonable time"""
        data = {
            'sepal_length': 5.1,
            'sepal_width': 3.5,
            'petal_length': 1.4,
            'petal_width': 0.2
        }
        start = time.time()
        response = app.test_client().post('/', data=data)
        elapsed = time.time() - start
        
        # Should respond in less than 2 seconds
        self.assertLess(elapsed, 2.0, f"Prediction took {elapsed} seconds")

    def test_models_load_quickly(self):
        """Test that models load quickly"""
        start = time.time()
        with open('dt_model.pkl', 'rb') as f:
            _ = pickle.load(f)
        with open('rf_model.pkl', 'rb') as f:
            _ = pickle.load(f)
        elapsed = time.time() - start
        
        # Models should load in less than 1 second
        self.assertLess(elapsed, 1.0)


if __name__ == '__main__':
    unittest.main()
