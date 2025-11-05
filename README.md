# 🌸 Iris Flower Classifier - ML Web App

A machine learning web application that predicts iris flower species using both **Scikit-learn Decision Tree** and **TensorFlow Neural Network** models, with a beautiful modern Flask frontend.

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Testing](#testing)
- [Deployment](#deployment)
- [API Endpoints](#api-endpoints)
- [Technologies Used](#technologies-used)
- [Future Enhancements](#future-enhancements)

---

## ✨ Features

✅ **Dual ML Models**
- Scikit-learn Decision Tree Classifier
- TensorFlow Keras Neural Network

✅ **Beautiful UI**
- Modern gradient design
- Responsive mobile-friendly layout
- Real-time prediction results
- Error handling and validation

✅ **Easy to Use**
- Simple form-based interface
- Quick predictions
- No technical knowledge required

✅ **Well Tested**
- Unit tests with unittest
- Integration tests
- Load testing capabilities
- cURL/API testing ready

---

## 📁 Project Structure

```
iris-classifier/
├── venv/                    # Virtual environment
├── templates/
│   └── index.html          # Frontend UI
├── models.py               # Model training script
├── app.py                  # Flask application
├── test_app.py             # Unit tests
├── test_with_requests.py   # API integration tests
├── load_test.py            # Load testing script
├── performance_test.py     # Performance benchmarking
├── dt_model.pkl            # Saved Decision Tree model
├── tf_model.h5             # Saved TensorFlow model
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

---

## 🚀 Installation

### Step 1: Clone or Download the Project

```bash
git clone https://github.com/yourusername/iris-classifier.git
cd iris-classifier
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Train Models

```bash
python models.py
```

This creates `dt_model.pkl` and `tf_model.h5`

---

## 💻 Usage

### Run the Flask App

```bash
python app.py
```

The app will start at `http://127.0.0.1:5000`

### Using the Web Interface

1. Open your browser and go to `http://127.0.0.1:5000`
2. Enter iris flower measurements:
   - Sepal Length (cm)
   - Sepal Width (cm)
   - Petal Length (cm)
   - Petal Width (cm)
3. Click "🚀 Predict Species"
4. View predictions from both models

### Example Measurements

| Species | Sepal Length | Sepal Width | Petal Length | Petal Width |
|---------|-------------|------------|-------------|------------|
| Setosa | 5.1 | 3.5 | 1.4 | 0.2 |
| Versicolor | 7.0 | 3.2 | 4.7 | 1.4 |
| Virginica | 6.3 | 3.3 | 6.0 | 2.5 |

---

## 🧪 Testing

### Unit Tests

```bash
python -m unittest test_app.py -v
```

Tests cover:
- Home page loading
- Form rendering
- Prediction functionality
- Error handling
- Model validation

### API Integration Tests

```bash
python test_with_requests.py
```

Tests all predictions and API endpoints

### Load Testing

Make sure Flask app is running in another terminal:

```bash
python load_test.py
```

Tests app with 20 concurrent requests

### Performance Testing

```bash
python performance_test.py
```

Benchmarks model prediction speed

### Manual Testing with PowerShell

```powershell
# Test with Invoke-WebRequest
$params = @{
	Uri = "http://127.0.0.1:5000/"
	Method = "POST"
	Body = @{
		sepal_length = 5.1
		sepal_width = 3.5
		petal_length = 1.4
		petal_width = 0.2
	}
}
Invoke-WebRequest @params
```

## 📦 requirements.txt

Create this file in your project root:

```
Flask==3.1.2
numpy==2.3.4
scikit-learn==1.7.2
tensorflow==2.20.0
gunicorn==21.2.0
```

---

## 🔗 API Endpoints

### GET /
Returns the home page with prediction form

**Response**: HTML page

---

### POST /
Submit prediction request

**Request Body**:
```
sepal_length=5.1&sepal_width=3.5&petal_length=1.4&petal_width=0.2
```

**Response**: HTML page with predictions
```
Decision Tree: setosa
TensorFlow NN: setosa
```

---

## 🛠️ Technologies Used

- **Backend**: Flask 3.1.2
- **ML Models**: Scikit-learn 1.7.2, TensorFlow 2.20.0
- **Data Processing**: NumPy 2.3.4
- **Deployment**: Gunicorn
- **Frontend**: HTML5, CSS3, Jinja2
- **Testing**: Python unittest, requests

---

## 📈 Model Performance

| Model | Type | Speed | Accuracy |
|-------|------|-------|----------|
| Decision Tree | Scikit-learn | Very Fast | ~97% |
| Neural Network | TensorFlow | Medium | ~98% |

Run `python performance_test.py` for detailed benchmarks

---

## 🎯 Future Enhancements

- [ ] Add confidence scores/probabilities
- [ ] Display model accuracy metrics
- [ ] Add prediction history database
- [ ] Create API documentation (Swagger/OpenAPI)
- [ ] Add more ML models (Random Forest, SVM, XGBoost)
- [ ] Add visualization of predictions
- [ ] User authentication
- [ ] Docker containerization
- [ ] Multi-language support

---

## 📝 License

This project is open source and available under the MIT License.

---

## 👨‍💻 Author

Created as a machine learning Flask web application project.

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📞 Support

For issues, questions, or suggestions, please open an issue on GitHub or contact the author.

---

## 🎓 Learning Resources

- [Flask Documentation](https://flask.palletsprojects.com/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Render Deployment Guide](https://render.com/docs)

---

**Happy Predicting! 🌸**
