from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load Iris data
iris = load_iris()
X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scikit-learn Decision Tree model
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)

# Save sklearn model
with open("dt_model.pkl", "wb") as f:
    pickle.dump(dt_model, f)

# TensorFlow Neural Network model
tf_model = Sequential(
    [
        Dense(10, activation="relu", input_shape=(4,)),
        Dense(10, activation="relu"),
        Dense(3, activation="softmax"),
    ]
)
tf_model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
tf_model.fit(X_train, y_train, epochs=20, verbose=0)

# Save TensorFlow model
tf_model.save("tf_model.h5")
