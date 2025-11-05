import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model 1: Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
pickle.dump(dt, open('dt_model.pkl', 'wb'))
print("✓ Decision Tree saved")

# Model 2: Random Forest
rf = RandomForestClassifier(n_estimators=10)
rf.fit(X_train, y_train)
pickle.dump(rf, open('rf_model.pkl', 'wb'))
print("✓ Random Forest saved")
