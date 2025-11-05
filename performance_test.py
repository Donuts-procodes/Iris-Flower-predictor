import time
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pickle
from tensorflow.keras.models import load_model

def performance_test():
    """Test model prediction speed"""
    
    # Load models
    with open('dt_model.pkl', 'rb') as f:
        dt_model = pickle.load(f)
    
    tf_model = load_model('tf_model.h5')
    
    # Test data
    iris = load_iris()
    X_test = iris.data[:10]  # Test on 10 samples
    
    print("Performance Test\n" + "="*50)
    
    # Decision Tree Performance
    print("\nDecision Tree Model:")
    start = time.time()
    for _ in range(100):
        dt_model.predict(X_test)
    dt_time = time.time() - start
    print(f"100 predictions on {len(X_test)} samples: {dt_time*1000:.2f} ms")
    print(f"Avg per prediction: {(dt_time/100)*1000:.4f} ms")
    
    # TensorFlow Performance
    print("\nTensorFlow Neural Network Model:")
    start = time.time()
    for _ in range(100):
        tf_model.predict(X_test, verbose=0)
    tf_time = time.time() - start
    print(f"100 predictions on {len(X_test)} samples: {tf_time*1000:.2f} ms")
    print(f"Avg per prediction: {(tf_time/100)*1000:.4f} ms")
    
    print(f"\nDecision Tree is {tf_time/dt_time:.2f}x faster")
    print("="*50)

if __name__ == '__main__':
    performance_test()
