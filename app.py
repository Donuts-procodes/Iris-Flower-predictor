from flask import Flask, render_template, request
import numpy as np
import pickle
import tensorflow as tf

app = Flask(__name__)

# Load models
with open('dt_model.pkl', 'rb') as f:
    dt_model = pickle.load(f)

tf_model = tf.keras.models.load_model('tf_model.h5')

iris_target_names = ['setosa', 'versicolor', 'virginica']

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            features = [float(request.form['sepal_length']),
                        float(request.form['sepal_width']),
                        float(request.form['petal_length']),
                        float(request.form['petal_width'])]
            features_np = np.array(features).reshape(1, -1)

            # Predict using Decision Tree
            dt_pred = dt_model.predict(features_np)[0]
            dt_pred_name = iris_target_names[dt_pred]

            # Predict using TensorFlow model
            tf_pred_probs = tf_model.predict(features_np)
            tf_pred = np.argmax(tf_pred_probs, axis=1)[0]
            tf_pred_name = iris_target_names[tf_pred]

            prediction = {
                'Decision Tree': dt_pred_name,
                'TensorFlow NN': tf_pred_name
            }
        except Exception as e:
            prediction = {'error': str(e)}

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
