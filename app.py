from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

with open('dt_model.pkl', 'rb') as f:
    dt_model = pickle.load(f)

with open('rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

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

            dt_pred = dt_model.predict(features_np)[0]
            dt_pred_name = iris_target_names[dt_pred]

            rf_pred = rf_model.predict(features_np)[0]
            rf_pred_name = iris_target_names[rf_pred]

            prediction = {
                'Decision Tree': dt_pred_name,
                'Random Forest': rf_pred_name
            }
        except Exception as e:
            prediction = {'error': str(e)}

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
