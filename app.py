from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

# Load the model and encoder
with open("linear_regression_model_bundle.pkl", "rb") as f:
    bundle = pickle.load(f)
    model = bundle["model"]
    encoder = bundle["encoder"]

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        input_data = {
            'Category': request.form['Category'],
            'Region': request.form['Region'],
            'Weather': request.form['Weather'],
            'Promotion': request.form['Promotion'],
            'Seasonality': request.form['Seasonality'],
            'Inventory': float(request.form['Inventory']),
            'Sales': float(request.form['Sales']),
            'Price': float(request.form['Price']),
            'Discount': float(request.form['Discount'])
        }

        # Create dataframe and transform
        input_df = pd.DataFrame([input_data])
        X_transformed = encoder.transform(input_df)

        # Predict
        prediction = model.predict(X_transformed)
        return render_template('index.html', prediction_text=f"Predicted Demand: {prediction[0]:.2f}")

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)

