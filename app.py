
from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd

# Load the model and encoder from the pickle file
with open('linear_regression_model_bundle.pkl', 'rb') as f:
    bundle = pickle.load(f)

model = bundle['model']
encoder = bundle['encoder']

app = Flask(__name__)

@app.route('/')
def home():
    return "Linear Regression Model is up and running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Expecting JSON input
        input_data = request.get_json()

        # Convert input JSON to DataFrame
        input_df = pd.DataFrame([input_data])

        # Apply encoder (if any preprocessing is needed)
        transformed_data = encoder.transform(input_df)

        # Make prediction
        prediction = model.predict(transformed_data)

        return jsonify({'prediction': prediction.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
