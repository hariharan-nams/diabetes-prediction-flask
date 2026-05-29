import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import pickle

# 1. Load the package containing both the model and the scaler
with open('diabetes_model_package.pkl', "rb") as file_obj:
    model_package = pickle.load(file_obj)

# Extract model and scaler individually
scaler = model_package["scaler"]
trained_model = model_package["model"]

# Median values computed during training phase
# Note: Ensure these values perfectly match your actual training data medians
MEDIAN_VALUES = {
    'Glucose': 117.0,
    'BloodPressure': 72.0,
    'SkinThickness': 23.0,
    'Insulin': 30.5,
    'BMI': 32.0
}

diabetes_app = Flask(__name__)

@diabetes_app.route('/')
def landing():
    return 'Welcome to Diabetes Prediction API'

@diabetes_app.route('/prediction', methods=["POST"])
def tree_prediction():
    try:
        # 1. Extract the JSON payload sent by the client/user
        data = request.get_json()

        # 2. Convert the incoming JSON object directly into a Pandas DataFrame row
        df = pd.DataFrame([data])

        # 3. Missing Value Handling (Impute structural 0s with training medians)
        for col, median_val in MEDIAN_VALUES.items():
            if col in df.columns:
                # Replace invalid 0 values with the pre-calculated median
                df[col] = df[col].replace(0, median_val)
                # Fill any missing/null columns with the median as a fallback
                df[col] = df[col].fillna(median_val)

        # 4. Feature Scaling (Crucial Step!)
        # Transform the incoming data using the pre-fitted training scaler rules
        df_scaled = scaler.transform(df)

        # 5. Generate Model Prediction
        prediction = trained_model.predict(df_scaled)[0]

        # 6. Return the finalized JSON response back to the client
        return jsonify({
            'status': 'success',
            'prediction': int(prediction),
            'result': 'Diabetes Positive' if int(prediction) == 1 else 'Diabetes Negative'
        })

    except Exception as e:
        # Catch errors gracefully and return the error message for debugging
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    diabetes_app.run(debug=True)
