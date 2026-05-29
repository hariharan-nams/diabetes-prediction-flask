import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import pickle

# 1. மாடல் மற்றும் ஸ்கேலர் அடங்கிய பேக்கேஜை லோடு செய்தல்
with open('diabetes_model_package.pkl', "rb") as file_obj:
    model_package = pickle.load(file_obj)

# தனித்தனியாகப் பிரித்தெடுத்தல்
scaler = model_package["scaler"]
trained_model = model_package["model"]

# ட்ரெய்னிங்கின் போது நாம் பயன்படுத்திய மீடியன் மதிப்புகள் (உதாரணத்திற்கு)
# குறிப்பு: உங்கள் உண்மையான ட்ரெய்னிங் தரவின் மீடியன் மதிப்புகளை இங்கு கொடுக்க வேண்டும்
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
        # 1. பயனரிடமிருந்து JSON தரவைப் பெறுதல்
        data = request.get_json()

        # 2. அதை DataFrame-ஆக மாற்றுதல்
        df = pd.DataFrame([data])

        # 3. Missing Value Handling (0-வை Median கொண்டு மாற்றுதல்)
        for col, median_val in MEDIAN_VALUES.items():
            if col in df.columns:
                # பயனர் அனுப்பிய மதிப்பில் 0 இருந்தால் அதை Median மதிப்பாக மாற்றுகிறோம்
                df[col] = df[col].replace(0, median_val)
                # ஒருவேளை அந்த காலமே விடுபட்டிருந்தால் அதற்கும் Median நிரப்புகிறோம்
                df[col] = df[col].fillna(median_val)

        # 4. Feature Scaling (மிக முக்கியம்!)
        # ட்ரெய்ன் செய்யப்பட்ட ஸ்கேலரை வைத்து புதிய தரவை மாற்றுகிறோம்
        df_scaled = scaler.transform(df)

        # 5. Prediction செய்தல்
        prediction = trained_model.predict(df_scaled)[0]

        # 6. முடிவை அனுப்புதல்
        return jsonify({
            'status': 'success',
            'prediction': int(prediction),
            'result': 'Diabetes Positive' if int(prediction) == 1 else 'Diabetes Negative'
        })

    except Exception as e:
        # ஏதாவது பிழை ஏற்பட்டால் அதை சரியாகக் காட்டுவதற்கு
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    diabetes_app.run(debug=True)