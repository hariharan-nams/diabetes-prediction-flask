import requests

url = "http://127.0.0.1:5000/prediction"

# Ensure keys perfectly match the feature names used during model training

payload = {
    "Pregnancies": 6,
    "Glucose": 148,
    "BloodPressure": 72,
    "SkinThickness": 35,
    "Insulin": 0,
    "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.627,
    "Age": 50
}

response = requests.post(url, json=payload)
# 4. Debugging safety net
if response.status_code == 200:
    print("Success!")
    print(response.json())  # This won't crash anymore!
else:
    print(f"Server returned error code: {response.status_code}")
    print("Raw response text from Flask:")
    print(response.text)