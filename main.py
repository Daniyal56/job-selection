from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow requests from APEX

# Load model
model = joblib.load("job_applicability_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Extract input values
    age = data['age']
    experience = data['experience']
    designation = data['designation']  # Assume it's already numeric
    qualification = data['qualification']  # Assume it's already numeric

    features = np.array([[age, experience, designation, qualification]])
    prediction = model.predict(features)[0]

    return jsonify({'result': int(prediction)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
