from flask import Flask,jsonify,request
import joblib
import numpy as np

from flask_cors import CORS


app=Flask(__name__)
model=joblib.load('drug_model.pkl')
label_encoder=joblib.load('label_encoder.pkl')

CORS(app)

@app.route('/')
def home():
    return "Drug prediction APi is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()  # Get data from the request
        features = np.array([[
            data['Age'],
            data['Na_to_K'],
            data['SEX_M'],
            data['BP_LOW'],
            data['BP_NORMAL'],
            data['Cholesterol_NORMAL']
        ]])

        prediction = model.predict(features)
        drug_name = label_encoder.inverse_transform(prediction)[0]

        return jsonify({"Predicted drug": drug_name})  # Send the response back
    except Exception as e:
        return jsonify({"error": str(e)}), 400  # Return error if any

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)