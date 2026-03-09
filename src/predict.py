import joblib
import numpy as np

model = joblib.load("models/fraud_model.pkl")
scaler = joblib.load("models/scaler.pkl")

def predict(transaction):
    transaction = np.array(transaction).reshape(1, -1)
    transaction = scaler.transform(transaction)

    prediction = model.predict(transaction)
    probability = model.predict_proba(transaction)[0][1]

    return prediction[0], probability