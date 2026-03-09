import joblib
import numpy as np

# Load model, scaler, and encoders
model = joblib.load("models/fraud_model.pkl")
scaler = joblib.load("models/scaler.pkl")
encoders = joblib.load("models/encoders.pkl")

def predict(data_dict):
    """
    data_dict should be a dictionary with keys:
    amount, transaction_type, merchant_category, location,
    device_used, time_since_last_transaction,
    spending_deviation_score, velocity_score,
    geo_anomaly_score, payment_channel
    """

    # Normalize categorical inputs (strip spaces, lowercase)
    for col in encoders.keys():
        if col in data_dict:
            data_dict[col] = str(data_dict[col]).strip().lower()

    # Encode categorical fields using saved encoders
    for col, encoder in encoders.items():
        if col in data_dict:
            # Handle unseen labels gracefully
            try:
                data_dict[col] = encoder.transform([data_dict[col]])[0]
            except ValueError:
                # If unseen category, map to a default (e.g., first class)
                data_dict[col] = 0

    # Collect features in the same order used during training
    ordered_features = [
        data_dict["amount"],
        data_dict["transaction_type"],
        data_dict["merchant_category"],
        data_dict["location"],
        data_dict["device_used"],
        data_dict["time_since_last_transaction"],
        data_dict["spending_deviation_score"],
        data_dict["velocity_score"],
        data_dict["geo_anomaly_score"],
        data_dict["payment_channel"]
    ]

    # Scale and predict
    data = np.array(ordered_features).reshape(1, -1)
    data_scaled = scaler.transform(data)
    pred = model.predict(data_scaled)[0]
    prob = model.predict_proba(data_scaled)[0][1]

    return pred, prob