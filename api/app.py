from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import joblib
import numpy as np

app = FastAPI(
    title="Fraud Detection API",
    description="🚨 Real-time Fraud Prediction Service",
    version="1.0.0"
)

# Load model + scaler + encoders
model = joblib.load("models/fraud_model.pkl")
scaler = joblib.load("models/scaler.pkl")
encoders = joblib.load("models/encoders.pkl")

# Home route with styled HTML form
@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
        <head>
            <title>Fraud Detection Dashboard</title>
            <style>
                body { font-family: Arial, sans-serif; background: #f4f6f9; text-align: center; }
                h1 { color: #2c3e50; }
                .card {
                    background: white; padding: 20px; margin: 30px auto;
                    width: 70%; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                }
                input, select {
                    margin: 8px; padding: 8px; width: 60%;
                    border-radius: 5px; border: 1px solid #ccc;
                }
                button {
                    background: #2c3e50; color: white; padding: 10px 20px;
                    border: none; border-radius: 5px; cursor: pointer;
                }
                button:hover { background: #34495e; }
            </style>
        </head>
        <body>
            <div class="card">
                <h1>🚀 Fraud Detection Form</h1>
                <form action="/predict" method="post">
                    <input type="number" step="0.01" name="amount" placeholder="Amount" required><br>
                    <input type="text" name="transaction_type" placeholder="Transaction Type" required><br>
                    <input type="text" name="merchant_category" placeholder="Merchant Category" required><br>
                    <input type="text" name="location" placeholder="Location" required><br>
                    <input type="text" name="device_used" placeholder="Device Used" required><br>
                    <input type="number" name="time_since_last_transaction" placeholder="Time Since Last Transaction" required><br>
                    <input type="number" step="0.01" name="spending_deviation_score" placeholder="Spending Deviation Score" required><br>
                    <input type="number" name="velocity_score" placeholder="Velocity Score" required><br>
                    <input type="number" step="0.01" name="geo_anomaly_score" placeholder="Geo Anomaly Score" required><br>
                    <input type="text" name="payment_channel" placeholder="Payment Channel" required><br>
                    <button type="submit">Predict Fraud</button>
                </form>
            </div>
        </body>
    </html>
    """

# Prediction route (accepts form data)
@app.post("/predict", response_class=HTMLResponse)
async def predict(
    amount: float = Form(...),
    transaction_type: str = Form(...),
    merchant_category: str = Form(...),
    location: str = Form(...),
    device_used: str = Form(...),
    time_since_last_transaction: int = Form(...),
    spending_deviation_score: float = Form(...),
    velocity_score: int = Form(...),
    geo_anomaly_score: float = Form(...),
    payment_channel: str = Form(...)
):
    # Convert categorical inputs using encoders
    input_data = []
    transaction_dict = {
        "amount": amount,
        "transaction_type": transaction_type,
        "merchant_category": merchant_category,
        "location": location,
        "device_used": device_used,
        "time_since_last_transaction": time_since_last_transaction,
        "spending_deviation_score": spending_deviation_score,
        "velocity_score": velocity_score,
        "geo_anomaly_score": geo_anomaly_score,
        "payment_channel": payment_channel
    }

    for col, val in transaction_dict.items():
        if col in encoders:
            le = encoders[col]
            val = le.transform([val])[0]
        input_data.append(val)

    # Scale numeric features
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # Return styled HTML result
    return f"""
    <html>
        <head><title>Prediction Result</title></head>
        <body style="font-family: Arial; text-align: center; background: #f4f6f9;">
            <div style="margin: 50px auto; width: 60%; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
                <h2>🔍 Fraud Prediction Result</h2>
                <p><b>Fraud Prediction:</b> {int(prediction)}</p>
                <p><b>Fraud Probability:</b> {round(float(probability), 4)}</p>
                <a href="/">⬅ Back to Form</a>
            </div>
        </body>
    </html>
    """
