from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from src.predict import predict   # make sure src/predict.py defines def predict(...)

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head>
            <title>Fraud Detection API</title>
        </head>
        <body>
            <h2>Fraud Detection Form</h2>
            <form action="/predict" method="post">
                Amount: <input type="text" name="amount"><br>
                Transaction Type: <input type="text" name="transaction_type"><br>
                Merchant Category: <input type="text" name="merchant_category"><br>
                Location: <input type="text" name="location"><br>
                Device Used: <input type="text" name="device_used"><br>
                Time Since Last Transaction: <input type="text" name="time_since_last_transaction"><br>
                Spending Deviation Score: <input type="text" name="spending_deviation_score"><br>
                Velocity Score: <input type="text" name="velocity_score"><br>
                Geo Anomaly Score: <input type="text" name="geo_anomaly_score"><br>
                Payment Channel: <input type="text" name="payment_channel"><br>
                <input type="submit" value="Predict">
            </form>
        </body>
    </html>
    """

@app.post("/predict", response_class=HTMLResponse)
def fraud_prediction(
    amount: float = Form(...),
    transaction_type: str = Form(...),
    merchant_category: str = Form(...),
    location: str = Form(...),
    device_used: str = Form(...),
    time_since_last_transaction: float = Form(...),
    spending_deviation_score: float = Form(...),
    velocity_score: float = Form(...),
    geo_anomaly_score: float = Form(...),
    payment_channel: str = Form(...)
):
    # Collect inputs into a dictionary
    data_dict = {
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

    # Run prediction
    pred, prob = predict(data_dict)

    return f"""
    <html>
        <body>
            <h2>Fraud Prediction Result</h2>
            <p><strong>Fraud Prediction:</strong> {int(pred)}</p>
            <p><strong>Fraud Probability:</strong> {float(prob):.4f}</p>
            <a href="/">Back to Home</a>
        </body>
    </html>
    """
