from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import predict

app = FastAPI()

class Transaction(BaseModel):
    amount: float
    time: float
    location: float
    merchant: float
    device: float

@app.get("/")
def home():
    return {"message": "Fraud Detection API"}

@app.post("/predict")
def fraud_prediction(transaction: Transaction):
    data = [
        transaction.amount,
        transaction.time,
        transaction.location,
        transaction.merchant,
        transaction.device,
    ]

    pred, prob = predict(data)

    return {
        "fraud_prediction": int(pred),
        "fraud_probability": float(prob),
    }