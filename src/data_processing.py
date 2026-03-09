import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import numpy as np

def preprocess_data(df):
    # Clean dataset: replace "-" with NaN
    df = df.replace("-", np.nan)

    # Drop irrelevant identifiers
    df = df.drop(columns=[
        "transaction_id", "timestamp", "sender_account", "receiver_account",
        "fraud_type", "ip_address", "device_hash"
    ], errors="ignore")

    # --- FIX: Clean target column before casting ---
    # Fill missing values with "False"
    df["is_fraud"] = df["is_fraud"].fillna("False")

    # Map True/False strings to integers
    df["is_fraud"] = df["is_fraud"].map({"True": 1, "False": 0})

    # If any unmapped values remain, fill them with 0
    df["is_fraud"] = df["is_fraud"].fillna(0).astype(int)

    # Encode categorical columns
    categorical_cols = [
        "transaction_type", "merchant_category", "location",
        "device_used", "payment_channel"
    ]
    encoders = {}
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    # Handle missing numeric values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].mean())

    # Features and target
    x = df.drop("is_fraud", axis=1)
    y = df["is_fraud"]

    # Scale numeric features
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # Save scaler + encoders
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(encoders, "models/encoders.pkl")

    # Train/test split
    x_train, x_test, y_train, y_test = train_test_split(
        x_scaled, y, test_size=0.2, random_state=42
    )

    return x_train, x_test, y_train, y_test
