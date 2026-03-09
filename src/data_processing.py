def preprocess_data(df):
    # Rename columns to match your FastAPI Transaction model
    df = df.rename(columns={
        "timestamp": "time",
        "merchant_id": "merchant",
        "device_id": "device"
    })

    X = df.drop("is_fraud", axis=1)
    y = df["is_fraud"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    joblib.dump(scaler, "models/scaler.pkl")

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test