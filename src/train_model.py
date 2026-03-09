import pandas as pd
import xgboost as xgb
import joblib
import mlflow
from src.data_processing import preprocess_data

DATA_PATH = "data/transactions.csv"

def train():
    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Preprocess
    x_train, x_test, y_train, y_test = preprocess_data(df)

    # Train XGBoost model
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    model.fit(x_train, y_train)

    # Evaluate
    accuracy = model.score(x_test, y_test)

    # Save model
    joblib.dump(model, "models/fraud_model.pkl")

    # Log with MLflow
    mlflow.start_run()
    mlflow.log_param("model_type", "XGBoost")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.end_run()

    print("Model trained with accuracy:", accuracy)

if __name__ == "__main__":
    train()
