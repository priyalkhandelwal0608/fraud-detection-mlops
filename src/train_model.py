import xgboost as xgb
import joblib
import mlflow
from src.data_processing import load_data, preprocess_data

DATA_PATH = "data/transactions.csv"

def train():
    df = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = preprocess_data(df)

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1
    )

    mlflow.start_run()
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)

    joblib.dump(model, "models/fraud_model.pkl")
    mlflow.sklearn.log_model(model, "fraud_model")

    mlflow.end_run()
    print("Model trained with accuracy:", accuracy)

if __name__ == "__main__":
    train()