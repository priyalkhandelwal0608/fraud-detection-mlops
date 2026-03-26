#  Fraud Detection MLOps Project

##  Overview
This project implements a **real‑time fraud detection system** using machine learning and MLOps best practices. It preprocesses transaction data, trains an XGBoost model, and serves predictions via a **FastAPI web app** with a user‑friendly HTML dashboard.

---

##  Features
- Data preprocessing: cleaning, encoding, scaling, handling missing values  
- Model training: XGBoost classifier with train/test split  
- Saved artifacts: model, scaler, encoders (`joblib`)  
- FastAPI service: `/predict` endpoint + styled HTML form  
- Modular, reproducible pipeline ready for deployment  

---


##  Project Structure
```text
FRAUD-DETECTION-MLOPS/
├── api/
│   └── app.py               # FastAPI server & Web UI
├── data/
│   └── transactions.csv     # Raw transaction dataset
├── docker/
│   └── Dockerfile           # Containerization instructions
├── models/                  # Saved .pkl artifacts (model, scaler)
├── monitoring/              # Scripts for drift and performance checks
├── pipelines/
│   └── training_pipeline.py # Orchestrates the full training flow
├── src/                     # Core logic modules
│   ├── data_processing.py   # Cleaning & Feature Engineering
│   ├── model.py             # Model architecture
│   └── train_model.py       # Training logic
├── mlflow.db                # MLflow experiment database
└── requirements.txt         # Project dependencies
