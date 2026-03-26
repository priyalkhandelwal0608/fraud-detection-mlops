#  Fraud Detection 
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

##  Technologies Used

###  Core Language & Environment
* **Python 3.11+**: High-level programming language for all logic and scripting.
* **Python-dotenv**: Manages environment variables for secure configuration (API keys, DB paths).

### Machine Learning & Data Science
* **XGBoost**: Gradient boosted decision trees for high-performance classification.
* **Scikit-learn**: Used for data splitting, scaling, and evaluation metrics.
* **Pandas & NumPy**: Essential libraries for high-speed data manipulation and numerical processing.
* **Joblib**: Efficient serialization for saving and loading trained models and scalers.

###  API & Web Interface
* **FastAPI**: Modern, high-performance web framework for building the prediction service.
* **Uvicorn**: Lightning-fast ASGI server implementation to host the application.
* **Jinja2**: Template engine used to render the styled HTML fraud-detection dashboard.


---
## Installation and run 
- pip install -r requirements.txt
- python -m pipelines.training_pipeline
- uvicorn api.app:app --reload
---

##  Project Structure
```text
FRAUD-DETECTION/
├── api/
│   └── app.py               # FastAPI server & Web UI
├── data/
│   └── transactions.csv     # Raw transaction dataset
├── models/                  # Saved .pkl artifacts (model, scaler)
├── monitoring/              # Scripts for drift and performance checks
├── pipelines/
│   └── training_pipeline.py # Orchestrates the full training flow
├── src/                     # Core logic modules
│   ├── data_processing.py   # Cleaning & Feature Engineering
│   ├── model.py             # Model architecture
│   └── train_model.py       # Training logic
└── requirements.txt         # Project dependencies
