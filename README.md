# Fraud Detection MLOps Project

## Overview
This project implements a real-time fraud detection system using:
- Kafka for transaction streaming
- Feature engineering and preprocessing
- XGBoost model for fraud classification
- FastAPI for serving predictions
- Evidently for monitoring data drift
- MLflow for experiment tracking
- Docker for containerization

## Project Structure
- `src/` → core ML logic
- `api/` → FastAPI service
- `pipelines/` → training orchestration
- `monitoring/` → drift detection
- `docker/` → Dockerfile
- `data/` → dataset

## Run Training
```bash
python pipelines/training_pipeline.py