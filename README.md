
# Bank Marketing Subscription Prediction

## Problem Description

This project predicts whether a bank customer will subscribe to a term deposit based on customer information and previous marketing campaign data.

The problem is a binary classification task.

## Dataset

The dataset is Bank Marketing from the UCI Machine Learning Repository.

Dataset link:
https://archive.ics.uci.edu/dataset/222/bank+marketing

The dataset contains 45,211 records and 17 original features.

The duration feature was removed because it is only known after a phone call and would cause data leakage.

## Data Preparation

- Missing values were handled using SimpleImputer.
- Numerical features were standardized.
- Categorical features were encoded using OneHotEncoder.
- Data was divided into training and testing sets.

## Models

The following models were evaluated:

- Logistic Regression
- Decision Tree
- Random Forest

GridSearchCV and StratifiedKFold were used for hyperparameter tuning and model selection.

ROC-AUC was used as the main evaluation metric.

## API

The prediction service was developed using FastAPI.

Available endpoints:

- GET /
- GET /health
- POST /predict

## Run Locally

Install dependencies:

pip install -r requirements.txt

Start the API:

uvicorn predict:app --host 0.0.0.0 --port 8000

Open:

http://127.0.0.1:8000/docs

## Docker

Build the image:

docker build -t bank-marketing-api .

Run the container:

docker run -p 8000:8000 bank-marketing-api

## Files

- notebook.ipynb: Data analysis, training and evaluation
- train.py: Model training script
- predict.py: FastAPI prediction service
- model.bin: Trained machine learning pipeline
- requirements.txt: Project dependencies
- Dockerfile: Docker configuration
