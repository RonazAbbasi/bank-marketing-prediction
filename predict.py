
import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Bank Marketing Prediction API")

model = joblib.load("model.bin")


class CustomerInput(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    balance: int
    housing: str
    loan: str
    contact: str
    day: int
    month: str
    campaign: int
    pdays: int
    previous: int
    poutcome: str


@app.get("/")
def home():
    return {"message": "Bank Marketing Prediction API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(customer: CustomerInput):

    data = pd.DataFrame([customer.model_dump()])

    prediction = int(model.predict(data)[0])
    probability = float(model.predict_proba(data)[0][1])

    return {
        "prediction": prediction,
        "subscription": "yes" if prediction == 1 else "no",
        "probability_yes": round(probability, 4)
    }
