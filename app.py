from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")
threshold = joblib.load("threshold.pkl")

@app.post("/predict")
def predict(transaction: dict):

    amount = transaction["Amount"]
    ttype = encoders["TransactionType"].transform(
        [transaction["TransactionType"]]
    )[0]
    location = encoders["Location"].transform(
        [transaction["Location"]]
    )[0]

    X = scaler.transform([[amount, ttype, location]])
    prob = model.predict_proba(X)[0][1]

    return {
        "fraud_probability": round(float(prob), 4),
        "is_fraud": int(prob >= threshold),
        "threshold": threshold
    }
