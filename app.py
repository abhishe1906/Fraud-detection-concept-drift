from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")
threshold = joblib.load("threshold.pkl")

from fastapi import FastAPI, HTTPException
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")
threshold = joblib.load("threshold.pkl")

@app.post("/predict")
def predict(transaction: dict):
    try:
        # Validate keys
        required_keys = ["Amount", "TransactionType", "Location"]
        for key in required_keys:
            if key not in transaction:
                raise ValueError(f"Missing key: {key}")

        amount = float(transaction["Amount"])

        # Safe encoding: TransactionType
        if transaction["TransactionType"] in encoders["TransactionType"].classes_:
            ttype = encoders["TransactionType"].transform(
                [transaction["TransactionType"]]
            )[0]
        else:
            ttype = 0  # fallback

        # Safe encoding: Location
        if transaction["Location"] in encoders["Location"].classes_:
            location = encoders["Location"].transform(
                [transaction["Location"]]
            )[0]
        else:
            location = 0  # fallback

        X = scaler.transform([[amount, ttype, location]])
        prob = model.predict_proba(X)[0][1]

        return {
            "fraud_probability": round(float(prob), 4),
            "is_fraud": int(prob >= threshold),
            "threshold": threshold
        }

    except Exception as e:
        # This makes debugging MUCH easier
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def read_root():
    return {
        "message": "Fraud Detection API is running",
        "endpoint": "/predict"
    }
