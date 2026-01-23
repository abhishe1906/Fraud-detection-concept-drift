import streamlit as st
import requests

st.set_page_config(page_title="Fraud Detection Demo")

st.title("Credit Card Fraud Detection")
st.write("Concept-drift aware fraud screening system")

amount = st.number_input("Transaction Amount", min_value=0.0)
ttype = st.selectbox("Transaction Type", ["online", "pos", "transfer"])
location = st.text_input("Location", "Delhi")

if st.button("Check Fraud Risk"):
    response = requests.post(
        "https://YOUR_FASTAPI_URL/predict",
        json={
            "Amount": amount,
            "TransactionType": ttype,
            "Location": location
        }
    )

    if response.status_code == 200:
        result = response.json()
        st.metric("Fraud Probability", result["fraud_probability"])
        st.write("Fraud Flag:", result["is_fraud"])
    else:
        st.error("API error")
