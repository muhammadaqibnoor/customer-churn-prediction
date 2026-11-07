from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "customer_churn_model.joblib"
META_PATH = ROOT / "models" / "model_metadata.json"

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="centered")
st.title("📊 Customer Churn Prediction")
st.write(
    "Enter a customer's service details to estimate churn risk. "
    "This is a machine-learning estimate, not a guarantee."
)

if not MODEL_PATH.exists():
    st.error("Trained model not found. Run `python src/train_model.py` first.")
    st.stop()

model = joblib.load(MODEL_PATH)
metadata = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}

with st.form("customer_form"):
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior citizen?", ["No", "Yes"])
        partner = st.selectbox("Has partner?", ["No", "Yes"])
        dependents = st.selectbox("Has dependents?", ["No", "Yes"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phone = st.selectbox("Phone service", ["Yes", "No"])
        multiple = st.selectbox("Multiple lines", ["No phone service", "No", "Yes"])
        internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online security", ["No internet service", "No", "Yes"])
        online_backup = st.selectbox("Online backup", ["No internet service", "No", "Yes"])
    with col2:
        device_protection = st.selectbox("Device protection", ["No internet service", "No", "Yes"])
        tech_support = st.selectbox("Tech support", ["No internet service", "No", "Yes"])
        streaming_tv = st.selectbox("Streaming TV", ["No internet service", "No", "Yes"])
        streaming_movies = st.selectbox("Streaming movies", ["No internet service", "No", "Yes"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless billing", ["No", "Yes"])
        payment = st.selectbox("Payment method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)",
            "Credit card (automatic)"
        ])
        monthly = st.number_input("Monthly charges", min_value=0.0, max_value=1000.0, value=70.0, step=1.0)
        total = st.number_input("Total charges", min_value=0.0, max_value=100000.0, value=float(monthly * tenure), step=10.0)

    submitted = st.form_submit_button("Estimate churn risk")

if submitted:
    row = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }])

    try:
        risk = float(model.predict_proba(row)[0, 1])
        prediction = int(risk >= 0.5)
        st.metric("Estimated churn probability", f"{risk:.1%}")
        if prediction:
            st.warning("The model classifies this customer as higher churn risk.")
        else:
            st.success("The model classifies this customer as lower churn risk.")
        st.caption(
            "Threshold is 50% for this demo. In a real business setting, choose a threshold "
            "based on retention costs and the cost of missing a customer who churns."
        )
        if metadata.get("selected_model"):
            st.caption(f"Model: {metadata['selected_model']}")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.info("Check that the app input fields match the dataset schema.")
