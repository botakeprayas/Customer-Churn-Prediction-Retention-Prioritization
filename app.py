import pandas as pd
import streamlit as st

from src.predict import HIGH, MEDIUM, load_model, risk_band

st.set_page_config(page_title="Churn Predictor", page_icon="📉")


@st.cache_resource
def get_model():
    return load_model()          # load the model once, not on every click


st.title("📉 Customer Churn Predictor")
st.caption("Enter the customer's details to see how likely they are to leave the company (churn risk).")

try:
    model = get_model()
except FileNotFoundError:
    st.error("Model file not found. Run `python -m src.train` in the terminal first.")
    st.stop()

with st.form("customer_form"):
    left, right = st.columns(2)

    with left:
        st.subheader("Customer")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior citizen?", ["No", "Yes"])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (months with the company)", 0, 72, 12)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless billing", ["Yes", "No"])
        payment = st.selectbox(
            "Payment method",
            ["Electronic check", "Mailed check",
             "Bank transfer (automatic)", "Credit card (automatic)"],
        )
        monthly = st.number_input("Monthly charges", 0.0, 200.0, 70.0, step=0.5)
        own_total = st.checkbox("Enter total charges manually")
        total_in = st.number_input("Total charges (used only if the box above is ticked)",
                                   0.0, 10000.0, 0.0, step=10.0)

    with right:
        st.subheader("Services")
        phone = st.selectbox("Phone service", ["Yes", "No"])
        multiple = st.selectbox("Multiple lines", ["No", "Yes"])
        internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
        online_security = st.selectbox("Online security", ["No", "Yes"])
        online_backup = st.selectbox("Online backup", ["No", "Yes"])
        device_protection = st.selectbox("Device protection", ["No", "Yes"])
        tech_support = st.selectbox("Tech support", ["No", "Yes"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
        streaming_movies = st.selectbox("Streaming movies", ["No", "Yes"])
        st.caption("If Internet service is 'No', the add-on services are automatically set to "
                   "'No internet service'. If Phone service is 'No', Multiple lines becomes "
                   "'No phone service'.")

    submitted = st.form_submit_button("Predict", type="primary")

if submitted:
    # Dataset rules: special values when there is no internet / no phone service
    no_net = internet == "No"
    net = lambda v: "No internet service" if no_net else v

    row = {
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": "No phone service" if phone == "No" else multiple,
        "InternetService": internet,
        "OnlineSecurity": net(online_security),
        "OnlineBackup": net(online_backup),
        "DeviceProtection": net(device_protection),
        "TechSupport": net(tech_support),
        "StreamingTV": net(streaming_tv),
        "StreamingMovies": net(streaming_movies),
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total_in if own_total else round(monthly * tenure, 2),
    }

    p = float(model.predict_proba(pd.DataFrame([row]))[0, 1])
    band = risk_band(p)

    st.divider()
    st.metric("Churn risk score", f"{p:.0%}")
    st.progress(min(max(p, 0.0), 1.0))

    if band == "High":
        st.error(f"🔴 HIGH risk (≥ {HIGH:.0%}). Offer the ₹500 retention offer.")
    elif band == "Medium":
        st.warning(f"🟡 MEDIUM risk ({MEDIUM:.0%} to {HIGH:.0%}). Decide on an offer based on your budget.")
    else:
        st.success(f"🟢 LOW risk (< {MEDIUM:.0%}). No offer needed.")