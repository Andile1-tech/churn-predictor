import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src.predict import predict_one


st.set_page_config(
    page_title="Churn Risk Analyzer",
    page_icon="📉",
    layout="wide",
)

st.title("📉 Churn Risk Analyzer")
st.caption(
    "Enter a customer's details to estimate their probability of churning. "
    "The model was trained on the Telco Customer Churn dataset."
)

with st.form("customer_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Account")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox(
            "Senior Citizen",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
        )
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"],
        )
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

    with col2:
        st.subheader("Services")
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["Yes", "No", "No phone service"],
        )
        internet = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"],
        )
        online_security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"],
        )
        online_backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"],
        )
        device_protection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"],
        )
        tech_support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"],
        )

    with col3:
        st.subheader("Billing")
        streaming_tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"],
        )
        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"],
        )
        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
        monthly = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            max_value=200.0,
            value=70.0,
            step=0.5,
        )
        total = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            max_value=10000.0,
            value=1000.0,
            step=10.0,
        )

    submitted = st.form_submit_button(
        "Predict Churn Risk",
        use_container_width=True,
    )


if submitted:
    customer = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
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
    }

    with st.spinner("Running prediction..."):
        result = predict_one(customer)

    prob = result["churn_probability"]

    st.divider()
    st.subheader("Prediction")

    m1, m2 = st.columns([1, 2])

    with m1:
        st.metric("Churn Probability", f"{prob:.1%}")

    with m2:
        if prob >= 0.7:
            st.error(f"🔴 {result['risk_label']} — this customer is likely to churn.")
        elif prob >= 0.4:
            st.warning(f"🟡 Medium risk — worth a retention touchpoint.")
        else:
            st.success(f"🟢 {result['risk_label']} — this customer is likely to stay.")

    st.progress(prob)
