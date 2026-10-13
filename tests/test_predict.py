import pytest

from src.predict import predict_batch, predict_one


HIGH_RISK = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 90.0,
    "TotalCharges": 90.0,
}

LOW_RISK = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "Yes",
    "tenure": 60,
    "PhoneService": "Yes",
    "MultipleLines": "Yes",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "Yes",
    "DeviceProtection": "Yes",
    "TechSupport": "Yes",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Two year",
    "PaperlessBilling": "No",
    "PaymentMethod": "Bank transfer (automatic)",
    "MonthlyCharges": 45.0,
    "TotalCharges": 2700.0,
}


def test_predict_one_returns_expected_keys():
    result = predict_one(HIGH_RISK)
    assert set(result.keys()) == {"churn_probability", "churn_prediction", "risk_label"}


def test_probability_is_in_unit_interval():
    for customer in (HIGH_RISK, LOW_RISK):
        result = predict_one(customer)
        assert 0.0 <= result["churn_probability"] <= 1.0


def test_prediction_is_binary():
    for customer in (HIGH_RISK, LOW_RISK):
        result = predict_one(customer)
        assert result["churn_prediction"] in (0, 1)


def test_high_risk_profile_scores_higher_than_low_risk_profile():
    high = predict_one(HIGH_RISK)["churn_probability"]
    low = predict_one(LOW_RISK)["churn_probability"]
    assert high > low
    assert high >= 0.7
    assert low <= 0.3


def test_predict_batch_matches_predict_one():
    batch = predict_batch([HIGH_RISK, LOW_RISK])
    single_high = predict_one(HIGH_RISK)
    single_low = predict_one(LOW_RISK)
    assert len(batch) == 2
    assert batch[0]["churn_probability"] == pytest.approx(
        single_high["churn_probability"]
    )
    assert batch[1]["churn_probability"] == pytest.approx(
        single_low["churn_probability"]
    )
