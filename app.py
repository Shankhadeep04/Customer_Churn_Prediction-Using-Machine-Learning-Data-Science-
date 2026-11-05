import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="wide")

DATA_PATH = Path("Raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

NUMERIC_FEATURES = ["MonthlyCharges", "TotalCharges", "tenure"]
CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod"
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

@st.cache_resource
def train_model():
    data = pd.read_csv(DATA_PATH)
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    data = data.dropna(subset=["TotalCharges"]).copy()

    X = data[FEATURES]
    y = data["Churn"].map({"Yes": 1, "No": 0})

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES)
    ])
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000))
    ])
    model.fit(X, y)
    return model, len(data)

st.title("📊 Customer Churn Prediction")
st.write("Enter a customer's service and account details to estimate the likelihood of churn.")
st.caption("Model: Logistic Regression with preprocessing trained from the Telco Customer Churn dataset.")

try:
    model, row_count = train_model()
    st.success(f"Model ready — trained on {row_count:,} customer records.")
except Exception as error:
    st.error("The model or dataset could not be loaded. Check the repository file path.")
    st.exception(error)
    st.stop()

with st.form("customer_details"):
    st.subheader("Customer details")
    c1, c2, c3 = st.columns(3)

    with c1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        partner = st.selectbox("Has partner?", ["Yes", "No"])
        dependents = st.selectbox("Has dependents?", ["Yes", "No"])
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)

    with c2:
        phone = st.selectbox("Phone service", ["Yes", "No"])
        multiple = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
        internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
        security = st.selectbox("Online security", ["No", "Yes", "No internet service"])
        backup = st.selectbox("Online backup", ["No", "Yes", "No internet service"])
        protection = st.selectbox("Device protection", ["No", "Yes", "No internet service"])

    with c3:
        support = st.selectbox("Tech support", ["No", "Yes", "No internet service"])
        tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        movies = st.selectbox("Streaming movies", ["No", "Yes", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless billing", ["Yes", "No"])
        payment = st.selectbox("Payment method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])

    c4, c5 = st.columns(2)
    with c4:
        monthly = st.number_input("Monthly charges ($)", min_value=0.0, value=70.0, step=1.0)
    with c5:
        total = st.number_input("Total charges ($)", min_value=0.0, value=840.0, step=10.0)

    submitted = st.form_submit_button("Predict churn", type="primary", use_container_width=True)

if submitted:
    customer = pd.DataFrame([{
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "tenure": tenure,
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": support,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment
    }])[FEATURES]

    prediction = int(model.predict(customer)[0])
    probability = float(model.predict_proba(customer)[0][1])

    st.subheader("Prediction result")
    left, right = st.columns(2)
    with left:
        if prediction == 1:
            st.error("Likely to churn")
        else:
            st.success("Likely to stay")
    with right:
        st.metric("Estimated churn probability", f"{probability:.1%}")
    st.progress(probability)
    st.caption("This is a machine-learning estimate, not a guarantee of customer behaviour.")
