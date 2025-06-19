import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")
st.title("📉 Smart Customer Churn Predictor")

# Load data
df = pd.read_csv("C:/Users/Lakhan Kariya/Dropbox/PC/Downloads/Customer_Churn_Prediction_App/Telco-Customer-Churn.csv")
features = ["gender", "SeniorCitizen", "MonthlyCharges", "tenure", "Contract", "InternetService", "TechSupport", "PaymentMethod"]
df = df[features + ["Churn"]]

# Sidebar Inputs
st.sidebar.header("Enter Customer Details")
gender = st.sidebar.selectbox("Gender", df["gender"].unique())
senior = st.sidebar.selectbox("Senior Citizen", [0, 1])
monthly_charges = st.sidebar.slider("Monthly Charges", 0, 200, 70)
tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
contract = st.sidebar.selectbox("Contract Type", df["Contract"].unique())
internet = st.sidebar.selectbox("Internet Service", df["InternetService"].unique())
tech_support = st.sidebar.selectbox("Tech Support", df["TechSupport"].unique())
payment_method = st.sidebar.selectbox("Payment Method", df["PaymentMethod"].unique())
model_name = st.sidebar.selectbox("Choose Model", ["RandomForest", "LogisticRegression", "XGBoost"])

# Load model & encoder
model_path = os.path.join("app", f"{model_name}.pkl")
encoder_path = os.path.join("app", "encoder.pkl")
model = joblib.load(model_path)
encoder = joblib.load(encoder_path)

# Predict
if st.sidebar.button("Predict Churn"):
    input_dict = {
        "gender": gender,
        "SeniorCitizen": senior,
        "MonthlyCharges": monthly_charges,
        "tenure": tenure,
        "Contract": contract,
        "InternetService": internet,
        "TechSupport": tech_support,
        "PaymentMethod": payment_method
    }

    input_df = pd.DataFrame([input_dict])
    expected_cols = ["gender", "SeniorCitizen", "MonthlyCharges", "tenure",
                     "Contract", "InternetService", "TechSupport", "PaymentMethod"]
    input_df = input_df[expected_cols]
    input_encoded = encoder.transform(input_df)

    prediction = model.predict(input_encoded)
    prob = model.predict_proba(input_encoded)[0][1]

    st.subheader("Prediction Result:")
    if prediction[0] == 1:
        st.warning(f"⚠️ Likely to Churn (Confidence: {prob:.2f})")
    else:
        st.success(f"✅ Likely to Stay (Confidence: {1 - prob:.2f})")

    # Show green marker on boxplot
    st.subheader("📉 Compare Tenure: Dataset vs Your Input")
    fig3, ax3 = plt.subplots()
    sns.boxplot(x="Churn", y="tenure", data=df, ax=ax3, palette={"Yes": "lightcoral", "No": "skyblue"})
    ax3.scatter(x=[0, 1], y=[tenure, tenure], color="lime", s=200, label="Your Input", marker="D")
    ax3.legend()
    st.pyplot(fig3)

# Static Charts
st.subheader("📊 Churn Distribution")
fig1, ax1 = plt.subplots()
df["Churn"].value_counts().plot.pie(autopct="%1.1f%%", ax=ax1, colors=["skyblue", "lightcoral"])
ax1.set_ylabel("")
st.pyplot(fig1)

st.subheader("📉 Tenure vs Churn (Full Dataset)")
fig2, ax2 = plt.subplots()
sns.boxplot(x="Churn", y="tenure", data=df, ax=ax2)
st.pyplot(fig2)
