import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import joblib
import os

# Load and prepare data
df = pd.read_csv("Telco-Customer-Churn.csv")
df = df.dropna()

X = df[["gender", "SeniorCitizen", "MonthlyCharges", "tenure", "Contract", "InternetService", "TechSupport", "PaymentMethod"]]
y = df["Churn"].map({"Yes": 1, "No": 0})  # Convert to binary

# ✅ Updated line
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X_encoded = encoder.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Train models
rf = RandomForestClassifier()
lr = LogisticRegression(max_iter=1000)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

rf.fit(X_train, y_train)
lr.fit(X_train, y_train)
xgb.fit(X_train, y_train)

# Save models
os.makedirs("app", exist_ok=True)
joblib.dump(rf, "app/RandomForest.pkl")
joblib.dump(lr, "app/LogisticRegression.pkl")
joblib.dump(xgb, "app/XGBoost.pkl")
joblib.dump(encoder, "app/encoder.pkl")

print("✅ Models & encoder saved in /app")
