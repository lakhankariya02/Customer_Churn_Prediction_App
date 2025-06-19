# Customer Churn Predictor

### How to Run

1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Download dataset from:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Save it as: `Telco-Customer-Churn.csv`

3. Train model:
```bash
python train_model.py
```

4. Run the app:
```bash
cd app
streamlit run app.py
```
🧩 Step 1: Open VS Code
Open VS Code.

Click File → Open Folder and select the extracted folder:
Customer_Churn_Prediction_App

🐍 Step 2: Create or Use a Virtual Environment (Recommended)
In the VS Code terminal:

# Create a virtual environment named 'venv'
python -m venv venv

# Activate it (Windows)
venv\\Scripts\\activate
You should see (venv) at the start of the terminal line.

📦 Step 3: Install Required Libraries

pip install -r requirements.txt
This installs:

streamlit

pandas

scikit-learn

joblib

🧠 Step 4: Download Dataset
Go to https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Download the CSV and place it in the project root as:

Telco-Customer-Churn.csv

🛠️ Step 5: Train the Model
In the VS Code terminal:

python train_model.py

You should see:
✅ Model and encoder saved in 'app/' folder.
🚀 Step 6: Run the Streamlit App

cd app
streamlit run app.py
Streamlit will launch your browser at:


http://localhost:8501
You’ll see a sidebar to enter customer details and get churn predictions.

