import streamlit as st
import pickle
import pandas as pd

import matplotlib.pyplot as plt

# ------------------ Load Model ------------------

with open("customer_churn_model.pkl", "rb") as f:
    model_data = pickle.load(f)

model = model_data["model"]
feature_names = model_data["feature_names"]

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# ------------------ UI ------------------

st.title("Customer Churn Prediction")

uploaded_file = st.file_uploader("Upload Customer CSV")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    predictions = model.predict(data)
    data["Churn Prediction"] = predictions
    st.info("Upload a CSV file with the same columns as the training dataset.")
    st.write(data)  
    
gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Partner", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.slider("Tenure (Months)", 0, 72, 12)

PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

OnlineSecurity = st.selectbox("Online Security", ["Yes", "No"])
OnlineBackup = st.selectbox("Online Backup", ["Yes", "No"])
DeviceProtection = st.selectbox("Device Protection", ["Yes", "No"])
TechSupport = st.selectbox("Tech Support", ["Yes", "No"])
StreamingTV = st.selectbox("Streaming TV", ["Yes", "No"])
StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No"])

Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
PaymentMethod = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", 
     "Bank transfer (automatic)", 
     "Credit card (automatic)"]
)



importances = model.feature_importances_
features = feature_names

fig, ax = plt.subplots()
ax.barh(features, importances)
ax.set_title("Feature Importance")
st.pyplot(fig)        


MonthlyCharges = st.number_input("Monthly Charges", 0.0, 200.0, 50.0)
TotalCharges = st.number_input("Total Charges", 0.0, 10000.0, 500.0)

# ------------------ Prediction ------------------

if st.button("Predict"):

    input_data = {
        'gender': gender,
        'SeniorCitizen': SeniorCitizen,
        'Partner': Partner,
        'Dependents': Dependents,
        'tenure': tenure,
        'PhoneService': PhoneService,
        'MultipleLines': MultipleLines,
        'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection,
        'TechSupport': TechSupport,
        'StreamingTV': StreamingTV,
        'StreamingMovies': StreamingMovies,
        'Contract': Contract,
        'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges
    }

    input_df = pd.DataFrame([input_data])

    # Encode categorical columns
    for column, encoder in encoders.items():
        input_df[column] = encoder.transform(input_df[column])

    # Maintain correct column order
    input_df = input_df[feature_names]

    prob = model.predict_proba(input_df)[:,1][0]
    prediction = 1 if prob > 0.5 else 0


    if prediction == 1:
        st.error(f"Customer likely to Churn (Probability: {round(prob,3)})")
    else:
        st.success(f"Customer likely to Stay (Probability: {round(prob,3)})")

        st.success(f"Customer likely to Stay (Probability: {round(prob,3)})")

    

    if prob > 0.5:
     st.write("Suggested Action:")
     st.write("• Offer long-term contract")
     st.write("• Provide discount or loyalty benefit")
     st.success(f"Customer likely to Stay (Probability: {round(prob,3)})")

   