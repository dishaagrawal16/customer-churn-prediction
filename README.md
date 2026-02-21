# Customer Churn Prediction

This project predicts whether a telecom customer is likely to churn using ML models and a Streamlit UI.

## 📊 Features
- Data cleaning and preprocessing
- SMOTE for class imbalance
- Model comparison (Random Forest, XGBoost)
- Hyperparameter tuning
- Evaluation with ROC-AUC
- Feature importance analysis
- Interactive Streamlit app

## 📈 Model Performance
- ROC-AUC: ~0.83
- Accuracy: ~0.78

## 🧠 Key Business Insights

- Customers with low tenure are more likely to churn.
- Month-to-month contract customers show higher churn risk.
- Customers without online security and backup services churn more.
- Long-term contracts reduce churn probability.

## 🛠 Tech Stack

- Python
- Pandas
- Scikit-learn
- XGBoost
- Streamlit
- 
## 🚀 How to Run

1. Clone the repo  
2. Install dependencies

 ## ▶️ How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
