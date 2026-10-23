import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="365 Automobiles Predictor", layout="centered")
st.title("🚘 365 Automobiles - Lead Conversion Predictor")

@st.cache_resource
def load_and_train():
    df = pd.read_csv("365_Automobiles_Sales.csv")
    df['target'] = df['purchased'].astype(str).str.lower().isin(['yes', 'true', '1']).astype(int)
    
    feature_cols = [
        'customer_age', 'monthly_income_aed', 'years_at_current_job', 
        'showroom_visits', 'days_in_pipeline', 'tradein_value_aed', 
        'salesperson_experience_yrs'
    ]
    
    data = df[feature_cols + ['target']].dropna()
    X = data[feature_cols]
    y = data['target']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = load_and_train()

st.sidebar.header("Lead Information")
customer_age = st.sidebar.slider("Customer Age", 18, 80, 35)
monthly_income = st.sidebar.number_input("Monthly Income (AED)", 5000, 100000, 25000)
years_job = st.sidebar.slider("Years at Job", 0, 30, 5)
visits = st.sidebar.slider("Showroom Visits", 0, 10, 2)
pipeline_days = st.sidebar.number_input("Days in Pipeline", 1, 180, 15)
tradein_val = st.sidebar.number_input("Trade-in Value (AED)", 0, 150000, 0)
sp_exp = st.sidebar.slider("Salesperson Exp (Years)", 0, 20, 4)

if st.button("Predict Lead Conversion"):
    input_data = pd.DataFrame([[
        customer_age, monthly_income, years_job, visits, 
        pipeline_days, tradein_val, sp_exp
    ]], columns=[
        'customer_age', 'monthly_income_aed', 'years_at_current_job', 
        'showroom_visits', 'days_in_pipeline', 'tradein_value_aed', 
        'salesperson_experience_yrs'
    ])
    
    prob = model.predict_proba(input_data)[0][1] * 100
    pred = model.predict(input_data)[0]
    
    st.markdown("---")
    if pred == 1:
        st.success(f"**High Potential Lead!** Likelihood to Purchase: **{prob:.1f}%**")
    else:
        st.warning(f"**Low Conversion Likelihood.** Likelihood to Purchase: **{prob:.1f}%**")