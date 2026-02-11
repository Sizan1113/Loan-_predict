import streamlit as st
import pandas as pd
from models import model_train

X_var, Y_var, X, Y, model, Y_pred = model_train()

st.title('Loan Amount Prediction')
st.sidebar.header("features details")
annual_income = st.sidebar.slider(
    'Annual_income',
    min_value=100,
    max_value=8000000,
    value=100000,
    step=10
)

if st.sidebar.button('Predict loan amount'):
    input_data = pd.DataFrame(
        [[annual_income]], columns=[X_var]
    )

    prediction = model.predict(input_data)[0]
    st.success(f"Predicted loan amount:{prediction:.2f}")#Display in alert
    st.metric(
        label="Loan amount",
        value=f"Rs .{prediction:.2f}"
    )
