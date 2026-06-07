# This app.py contains our modular web page structure of streamlit webframe.

# iterface of the web  aapp using streamlit.


import streamlit as st
import joblib
import numpy as np

scaler = joblib.load('scaler.pkl')
model = joblib.load("model.pkl")

st.title('Churn Prediction App')

st.divider()

st.write('Please enter the values and press the prediction button,in order to get the prediction: ')

st.divider()

age = st.number_input('Enter age', min_value=10, max_value=100, value=30)


tenure = st.number_input(f"Enter Tenure",min_value=0, max_value=130, value=10)

monthly_charge = st.number_input("Enter Monthly Charges", min_value=30, max_value=150)

gender = st.selectbox("Enter the Gender",["Male","Female"])

st.divider()

predict_button = st.button("Predict")

st.divider()

if predict_button:
    gender_selected =1 if gender == 'Female' else 0

    X = [age, gender_selected, tenure, monthly_charge]

    X1 = np.array(X)

    X_array = scaler.transform([X1])

    prediction = model.predict(X_array)[0]

    predicted = "Yes" if prediction == 1 else "No"

    st.balloons()

    st.write(f"Predicted: {predicted}")

else:
    st.write("Please enter the values and predict button.")
# Our model is ready, Let's hit stremlit run app.py in terminal 🙂
