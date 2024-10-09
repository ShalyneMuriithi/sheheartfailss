import pandas as pd
import streamlit as st
import joblib

# Load the trained logistic regression model
logreg = joblib.load('logistic_model.pkl')

# Define the Streamlit app
st.title('Mortality Prediction Caused by Heart Failure')

st.write("""
This application predicts the risk of heart failure based on various predictors.
Please provide the input data to get a prediction.
""")

# Define a function to get user input features
def user_input_features():
    # Create a form for user inputs
    with st.form(key='prediction_form'):
        st.header('Input Features')

        age = st.slider('Age', min_value=0, max_value=120, value=60)
        anaemia = st.selectbox('Anaemia (1 = Yes, 0 = No)', [0, 1])
        creatinine_phosphokinase = st.slider('Creatinine Phosphokinase (mcg/L)', value=250)
        diabetes = st.selectbox('Diabetes (1 = Yes, 0 = No)', [0, 1])
        ejection_fraction = st.slider('Ejection Fraction (%)', value=50)
        high_blood_pressure = st.selectbox('High Blood Pressure (1 = Yes, 0 = No)', [0, 1])
        platelets = st.slider('Platelets (kiloplatelets/mL)', value=250.0)
        serum_creatinine = st.slider('Serum Creatinine (mg/dL)', value=1.0)
        serum_sodium = st.slider('Serum Sodium (mEq/L)', value=135)
        sex = st.selectbox('Sex (1 = Male, 0 = Female)', [0, 1])
        smoking = st.selectbox('Smoking (1 = Yes, 0 = No)', [0, 1])
        time = st.slider('Follow-up Period (days)', value=150)

        # Submit button for the form
        submit_button = st.form_submit_button(label='Predict')

        # Create a DataFrame for input
        features = pd.DataFrame({
            'age': [age],
            'anaemia': [anaemia],
            'creatinine_phosphokinase': [creatinine_phosphokinase],
            'diabetes': [diabetes],
            'ejection_fraction': [ejection_fraction],
            'high_blood_pressure': [high_blood_pressure],
            'platelets': [platelets],
            'serum_creatinine': [serum_creatinine],
            'serum_sodium': [serum_sodium],
            'sex': [sex],
            'smoking': [smoking],
            'time': [time]
        })
        
        return features, submit_button

# Get user input
input_df, submit_button = user_input_features()

if submit_button:
    # Display user inputs
    st.write("User Inputs:")
    st.write(input_df)

    # Prediction
    prediction = logreg.predict(input_df)
    st.write(f'Prediction: {"Death" if prediction[0] == 1 else "Survival"}')
