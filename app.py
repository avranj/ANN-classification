import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
from sklearn.preprocessing import OneHotEncoder
import streamlit as st


model=tf.keras.models.load_model('model1.h5')

# load the encoder and scaler
with open('lgender.pkl','rb') as file:
    lgender=pickle.load(file)

with open('ohe.pkl','rb') as file:
    ohe=pickle.load(file) 

with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file) 


# Streamlit app start
st.title('Customer Churn Prediction')

# Here we are creating paranthesis to get the values
# User input
geography = st.selectbox('Geography', ohe.categories_[0])# ohe.categories_[0] will display values which are inside it to select
gender = st.selectbox('Gender', lgender.classes_)# same with lgender.classes_
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [lgender.transform([gender])[0]],# taking only 0th value which is first value because bydefault two values will be in the option
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode 'Geography'
geo_encoded = ohe.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=ohe.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)


# Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')




