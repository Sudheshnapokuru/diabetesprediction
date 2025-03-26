# IMPORT STATEMENTS
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib  # To save/load model efficiently
import os

# Load Dataset
DATA_PATH = "diabetes.csv"

if not os.path.exists(DATA_PATH):
    st.error(f"Dataset file '{DATA_PATH}' not found! Please ensure it exists.")
    st.stop()

df = pd.read_csv(DATA_PATH)

# HEADINGS
st.title('Diabetes Checkup')
st.sidebar.header('Patient Data')
st.subheader('Training Data Stats')
st.write(df.describe())

# Splitting Data
X = df.drop(columns=['Outcome'])
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load or Train Model
MODEL_PATH = "random_forest_model.pkl"

if os.path.exists(MODEL_PATH):
    rf = joblib.load(MODEL_PATH)
else:
    st.warning("Model file not found. Training a new model...")
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    joblib.dump(rf, MODEL_PATH)

# FUNCTION: Collect User Input
def user_report():
    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose', 0, 200, 120)
    bp = st.sidebar.slider('Blood Pressure', 0, 122, 70)
    skinthickness = st.sidebar.slider('Skin Thickness', 0, 100, 20)
    insulin = st.sidebar.slider('Insulin', 0, 846, 79)
    bmi = st.sidebar.slider('BMI', 0, 67, 20)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.4, 0.47)
    age = st.sidebar.slider('Age', 21, 88, 33)

    user_report_data = {
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [bp],  # Match dataset
        'SkinThickness': [skinthickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [dpf],  # Match dataset
        'Age': [age]
    }

    return pd.DataFrame(user_report_data)

# Get User Data
user_data = user_report()
st.subheader('Patient Data')
st.write(user_data)

# Ensure input matches training features
missing_features = set(X.columns) - set(user_data.columns)
if missing_features:
    st.error(f"Missing features in input: {missing_features}")
    st.stop()

# Make Prediction
user_result = rf.predict(user_data)

# OUTPUT
st.subheader('Your Report:')
st.title('You are Diabetic' if user_result[0] == 1 else 'You are not Diabetic')

# Model Accuracy
st.subheader('Model Accuracy:')
accuracy = accuracy_score(y_test, rf.predict(X_test)) * 100
st.write(f"{accuracy:.2f}%")
