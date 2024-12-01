import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load('student_exam_model.pkl')

# Define the prediction function
def predict_output(study_hours, previous_exam_score):
    input_data = np.array([study_hours, previous_exam_score]).reshape(1, -1)
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        return "You will pass in your upcoming exams based on your preparations."
    else:
        return "Sorry, you will fail in the exam. Increase your reading time."

# Streamlit app UI
st.title("Student Exam Prediction")
st.write("""
This app predicts whether a student will pass or fail based on:
- **Study hours** invested
- **Previous exam scores**
""")

# Input form
study_hours = st.number_input("Enter Study Hours", min_value=0.0, step=0.5)
previous_exam_score = st.number_input("Enter Previous Exam Score", min_value=0, max_value=100, step=1)

if st.button("Predict"):
    result = predict_output(study_hours, previous_exam_score)
    st.success(result)
