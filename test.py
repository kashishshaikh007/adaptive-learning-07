import streamlit as st
import numpy as np
import joblib

# Load the pre-trained model
@st.cache_resource  # Cache the model to improve performance
def load_model():
    """Load the trained model from the file."""
    return joblib.load("website.pkl")

model = load_model()

# Define mapping dictionaries
subject_map = {0: "Biology", 1: "Chemistry", 2: "Maths", 3: "Physics"}
difficulty_map = {0: "Easy", 1: "Hard", 2: "Medium"}
publisher_map = {0: "Career 360", 1: "Embibe", 2: "Text Book", 3: "Topper"}

# Define prediction function
def predict_output(year, subject, difficulty, publisher, rating, topics_covered):
    """Predict the output based on user inputs."""
    input_data = np.array([year, subject, difficulty, publisher, rating, topics_covered]).reshape(1, -1)
    prediction = model.predict(input_data)
    return prediction

# Streamlit UI
st.title("Material Recommendation Predictor for Kcet students")
st.write("Use this app to predict material recommendations based on various attributes.")

# Input fields
year = st.number_input("Year", min_value=2000, max_value=2100, value=2024)

# Dropdowns for categorical variables
subject = st.selectbox("Subject", options=list(subject_map.keys()), format_func=lambda x: subject_map[x])
difficulty = st.selectbox("Difficulty", options=list(difficulty_map.keys()), format_func=lambda x: difficulty_map[x])
publisher = st.selectbox("Publisher", options=list(publisher_map.keys()), format_func=lambda x: publisher_map[x])

rating = st.slider("Rating", min_value=0.0, max_value=5.0, value=4.7, step=0.1)
topics_covered = st.number_input("Topics Covered", min_value=0, max_value=10, value=2)

# Prediction button
if st.button("Predict"):
    result = predict_output(year, subject, difficulty, publisher, rating, topics_covered)
    
    # Displaying human-readable names instead of encoded values
    st.success(f"Predicted Output: {result[0]}")
    st.write("### Input Summary")
    st.write(f"- Year: {year}")
    st.write(f"- Subject: {subject_map[subject]}")
    st.write(f"- Difficulty: {difficulty_map[difficulty]}")
    st.write(f"- Publisher: {publisher_map[publisher]}")
    st.write(f"- Rating: {rating}")
    st.write(f"- Topics Covered: {topics_covered}")
