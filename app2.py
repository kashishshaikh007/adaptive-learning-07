import streamlit as st
import numpy as np
import joblib

# Load the pre-trained model
@st.cache_resource  # Cache the model to improve performance
def load_model():
    """Load the trained model from the file."""
    return joblib.load("website.pkl")

model = load_model()

# Define prediction function
def predict_output(year, subject, difficulty, publisher, rating, topics_covered):
    """Predict the output based on user inputs."""
    input_data = np.array([year, subject, difficulty, publisher, rating, topics_covered]).reshape(1, -1)
    prediction = model.predict(input_data)
    return prediction

# Streamlit UI
st.title("Book Recommendation Predictor")
st.write("Use this app to predict book recommendations based on various attributes.")

# Input fields
year = st.number_input("Year", min_value=2000, max_value=2100, value=2024)
subject = st.number_input("Subject (Encoded)", min_value=0, max_value=10, value=3)
difficulty = st.number_input("Difficulty (Encoded)", min_value=0, max_value=5, value=2)
publisher = st.number_input("Publisher (Encoded)", min_value=0, max_value=5, value=1)
rating = st.slider("Rating", min_value=0.0, max_value=5.0, value=4.7, step=0.1)
topics_covered = st.number_input("Topics Covered", min_value=0, max_value=10, value=2)

# Prediction button
if st.button("Predict"):
    result = predict_output(year, subject, difficulty, publisher, rating, topics_covered)
    st.success(f"Predicted Output: {result[0]}")
