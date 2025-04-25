import streamlit as st
import pickle
import numpy as np
import pandas as pd

def load_model():
    try:
        with open('saved_steps.pkl', 'rb') as file:
            data = pickle.load(file)
        return data
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        raise


# Load the model and label encoders
data = load_model()

regressor = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]


def show_predict_page():
    st.title("Software Developer Salary Prediction")

    st.write("""### We need some information to predict the salary""")

    countries = (
        "United States",
        "India",
        "United Kingdom",
        "Germany",
        "Canada",
        "Brazil",
        "France",
        "Spain",
        "Australia",
        "Netherlands",
        "Poland",
        "Italy",
        "Russian Federation",
        "Sweden",
    )

    education = (
        "Less than a Bachelors",
        "Bachelor’s degree",
        "Master’s degree",
        "Post grad",
    )

    # Get user input
    country = st.selectbox("Country", countries)
    education_level = st.selectbox("Education Level", education)
    experience = st.slider("Years of Experience", 0, 50, 3)

    # Prepare input data for prediction
    input_data = {
        'Country': [country],
        'EdLevel': [education_level],
        'YearsCodePro': [experience],
    }

    # Convert the input data to a DataFrame with the correct column names
    input_df = pd.DataFrame(input_data)

    # Transform inputs using label encoders
    try:
        input_df['Country'] = le_country.transform(input_df['Country'])
        input_df['EdLevel'] = le_education.transform(input_df['EdLevel'])

        # Make the prediction using the regressor
        salary = regressor.predict(input_df)

        st.subheader(f"The estimated salary is ${salary[0]:.2f}")

    except Exception as e:
        st.error(f"Error in prediction: {e}")
