import streamlit as st
import pickle
import numpy as np
import pandas as pd  # Make sure to import pandas

# Load the model and encoders
def load_model():
    try:
        with open('saved_steps.pkl', 'rb') as file:
            data = pickle.load(file)
        return data
    except FileNotFoundError:
        st.error("Model file not found. Please make sure 'saved_steps.pkl' is uploaded.")
        st.stop()

# Load the model data
data = load_model()

# Extract the model and encoders from the loaded data
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

    # Take user input
    country = st.selectbox("Country", countries, key="country_selectbox")
    education_level = st.selectbox("Education Level", education, key="education_selectbox")
    experience = st.slider("Years of Experience", 0, 50, 3, key="experience_slider")

    # When the button is pressed, make a prediction
    ok = st.button("Calculate Salary", key="calculate_button")
    if ok:
        # Prepare the input data in the format the model expects (as a DataFrame)
        input_data = pd.DataFrame([[country, education_level, experience]], columns=["Country", "EdLevel", "YearsCodePro"])

        # Apply the label encoders to the input data
        try:
            input_data["Country"] = le_country.transform(input_data["Country"])
            input_data["EdLevel"] = le_education.transform(input_data["EdLevel"])

            # Use the trained model to predict the salary
            salary = regressor.predict(input_data)

            # Display the predicted salary
            st.subheader(f"The estimated salary is ${salary[0]:.2f}")
        
        except Exception as e:
            st.error(f"Error occurred while processing your request: {e}")

# Call the function to display the prediction page
show_predict_page()
