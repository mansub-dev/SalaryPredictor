import streamlit as st
import pickle
import numpy as np

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

    # Remove the 'key' argument
    country = st.selectbox("Country", countries)
    education_level = st.selectbox("Education Level", education)

    experience = st.slider("Years of Experience", 0, 50, 3)

    # Prediction button
    ok = st.button("Calculate Salary")
    if ok:
        # Prepare input data for prediction
        X = np.array([[country, education_level, experience]])

        # Transform inputs using label encoders
        try:
            X[:, 0] = le_country.transform(X[:, 0])  # Transform country
            X[:, 1] = le_education.transform(X[:, 1])  # Transform education level
            X = X.astype(float)  # Ensure the data is of type float

            # Make the prediction
            salary = regressor.predict(X)
            st.subheader(f"The estimated salary is ${salary[0]:.2f}")

        except Exception as e:
            st.error(f"Error in prediction: {e}")

