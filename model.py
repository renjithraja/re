import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the dataset to get unique values for selections
df = pd.read_csv('zomato.csv')  # Update with the path to your dataset
df_cleaned = df.drop(columns=['Unnamed: 0.1', 'Unnamed: 0', 'restaurant name', 'local address'])

# Load the trained model and preprocessor
model_filename = 'random_forest_model.joblib'
preprocessor_filename = 'preprocessor.joblib'

# Load model and preprocessor
model = joblib.load(model_filename)
preprocessor = joblib.load(preprocessor_filename)  # Load the preprocessor

# Extract unique values for selection boxes from the dataset
restaurant_types = df_cleaned['restaurant type'].dropna().unique()  # Remove NaNs for unique selection
cuisines_types = df_cleaned['cuisines type'].dropna().unique()
areas = df_cleaned['area'].dropna().unique()

# Streamlit app title
st.title("Restaurant Recommendation System")

# User input for features
st.sidebar.header("Input Features")

# User input fields (excluding avg cost)
restaurant_type = st.sidebar.selectbox("Restaurant Type", options=restaurant_types)
online_order = st.sidebar.selectbox("Online Order Available", options=["Yes", "No"])
table_booking = st.sidebar.selectbox("Table Booking Available", options=["Yes", "No"])
cuisines_type = st.sidebar.selectbox("Cuisines Type", options=cuisines_types)  # Use selectbox for single cuisine
area = st.sidebar.selectbox("Area", options=areas)

# Button to predict
if st.sidebar.button("Predict"):
    # Prepare the input data (excluding avg cost)
    input_data = pd.DataFrame({
        'restaurant type': [str(restaurant_type)],  # Ensure string type
        'online_order': [1 if online_order == "Yes" else 0],
        'table booking': [1 if table_booking == "Yes" else 0],
        'cuisines type': [str(cuisines_type)],  # Ensure string type
        'area': [str(area)]  # Ensure string type
    })

    # Debugging: Show the input data before preprocessing
    st.write("Input Data:", input_data)

    # Check for the columns before transformation
    required_columns = ['restaurant type', 'online_order', 'table booking', 'cuisines type', 'area']
    missing_columns = [col for col in required_columns if col not in input_data.columns]

    if missing_columns:
        st.error(f"Missing columns in input data: {missing_columns}")
    else:
        # Preprocess the input data
        try:
            input_data_preprocessed = preprocessor.transform(input_data)  # Use the preprocessor
        except ValueError as ve:
            st.error(f"ValueError: {ve}")
            st.write("Check if the input data has categories that were not seen during training.")
            st.stop()
        except Exception as e:
            st.error(f"An error occurred during preprocessing: {e}")
            st.stop()

        # Make prediction
        try:
            prediction = model.predict(input_data_preprocessed)

            # Display the result
            if prediction[0]:
                st.success("This restaurant is recommended!")
            else:
                st.warning("This restaurant is not recommended.")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

# Footer
st.sidebar.text("Created by [Your Name]")  # Update with your name or other details
