import streamlit as st
import pandas as pd
import joblib

# Load the cleaned data
def load_data():
    return pd.read_csv('zomato.csv') 

df = pd.read_csv('zomato.csv')  
df_cleaned = df.drop(columns=['Unnamed: 0.1', 'Unnamed: 0', 'restaurant name', 'local address'])

# Load the trained model and preprocessor
model_filename = 'random_forest_model.joblib'
preprocessor_filename = 'preprocessor.joblib'

# Extract unique values for selection boxes from the dataset
restaurant_types = df_cleaned['restaurant type'].dropna().unique()  # Remove NaNs for unique selection
cuisines_types = df_cleaned['cuisines type'].dropna().unique()
areas = df_cleaned['area'].dropna().unique()

# Load model and preprocessor
model = joblib.load(model_filename)
preprocessor = joblib.load(preprocessor_filename)  

# Streamlit app for nearby restaurant recommendations
def main():
    st.title('Nearby Restaurant Prediction System')

    # Load data
    df = load_data()

    # User inputs
    st.sidebar.header('Filter Options')
    selected_area = st.sidebar.selectbox('Select Area', df['area'].unique())
    max_cost = st.sidebar.slider('Max Budget (Two People)', 
                                  int(df['avg cost (two people)'].min()), 
                                  int(df['avg cost (two people)'].max()), 
                                  int(df['avg cost (two people)'].median()))
    selected_cuisines = st.sidebar.multiselect('Preferred Cuisines', df['cuisines type'].unique())
    
    # Button to show filtered results
    if st.sidebar.button('Predict Restaurants'):
        # Filter data based on user input
        filtered_data = df[(df['area'] == selected_area) & 
                           (df['avg cost (two people)'] <= max_cost)]
        
        # If cuisines are selected, filter further
        if selected_cuisines:
            filtered_data = filtered_data[filtered_data['cuisines type'].isin(selected_cuisines)]
        
        # Show filtered results
        st.header(f"Restaurants in {selected_area} within your budget")
        
        if not filtered_data.empty:
            for index, row in filtered_data.iterrows():
                st.subheader(row['restaurant name'])
                st.write(f"Type: {row['restaurant type']}")
                st.write(f"Rating: {row['rate (out of 5)']} ({row['num of ratings']} ratings)")
                st.write(f"Average Cost for Two: ₹{row['avg cost (two people)']}")
                st.write(f"Cuisines: {row['cuisines type']}")
                st.write(f"Online Order Available: {row['online_order']}")
                st.write(f"Table Booking: {row['table booking']}")
                st.write(f"Location: {row['local address']}")
                st.write("---")
        else:
            st.write("No restaurants match your criteria.")

if __name__ == '__main__':
    main()
