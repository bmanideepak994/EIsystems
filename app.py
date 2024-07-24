import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Load the trained model from the file
with open('car_safety_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the dataset for visualization purposes
data = pd.read_csv('cars.csv')

# Streamlit app
st.title('Car Safety Predictor')

# Input for car price
price = st.number_input('Enter Car Price', min_value=0.0, format="%.2f")

# Predict safety percentage
if st.button('Predict Safety'):
    prediction = model.predict([[price]])
    st.write(f'The predicted safety percentage for the car priced at {price} is {prediction[0]:.2f}%')

    # Plot the comparison
    fig, ax = plt.subplots()
    ax.scatter(data['Price'], data['Safety_Percentage'], label='Existing Data', color='blue')
    ax.scatter([price], [prediction[0]], label='Predicted Safety', color='red')
    ax.set_xlabel('Price')
    ax.set_ylabel('Safety Percentage')
    ax.legend()
    st.pyplot(fig)

# Display the data table for reference
st.write('Data Table:')
st.write(data.head())
