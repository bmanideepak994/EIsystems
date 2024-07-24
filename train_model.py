import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load the dataset
data = pd.read_csv('cars.csv')

# Split the dataset into features and target variable
X = data[['Price']]  # Feature: Car Price
y = data['Safety_Percentage']  # Target: Safety Percentage

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model to a file using pickle
with open('car_safety_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model trained and saved as 'car_safety_model.pkl'.")
