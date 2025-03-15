import streamlit as st
import joblib
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd


data = pd.read_csv('Breast_cancer_data.csv')

# Load the trained model
model = joblib.load('Regression_model.pkl')

# Fetch historical stock data for slider values
data = yf.download('MSFT', start='2020-01-01', end='2024-12-31')
data = data[['Close', 'Open', 'Low', 'High', 'Volume']]  # Use relevant columns for input

# Prepare data for evaluation
features = data[['Open', 'Low', 'High', 'Volume']]
target = data['Close']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Compute evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Function to make predictions
def predict(features):
    prediction = model.predict([features])
    return prediction[0]  # Return the scalar value

# Streamlit app
st.title('Stock Price Prediction')

# Input features from the user
st.sidebar.header('User Input Features')

def user_input_features():
    Open_Price = st.sidebar.slider('Open Price', float(data['Open'].min()), float(data['Open'].max()), float(data['Open'].mean()))
    Low_Price = st.sidebar.slider('Low Price', float(data['Low'].min()), float(data['Low'].max()), float(data['Low'].mean()))
    High_Price = st.sidebar.slider('High Price', float(data['High'].min()), float(data['High'].max()), float(data['High'].mean()))
    Volume = st.sidebar.slider('Volume', float(data['Volume'].min()), float(data['Volume'].max()), float(data['Volume'].mean()))
    features = np.array([Open_Price, High_Price, Low_Price, Volume])
    return features

features = user_input_features()

# Make a prediction
prediction = predict(features)

# Display the result
st.write('Predicted Stock Price:')
st.write(prediction)  # Correctly format the prediction

# Optional: Display historical stock price chart
st.subheader('Historical Stock Prices')
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['Close'], label='Close Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Microsoft Stock Price Over Time')
plt.legend()
st.pyplot(plt)

# Display model evaluation metrics
st.subheader('Model Evaluation Metrics:')
st.write('Mean Squared Error (MSE): ')
st.write(mse)
st.write('R^2 Score: ')
st.write(r2)
