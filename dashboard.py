#Install Dependencies
#USE - pip install pandas numpy matplotlib seaborn plotly scikit-learn statsmodels prophet tensorflow streamlit

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import streamlit as st

# 📥 Load Data
data = pd.read_csv("C:/Users/sengu/Desktop/ds PRO/NFLX.csv")
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# 📊 Visualize Trends
plt.figure(figsize=(12,6))
plt.plot(data['Close'])
plt.title('Stock Price Over Time')
plt.show()

# 📏 ARIMA Forecasting
model_arima = ARIMA(data['Close'], order=(5,1,0))
result_arima = model_arima.fit()
forecast_arima = result_arima.forecast(steps=30)

# Calculate RMSE for ARIMA
rmse_arima = np.sqrt(mean_squared_error(data['Close'][-30:], forecast_arima)) # Calculate RMSE for the last 30 data points


# 📏 SARIMA Forecasting
model_sarima = SARIMAX(data['Close'], order=(1,1,1), seasonal_order=(1,1,0,12))
result_sarima = model_sarima.fit()
forecast_sarima = result_sarima.forecast(30)

# Calculate RMSE for SARIMA
rmse_sarima = np.sqrt(mean_squared_error(data['Close'][-30:], forecast_sarima)) # Calculate RMSE for the last 30 data points

# 📏 Prophet Forecasting
prophet_df = data.reset_index()[['Date', 'Close']]
prophet_df.columns = ['ds', 'y']
model_prophet = Prophet()
model_prophet.fit(prophet_df)
future = model_prophet.make_future_dataframe(periods=30)
forecast_prophet = model_prophet.predict(future)

# Calculate RMSE for Prophet
rmse_prophet = np.sqrt(mean_squared_error(data['Close'][-30:], forecast_prophet['yhat'][-30:]))  # Calculate RMSE for the last 30 data points

# 📏 LSTM Forecasting
scaler = MinMaxScaler(feature_range=(0,1))
data_scaled = scaler.fit_transform(data[['Close']])

X, y = [], []
for i in range(60, len(data_scaled)):
    X.append(data_scaled[i-60:i, 0])
    y.append(data_scaled[i, 0])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

model_lstm = Sequential()
model_lstm.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model_lstm.add(LSTM(units=50))
model_lstm.add(Dense(1))

model_lstm.compile(loss='mean_squared_error', optimizer='adam')
model_lstm.fit(X, y, epochs=10, batch_size=32)

# Make predictions for LSTM
# Define scaled_data here, before it's used
scaled_data = scaler.fit_transform(data[['Close']].values)  # Reshape if necessary
train_size = int(len(scaled_data) * 0.8)
train, test = scaled_data[:train_size], scaled_data[train_size:]


def create_dataset(dataset, time_step=60):
    X, Y = [], []
    for i in range(len(dataset)-time_step-1):
        X.append(dataset[i:(i+time_step), 0])
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 60
_, X_test = create_dataset(test, time_step) # Recalculate X_test
# The original line causing the error:
# X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Reshape X_test to 3D, but with only one feature
X_test = X_test.reshape(X_test.shape[0], 1, 1)


inputs = data_scaled[len(data_scaled) - len(X_test) - 60:]
inputs = inputs.reshape(-1, 1)
inputs = scaler.transform(inputs)
X_test_lstm = []
for i in range(60, inputs.shape[0]):
    X_test_lstm.append(inputs[i - 60:i, 0])
X_test_lstm = np.array(X_test_lstm)
X_test_lstm = np.reshape(X_test_lstm, (X_test_lstm.shape[0], X_test_lstm.shape[1], 1))

lstm_predictions = model_lstm.predict(X_test_lstm)
lstm_predictions = scaler.inverse_transform(lstm_predictions)


# Calculate RMSE for LSTM
rmse_lstm = np.sqrt(mean_squared_error(data['Close'][-30:], lstm_predictions[-30:]))  # Calculate RMSE for the last 30 data points



# 📊 Streamlit Dashboard
st.title("Netflix Stock Market Forecast Dashboard")
st.line_chart(data['Close'])
st.subheader("ARIMA Forecast")
st.line_chart(forecast_arima)
st.subheader("SARIMA Forecast")
st.line_chart(forecast_sarima)
st.subheader("Prophet Forecast")
st.line_chart(forecast_prophet[['ds', 'yhat']].set_index('ds').tail(30))

st.subheader("Model Accuracy (RMSE)")
st.write(f'ARIMA RMSE: {rmse_arima:.2f}')
st.write(f'SARIMA RMSE: {rmse_sarima:.2f}')
st.write(f'Prophet RMSE: {rmse_prophet:.2f}')
st.write(f'LSTM RMSE: {rmse_lstm:.2f}')

st.success("All forecasts generated successfully!")