# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 11:44:48 2023

@author: mrslk
"""

# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


data = pd.read_csv('water.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values


scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)



timesteps = 10
X_reshaped = np.array([X_normalized[i-timesteps:i, :] for i in range(timesteps, len(X_normalized))])
y_reshaped = y[timesteps:]


X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_reshaped, test_size=0.2, random_state=42)


model = Sequential([
    LSTM(units=50, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')


model.fit(X_train, y_train, epochs=100, batch_size=32)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)

print(f'Mean Squared Error: {mse}')

plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual')
plt.scatter(range(len(y_test)), y_pred, color='red', label='Predicted')
plt.xlabel('Time')
plt.ylabel('Target')
plt.legend()
plt.show()
