# Introducing the advanced Deep Learning Model i.e. LSTM network.
# It has been implemented as the advanced network for the previously suggested Machine Learning Model.
# Though the value of error is precise for the machine learning model, the LSTM network is developed for its robust nature to handle a huge amount of dataset.
# LSTM which is a Recurrent Neural Network, is designed to capture long-term dependencies in sequential data.
# In the beginning of the training, the network deviated a lot with corresponding to the observed values.
# Later on multiple modification such as (no.of LSTM units per LSTM layer, epoch_size,dropout_rate), the LSTM network predicted precise labels of data.
# NOTE:
# (1.Increasing no.of LSTM units will make the network more relaible)
# (2.Raised dropout_rate ensure that the network do not overfit)
# (3.Optimal no.of epochs will ensure that the network is well trained)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load the dataset
dt = pd.read_csv("/content/TEST1.csv")
cols = ['ACCESS-CM2', 'ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CMCC-CM2-SR5', 'CMCC-ESM2', 'IMD_MAM']

# Dropping any rows with missing values
dt.dropna(inplace=True)

# Defining features and target variable
X = dt[cols[:-1]].values
y = dt['IMD_MAM'].values

# Normalizing the features and target variable
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

# Reshaping features for LSTM input shape 
X_scaled = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])

# Defining the LSTM model
model = Sequential()
model.add(LSTM(250, activation='relu', input_shape=(X_scaled.shape[1], X_scaled.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(250, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

history = model.fit(X_scaled, y_scaled, epochs=250, batch_size=128, verbose=1, shuffle=False)

y_pred_scaled = model.predict(X_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

mse = mean_squared_error(y, y_pred)



# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(y, label='True', color='blue')
plt.plot(y_pred, label='Predicted', color='red')
plt.title(f'True vs. Predicted Values (MSE: {mse:.4f})')
plt.xlabel('Index')
plt.ylabel('IMD_MAM')
plt.legend()
plt.grid(True)
plt.show()


dt['Predicted_IMD_MAM'] = y_pred  

dt.to_csv("/content/LSMT_TEST1_with_predictions72.csv", index=False)

print("Predicted values added to the CSV file.")
