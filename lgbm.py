import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

# Load the dataset
dt = pd.read_csv("/content/TG_tasmax_MAM_2001_2022test.csv")
cols = ['EC-Earth3-Veg-LR', 'GFDL-CM4_gr2', 'INM-CM4-8', 'MPI-ESM1-2-HR', 'NorESM2-MM']

# Define features and target variable
X = dt[cols]
y = dt['IMD']

# Instantiate the LightGBM Regressor model
lgb_regressor = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)

lgb_regressor.fit(X, y)

y_pred = lgb_regressor.predict(X)

mse = mean_squared_error(y, y_pred)
print("Mean Squared Error:", mse)

# Plot true vs. predicted values
plt.figure(figsize=(16,8))
plt.plot(y, color='red', label='True')
plt.plot(y_pred, color='blue', label='Predicted')
plt.title(f'Real vs Prediction - MSE {mse}', fontsize=20)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.legend(fontsize=16)
plt.grid(True)
plt.show()



