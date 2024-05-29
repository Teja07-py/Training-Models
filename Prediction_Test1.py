The given dataset consists of a huge no.of datapoints.
# Considering entire dataset, a subset from the actual dataset is taken so as to analyse and work on the implementation for the best suitable machine learning model which predicts the data labels in an efficient manner with smaller rate MAE possibly.


# A subset consisting approximately 500 datapoints is taken for training and testing the machine learning model.
# Observations noted:

# // Various models (say: XGBoost,Lightgbm,GradientBoosting) responded with different predictions.

# // The label predictions were appropraite for the less no.of data points.

# // As the value of datapoints increased, models have depicted its predictions to the point.

# // After a certain value (say 400) the issue of overfitting araised.

# // Assuming 400 as a threshold value, modifications were done to the algorithm (such as estimator's count,learning_rate).

# // Significantly, after the modifications in addition with GridSearch, models were well trained on the subset and predicted accurate labels.



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
dt = pd.read_csv("/content/TEST1.csv")
cols = ['ACCESS-CM2', 'ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CMCC-CM2-SR5', 'CMCC-ESM2']

# Define features and target variable
X = dt[cols]
y = dt['IMD_MAM']

# Instantiate the GradientBoostingRegressor model
gb_regressor = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)

gb_regressor.fit(X, y)

y_pred = gb_regressor.predict(X)

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


dt['Predicted_IMD_MAM'] = y_pred  

# Save the updated dataframe to a new CSV file
dt.to_csv("/content/TEST1_with_predictions35.csv", index=False)

print("Predicted values added to the CSV file.")
