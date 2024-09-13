import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

data = pd.read_csv('/content/drive/MyDrive/saledataset.csv')

data.to_csv('/content/drive/MyDrive/saledataset.csv', index=False)


data['Date'] = pd.to_datetime(data['Date'])

from google.colab import drive
drive.mount('/content/drive')

data = data.dropna()

data['month'] = data['Date'].dt.month
data['year'] = data['Date'].dt.year

features = data[['Advertising', 'Economic_indicator', 'month', 'year']]
target = data['Sales ']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"\nMean Squared Error (MSE): {mse}")
print("\nCoefficient of Determination {R2} :")
print(f"R-squared (R2 ): {r2}")

plt.figure(figsize=(15, 10))
plt.plot(data['Date'], data['Sales '], label='Actual Sale',color='blue')
plt.plot(data['Date'][X_train.shape[0]:], y_pred, label='Predicted Sale', color='red')
plt.xlabel('\nMontly data of per year')
plt.ylabel('\nSales  per unit')
plt.title('\nSales Prediction using Linear Regression of a product company')
plt.legend()
plt.show()