# Sales Prediction Using Linear Regression

Welcome to the **Sales Prediction using Linear Regression** project! 📊

This project leverages **Linear Regression** to predict sales based on advertising expenditure, economic indicators, and time-related features like the month and year. By analyzing these factors, the model provides valuable insights into the relationship between these variables and sales, helping businesses make data-driven decisions.

## Table of Contents
- [Project Overview](#project-overview)
- [Installation Guide](#installation-guide)
- [How to Use](#how-to-use)
- [Dataset Information](#dataset-information)
- [Model Architecture](#model-architecture)
- [Model Evaluation](#model-evaluation)
- [Visualizations](#visualizations)
- [Contributing](#contributing)

---
## Project Overview

The objective of this project is to build a **Sales Prediction Model** that accurately forecasts sales based on three key factors:
- **Advertising Expenditure**: The amount spent on advertising campaigns.
- **Economic Indicators**: Metrics like GDP growth or inflation, reflecting the economic climate.
- **Time-related Features**: Data from the `Date` feature, including month and year.

By using **Linear Regression**, the model uncovers patterns and dependencies between these factors and sales, providing future sales predictions for strategic planning.

---
## Installation Guide 🚀

To get started, follow these steps to set up and run the project locally:
   -Navigate to the Project Directory:
      cd sales-prediction
   -Install Dependencies
      Install the required libraries by running:
      pip install -r requirements.txt

---
## How to Use 🛠️
  
   - **Load the Dataset**
   Load your dataset from a CSV file
   - **Preprocess the Data**
   Convert the Date column to datetime.
   Handle missing data and extract relevant features.
   - **Select Features and Split Data**
   Define the features (independent variables) and the target variable (sales), then split the data into training and test sets.
   - **Make Predictions and Evaluate the Model**
   Use the trained model to make predictions and evaluate its performance.
   - **Visualize the Results**
   Plot the actual vs predicted sales using matplotlib

---
## Dataset Information 📊
The dataset should be a CSV file containing the following columns:

- **Date**: The date of the sales record.

- **Advertising:** Advertising expenditure for the corresponding date.

- **Economic_indicator:** A numeric value representing the economic situation (e.g., GDP growth).

- **Sales:** The actual sales value for the corresponding date.

---
## Model Architecture 🧠
The Linear Regression model aims to predict the Sales based on the following features:

- **Advertising Expenditure:** Amount spent on advertising.

- **Economic Indicator:** Economic metrics like inflation or GDP.

- **Month:** Extracted from the Date feature to capture seasonality.

- **Year:** Captures long-term trends based on the year.

The model then learns the relationship between these features and sales, predicting future sales.

---
## Visualizations 📉
The model includes a visualization of actual vs predicted sales to help assess its performance. Below is an example of how the chart looks:

- **Blue line:** Actual sales

- **Red line:** Predicted sales


---
## Model Evaluation 📈
We evaluate the model using the following metrics:

- **Mean Absolute Error (MAE):** The average magnitude of errors in predictions.

- **Mean Squared Error (MSE):** Penalizes larger errors more than MAE.

- **R-squared (R²):** A measure of how well the model explains the variance in the data. R² values close to 1 indicate a good fit.


---
## Contributing 💡
We welcome contributions! If you would like to improve this project, feel free to:

- Fork the repository.

- Create a new branch for your feature or bug fix.

- Make the necessary changes.

- Open a pull request with a description of your changes.

- Feel free to open issues or suggestions via the Issues page.

  ---
1. **Clone the Repository**
   ```bash
   git clone https://github.com/Ankitajoshi2002/Sale_Prediction_using_LinearRegression_Model.git



