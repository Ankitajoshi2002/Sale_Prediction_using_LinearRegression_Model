# Sales Prediction Using Linear Regression

This repository contains a Python script for predicting sales based on advertising, economic indicators, and time-related features using a Linear Regression model.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Model](#model)
- [Evaluation](#evaluation)
- [Visualization](#visualization)
- [Contributing](#contributing)
- [License](#license)

## Introduction
This project aims to predict sales by analyzing various factors such as advertising, economic indicators, and time-related features (month and year). The Linear Regression model is used to understand the relationship between these variables and sales.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sales-prediction.git
   ```
2. Navigate to the project directory:
   ```bash
   cd sales-prediction
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Load the dataset:
   ```python
   data = pd.read_csv('/content/drive/MyDrive/saledataset.csv')
   ```
2. Preprocess the data and train the model:
   ```python
   # Your preprocessing and training code here
   ```
3. Evaluate the model and visualize the results:
   ```python
   # Your evaluation and visualization code here
   ```

## Data
The dataset should contain the following columns:
- `Date`: The date of the sales record.
- `Advertising`: Advertising expenditure.
- `Economic_indicator`: Economic indicator value.
- `Sales`: Sales value.

## Model
The Linear Regression model is used to predict sales based on the features:
- `Advertising`
- `Economic_indicator`
- `month`
- `year`

## Evaluation
The model is evaluated using the following metrics:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R-squared (R2)

## Visualization
The script includes a visualization of actual vs. predicted sales over time using `matplotlib`.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

