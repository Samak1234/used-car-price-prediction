# Used Car Price Prediction

End-to-end machine learning project for predicting used-car selling prices, comparing regression models, analyzing prediction errors, and deploying an interactive web application with Streamlit.

## Live Demo

Try the deployed application:

https://used-car-price-prediction-samak1234.streamlit.app/

## Project Overview

This project explores a complete regression workflow for estimating the resale value of used cars.

It goes beyond training a single model and includes:

- data cleaning
- feature engineering
- categorical encoding
- model training
- model comparison
- prediction error analysis
- residual analysis
- overfitting analysis
- basic hyperparameter experimentation
- Streamlit application development
- cloud deployment

The deployed application allows users to enter vehicle details and generate estimated selling prices using different regression models.

## Models

The project currently includes:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

Each model is evaluated using the same regression metrics so their performance can be compared fairly.

## Model Performance

Current results on the test split:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 1.47 | 2.52 | 0.75 |
| Decision Tree | 1.33 | 2.92 | 0.67 |
| Random Forest | 1.41 | 3.32 | 0.57 |

Linear Regression currently gives the strongest overall performance based on RMSE and R².

Decision Tree produces the lowest MAE on the current split.

These results are based on the current train/test split and are not treated as final model-selection conclusions. Cross-validation is planned as the next step for more reliable comparison.

## Features Used

The models use the following input features:

- Present Price
- Kilometers Driven
- Previous Owners
- Car Age
- Fuel Type
- Seller Type
- Transmission

`Car_Name` is currently excluded from the model.

The original `Year` feature is transformed into:

```text
Car_Age = Maximum Year in Dataset - Vehicle Year
