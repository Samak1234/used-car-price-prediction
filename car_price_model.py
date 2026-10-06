import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold



# Load dataset
df = pd.read_csv("car_prediction_data.csv")


# Explore dataset
print(df.head())
print(df.shape)
print(df.columns)
df.info()


# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())


# Check for duplicate rows
print("\nDuplicate rows:", df.duplicated().sum())


# Display statistical summary
print("\nStatistical summary:")
print(df.describe())


# Remove duplicate rows
df = df.drop_duplicates()

print("\nDataset shape after removing duplicates:")
print(df.shape)

# Inspect Car_Name cardinality
print("\nNumber of unique car names:")
print(df["Car_Name"].nunique())

print("\nCar name frequency:")
print(df["Car_Name"].value_counts().to_string())


print("\nNumber of car names appearing only once:")
print((df["Car_Name"].value_counts() == 1).sum())

# Group rare car names
car_name_counts = df["Car_Name"].value_counts()

df["Car_Name_Grouped"] = df["Car_Name"].where(
    df["Car_Name"].map(car_name_counts) >= 3,
    "Other"
)

print("\nGrouped car name frequency:")
print(df["Car_Name_Grouped"].value_counts().to_string())

print("\nNumber of grouped car name categories:")
print(df["Car_Name_Grouped"].nunique())

# Create Car_Age feature
df["Car_Age"] = df["Year"].max() - df["Year"]

print("\nYear and Car_Age:")
print(df[["Year", "Car_Age"]].head())


# Separate features and target
X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())


# Remove Car_Name and Year
# Keep Car_Name_Grouped so it can be one-hot encoded
X = X.drop(["Car_Name", "Year"], axis=1)

print("\nFeatures after removing Car_Name and Year:")
print(X.head())


# Convert categorical columns into numerical columns
X = pd.get_dummies(X, drop_first=True)

print("\nFeatures after encoding:")
print(X.head())

print("\nFeature columns:")
print(X.columns)

print("\nFeature data types:")
print(X.dtypes)


# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -----------------------------
# Linear Regression
# -----------------------------

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

print("\nLinear Regression training completed!")


# Predict selling prices
y_pred = model.predict(X_test)

print("\nLinear Regression Predictions:")
print(y_pred)


# Compare actual and predicted prices
results = pd.DataFrame({
    "Actual Price": y_test,
    "Predicted Price": y_pred
})

print("\nActual vs Predicted Prices:")
print(results.head(10))


# Calculate Linear Regression evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\nLinear Regression Evaluation Results:")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R² Score:", r2)

# -----------------------------
# Linear Regression Cross-Validation
# -----------------------------

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

linear_cv_scores = cross_val_score(
    LinearRegression(),
    X,
    y,
    cv=kf,
    scoring="r2"
)

print("\nLinear Regression 5-Fold Cross-Validation R² Scores:")
print(linear_cv_scores)

print("Average R²:", linear_cv_scores.mean())


# Calculate prediction errors
results["Error"] = (
    results["Actual Price"]
    - results["Predicted Price"]
)

print("\nPrediction Errors:")
print(results.head(10))


# Calculate absolute errors
results["Absolute Error"] = results["Error"].abs()

print("\nPrediction Errors with Absolute Values:")
print(results.head(10))


# Find predictions with the largest errors
largest_errors = results.sort_values(
    by="Absolute Error",
    ascending=False
)

print("\nTop 5 Largest Prediction Errors:")
print(largest_errors.head(5))


# Get original car details for the five largest errors
worst_indices = largest_errors.head(5).index
worst_cars = df.loc[worst_indices]

print("\nCars with the Largest Prediction Errors:")
print(worst_cars)


# Plot actual vs predicted prices
plt.scatter(y_test, y_pred)

# Create perfect prediction line
min_price = min(y_test.min(), y_pred.min())
max_price = max(y_test.max(), y_pred.max())

plt.plot(
    [min_price, max_price],
    [min_price, max_price]
)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices")

plt.show()


# Plot residuals
plt.scatter(y_pred, results["Error"])

# Add zero-error reference line
plt.axhline(y=0)

plt.xlabel("Predicted Price")
plt.ylabel("Error")
plt.title("Residual Plot")

plt.show()


# Check Linear Regression coefficients
coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print("\nLinear Regression Coefficients:")
print(
    coefficients.sort_values(
        by="Coefficient",
        ascending=False
    )
)


# -----------------------------
# Decision Tree Regressor
# -----------------------------

# Create Decision Tree model
tree_model = DecisionTreeRegressor(
    random_state=42
)

# Train Decision Tree model
tree_model.fit(X_train, y_train)

print("\nDecision Tree training completed!")


# Make Decision Tree predictions
tree_pred = tree_model.predict(X_test)

print("\nDecision Tree Predictions:")
print(tree_pred)


# Calculate Decision Tree evaluation metrics
tree_mae = mean_absolute_error(
    y_test,
    tree_pred
)

tree_mse = mean_squared_error(
    y_test,
    tree_pred
)

tree_rmse = tree_mse ** 0.5

tree_r2 = r2_score(
    y_test,
    tree_pred
)

print("\nDecision Tree Evaluation Results:")
print("MAE:", tree_mae)
print("MSE:", tree_mse)
print("RMSE:", tree_rmse)
print("R² Score:", tree_r2)

# -----------------------------
# Decision Tree Cross-Validation
# -----------------------------

tree_cv_scores = cross_val_score(
    DecisionTreeRegressor(random_state=42),
    X,
    y,
    cv=kf,
    scoring="r2"
)

print("\nDecision Tree 5-Fold Cross-Validation R² Scores:")
print(tree_cv_scores)

print("Average R²:", tree_cv_scores.mean())

# -----------------------------
# Random Forest max_depth comparison
# -----------------------------

for depth in [3, 5, 7, 10, None]:

    forest_model = RandomForestRegressor(
        max_depth=depth,
        random_state=42
    )

    forest_model.fit(X_train, y_train)

    forest_pred = forest_model.predict(X_test)

    forest_mae = mean_absolute_error(y_test, forest_pred)
    forest_mse = mean_squared_error(y_test, forest_pred)
    forest_rmse = forest_mse ** 0.5
    forest_r2 = r2_score(y_test, forest_pred)

    print(f"\nRandom Forest max_depth={depth}")
    print("MAE:", forest_mae)
    print("RMSE:", forest_rmse)
    print("R² Score:", forest_r2)

# -----------------------------
# Random Forest Cross-Validation
# -----------------------------

forest_cv_scores = cross_val_score(
    RandomForestRegressor(random_state=42),
    X,
    y,
    cv=kf,
    scoring="r2"
)

print("\nRandom Forest 5-Fold Cross-Validation R² Scores:")
print(forest_cv_scores)

print("Average R²:", forest_cv_scores.mean())

# -----------------------------
# Model Comparison
# -----------------------------

comparison = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Decision Tree",
        "Random Forest"
    ],
    "MAE": [
        mae,
        tree_mae,
        forest_mae
    ],
    "RMSE": [
        rmse,
        tree_rmse,
        forest_rmse
    ],
    "R²": [
        r2,
        tree_r2,
        forest_r2
    ]
})

print("\nModel Comparison:")
print(comparison)

