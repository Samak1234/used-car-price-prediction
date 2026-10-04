import streamlit as st

#---------------------------------------------------------------------------------
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("car_prediction_data.csv")

df = df.drop_duplicates()

df["Car_Age"] = df["Year"].max() - df["Year"]

X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

X = X.drop(["Car_Name", "Year"], axis=1)

X = pd.get_dummies(X, drop_first=True)

model = LinearRegression()
model.fit(X, y)

#-----------------------------------------------------------------------------------
st.title("Used Car Price Estimator")

st.write(
    "Estimate the resale value of a used car using a machine learning model."
)

st.divider()

present_price = st.number_input(
    "Present Price (in lakh)",
    min_value=0.0,
    step=0.1
)

kms_driven = st.number_input(
    "Kilometers Driven",
    min_value=0,
    step=1000
)

car_age = st.number_input(
    "Car Age",
    min_value=0,
    step=1
)

owners = st.number_input(
    "Previous Owners",
    min_value=0,
    step=1
)

fuel_type = st.selectbox(
    "Fuel Type",
    ["Petrol", "Diesel", "CNG"]
)

seller_type = st.selectbox(
    "Seller Type",
    ["Dealer", "Individual"]
)

transmission = st.selectbox(
    "Transmission",
    ["Manual", "Automatic"]
)

predict_button = st.button("Predict Price")

if predict_button:

    if present_price <= 0:
        st.error("Present price must be greater than 0.")

    elif kms_driven < 0:
        st.error("Kilometers driven cannot be negative.")

    elif car_age < 0:
        st.error("Car age cannot be negative.")

    elif owners < 0:
        st.error("Number of owners cannot be negative.")

    else:
        input_data = pd.DataFrame({
            "Present_Price": [present_price],
            "Kms_Driven": [kms_driven],
            "Owner": [owners],
            "Car_Age": [car_age],
            "Fuel_Type_Diesel": [fuel_type == "Diesel"],
            "Fuel_Type_Petrol": [fuel_type == "Petrol"],
            "Seller_Type_Individual": [seller_type == "Individual"],
            "Transmission_Manual": [transmission == "Manual"]
        })

        input_data = input_data[X.columns]

        predicted_price = model.predict(input_data)[0]

        if predicted_price < 0:
            st.warning(
                "The model produced an unrealistic negative prediction for these inputs."
            )
        else:
            st.success(
                f"Estimated Selling Price: ₹{predicted_price:.2f} lakh"
            )