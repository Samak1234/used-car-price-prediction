import streamlit as st
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


# -----------------------------
# Page setup
# -----------------------------

st.set_page_config(
    page_title="Used Car Price Estimator",
    page_icon="🚗",
    layout="centered"
)


# -----------------------------
# Minimal custom styling
# -----------------------------

st.markdown(
    """
    <style>
        .block-container {
            max-width: 900px;
            padding-top: 3rem;
            padding-bottom: 3rem;
        }

        h1 {
            font-size: 2.6rem !important;
            font-weight: 700 !important;
            letter-spacing: -0.03em;
        }

        .subtitle {
            color: #6b7280;
            font-size: 1.05rem;
            margin-top: -0.5rem;
            margin-bottom: 2rem;
        }

        div.stButton > button {
            width: 100%;
            border-radius: 10px;
            height: 3rem;
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------
# Load and prepare dataset
# -----------------------------

df = pd.read_csv("car_prediction_data.csv")

df = df.drop_duplicates()

df["Car_Age"] = df["Year"].max() - df["Year"]

X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

X = X.drop(["Car_Name", "Year"], axis=1)

X = pd.get_dummies(X, drop_first=True)


# -----------------------------
# Train models
# -----------------------------

linear_model = LinearRegression()
linear_model.fit(X, y)

tree_model = DecisionTreeRegressor(
    random_state=42
)
tree_model.fit(X, y)

forest_model = RandomForestRegressor(
    random_state=42
)
forest_model.fit(X, y)


# -----------------------------
# Header
# -----------------------------

st.title("Used Car Price Estimator")

st.markdown(
    '<p class="subtitle">Estimate the resale value of a used car using a machine learning model.</p>',
    unsafe_allow_html=True
)

st.divider()


# -----------------------------
# Input form
# -----------------------------

with st.form("car_price_form"):

    model_choice = st.selectbox(
        "Prediction Model",
        [
            "Linear Regression",
            "Decision Tree",
            "Random Forest"
        ]
    )

    col1, col2 = st.columns(2)

    with col1:
        present_price = st.number_input(
            "Present Price (₹ lakh)",
            min_value=0.0,
            step=0.1
        )

        car_age = st.number_input(
            "Car Age",
            min_value=0,
            step=1
        )

        fuel_type = st.selectbox(
            "Fuel Type",
            ["Petrol", "Diesel", "CNG"]
        )

        transmission = st.selectbox(
            "Transmission",
            ["Manual", "Automatic"]
        )

    with col2:
        kms_driven = st.number_input(
            "Kilometers Driven",
            min_value=0,
            step=1000
        )

        owners = st.number_input(
            "Previous Owners",
            min_value=0,
            step=1
        )

        seller_type = st.selectbox(
            "Seller Type",
            ["Dealer", "Individual"]
        )

    st.write("")

    predict_button = st.form_submit_button(
        "Predict Price",
        use_container_width=True
    )


# -----------------------------
# Prediction
# -----------------------------

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

        # Match the training column order
        input_data = input_data[X.columns]

        # Select model
        if model_choice == "Linear Regression":
            model = linear_model

        elif model_choice == "Decision Tree":
            model = tree_model

        else:
            model = forest_model

        # Make prediction
        predicted_price = model.predict(input_data)[0]

        if predicted_price < 0:
            st.warning(
                "The model produced an unrealistic negative prediction for these inputs."
            )

        else:
            st.subheader("Estimated Selling Price")

            st.success(
                f"₹{predicted_price:.2f} lakh"
            )

            st.caption(
                f"Predicted using {model_choice}"
            )


# -----------------------------
# Model details
# -----------------------------

st.divider()

st.subheader("Model details")

st.write(
    "This project compares Linear Regression, "
    "Decision Tree, and Random Forest regression models."
)

col1, col2, col3 = st.columns(3)

col1.metric(
    "Linear Regression R²",
    "0.75"
)

col2.metric(
    "Decision Tree R²",
    "0.67"
)

col3.metric(
    "Random Forest R²",
    "0.57"
)