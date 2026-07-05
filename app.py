import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv("car data.csv")

# -------------------------------
# Preprocessing
# -------------------------------

# Create Car Age
df['Car_Age'] = 2026 - df['Year']
df.drop('Year', axis=1, inplace=True)

# Save original names for dropdown
car_names = df['Car_Name'].unique()

# Encoding
le_car = LabelEncoder()
df['Car_Name'] = le_car.fit_transform(df['Car_Name'])

le_fuel = LabelEncoder()
df['Fuel_Type'] = le_fuel.fit_transform(df['Fuel_Type'])

le_sell = LabelEncoder()
df['Selling_type'] = le_sell.fit_transform(df['Selling_type'])

le_trans = LabelEncoder()
df['Transmission'] = le_trans.fit_transform(df['Transmission'])

# Features & Target
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Model (Random Forest)
# -------------------------------
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions for evaluation
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🚗 Car Price Predictor (Advanced ML)")

st.subheader("📊 Model Performance")
st.write(f"MAE: {mae:.2f}")
st.write(f"R² Score: {r2:.2f}")

# -------------------------------
# Graph (Actual vs Predicted)
# -------------------------------
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred)
ax.set_xlabel("Actual Price")
ax.set_ylabel("Predicted Price")
ax.set_title("Actual vs Predicted Price")

st.pyplot(fig)

# -------------------------------
# User Input Section
# -------------------------------
st.subheader("🔍 Enter Car Details")

car_name = st.selectbox("Car Brand", car_names)
present_price = st.number_input("Present Price (Lakhs)", min_value=0.0)
kms_driven = st.number_input("KMs Driven", min_value=0)
owner = st.selectbox("Owner", [0, 1, 2, 3])

fuel_type = st.selectbox("Fuel Type", le_fuel.classes_)
selling_type = st.selectbox("Selling Type", le_sell.classes_)
transmission = st.selectbox("Transmission", le_trans.classes_)

car_age = st.slider("Car Age (Years)", 0, 20)

# Encode Inputs
car_name_enc = le_car.transform([car_name])[0]
fuel_enc = le_fuel.transform([fuel_type])[0]
sell_enc = le_sell.transform([selling_type])[0]
trans_enc = le_trans.transform([transmission])[0]

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Price"):
    input_data = np.array([[present_price, kms_driven,
                            fuel_enc, sell_enc, trans_enc,
                            owner, car_name_enc, car_age]])

    prediction = model.predict(input_data)

    st.success(f"💰 Estimated Car Price: {prediction[0]:.2f} Lakhs")