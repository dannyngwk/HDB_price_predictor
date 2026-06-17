import numpy as np
import pickle
import streamlit as st

# 1. Load the "frozen" model we built earlier
with open("house_model.pkl", "rb") as f:
    model = pickle.load(f)

# 2. Design the Web App Interface
st.title("🏡 House Price Predictor App")
st.write(
    "Adjust the sliders below to estimate the market value of a house."
)

# 3. Create input widgets for the user
income = st.slider(
    "Median Neighborhood Income (in tens of thousands, e.g., 5 = $50k)",
    min_value=1.0,
    max_value=15.0,
    value=5.0,
    step=0.5,
)
rooms = st.slider(
    "Total Rooms in the House",
    min_value=1,
    max_value=10,
    value=5,
    step=1,
)
age = st.slider(
    "Age of the House (Years)",
    min_value=1,
    max_value=50,
    value=20,
    step=1,
)

# 4. When the user clicks the button, make a prediction
if st.button("Predict House Value"):
    # Format inputs exactly how the model expects them
    input_data = np.array([[income, rooms, age]])

    # Make the prediction
    prediction = model.predict(input_data)[0]

    # Display the result beautifully
    st.success(f"💰 Estimated House Value: ${prediction:,.2f}")