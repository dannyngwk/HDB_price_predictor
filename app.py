import pandas as pd
import pickle
import streamlit as st

# 1. Load our trained HDB prediction brain
with open("house_model.pkl", "rb") as f:
    model = pickle.load(f)

# Pull the town/flat-type options straight from the fitted encoder so the
# dashboard's dropdowns always match what the model was trained on
cat_encoder = model.named_steps["preprocessor"].named_transformers_["cat"]
town_options = sorted(cat_encoder.categories_[0])
flat_type_options = sorted(cat_encoder.categories_[1])

# 2. Design the localized App Interface
st.title("🏡 Singapore HDB Resale Price Predictor")
st.write(
    "Input the flat's structural attributes below to calculate its estimated market value."
)

# 3. Create input parameters matching local HDB profiles
sqm = st.slider(
    "Floor Area (Square Metres - sqm)",
    min_value=30,
    max_value=160,
    value=90,
    step=1,
)
lease_year = st.slider(
    "Lease Commencement Year",
    min_value=1970,
    max_value=2025,
    value=2000,
    step=1,
)
floor = st.slider(
    "Floor Level (Storey)", min_value=1, max_value=50, value=5, step=1
)
remaining_lease_years = st.slider(
    "Remaining Lease (Years)", min_value=40, max_value=99, value=70, step=1
)
town = st.selectbox("Town", town_options)
flat_type = st.selectbox("HDB Flat Type", flat_type_options)

# 4. Perform calculation on submission
if st.button("Predict Resale Price"):
    # Format user parameters to match training data positions
    input_data = pd.DataFrame(
        [
            {
                "floor_area_sqm": sqm,
                "lease_commence_date": lease_year,
                "floor_level": floor,
                "remaining_lease_years": remaining_lease_years,
                "town": town,
                "flat_type": flat_type,
            }
        ]
    )

    # Generate calculation
    prediction = model.predict(input_data)[0]

    # Display localized output
    st.success(f"🇸🇬 Estimated Resale Price: S${prediction:,.2f}")