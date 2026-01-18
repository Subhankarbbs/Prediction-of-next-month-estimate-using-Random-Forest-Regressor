import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ==========================================
# 1. SETUP PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Biometric Predictor", layout="centered")

st.title("📊 Biometric Enrollment Predictor")
st.write("Enter the location and date details below to get a forecast.")

# ==========================================
# 2. LOAD THE SAVED MODEL
# ==========================================
@st.cache_resource
def load_artifacts():
    try:
        # Load the file we saved earlier
        artifacts = joblib.load('biometric_model_v1.pkl')
        return artifacts
    except FileNotFoundError:
        return None

artifacts = load_artifacts()

# Check if model loaded correctly
if artifacts is None:
    st.error("❌ Error: 'biometric_model_v1.pkl' not found. Please run the training code first to generate the model file.")
    st.stop()

# Extract the pieces from the loaded file
model = artifacts['model']
le_state = artifacts['le_state']
le_district = artifacts['le_district']

# ==========================================
# 3. CREATE INPUT FORM
# ==========================================
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        # distinct lists from the encoders (le.classes_)
        # This creates a Dropdown so users can't make spelling mistakes
        selected_state = st.selectbox("Select State", options=le_state.classes_)
        selected_year = st.number_input("Year", min_value=2024, max_value=2030, value=2025, step=1)

    with col2:
        selected_district = st.selectbox("Select District", options=le_district.classes_)
        selected_month = st.number_input("Month", min_value=1, max_value=12, value=3, step=1)

    # Submit Button
    submitted = st.form_submit_button("🔮 Predict Enrollment")

# ==========================================
# 4. PREDICTION LOGIC
# ==========================================
if submitted:
    # A. Prepare the input data
    input_data = pd.DataFrame({
        'state': [selected_state],
        'district': [selected_district],
        'year': [selected_year],
        'month': [selected_month]
    })

    # B. Encoding (Transform text to numbers using the loaded encoders)
    # We use a try-except block just in case something weird happens
    try:
        input_data['state_encoded'] = le_state.transform(input_data['state'])
        input_data['district_encoded'] = le_district.transform(input_data['district'])
        
        # C. Select features in the exact order the model expects
        # (We saved the feature names list in the pkl file, so we use that order)
        features = artifacts.get('features', ['state_encoded', 'district_encoded', 'year', 'month'])
        X_input = input_data[features]

        # D. Get Prediction
        prediction = model.predict(X_input)
        
        # E. Display Results
        st.success("✅ Prediction Generated Successfully!")
        
        # Create 3 columns for a nice display
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.metric(label="Predicted Bio 5-17", value=int(prediction[0][0]))
            
        with res_col2:
            st.metric(label="Predicted Bio 17+", value=int(prediction[0][1]))

        # Debug info (optional - shows what the model actually saw)
        with st.expander("See technical details"):
            st.write("Model Input Vector:", X_input)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")