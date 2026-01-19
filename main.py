import streamlit as st
import pandas as pd
import os
import joblib

# --- IMPORT YOUR CUSTOM MODULES ---
# We import the specific functions to run them inside the app
import data_cleanning as dc
import train_model as tm

# --- CONFIGURATION ---
MASTER_DATA_PATH = 'Dataset_Cleaned.csv'
DISTRICT_FILE_PATH = 'districts.txt'
MODEL_PATH = 'biometric_model_v1.pkl'

# --- PAGE SETUP ---
st.set_page_config(page_title="Biometric Data Manager", layout="wide")
st.title("🧬 Biometric Data System")

# --- HELPER: LOAD ARTIFACTS ---
def load_artifacts():
    """Safely loads the model and encoders."""
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

# --- TABS ---
tab1, tab2 = st.tabs(["🔄 Update & Retrain", "🔮 Predict Outcomes"])

# ==========================================
# TAB 1: UPLOAD, CLEAN, AND RETRAIN
# ==========================================
with tab1:
    st.header("Upload Monthly Data")
    
    uploaded_file = st.file_uploader("Upload Monthly Dataset (CSV)", type=["csv"])
    
    if uploaded_file is not None:
        # 1. Load the uploaded data into a dataframe
        try:
            new_data = pd.read_csv(uploaded_file)
            st.subheader("1. Data Preview")
            st.dataframe(new_data.head())
            
            # Button to trigger the pipeline
            if st.button("Validate, Merge & Retrain", type="primary"):
                
                # --- STEP 1: VALIDATION (Using data_cleanning.py) ---
                st.info("running validation checks...")
                try:
                    # Run checks directly on the dataframe
                    # We capture print statements isn't easy, so we rely on the
                    # Validation functions RAISING errors if something is wrong.
                    
                    dc.inspect_nulls(new_data)
                    dc.check_placeholders(new_data)
                    dc.check_numeric_logic(new_data)
                    
                    # For district validation, we need the txt file
                    if os.path.exists(DISTRICT_FILE_PATH):
                        dc.validate_districts(new_data, DISTRICT_FILE_PATH)
                    else:
                        st.warning(f"⚠️ {DISTRICT_FILE_PATH} not found. Skipping district validation.")

                    st.success("✅ Validation Passed!")

                    # --- STEP 2: MERGE DATA ---
                    with st.spinner("Merging with master dataset..."):
                        if os.path.exists(MASTER_DATA_PATH):
                            master_df = pd.read_csv(MASTER_DATA_PATH)
                            # Append new data
                            updated_df = pd.concat([master_df, new_data], ignore_index=True)
                        else:
                            updated_df = new_data
                        
                        # Save the updated master file
                        updated_df.to_csv(MASTER_DATA_PATH, index=False)
                        st.write(f"Master dataset updated. Total rows: {len(updated_df)}")

                    # --- STEP 3: RETRAIN MODEL (Using train_model.py) ---
                    with st.spinner("Retraining model..."):
                        # We call your function. 
                        # Note: Your function re-loads data from disk, so we point it to the master file
                        tm.train_and_save_model(MASTER_DATA_PATH, MODEL_PATH)
                        st.success("✅ Model Retrained & Saved Successfully!")

                except ValueError as e:
                    # This catches the specific errors raised in your cleaning script
                    st.error(f"⛔ Validation Failed: {e}")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")

        except Exception as e:
            st.error(f"Could not read uploaded file: {e}")

# ==========================================
# TAB 2: PREDICTION
# ==========================================
with tab2:
    st.header("Predict Biometric Data")
    
    artifacts = load_artifacts()
    
    if artifacts is None:
        st.warning("⚠️ No model found. Please train the model in the 'Update' tab first.")
    else:
        # Unpack the artifacts
        model = artifacts['model']
        le_state = artifacts['le_state']
        le_district = artifacts['le_district']
        
        # --- 1. LOAD MASTER DATA ---
        if os.path.exists(MASTER_DATA_PATH):
            df_master = pd.read_csv(MASTER_DATA_PATH)
            df_master['state'] = df_master['state'].astype(str)
            df_master['district'] = df_master['district'].astype(str)
            state_options = sorted(df_master['state'].unique())
        else:
            df_master = pd.DataFrame()
            state_options = le_state.classes_

        # --- 2. CREATE INPUTS ---
        col1, col2 = st.columns(2)
        
        with col1:
            selected_state = st.selectbox("State", state_options)
            
            # Filter districts based on state
            if not df_master.empty:
                districts_for_state = df_master[df_master['state'] == selected_state]['district']
                district_options = sorted(districts_for_state.unique())
            else:
                district_options = le_district.classes_
            
            selected_district = st.selectbox("District", district_options)
            
        with col2:
            year = st.number_input("Year", min_value=2000, max_value=2030, value=2025)
            
            # --- NEW: Map Month Names to Numbers ---
            # We map names to numbers, excluding Jan(1), Feb(2), Aug(8)
            month_map = {
                "March": 3,
                "April": 4,
                "May": 5,
                "June": 6,
                "July": 7,
                "September": 9,
                "October": 10,
                "November": 11,
                "December": 12
            }
            
            # User sees the Names (Keys)
            selected_month_name = st.selectbox("Month", list(month_map.keys()))
            
            # We get the Number (Value) for the model
            month_number = month_map[selected_month_name]

        # --- 3. PREDICT ---
        if st.button("Predict"):
            try:
                if selected_state not in le_state.classes_:
                    st.error(f"Error: Model needs retraining. State '{selected_state}' is unknown.")
                elif selected_district not in le_district.classes_:
                    st.error(f"Error: Model needs retraining. District '{selected_district}' is unknown.")
                else:
                    state_enc = le_state.transform([selected_state])[0]
                    district_enc = le_district.transform([selected_district])[0]
                    
                    # Use 'month_number' here, not the name!
                    features = pd.DataFrame([[state_enc, district_enc, year, month_number]], 
                                          columns=['state_encoded', 'district_encoded', 'year', 'month'])
                    
                    prediction = model.predict(features)
                    
                    st.subheader(f"Prediction for {selected_month_name} {year}")
                    res_5_17 = prediction[0][0]
                    res_17_plus = prediction[0][1]
                    
                    metric_col1, metric_col2 = st.columns(2)
                    metric_col1.metric("Bio Age 5-17", f"{res_5_17:.2f}")
                    metric_col2.metric("Bio Age 17+", f"{res_17_plus:.2f}")
                    
            except Exception as e:
                st.error(f"Prediction Error: {e}")