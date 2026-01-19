import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os
import joblib

# Try importing custom modules for the Manager section
# These must be in the same directory as this file
try:
    import data_cleanning as dc
    import train_model as tm
except ImportError:
    pass # Will handle this gracefully in the app if modules are missing

# ==========================================
# 1. GLOBAL APP CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Aadhaar 360 | Integrated System",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional UI 
# (Sidebar CSS removed so it uses the Default Streamlit Theme)
st.markdown("""
<style>
    /* Main Headings */
    .main-header {
        font-size: 2.5rem; 
        color: #3B82F6; 
        font-weight: 800; 
        margin-bottom: 0px;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    
    /* Sub-text */
    .sub-text {
        font-size: 1.1rem; 
        font-weight: 500;
        margin-bottom: 20px;
        opacity: 0.8; 
    }
    
    /* Metrics Styling */
    [data-testid="stMetricValue"] {
        font-size: 24px;
        font-weight: 700;
    }
    
    /* Card-like Containers (Adaptive Background) */
    .css-1r6slb0, [data-testid="stForm"] {
        background-color: rgba(128, 128, 128, 0.05); /* Works in both Light & Dark modes */
        border: 1px solid rgba(128, 128, 128, 0.1);
        border-radius: 10px;
        padding: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. MAIN NAVIGATION
# ==========================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/c/cf/Aadhaar_Logo.svg", width=120)
    st.title("System Navigation")
    
    # This toggle determines which 'App' we are looking at
    app_mode = st.radio(
        "Select Module:", 
        ["📊 Aadhaar 360 Dashboard", "⚙️ Biometric Model Manager"]
    )
    st.divider()

# ==========================================
# 3. MODULE: AADHAAR 360 DASHBOARD
# ==========================================
if app_mode == "📊 Aadhaar 360 Dashboard":

    # --- Data Engine (Cached) ---
    @st.cache_data
    def load_and_process_analytics_data():
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(current_dir, 'aadhaar_district_analytics_final_cleaned.csv')
            df = pd.read_csv(file_path)
        except FileNotFoundError:
            st.error("⚠️ Data file 'aadhaar_district_analytics_final_cleaned.csv' not found.")
            return pd.DataFrame()

        # Feature Engineering
        if 'bio_age_5_17' in df.columns and 'demo_age_5_17' in df.columns:
            df['Child_Bio_Intensity'] = df['bio_age_5_17'] / (df['demo_age_5_17'] + 1)
        
        if 'bio_age_17_' in df.columns and 'demo_age_17_' in df.columns:
            df['Adult_Bio_Intensity'] = df['bio_age_17_'] / (df['demo_age_17_'] + 1)

        # Clustering (ML)
        features = ['UER_Score', 'Catch_Up_Index', 'Adult_Entry_Rate', 'CV_Volatility']
        available_features = [f for f in features if f in df.columns]
        
        if len(available_features) == 4:
            X = df[available_features].fillna(0).replace([np.inf, -np.inf], 0)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            kmeans = KMeans(n_clusters=4, random_state=42)
            df['Cluster_ID'] = kmeans.fit_predict(X_scaled)
        else:
            df['Cluster_ID'] = 0
            
        return df

    df = load_and_process_analytics_data()

    if not df.empty:
        # --- Sidebar: Dashboard Controls ---
        with st.sidebar:
            st.markdown("### 🎛️ Dashboard Controls")
            user_role = st.radio("Portal Mode:", ["👤 Citizen Utility", "👮 Admin Command Center"])
            
            st.markdown("### 📍 Location Filter")
            selected_state = st.selectbox("Select State", sorted(df['state'].unique()))
            district_list = sorted(df[df['state'] == selected_state]['district'].unique())
            selected_district = st.selectbox("Select District", district_list)
            
            # Get Specific Data Row
            row = df[(df['state'] == selected_state) & (df['district'] == selected_district)].iloc[0]

        # --- VIEW: CITIZEN UTILITY ---
        if user_role == "👤 Citizen Utility":
            st.markdown(f"<div class='main-header'>Aadhaar Seva: {selected_district}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='sub-text'>Real-time service updates and guidance for <b>{selected_state}</b> residents.</div>", unsafe_allow_html=True)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Reliability Card
                st.subheader("🏥 Center Status")
                volatility = row.get('CV_Volatility', 0)
                
                if volatility < 1.5:
                    st.success("✅ **Operational:** Centers are open regularly (9 AM - 5 PM).")
                elif volatility < 4:
                    st.warning("⚠️ **High Traffic:** Expect delays. Centers usually busy.")
                else:
                    st.error("🛑 **Irregular Service:** Centers likely operating on Camp Mode.")

                # Interactive Crowd Chart
                st.subheader("⏳ Predicted Wait Times")
                np.random.seed(hash(selected_district) % 2**32) 
                days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                traffic = [np.random.randint(50, 100) for _ in range(5)] + [np.random.randint(10, 40), 0]
                
                fig = px.bar(
                    x=days, y=traffic, 
                    labels={'x': 'Day', 'y': 'Busy Level (%)'},
                    color=traffic,
                    color_continuous_scale=['#22c55e', '#facc15', '#ef4444'],
                    height=250,
                    title=f"Traffic Pattern for {selected_district}"
                )
                fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Action Card
                with st.container(border=True):
                    st.markdown("### 📱 Action Guide")
                    adult_bio = row.get('Adult_Bio_Intensity', 0)
                    
                    if adult_bio < 1:
                        st.info("💡 **Recommendation: Online**")
                        st.write("Most updates in your area are address changes.")
                        st.link_button("🔗 Login to myAadhaar", "https://myaadhaar.uidai.gov.in/")
                    else:
                        st.warning("🛠️ **Recommendation: Visit Center**")
                        st.write("High biometric updates detected in your area.")
                        st.link_button("📍 Locate Nearest Center", "https://bhuvan.nrsc.gov.in/aadhaar/")
                
                # Digital Score
                with st.container(border=True):
                    demo_total = row.get('demo_age_5_17', 0) + row.get('demo_age_17_', 0)
                    bio_total = row.get('bio_age_5_17', 0) + row.get('bio_age_17_', 0)
                    total_updates = demo_total + bio_total
                    
                    digital_score = (demo_total / total_updates * 100) if total_updates > 0 else 0
                    
                    st.metric("Digital Maturity Score", f"{digital_score:.0f}/100")
                    if digital_score > 70:
                        st.caption("🌟 High Digital Adoption")
                        st.progress(int(digital_score) / 100)
                    else:
                        st.caption("⚠️ Low Digital Adoption")
                        st.progress(int(digital_score) / 100)

        # --- VIEW: ADMIN COMMAND CENTER ---
        else:
            col_head, col_btn = st.columns([4, 1])
            with col_head:
                st.markdown(f"<div class='main-header'>Command Center | {selected_district}</div>", unsafe_allow_html=True)
            with col_btn:
                csv_data = row.to_frame().T.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Export Report", data=csv_data, file_name=f"Report_{selected_district}.csv", mime="text/csv")

            # Cluster Banner
            c_id = row.get('Cluster_ID', 0)
            clusters = {
                0: {"label": "Standard Operations", "color": "#2563eb", "msg": "Routine monitoring."},
                1: {"label": "High Growth Zone", "color": "#16a34a", "msg": "Deploy ECMP Kits for new enrolments."},
                2: {"label": "Catch-up / Crisis", "color": "#d97706", "msg": "High backlog. Deploy Hospital Teams."},
                3: {"label": "Fraud Risk / Anomaly", "color": "#dc2626", "msg": "Audit Required: High Adult Entry."}
            }
            info = clusters.get(c_id, clusters[0])
            st.markdown(f"""
            <div style="background-color:{info['color']}; padding:15px; border-radius:10px; color:white; margin-bottom:20px;">
                <strong>STATUS: {info['label']}</strong><br>{info['msg']}
            </div>
            """, unsafe_allow_html=True)

            # KPI Metrics
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("UER Score", f"{row.get('UER_Score', 0):.2f}")
            k2.metric("Catch-up Index", f"{row.get('Catch_Up_Index', 0):.1f}")
            k3.metric("Adult Entry Rate", f"{row.get('Adult_Entry_Rate', 0)*100:.1f}%")
            k4.metric("Bio Failure Proxy", f"{row.get('Adult_Bio_Intensity', 0):.2f}")

            st.markdown("---")
            
            # Analytics Tabs
            tab1, tab2, tab3 = st.tabs(["📊 Resource Planning", "📉 Demographics", "🔍 Fraud & Anomalies"])

            with tab1:
                st.subheader("Resource Allocation Strategy")
                c1, c2 = st.columns(2)
                with c1:
                    types_data = pd.DataFrame({
                        'Type': ['Demographic', 'Biometric'],
                        'Volume': [
                            row.get('demo_age_5_17', 0) + row.get('demo_age_17_', 0),
                            row.get('bio_age_5_17', 0) + row.get('bio_age_17_', 0)
                        ]
                    })
                    fig = px.pie(types_data, values='Volume', names='Type', title="Workload Split", hole=0.4)
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    st.info("💡 **Strategic Advice**")
                    if row.get('UER_Score', 0) > 5:
                        st.write("• **Market Saturated:** Focus on UCL (Update Client Lite).")
                    else:
                        st.write("• **High Growth:** Deploy GPS Enrolment Kits.")

            with tab2:
                st.subheader("Population Dynamics")
                age_df = pd.DataFrame({
                    'Group': ['Infants (0-5)', 'School (5-17)', 'Adults (18+)'],
                    'Enrolments': [row.get('age_0_5', 0), row.get('age_5_17', 0), row.get('age_18_greater', 0)]
                })
                fig = px.bar(age_df, x='Group', y='Enrolments', color='Group', title="New Enrolment by Age")
                st.plotly_chart(fig, use_container_width=True)

            with tab3:
                st.subheader("Fraud Detection Radar")
                enrol_tot = row.get('Enrol_Total', 0)
                update_tot = row.get('Update_Total', 0)
                ghost_proxy = enrol_tot / (update_tot + 1)
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Ghost Village Proxy", f"{ghost_proxy:.2f}")
                c2.metric("Hardware Risk", f"{row.get('Adult_Bio_Intensity', 0):.2f}")
                c3.metric("Operator Error Rate", f"{row.get('Correction_Intensity', 0):.2f}")

# ==========================================
# 4. MODULE: BIOMETRIC MODEL MANAGER
# ==========================================
elif app_mode == "⚙️ Biometric Model Manager":
    
    st.markdown("<div class='main-header'>Biometric Data Manager</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-text'>Upload new datasets, retrain the prediction model, and forecast operational needs.</div>", unsafe_allow_html=True)

    # --- Configuration ---
    MASTER_DATA_PATH = 'Dataset_Cleaned.csv'
    DISTRICT_FILE_PATH = 'districts.txt'
    MODEL_PATH = 'biometric_model_v1.pkl'

    # --- Helper: Load Artifacts ---
    def load_artifacts():
        if os.path.exists(MODEL_PATH):
            return joblib.load(MODEL_PATH)
        return None

    # --- Tabs ---
    tab_m1, tab_m2 = st.tabs(["🔄 Update & Retrain", "🔮 Predict Outcomes"])

    # --- TAB 1: UPLOAD & RETRAIN ---
    with tab_m1:
        st.subheader("Upload Monthly Data")
        
        # Check for dependencies
        try:
            import data_cleanning
            import train_model
            has_modules = True
        except ImportError:
            has_modules = False
            st.error("⚠️ Modules `data_cleanning.py` and `train_model.py` are missing. Please ensure they are in the app directory.")

        uploaded_file = st.file_uploader("Upload Monthly Dataset (CSV)", type=["csv"])
        
        if uploaded_file is not None and has_modules:
            try:
                new_data = pd.read_csv(uploaded_file)
                st.write("**Data Preview:**")
                st.dataframe(new_data.head())
                
                if st.button("Validate, Merge & Retrain", type="primary"):
                    st.info("Running validation checks...")
                    try:
                        # 1. Validation
                        dc.inspect_nulls(new_data)
                        dc.check_placeholders(new_data)
                        dc.check_numeric_logic(new_data)
                        
                        if os.path.exists(DISTRICT_FILE_PATH):
                            dc.validate_districts(new_data, DISTRICT_FILE_PATH)
                        else:
                            st.warning(f"⚠️ {DISTRICT_FILE_PATH} not found. Skipping district validation.")

                        st.success("✅ Validation Passed!")

                        # 2. Merge Data
                        with st.spinner("Merging with master dataset..."):
                            if os.path.exists(MASTER_DATA_PATH):
                                master_df = pd.read_csv(MASTER_DATA_PATH)
                                updated_df = pd.concat([master_df, new_data], ignore_index=True)
                            else:
                                updated_df = new_data
                            
                            updated_df.to_csv(MASTER_DATA_PATH, index=False)
                            st.write(f"Master dataset updated. Total rows: {len(updated_df)}")

                        # 3. Retrain Model
                        with st.spinner("Retraining model..."):
                            tm.train_and_save_model(MASTER_DATA_PATH, MODEL_PATH)
                            st.success("✅ Model Retrained & Saved Successfully!")

                    except ValueError as e:
                        st.error(f"⛔ Validation Failed: {e}")
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {e}")

            except Exception as e:
                st.error(f"Could not read uploaded file: {e}")

    # --- TAB 2: PREDICTION ---
    with tab_m2:
        st.subheader("Predict Biometric Data")
        
        artifacts = load_artifacts()
        
        if artifacts is None:
            st.warning("⚠️ No model found. Please train the model in the 'Update & Retrain' tab first.")
        else:
            model = artifacts['model']
            le_state = artifacts['le_state']
            le_district = artifacts['le_district']
            
            # Load Data for Dropdowns
            if os.path.exists(MASTER_DATA_PATH):
                df_master = pd.read_csv(MASTER_DATA_PATH)
                df_master['state'] = df_master['state'].astype(str)
                df_master['district'] = df_master['district'].astype(str)
                state_options = sorted(df_master['state'].unique())
            else:
                df_master = pd.DataFrame()
                state_options = le_state.classes_

            # Inputs
            with st.container(border=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    pred_state = st.selectbox("State", state_options, key="pred_state")
                    
                    if not df_master.empty:
                        districts_for_state = df_master[df_master['state'] == pred_state]['district']
                        district_options = sorted(districts_for_state.unique())
                    else:
                        district_options = le_district.classes_
                    
                    pred_district = st.selectbox("District", district_options, key="pred_dist")
                    
                with col2:
                    pred_year = st.number_input("Year", min_value=2000, max_value=2030, value=2025)
                    
                    month_map = {
                        "March": 3, "April": 4, "May": 5, "June": 6, "July": 7,
                        "September": 9, "October": 10, "November": 11, "December": 12
                    }
                    pred_month_name = st.selectbox("Month", list(month_map.keys()))
                    pred_month_num = month_map[pred_month_name]

                if st.button("Generate Prediction", type="primary"):
                    try:
                        if pred_state not in le_state.classes_:
                            st.error(f"Error: Model unknown state '{pred_state}'")
                        elif pred_district not in le_district.classes_:
                            st.error(f"Error: Model unknown district '{pred_district}'")
                        else:
                            state_enc = le_state.transform([pred_state])[0]
                            district_enc = le_district.transform([pred_district])[0]
                            
                            features = pd.DataFrame(
                                [[state_enc, district_enc, pred_year, pred_month_num]], 
                                columns=['state_encoded', 'district_encoded', 'year', 'month']
                            )
                            
                            prediction = model.predict(features)
                            res_5_17 = prediction[0][0]
                            res_17_plus = prediction[0][1]
                            
                            st.markdown("---")
                            st.subheader(f"Forecast: {pred_month_name} {pred_year}")
                            
                            mc1, mc2 = st.columns(2)
                            mc1.metric("Predicted Bio Age 5-17", f"{res_5_17:.0f}", help="Estimated biometric updates for school children")
                            mc2.metric("Predicted Bio Age 17+", f"{res_17_plus:.0f}", help="Estimated biometric updates for adults")
                            
                    except Exception as e:
                        st.error(f"Prediction Error: {e}")