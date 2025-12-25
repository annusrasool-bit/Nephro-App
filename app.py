import streamlit as st
import pandas as pd
import joblib
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import datetime

# ---------------------------------------------------------
# 1. SETUP & CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(page_title="Nephro-AI CDSS", page_icon="🏥")

# Load the Brain (Model)
@st.cache_resource
def load_model():
    try:
        return joblib.load('Nephro_Brain_Final.pkl')
    except:
        st.error("⚠️ Model file not found. Please upload 'Nephro_Brain_Final.pkl' to your GitHub.")
        return None

model = load_model()

# Connect to Google Sheets (The Memory)
def add_to_database(data_row):
    try:
        # Use Streamlit's Secret Manager for Cloud Deployment
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds_dict = dict(st.secrets["gcp_service_account"]) # Reads from Cloud Secrets
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        sheet = client.open("Nephro_DB").sheet1
        sheet.append_row(data_row)
        return True
    except Exception as e:
        st.error(f"Database Error: {e}")
        return False

# ---------------------------------------------------------
# 2. THE USER INTERFACE (UI)
# ---------------------------------------------------------
st.title("🏥 Nephro-AI Assistant")
st.caption("Real-time Dialysis Prediction & Data Logging")

with st.form("patient_form"):
    st.subheader("Patient Vitals & Labs")
    
    col1, col2 = st.columns(2)
    with col1:
        cr = st.number_input("Creatinine (mg/dL)", min_value=0.0, value=2.0, step=0.1)
        delta_cr = st.number_input("Delta Cr (24h change)", value=0.0, step=0.1)
        k = st.number_input("Potassium (mEq/L)", min_value=0.0, value=4.5, step=0.1)
        bicarb = st.number_input("Bicarbonate (mEq/L)", min_value=0.0, value=24.0, step=1.0)
    
    with col2:
        bun = st.number_input("BUN (mg/dL)", min_value=0.0, value=40.0, step=1.0)
        ph = st.number_input("pH Level", min_value=6.8, max_value=7.6, value=7.4, step=0.01)
        uo = st.number_input("Urine Output 24h (ml)", min_value=0.0, value=1500.0, step=50.0)
        
    st.subheader("Clinical Signs")
    fluid = st.selectbox("Fluid Overload Grade", [0, 1, 2, 3], help="0=None, 3=Anasarca")
    enceph = st.checkbox("Uremic Encephalopathy Present?")
    
    # Submission Button
    submitted = st.form_submit_button("Analyze & Save Case")

# ---------------------------------------------------------
# 3. PREDICTION & LOGIC
# ---------------------------------------------------------
if submitted and model:
    # 1. Prepare Data for Model
    input_data = pd.DataFrame({
        'creatinine': [cr],
        'delta_Cr_24h': [delta_cr],
        'potassium': [k],
        'bicarbonate': [bicarb],
        'bun': [bun],
        'ph_level': [ph],
        'fluid_overload_grade': [fluid],
        'uremic_encephalopathy': [1 if enceph else 0],
        'urine_output_24h': [uo]
    })

    # 2. Predict
    risk_prob = model.predict_proba(input_data)[0][1]
    
    # 3. Display Result
    st.divider()
    st.markdown(f"### 🔍 AI Risk Assessment: **{risk_prob:.1%}**")
    
    if risk_prob > 0.75:
        st.error("🚨 HIGH RISK: Consider Dialysis Initiation")
    elif risk_prob > 0.40:
        st.warning("⚠️ MODERATE RISK: Monitor Closely")
    else:
        st.success("✅ LOW RISK: Conservative Management")

    # 4. Save to Cloud (Google Sheet)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # We save inputs AND the AI's prediction
    log_row = [timestamp, cr, delta_cr, k, bicarb, bun, ph, fluid, int(enceph), uo, round(risk_prob, 3)]
    
    if add_to_database(log_row):
        st.toast("✅ Case logged to Google Cloud Database!", icon="☁️")