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

# Load the Brain
@st.cache_resource
def load_model():
    try:
        return joblib.load('Nephro_Brain_Final.pkl')
    except:
        return None

model = load_model()

# Database Function
def add_to_database(data_row):
    try:
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds_dict = dict(st.secrets["gcp_service_account"])
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        sheet = client.open("Nephro_DB").sheet1
        sheet.append_row(data_row)
        return True
    except Exception as e:
        st.error(f"Database Error: {e}")
        return False

# ---------------------------------------------------------
# 2. THE INTERFACE
# ---------------------------------------------------------
st.title("🏥 Nephro-AI Assistant")
st.caption("Clinical Decision Support System")

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
    
    st.divider()
    st.markdown("### 💾 Data Options")
    save_data = st.checkbox("Contribute this case to AI Training Database?", value=False)
    
    # Submission Button
    submitted = st.form_submit_button("Run Analysis")

# ---------------------------------------------------------
# 3. LOGIC
# ---------------------------------------------------------
if submitted:
    if model:
        # 1. Create Dataframe with correct names
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

        # --- THE FIX: FORCE EXACT COLUMN ORDER ---
        # We tell the dataframe to arrange columns exactly how the brain learned them
        try:
            input_data = input_data[model.feature_names_in_]
        except AttributeError:
            # If the model is old and doesn't store names, we skip reordering
            pass
        
        # DEBUG: Show the doctor what the AI sees (Optional - helps you check)
        with st.expander("Show Technical Debug Data"):
            st.write("Sending this data to the Brain:", input_data)

        # 2. Predict
        risk_prob_raw = model.predict_proba(input_data)[0][1]
        risk_prob = float(risk_prob_raw)
        
        # 3. Show Result
        st.divider()
        st.metric(label="Dialysis Probability", value=f"{risk_prob:.1%}")
        
        if risk_prob > 0.75:
            st.error("🚨 HIGH RISK: Consider Dialysis Initiation")
        elif risk_prob > 0.40:
            st.warning("⚠️ MODERATE RISK: Monitor Closely")
        else:
            st.success("✅ LOW RISK: Conservative Management")

        # 4. Save
        if save_data:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_row = [
                str(timestamp), float(cr), float(delta_cr), float(k), 
                float(bicarb), float(bun), float(ph), int(fluid), 
                int(enceph), float(uo), round(risk_prob, 3)
            ]
            
            with st.spinner("Saving to Research Database..."):
                if add_to_database(log_row):
                    st.toast("✅ Saved for training!", icon="🧬")
    else:
        st.error("⚠️ AI Brain (Model) not found. Check GitHub files.")
