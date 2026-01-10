import streamlit as st
import pandas as pd
import joblib
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import datetime
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import google.generativeai as genai

# ---------------------------------------------------------
# 1. SETUP & CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(page_title="Nephro-AI: Kinetic Research", page_icon="📉", layout="centered")

# Custom CSS for Medical/Research Look
st.markdown("""
    <style>
    .stNumberInput > label {font-size: 105%; font-weight: bold; color: #304ffe;}
    .stSelectbox > label {font-size: 105%; font-weight: bold; color: #304ffe;}
    div[data-testid="stExpander"] details summary {font-weight: bold;}
    </style>
    """, unsafe_allow_html=True)

# Load the Brain (Cached)
@st.cache_resource
def load_model():
    try:
        return joblib.load('Nephro_Brain_Final.pkl')
    except Exception as e:
        st.error(f"⚠️ Model Loading Error: {e}")
        return None

model = load_model()

# Database Connection (Research Grade)
def add_to_database(data_row):
    try:
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        # Check if secrets exist
        if "gcp_service_account" in st.secrets:
            creds_dict = dict(st.secrets["gcp_service_account"])
            creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
            client = gspread.authorize(creds)
            sheet = client.open("Nephro_DB").sheet1
            sheet.append_row(data_row)
            return True
        else:
            return False
    except Exception as e:
        return False

# ---------------------------------------------------------
# 2. THE KINETIC INTERFACE
# ---------------------------------------------------------
st.title("📉 Nephro-AI: Kinetic")
st.caption("Validated Research Protocol | Kinetic-Hybrid Logic")

with st.form("patient_form"):
    
    # --- A. RESEARCH IDENTIFIERS ---
    col_id1, col_id2 = st.columns([1, 2])
    with col_id1:
        mr_number = st.text_input("MR Number", placeholder="e.g. 123-456")
    with col_id2:
        st.info("ℹ️ **Protocol:** Input previous 24h labs to calculate Kinetic Delta.")

    st.divider()
    
    # --- B. KINETIC RENAL FUNCTION (CORE LOGIC) ---
    st.subheader("1. Kinetic Renal Function")
    st.markdown("Use the Delta value to indicate trajectory (Positive = Worsening, Negative = Recovery).")
    
    k_col1, k_col2 = st.columns(2)
    with k_col1:
        cr = st.number_input("Current Creatinine (mg/dL)", min_value=0.0, value=2.5, step=0.1)
    with k_col2:
        # CRITICAL: This input drives the new research logic
        delta_cr = st.number_input("⚠️ Delta Cr (24h Change)", 
                                   value=0.0, step=0.1, 
                                   help="Positive (+) = Injury. Negative (-) = Recovery. Zero = Stable CKD.")

    # --- C. METABOLIC PROFILE ---
    st.subheader("2. Metabolic & Fluid Profile")
    
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        k = st.number_input("Potassium (mEq/L)", min_value=0.0, value=4.5, step=0.1, help="Critical cutoff: 6.0")
        bicarb = st.number_input("Bicarbonate (mEq/L)", min_value=0.0, value=24.0, step=1.0)
        fluid = st.selectbox("Fluid Overload Grade", [0, 1, 2, 3], help="0=None, 3=Anasarca")
    with m_col2:
        bun = st.number_input("BUN (mg/dL)", min_value=0.0, value=45.0, step=1.0)
        ph = st.number_input("pH Level", min_value=6.8, max_value=7.6, value=7.35, step=0.01)
        uo = st.number_input("Urine Output 24h (ml)", min_value=0.0, value=1200.0, step=50.0)
        
    enceph = st.checkbox("Uremic Encephalopathy / Pericarditis?", help="Clinical emergency indication.")
    
    st.divider()
    save_data = st.checkbox("Log to Validation Database?", value=True)
    submitted = st.form_submit_button("Run Kinetic Analysis")

# ---------------------------------------------------------
# 3. LOGIC ENGINE & PREDICTION
# ---------------------------------------------------------
if submitted:
    if model:
        # 1. Prepare Data (Strict Column Matching)
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
        
        # 2. Generate Prediction
        risk_prob = float(model.predict_proba(input_data)[0][1])
        st.session_state['risk_prob'] = risk_prob
        st.session_state['patient_context'] = {
            'cr': cr, 'delta': delta_cr, 'k': k, 'ph': ph, 'uo': uo, 
            'fluid': fluid, 'enceph': enceph, 'bun': bun,
            'input_data': input_data
        }
        
        # 3. Log to Database
        if save_data:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # LOGGING ORDER: MR, Time, Cr, Delta, K, Bicarb, BUN, pH, Fluid, Enceph, UO, Risk
            log_row = [
                str(mr_number), str(timestamp), 
                float(cr), float(delta_cr), float(k), 
                float(bicarb), float(bun), float(ph), 
                int(fluid), int(enceph), float(uo), 
                round(risk_prob, 3)
            ]
            if add_to_database(log_row):
                st.toast(f"✅ Patient {mr_number} logged successfully!", icon="💾")
            else:
                st.error("❌ Database Error: Check Google Sheet Connection.")
    else:
        st.error("🚨 Brain File Missing. Please ensure 'Nephro_Brain_Final.pkl' is uploaded.")

# ---------------------------------------------------------
# 4. RESULTS DASHBOARD (Visuals + Interpretation)
# ---------------------------------------------------------
if 'risk_prob' in st.session_state:
    risk_prob = st.session_state['risk_prob']
    input_data = st.session_state['patient_context']['input_data']
    
    st.divider()
    
    # --- VISUAL A: THE URGENCY GAUGE ---
    # Logic: Green < 40% | Yellow 40-75% | Red > 75%
    if risk_prob > 0.75:
        color = "#FF4B4B" # Red
        status = "High Urgency"
    elif risk_prob > 0.40:
        color = "#fabc02" # Yellow
        status = "Monitor Closely"
    else:
        color = "#00C851" # Green
        status = "Conservative Management"

    col_g1, col_g2 = st.columns([1, 2])
    with col_g1:
        # Custom HTML Number Display
        st.markdown(f"<div style='text-align: center; color: {color};'>"
                    f"<span style='font-size: 50px; font-weight: bold;'>{risk_prob:.1%}</span>"
                    f"<br><span>{status}</span></div>", unsafe_allow_html=True)
    
    with col_g2:
        # --- VISUAL B: KINETIC INTERPRETATION (Text Analysis) ---
        st.subheader("💡 Kinetic Analysis")
        
        # Calculate SHAP for Explanation
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_data)
        
        # Extract Top 3 Drivers
        feature_names = input_data.columns
        importance = pd.DataFrame({
            'feature': feature_names,
            'shap_value': shap_values[0],
            'actual_value': input_data.iloc[0].values
        })
        importance['abs_shap'] = importance['shap_value'].abs()
        top_drivers = importance.sort_values('abs_shap', ascending=False).head(3)
        
        # Loop through drivers to create sentences
        for index, row in top_drivers.iterrows():
            feat = row['feature']
            val = row['actual_value']
            shap_val = row['shap_value']
            
            # --- CUSTOM LOGIC FOR DELTA (The Research Requirement) ---
            if feat == 'delta_Cr_24h':
                if val < 0:
                    st.success(f"📉 **Recovery Trend:** Negative Delta ({val}) is a strong protective factor.")
                elif val == 0:
                    st.info(f"⚖️ **Stable Kinetic:** Delta is zero, suggesting CKD stability rather than acute injury.")
                else:
                    st.warning(f"📈 **Acute Deterioration:** Positive Delta (+{val}) indicates active injury.")
            
            # --- CUSTOM LOGIC FOR URINE ---
            elif feat == 'urine_output_24h':
                if val < 500:
                    st.warning(f"💧 **Oliguria:** Low output ({val}ml) is a major risk driver.")
                else:
                    st.success(f"🌊 **Good Output:** Urine volume ({val}ml) is preserving renal function.")
            
            # --- GENERIC LOGIC ---
            else:
                impact = "increasing risk" if shap_val > 0 else "lowering risk"
                icon = "⚠️" if shap_val > 0 else "🛡️"
                st.write(f"{icon} **{feat}** ({val}) is {impact}.")

    # --- VISUAL C: RISK FACTOR BAR CHART (Restored) ---
    st.divider()
    st.subheader("📊 Risk Factor Breakdown")
    
    # Split into Positive (Red) and Negative (Green) impacts
    pos_factors = importance[importance['shap_value'] > 0].sort_values('shap_value')
    neg_factors = importance[importance['shap_value'] < 0].sort_values('shap_value')
    
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(y=pos_factors['feature'], x=pos_factors['shap_value'], 
                             orientation='h', name='Increases Risk', marker_color='#FF4B4B'))
    fig_bar.add_trace(go.Bar(y=neg_factors['feature'], x=neg_factors['shap_value'], 
                             orientation='h', name='Decreases Risk', marker_color='#00C851'))
    
    fig_bar.update_layout(barmode='relative', height=300, margin=dict(l=0, r=0, t=20, b=20))
    st.plotly_chart(fig_bar, use_container_width=True)

    # --- VISUAL D: WATERFALL PLOT (Scientific Proof) ---
    with st.expander("🔬 View Detailed Statistical Waterfall (For Publication)"):
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.plots.waterfall(
            shap.Explanation(values=shap_values[0], base_values=explainer.expected_value, 
                             data=input_data.iloc[0], feature_names=input_data.columns),
            show=False
        )
        st.pyplot(fig)

# ---------------------------------------------------------
# 5. NEPHRO-GPT (CHATBOT)
# ---------------------------------------------------------
st.divider()
st.subheader("🤖 Nephro-GPT Consultant")

if 'risk_prob' in st.session_state:
    try:
        if "GOOGLE_API_KEY" in st.secrets:
            genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
            model_ai = genai.GenerativeModel('gemini-1.5-flash')
            
            if "messages" not in st.session_state:
                st.session_state.messages = []

            # Display Chat History
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Chat Input
            if prompt := st.chat_input("Ask about management (e.g., 'Calcium Gluconate dose?')..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Context-Aware System Prompt
                ctx = st.session_state['patient_context']
                system_prompt = f"""
                ACT AS A SENIOR NEPHROLOGIST.
                Patient Data: Cr {ctx['cr']}, Delta {ctx['delta']}, K {ctx['k']}, pH {ctx['ph']}, UO {ctx['uo']}.
                AI Risk Score: {st.session_state['risk_prob']:.1%}.
                User Question: {prompt}
                Keep answer clinical, brief, and guideline-based.
                """
                
                with st.chat_message("assistant"):
                    with st.spinner("Consulting guidelines..."):
                        response = model_ai.generate_content(system_prompt)
                        st.markdown(response.text)
                        st.session_state.messages.append({"role": "assistant", "content": response.text})
        else:
            st.info("ℹ️ Chatbot requires API Key in Streamlit Secrets.")
    except Exception as e:
        st.error(f"Chatbot Error: {e}")

# ---------------------------------------------------------
# 6. FOOTER
# ---------------------------------------------------------
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.8em;'>
    <b>Nephro-AI Kinetic Research Tool</b><br>
    Developed for Validation Study | Based on KDIGO 2012 & Kinetic Modeling<br>
    <i>Not for unverified clinical use.</i>
</div>
""", unsafe_allow_html=True)
