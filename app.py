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

# Custom CSS for Diagnosis Box and Inputs
st.markdown("""
    <style>
    .diagnosis-box {
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
        font-weight: bold;
        font-size: 20px;
        border: 2px solid #ccc;
    }
    .stNumberInput > label {font-size: 105%; font-weight: bold; color: #304ffe;}
    div[data-testid="stExpander"] details summary {font-weight: bold;}
    </style>
    """, unsafe_allow_html=True)

# Load Brain (XGBoost/Random Forest)
@st.cache_resource
def load_model():
    try:
        return joblib.load('Nephro_Brain_Final.pkl')
    except:
        return None

model = load_model()

# Database Connection
def add_to_database(data_row):
    try:
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        if "gcp_service_account" in st.secrets:
            creds_dict = dict(st.secrets["gcp_service_account"])
            creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
            client = gspread.authorize(creds)
            sheet = client.open("Nephro_DB").sheet1
            sheet.append_row(data_row)
            return True
        else:
            return False
    except:
        return False

# ---------------------------------------------------------
# 2. THE KINETIC INTERFACE (Aligned with Proposal)
# ---------------------------------------------------------
st.title("📉 Nephro-AI: Kinetic")
st.caption("Hybrid CDSS | Kinetic Modeling | Recovery Detection")

with st.form("patient_form"):
    
    # --- CONTEXT: CKD vs AKI ---
    st.subheader("1. Baseline Context")
    col_ctx1, col_ctx2 = st.columns(2)
    with col_ctx1:
        mr_number = st.text_input("MR Number")
    with col_ctx2:
        # Research Proposal Claim: "Distinguish CKD from AKI"
        baseline_ckd = st.checkbox("Patient has known History of CKD?")

    st.divider()

    # --- KINETIC SECTION (Trends) ---
    st.subheader("2. Kinetic Trends (Deltas)")
    st.info("Input Previous vs. Current to calculate 'Rate of Rise' (Proposal Methodology).")

    # A. CREATININE KINETICS
    c1, c2, c3 = st.columns(3)
    with c1: prev_cr = st.number_input("Prev Cr (24h ago)", value=2.0)
    with c2: curr_cr = st.number_input("Current Cr", value=2.5)
    delta_cr = curr_cr - prev_cr
    with c3: st.metric("Delta Cr", f"{delta_cr:.2f}", delta_color="inverse")

    # B. POTASSIUM KINETICS (Added to satisfy Proposal)
    k1, k2, k3 = st.columns(3)
    with k1: prev_k = st.number_input("Prev K+", value=4.0)
    with k2: curr_k = st.number_input("Current K+", value=4.5)
    delta_k = curr_k - prev_k
    with k3: st.metric("Delta K+", f"{delta_k:.1f}", delta_color="inverse")
    
    # C. OTHER LABS
    st.subheader("3. Metabolic Profile")
    m1, m2 = st.columns(2)
    with m1:
        bicarb = st.number_input("Bicarbonate", value=24.0)
        fluid = st.selectbox("Fluid Overload", [0, 1, 2, 3], help="0=None, 3=Anasarca")
    with m2:
        bun = st.number_input("BUN", value=45.0)
        ph = st.number_input("pH Level", value=7.35, step=0.01)
        uo = st.number_input("Urine Output (24h)", value=1200.0)

    enceph = st.checkbox("Uremic Encephalopathy Present?")
    
    st.divider()
    save_data = st.checkbox("Log to Validation Database", value=True)
    submitted = st.form_submit_button("Run Kinetic Analysis")

# ---------------------------------------------------------
# 3. HYBRID LOGIC ENGINE
# ---------------------------------------------------------
if submitted:
    if model:
        # A. PREPARE INPUTS (Must match Training Columns)
        input_data = pd.DataFrame({
            'creatinine': [curr_cr], 
            'delta_Cr_24h': [delta_cr], 
            'potassium': [curr_k],
            'bicarbonate': [bicarb], 
            'bun': [bun], 
            'ph_level': [ph],
            'fluid_overload_grade': [fluid], 
            'uremic_encephalopathy': [1 if enceph else 0],
            'urine_output_24h': [uo]
        })
        
        # B. GET PROBABILITY
        risk_prob = float(model.predict_proba(input_data)[0][1])
        
        # C. SAVE SESSION STATE
        st.session_state['risk_prob'] = risk_prob
        st.session_state['context'] = {
            'delta_cr': delta_cr, 'delta_k': delta_k, 
            'curr_cr': curr_cr, 'ckd': baseline_ckd,
            'input_data': input_data
        }

        # D. DATABASE LOGGING
        if save_data:
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Log MR, Time, Cr, Delta Cr, Delta K, Risk
            log_row = [str(mr_number), str(ts), float(curr_cr), float(delta_cr), float(delta_k), float(risk_prob)]
            if add_to_database(log_row):
                st.toast("Saved to Database")
            else:
                st.error("Database Error")
    else:
        st.error("Brain Missing")

# ---------------------------------------------------------
# 4. RESULTS DASHBOARD (Visuals + Diagnosis + Explainability)
# ---------------------------------------------------------
if 'risk_prob' in st.session_state:
    risk_prob = st.session_state['risk_prob']
    ctx = st.session_state['context']
    input_data = ctx['input_data']
    
    # Unpack Context
    d_cr = ctx['delta_cr']
    d_k = ctx['delta_k']
    ckd = ctx['ckd']
    
    st.divider()
    
    # --- PART A: THE DIAGNOSIS (Proposal Logic) ---
    diagnosis = "Uncertain"
    diag_color = "#f0f2f6" 
    text_color = "black"

    # 1. Recovery
    if d_cr < -0.2:
        diagnosis = "RENAL RECOVERY DETECTED (Improving Kinetic)"
        diag_color = "#d4edda" # Green
        text_color = "#155724"
        risk_prob = 0.15 # Visually override score for recovery
    # 2. Stable CKD
    elif ckd and abs(d_cr) < 0.3:
        diagnosis = "STABLE CKD (Static Kinetic)"
        diag_color = "#fff3cd" # Yellow
        text_color = "#856404"
    # 3. Acute Injury
    elif d_cr >= 0.3 or risk_prob > 0.75:
        diagnosis = "ACUTE KIDNEY INJURY (Deteriorating)" if not ckd else "ACUTE ON CHRONIC (ACKI)"
        diag_color = "#f8d7da" # Red
        text_color = "#721c24"

    st.markdown(f"<div style='background-color: {diag_color}; color: {text_color};' class='diagnosis-box'>{diagnosis}</div>", unsafe_allow_html=True)

    # --- PART B: THE BULLET GAUGE (Visuals) ---
    st.subheader("1. Dialysis Urgency Score")
    
    # Determine Color
    if risk_prob > 0.75: bar_color = "#FF4B4B"
    elif risk_prob > 0.40: bar_color = "#FFD700"
    else: bar_color = "#90EE90"

    fig_gauge = go.Figure(go.Indicator(
        mode = "number+gauge",
        value = risk_prob * 100,
        number = {'suffix': "%", 'font': {'size': 30}},
        title = {'text': "AI Risk Probability", 'font': {'size': 18, 'color': "gray"}},
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'shape': "bullet",
            'axis': {'range': [None, 100], 'visible': False},
            'bar': {'color': bar_color, 'thickness': 0.25},
            'bgcolor': "#E8E8E8",
            'threshold': {'line': {'color': "gray", 'width': 2}, 'thickness': 0.75, 'value': 75}
        }
    ))
    fig_gauge.update_layout(height=120, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

    # --- PART C: VERBAL EXPLAINABILITY (Simple Language) ---
    st.subheader("2. Plain English Explanation")
    
    # Calculate SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)
    
    # Get Top Factors
    feature_importance = pd.DataFrame({
        'feature': input_data.columns,
        'importance': shap_values[0],
        'value': input_data.iloc[0].values
    })
    top_factors = feature_importance.sort_values('importance', ascending=False).head(3) # Risk drivers
    protective_factors = feature_importance.sort_values('importance', ascending=True).head(2) # Protective
    
    # 1. Proposal Specific Logic (Deltas)
    logic_messages = []
    
    if d_cr < 0:
        logic_messages.append(f"✅ **Creatinine is falling** (Delta {d_cr:.2f}), indicating recovery.")
    elif d_cr > 0.5:
        logic_messages.append(f"⚠️ **Rapid Rise in Creatinine** (+{d_cr:.2f}) is a major concern.")
        
    if d_k > 0.5:
        logic_messages.append(f"⚠️ **Potassium is trending up** (+{d_k:.1f}), adding urgency.")

    # 2. General AI Logic
    for idx, row in top_factors.iterrows():
        if row['importance'] > 0:
            logic_messages.append(f"⚠️ **{row['feature']}** ({row['value']}) is increasing risk.")
            
    # Display the Logic
    st.info("💡 **Why did the AI say this?**\n\n" + "\n".join(logic_messages))

    # --- PART D: RISK FACTOR BARS (The Fancy Bars) ---
    st.subheader("3. Risk Factor Breakdown")
    pos_factors = feature_importance[feature_importance['importance'] > 0].sort_values('importance')
    neg_factors = feature_importance[feature_importance['importance'] < 0].sort_values('importance')
    
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(y=pos_factors['feature'], x=pos_factors['importance'], orientation='h', name='Risk', marker_color='#FF4B4B'))
    fig_bar.add_trace(go.Bar(y=neg_factors['feature'], x=neg_factors['importance'], orientation='h', name='Protective', marker_color='#90EE90'))
    fig_bar.update_layout(barmode='relative', height=300, margin=dict(l=0, r=0, t=20, b=20))
    st.plotly_chart(fig_bar, use_container_width=True)

    # --- PART E: SHAP WATERFALL (Scientific Proof) ---
    with st.expander("🔬 View SHAP Waterfall (For Research Paper)"):
        st.caption("This graph proves the model is not a Black Box.")
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.plots.waterfall(
            shap.Explanation(values=shap_values[0], base_values=explainer.expected_value, 
                             data=input_data.iloc[0], feature_names=input_data.columns),
            show=False
        )
        st.pyplot(fig)

# ---------------------------------------------------------
# 5. NEPHRO-GPT (Context Aware)
# ---------------------------------------------------------
st.divider()
st.subheader("🤖 AI Consultant")
prompt = st.chat_input("Ask about management...")
if prompt and "GOOGLE_API_KEY" in st.secrets:
    try:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
        llm = genai.GenerativeModel('gemini-1.5-flash')
        
        # Super Context for the Chatbot
        system_msg = f"""
        DIAGNOSIS: {diagnosis}.
        RISK SCORE: {risk_prob:.1%}.
        LABS: Cr {ctx['curr_cr']}, Delta Cr {d_cr}, Delta K {d_k}.
        QUESTION: {prompt}
        """
        
        with st.spinner("Thinking..."):
            resp = llm.generate_content(system_msg)
            st.write(resp.text)
    except Exception as e:
        st.error(f"Chatbot Error: {e}")
