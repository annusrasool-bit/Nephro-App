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
st.set_page_config(page_title="Nephro-AI: Clinical Research", page_icon="🏥", layout="centered")

# Professional Medical CSS
st.markdown("""
    <style>
    .main-header {font-family: 'Helvetica', sans-serif; color: #0d47a1;}
    .diagnosis-box {
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 20px;
        text-align: center;
        font-weight: 700;
        font-size: 22px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .warning-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #ffc107;
        margin-top: 10px;
        font-size: 16px;
    }
    .stNumberInput > label {font-weight: 600; color: #1565c0;}
    </style>
    """, unsafe_allow_html=True)

# Load the AI Brain (Layer 1)
@st.cache_resource
def load_model():
    try:
        return joblib.load('Nephro_Brain_Final.pkl')
    except Exception as e:
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
st.title("🏥 Nephro-AI: Kinetic CDSS")
st.markdown("**Research Protocol:** Hybrid Kinetic Analysis (Delta-Aware Logic)")

with st.form("patient_form"):
    
    # --- SECTION A: PATIENT CONTEXT (Proposal Requirement 1) ---
    st.subheader("1. Clinical Context")
    c1, c2 = st.columns([1, 2])
    with c1:
        mr_number = st.text_input("MR Number / ID")
    with c2:
        # Crucial for distinguishing AKI vs CKD logic
        patient_type = st.radio("Baseline Renal Function:", 
                                ["De Novo AKI (No prior history)", "Acute on Chronic (Known CKD)"],
                                horizontal=True)
        is_ckd = True if "Chronic" in patient_type else False

    st.divider()

    # --- SECTION B: KINETIC LABS (Proposal Requirement 2) ---
    st.subheader("2. Kinetic Laboratory Data (24h Trend)")
    st.info("Input **Previous (24h ago)** and **Current** values to calculate Kinetics.")

    # 1. CREATININE (The Star Variable)
    k1, k2, k3 = st.columns(3)
    with k1: prev_cr = st.number_input("Prev Creatinine", value=1.5, format="%.2f")
    with k2: curr_cr = st.number_input("Current Creatinine", value=2.0, format="%.2f")
    delta_cr = curr_cr - prev_cr
    with k3: st.metric("Delta Cr", f"{delta_cr:+.2f}", delta_color="inverse")

    # 2. POTASSIUM (Safety Rule Variable)
    p1, p2, p3 = st.columns(3)
    with p1: prev_k = st.number_input("Prev Potassium", value=4.0, format="%.1f")
    with p2: curr_k = st.number_input("Current Potassium", value=4.8, format="%.1f")
    delta_k = curr_k - prev_k
    with p3: st.metric("Delta K+", f"{delta_k:+.1f}", delta_color="inverse")

    # 3. BUN (Urea)
    b1, b2, b3 = st.columns(3)
    with b1: prev_bun = st.number_input("Prev BUN", value=30.0)
    with b2: curr_bun = st.number_input("Current BUN", value=40.0)
    delta_bun = curr_bun - prev_bun
    with b3: st.metric("Delta BUN", f"{delta_bun:+.1f}", delta_color="inverse")

    # 4. ACID-BASE (pH & Bicarb)
    ab1, ab2, ab3, ab4 = st.columns(4)
    with ab1: prev_ph = st.number_input("Prev pH", value=7.35)
    with ab2: curr_ph = st.number_input("Current pH", value=7.30)
    delta_ph = curr_ph - prev_ph
    with ab3: curr_bicarb = st.number_input("Curr Bicarb", value=22.0)
    with ab4: st.metric("Delta pH", f"{delta_ph:+.2f}", delta_color="normal")

    st.divider()

    # --- SECTION C: CLINICAL SIGNS ---
    st.subheader("3. Clinical Parameters")
    cs1, cs2, cs3 = st.columns(3)
    with cs1: fluid = st.selectbox("Fluid Overload", [0, 1, 2, 3], help="0=None, 3=Anasarca")
    with cs2: uo = st.number_input("Urine Output (24h)", value=1000.0)
    with cs3: enceph = st.checkbox("Uremic Encephalopathy?")

    submit_btn = st.form_submit_button("Run Hybrid Analysis")

# ---------------------------------------------------------
# 3. HYBRID ENGINE EXECUTION
# ---------------------------------------------------------
if submit_btn:
    if model:
        # -------------------------------------------------
        # LAYER 1: THE AI BRAIN (XGBoost)
        # -------------------------------------------------
        # We pass strictly the 9 variables the brain was trained on.
        # Note: Brain sees CURRENT values. Trends are handled in Layer 2.
        input_data = pd.DataFrame({
            'creatinine': [curr_cr], 
            'delta_Cr_24h': [delta_cr], 
            'potassium': [curr_k],
            'bicarbonate': [curr_bicarb], 
            'bun': [curr_bun], 
            'ph_level': [curr_ph],
            'fluid_overload_grade': [fluid], 
            'uremic_encephalopathy': [1 if enceph else 0],
            'urine_output_24h': [uo]
        })
        
        # Calculate Risk Probability
        risk_prob = float(model.predict_proba(input_data)[0][1])

        # -------------------------------------------------
        # LAYER 2: THE SAFETY RULES (Deterministic Logic)
        # -------------------------------------------------
        safety_warnings = []
        
        # Rule A: Potassium Velocity
        if delta_k >= 0.5:
            safety_warnings.append(f"⚠️ **Rapid Potassium Rise** (+{delta_k:.1f} mEq/L). Suggests catabolic state.")
        
        # Rule B: Acidosis Velocity
        if delta_ph <= -0.1:
            safety_warnings.append(f"⚠️ **Worsening Acidosis** (pH dropped by {abs(delta_ph):.2f}). Monitor buffering capacity.")
            
        # Rule C: Urea Velocity
        if delta_bun >= 15:
            safety_warnings.append(f"⚠️ **Significant BUN Surge** (+{delta_bun} mg/dL). High uremic load.")
            
        # Rule D: Recovery Override
        is_recovering = False
        if delta_cr <= -0.2:
            is_recovering = True
            risk_prob = 0.15 # Visually override AI score for recovery

        # -------------------------------------------------
        # SAVE STATE & LOGGING
        # -------------------------------------------------
        st.session_state['result'] = {
            'risk': risk_prob,
            'input': input_data,
            'warnings': safety_warnings,
            'context': {'is_ckd': is_ckd, 'delta_cr': delta_cr, 'mr': mr_number},
            'deltas': {'k': delta_k, 'bun': delta_bun, 'ph': delta_ph}
        }
        
        # Log to Database (Expanded Validation Set)
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_row = [
            str(mr_number), str(ts), 
            float(curr_cr), float(delta_cr), float(delta_k), 
            float(risk_prob), str(patient_type)
        ]
        add_to_database(log_row)

    else:
        st.error("❌ Critical Error: AI Brain (.pkl) not found.")

# ---------------------------------------------------------
# 4. RESULTS DASHBOARD (Professional View)
# ---------------------------------------------------------
if 'result' in st.session_state:
    res = st.session_state['result']
    risk = res['risk']
    warnings = res['warnings']
    ctx = res['context']
    
    st.divider()

    # --- A. DIAGNOSTIC CONCLUSION ---
    diag_text = "Uncertain"
    bg_color = "#f8f9fa"
    text_color = "#212529"

    # Logic Matrix
    if ctx['delta_cr'] < -0.1:
        diag_text = "DIAGNOSIS: RENAL RECOVERY PHASE (Improving Kinetics)"
        bg_color = "#d4edda" # Green
        text_color = "#155724"
    elif ctx['is_ckd'] and abs(ctx['delta_cr']) < 0.3:
        diag_text = "DIAGNOSIS: STABLE CKD (Baseline Dysfunction)"
        bg_color = "#fff3cd" # Yellow
        text_color = "#856404"
    elif risk > 0.75 or (ctx['delta_cr'] > 0.3):
        prefix = "ACUTE ON CHRONIC" if ctx['is_ckd'] else "DE NOVO AKI"
        diag_text = f"DIAGNOSIS: {prefix} - URGENT EVALUATION"
        bg_color = "#f8d7da" # Red
        text_color = "#721c24"

    st.markdown(f"<div style='background-color: {bg_color}; color: {text_color};' class='diagnosis-box'>{diag_text}</div>", unsafe_allow_html=True)

    # --- B. RISK GAUGE ---
    c_gauge, c_narrative = st.columns([1, 1])
    
    with c_gauge:
        # Determine Gauge Color
        if risk > 0.70: gauge_color = "#dc3545"
        elif risk > 0.40: gauge_color = "#ffc107"
        else: gauge_color = "#28a745"

        fig = go.Figure(go.Indicator(
            mode = "number+gauge", value = risk * 100,
            number = {'suffix': "%", 'font': {'size': 40}},
            title = {'text': "AI Dialysis Score", 'font': {'size': 20, 'color': "gray"}},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': gauge_color},
                'steps': [{'range': [0, 100], 'color': "#e9ecef"}],
                'threshold': {'line': {'color': "black", 'width': 3}, 'thickness': 0.75, 'value': 75}
            }
        ))
        fig.update_layout(height=250, margin=dict(l=20,r=20,t=40,b=20))
        st.plotly_chart(fig, use_container_width=True)

    # --- C. CLINICAL NARRATIVE (SHAP Engine) ---
    with c_narrative:
        st.subheader("Clinical Analysis")
        
        # Calculate SHAP
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(res['input'])
        
        # Extract Logic
        feats = res['input'].columns
        vals = shap_vals[0]
        impact_df = pd.DataFrame({'feature': feats, 'shap': vals, 'value': res['input'].iloc[0].values})
        
        top_driver = impact_df.sort_values('shap', ascending=False).iloc[0]
        
        # Narrative Generation
        narrative = []
        if ctx['delta_cr'] < 0:
            narrative.append(f"✅ **Protective Kinetic:** Negative Creatinine Delta ({ctx['delta_cr']:.2f}) indicates washout/recovery.")
        else:
            narrative.append(f"ℹ️ **Kinetic Trend:** Creatinine Delta is {ctx['delta_cr']:+.2f}.")
            
        if top_driver['shap'] > 0:
            narrative.append(f"⚠️ **Primary Risk Driver:** {top_driver['feature']} ({top_driver['value']}) is significantly increasing the urgency score.")
            
        st.markdown("\n\n".join(narrative))

    # --- D. SAFETY LAYER ALERTS (The 2nd Layer) ---
    if warnings:
        st.subheader("⚠️ Kinetic Safety Alerts (Rule-Based)")
        for w in warnings:
            st.markdown(f"<div class='warning-box'>{w}</div>", unsafe_allow_html=True)
    elif risk < 0.4:
        st.success("✅ No alarming kinetic velocity detected in K+, BUN, or pH.")

    # --- E. VISUAL EVIDENCE (SHAP) ---
    with st.expander("🔬 View Statistical Evidence (SHAP Waterfall)"):
        fig, ax = plt.subplots(figsize=(8, 6))
        shap.plots.waterfall(
            shap.Explanation(values=shap_vals[0], base_values=explainer.expected_value, 
                             data=res['input'].iloc[0], feature_names=res['input'].columns),
            show=False
        )
        st.pyplot(fig)

# ---------------------------------------------------------
# 5. NEPHRO-GPT (Consultant Mode)
# ---------------------------------------------------------
st.divider()
st.subheader("🤖 Nephro-GPT: Clinical Assistant")

if "result" in st.session_state:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about management (e.g. 'Dose of Lasix?')..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if "GOOGLE_API_KEY" in st.secrets:
            genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
            llm = genai.GenerativeModel('gemini-1.5-flash')
            
            # Context Injection
            sys_prompt = f"""
            Role: Senior Nephrologist.
            Patient: {ctx['mr']}. Diagnosis: {diag_text}.
            AI Score: {risk:.2f}.
            Labs: Cr {curr_cr} (Delta {ctx['delta_cr']}), K {curr_k}.
            Warnings: {warnings}.
            Question: {prompt}
            Response: Brief, clinical, guideline-based.
            """
            
            with st.chat_message("assistant"):
                with st.spinner("Analyzing guidelines..."):
                    response = llm.generate_content(sys_prompt)
                    st.markdown(response.text)
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
