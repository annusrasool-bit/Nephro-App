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
st.set_page_config(page_title="Nephro-AI CDSS", page_icon="🏥", layout="centered")

# Load the Brain (Cached for Speed)
@st.cache_resource
def load_model_v3():
    try:
        return joblib.load('Nephro_Brain_Final.pkl')
    except:
        return None

model = load_model_v3()

# Database Connection (Safe Mode)
def add_to_database(data_row):
    try:
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        # Check if secrets exist before connecting
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
# 2. THE INTERFACE (With Clinical Tooltips)
# ---------------------------------------------------------
st.title("🏥 Nephro-AI Assistant")
st.caption("Clinical Decision Support System with Explainability")

with st.form("patient_form"):
    st.subheader("Patient Vitals & Labs")
    
    col1, col2 = st.columns(2)
    with col1:
        # Creatinine
        cr = st.number_input(
            "Creatinine (mg/dL)", 
            min_value=0.0, value=2.0, step=0.1,
            help="Current serum creatinine. \n\nReference: 0.7 - 1.3 mg/dL. \nValues > 4.0 often indicate severe renal failure."
        )
        
        # Delta Cr (24h)
        delta_cr = st.number_input(
            "Delta Cr (24h change)", 
            value=0.0, step=0.1,
            help="Calculation: (Current Cr) - (Cr 24h ago). \n\nPositive = Worsening (AKI velocity). \n0.0 = Stable."
        )
        
        # Potassium
        k = st.number_input(
            "Potassium (mEq/L)", 
            min_value=0.0, value=4.5, step=0.1,
            help="Serum Potassium. \n\n⚠️ Critical > 6.0 mEq/L (Arrhythmia Risk). \nTarget range: 3.5 - 5.0."
        )
        
        # Bicarbonate
        bicarb = st.number_input(
            "Bicarbonate (mEq/L)", 
            min_value=0.0, value=24.0, step=1.0,
            help="Serum HCO3. \n\nReference: 22 - 29 mEq/L. \nValues < 15 indicate severe metabolic acidosis."
        )
    
    with col2:
        # BUN
        bun = st.number_input(
            "BUN (mg/dL)", 
            min_value=0.0, value=40.0, step=1.0,
            help="Blood Urea Nitrogen. \n\nValues > 80-100 mg/dL suggest Uremic Toxicity (bleeding risk, pericarditis)."
        )
        
        # pH Level
        ph = st.number_input(
            "pH Level", 
            min_value=6.8, max_value=7.6, value=7.4, step=0.01,
            help="Arterial or Venous pH. \n\n⚠️ Critical < 7.15. \nAcidosis causes enzyme denaturation and myocardial depression."
        )
        
        # Urine Output
        uo = st.number_input(
            "Urine Output 24h (ml)", 
            min_value=0.0, value=1500.0, step=50.0,
            help="Total volume in last 24 hours. \n\nOliguria: < 400ml. \nAnuria: < 100ml. \nPolyuria: > 3000ml."
        )
        
    st.subheader("Clinical Signs")
    
    # Fluid Overload
    fluid = st.selectbox(
        "Fluid Overload Grade", 
        [0, 1, 2, 3], 
        help="0 = No Edema \n1 = Mild Pedal Edema \n2 = Pulmonary Crackles / Facial Edema \n3 = Anasarca / Respiratory Distress (Requires O2)"
    )
    
    # Encephalopathy
    enceph = st.checkbox(
        "Uremic Encephalopathy Present?",
        help="Check if patient has: \n- Confusion / Altered Mental Status \n- Asterixis (Flapping Tremor) \n- Seizures attributed to Uremia."
    )
    
    st.divider()
    st.markdown("### 💾 Data Options")
    save_data = st.checkbox("Contribute this case to AI Training Database?", value=False)
    
    # Submission Button
    submitted = st.form_submit_button("Run Analysis")

# ---------------------------------------------------------
# 3. LOGIC ENGINE & DATA SAVING
# ---------------------------------------------------------
if submitted:
    if model:
        # 1. Prepare Data
        input_data = pd.DataFrame({
            'creatinine': [cr], 'delta_Cr_24h': [delta_cr], 'potassium': [k],
            'bicarbonate': [bicarb], 'bun': [bun], 'ph_level': [ph],
            'fluid_overload_grade': [fluid], 'uremic_encephalopathy': [1 if enceph else 0],
            'urine_output_24h': [uo]
        })
        
        # Align columns
        try:
            input_data = input_data[model.feature_names_in_]
        except:
            pass
        
        # 2. Predict Probability
        risk_prob_raw = model.predict_proba(input_data)[0][1]
        risk_prob = float(risk_prob_raw)
        
        # 3. SAVE TO MEMORY (Critical for Chatbot stability)
        st.session_state['risk_prob'] = risk_prob
        st.session_state['patient_context'] = {
            'cr': cr, 'bun': bun, 'k': k, 'ph': ph,
            'fluid': fluid, 'enceph': enceph, 'uo': uo,
            'input_data': input_data # Save dataframe for graphs
        }
        
        # 4. Save to Cloud Database
        if save_data:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_row = [
                str(timestamp), float(cr), float(delta_cr), float(k), 
                float(bicarb), float(bun), float(ph), int(fluid), 
                int(enceph), float(uo), round(risk_prob, 3)
            ]
            if add_to_database(log_row):
                st.toast("✅ Saved for training!", icon="🧬")

    else:
        st.error("⚠️ AI Brain (Model) not found. Check GitHub files.")

# ---------------------------------------------------------
# 4. DASHBOARD & VISUALS
# ---------------------------------------------------------
# We check session_state so results persist during chat
if 'risk_prob' in st.session_state:
    risk_prob = st.session_state['risk_prob']
    input_data = st.session_state['patient_context']['input_data']
    
    st.divider()
    st.subheader("1. Clinical Assessment")

    # --- VISUAL 1: MINIMALIST BULLET TRACK ---
    if risk_prob > 0.75:
        bar_color = "#FF4B4B" # Red
    elif risk_prob > 0.40:
        bar_color = "#FFD700" # Yellow
    else:
        bar_color = "#90EE90" # Green

    fig_gauge = go.Figure(go.Indicator(
        mode = "number+gauge",
        value = risk_prob * 100,
        number = {'suffix': "%", 'font': {'size': 30, 'family': "Arial"}},
        title = {'text': "Dialysis Urgency", 'font': {'size': 18, 'color': "gray"}},
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'shape': "bullet",
            'axis': {'range': [None, 100], 'visible': False}, 
            'bar': {'color': bar_color, 'thickness': 0.25},   
            'bgcolor': "#E8E8E8",                             
            'steps': [{'range': [0, 100], 'color': "#f0f2f6"}],
            'threshold': {'line': {'color': "gray", 'width': 2}, 'thickness': 0.75, 'value': 75}
        }
    ))
    fig_gauge.update_layout(height=120, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

    # --- VISUAL 2: CONSULTANT NOTE ---
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)
    
    feature_importance = pd.DataFrame({
        'feature': input_data.columns,
        'importance': shap_values[0],
        'value': input_data.iloc[0].values
    })
    feature_importance['abs_importance'] = feature_importance['importance'].abs()
    top_factors = feature_importance.sort_values('abs_importance', ascending=False).head(3)
    
    reasoning_text = []
    for index, row in top_factors.iterrows():
        direction = "increased" if row['importance'] > 0 else "decreased"
        reasoning_text.append(f"**{row['feature']} ({row['value']})** {direction} risk")
        
    st.info(f"💡 **AI Logic:** Decision primarily driven by: {', '.join(reasoning_text)}")

    # --- VISUAL 3: RISK BAR CHART ---
    st.subheader("2. Risk Factor Breakdown")
    pos_factors = feature_importance[feature_importance['importance'] > 0].sort_values('importance')
    neg_factors = feature_importance[feature_importance['importance'] < 0].sort_values('importance')
    
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(y=pos_factors['feature'], x=pos_factors['importance'], orientation='h', name='Risk', marker_color='#FF4B4B'))
    fig_bar.add_trace(go.Bar(y=neg_factors['feature'], x=neg_factors['importance'], orientation='h', name='Protective', marker_color='#90EE90'))
    fig_bar.update_layout(barmode='relative', height=300, margin=dict(l=0, r=0, t=20, b=20))
    st.plotly_chart(fig_bar, use_container_width=True)

    # --- VISUAL 4: HIDDEN WATERFALL (DEEP DIVE) ---
    with st.expander("🔬 View Raw Statistical Waterfall (Complex)", expanded=False):
        st.markdown("**Raw SHAP Trace:**")
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.plots.waterfall(
            shap.Explanation(values=shap_values[0], base_values=explainer.expected_value, 
                            data=input_data.iloc[0], feature_names=input_data.columns),
            show=False
        )
        st.pyplot(fig) 
            # ---------------------------------------------------------
            # 5. DYNAMIC CLINICAL INTERPRETATION (VERBAL SUMMARY)
            # ---------------------------------------------------------
            st.divider()
            st.markdown("### 🗣️ Verbalized Analysis")
            
            # 1. Sort factors into "Hurting" (Positive) and "Helping" (Negative)
            # We assume 'feature_importance' dataframe was created in the visual sections above. 
            # If not, we recreate it briefly here to be safe:
            feature_importance = pd.DataFrame({
                'feature': input_data.columns,
                'importance': shap_values[0],
                'value': input_data.iloc[0].values
            })
            
            pos_drivers = feature_importance[feature_importance['importance'] > 0].sort_values('importance', ascending=False).head(2)
            neg_drivers = feature_importance[feature_importance['importance'] < 0].sort_values('importance', ascending=True).head(2)
            
            # 2. Helper to format text (e.g., "Potassium (6.5)")
            def fmt(row): return f"**{row['feature']}** is {row['value']}"
            
            # 3. Build the Sentence
            if risk_prob < 0.50:
                # SCENARIO: LOW RISK (Holding Off)
                if not pos_drivers.empty:
                    bad_factors = " and ".join([fmt(row) for _, row in pos_drivers.iterrows()])
                    opening = f"Although {bad_factors}, "
                else:
                    opening = "With no major risk factors present, "
                
                good_factors = " and ".join([fmt(row) for _, row in neg_drivers.iterrows()])
                summary = f"{opening}the AI recommends **holding off on dialysis**. This is primarily because {good_factors}, which provides a protective stability."
                st.success(f"🛡️ **Interpretation:** {summary}")
    
            else:
                # SCENARIO: HIGH RISK (Initiate)
                bad_factors = " and ".join([fmt(row) for _, row in pos_drivers.iterrows()])
                if not neg_drivers.empty:
                    good_factors = " and ".join([fmt(row) for _, row in neg_drivers.iterrows()])
                    ending = f"Despite {good_factors}, the risk remains critical."
                else:
                    ending = "There are no significant protective factors buffering this risk."
                    
                summary = f"The AI recommends **urgent consideration for dialysis**. This is driven primarily because {bad_factors}. {ending}"
                st.error(f"🚨 **Interpretation:** {summary}")

# ---------------------------------------------------------
# 5. FOOTER: GUIDELINES & REFERENCES
# ---------------------------------------------------------
st.divider()
with st.expander("ℹ️ Evidence, Guidelines & Creator Info"):
    st.markdown("""
    ### 🧠 How this AI Works
    This tool utilizes a **Hybrid Clinical-AI Model** trained on 3,000 clinically validated scenarios based on **KDIGO Emergency Criteria**.
    
    ### 📚 Key References
    1.  **KDIGO 2012:** *Clinical Practice Guideline for Acute Kidney Injury*.
    2.  **The IDEAL Study:** *Initiation of Dialysis Early and Late*. N Engl J Med 2010.
    
    ### 👨‍⚕️ About the Creator
    **Dr. [Annus Rasool]** *Nephrology Resident & AI Developer*
    *Disclaimer: This tool is a Clinical Decision Support System (CDSS) for educational purposes only.*
    """)

# ---------------------------------------------------------
# 6. NEPHRO-GPT (AI CONSULTANT)
# ---------------------------------------------------------
st.divider()
st.subheader("🤖 Nephro-GPT: Management Assistant")

if 'risk_prob' in st.session_state:
    risk_p = st.session_state['risk_prob']
    ctx = st.session_state['patient_context']
    
    # 1. Setup API (Safe Check)
    try:
        if "GOOGLE_API_KEY" in st.secrets:
            genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
            model_ai = genai.GenerativeModel('gemini-1.5-flash')
        else:
            st.warning("⚠️ Chatbot disabled: Add 'GOOGLE_API_KEY' to Streamlit Secrets.")
            model_ai = None
    except:
        model_ai = None

    # 2. Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 3. SBAR Prompt (Auto-generated)
    sbar_context = f"""
    ACT AS A SENIOR NEPHROLOGIST.
    PATIENT SBAR:
    - Assessment: Cr {ctx['cr']}, K {ctx['k']}, pH {ctx['ph']}, Fluid {ctx['fluid']}, UO {ctx['uo']}
    - Dialysis Risk: {risk_p:.1%}
    - Task: Provide brief management steps based on KDIGO.
    """

    # 4. Show Chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 5. Handle Input
    if prompt := st.chat_input("Ask about management (e.g., 'Dose of Lasix?')..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if model_ai:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = model_ai.generate_content(sbar_context + "\n\nUser Question: " + prompt)
                        st.markdown(response.text)
                        st.session_state.messages.append({"role": "assistant", "content": response.text})
                    except Exception as e:
                        st.error(f"AI Error: {e}")
else:
    st.info("👆 **Please run the analysis above** to activate the AI Consultant.")
