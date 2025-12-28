import streamlit as st
import pandas as pd
import joblib
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import datetime
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# ---------------------------------------------------------
# 1. SETUP & CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(page_title="Nephro-AI CDSS", page_icon="🏥", layout="centered")

# Load the Brain
@st.cache_resource
def load_model_v3():
    try:
        return joblib.load('Nephro_Brain_Final.pkl')
    except:
        return None

model = load_model_v3()

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
st.caption("Clinical Decision Support System with Explainability")

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
# 3. LOGIC & HYBRID DASHBOARD
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
        
        # --- SAVE TO MEMORY (CRITICAL FIX) ---
        st.session_state['risk_prob'] = risk_prob
        st.session_state['patient_context'] = {
            'cr': cr, 'bun': bun, 'k': k, 'ph': ph,
            'fluid': fluid, 'enceph': enceph, 'uo': uo
        }
        # -------------------------------------

        # --- LAYER 1: THE CLINICAL DASHBOARD ---
        st.divider()
        st.subheader("1. Clinical Assessment")
        
        # (Rest of your Dashboard/Visuals code goes here - no changes needed to visuals)
        # ... [Keep your Gauge and Charts code] ...
        
      # -----------------------------------------------------
        # VISUAL 1: MINIMALIST RISK STRIP (BULLET CHART)
        # -----------------------------------------------------
        st.divider()
        st.subheader("1. Clinical Assessment")
        
        # Determine color based on risk
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
                'axis': {'range': [None, 100], 'visible': False}, # Hide ugly ticks
                'bar': {'color': bar_color, 'thickness': 0.25},   # The patient's score
                'bgcolor': "#E8E8E8",                             # Grey background track
                'steps': [
                    {'range': [0, 100], 'color': "#f0f2f6"}      # Subtle background
                ],
                # Add threshold markers for context (Low/Mod/High cutoffs)
                'threshold': {
                    'line': {'color': "gray", 'width': 2},
                    'thickness': 0.75,
                    'value': 75 # Mark the "High Risk" line
                }
            }
        ))
        
        # Adjust height to be slim
        fig_gauge.update_layout(height=120, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
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

        # --- LAYER 2: THE SCIENTIFIC DEEP DIVE (Collapsible) ---
        # This keeps the dashboard clean, but holds the complex stats for review.
        
        with st.expander("🔬 View Advanced Statistical Analysis (SHAP)", expanded=False):
            st.markdown("### Statistical Breakdown")
            st.caption("This graph shows the exact mathematical contribution of every variable (Shapley Values).")
            
            # 1. Simplified Bar Chart (Easier to read)
            pos_factors = feature_importance[feature_importance['importance'] > 0].sort_values('importance')
            neg_factors = feature_importance[feature_importance['importance'] < 0].sort_values('importance')
            
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(y=pos_factors['feature'], x=pos_factors['importance'], orientation='h', name='Risk Factors', marker_color='#FF4B4B'))
            fig_bar.add_trace(go.Bar(y=neg_factors['feature'], x=neg_factors['importance'], orientation='h', name='Protective Factors', marker_color='#90EE90'))
            fig_bar.update_layout(title="Risk Drivers (Red) vs Protective Factors (Green)", barmode='relative')
            st.plotly_chart(fig_bar, use_container_width=True)
            
            # 2. The Original Waterfall Plot (For Authenticity)
            st.markdown("---")
            st.markdown("**Raw Waterfall Trace:**")
            fig, ax = plt.subplots(figsize=(8, 5))
            shap.plots.waterfall(
                shap.Explanation(values=shap_values[0], base_values=explainer.expected_value, 
                                data=input_data.iloc[0], feature_names=input_data.columns),
                show=False
            )
            st.pyplot(fig)

        # Save to Cloud
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
# 4. FOOTER: GUIDELINES & REFERENCES
# ---------------------------------------------------------
st.divider()
with st.expander("ℹ️ Evidence, Guidelines & Creator Info"):
    st.markdown("""
    ### 🧠 How this AI Works
    This tool utilizes a **Hybrid Clinical-AI Model** trained on 3,000 clinically validated scenarios...
    *(Your references from the previous step go here)*
    """)



import google.generativeai as genai # <--- Add this to your TOP imports

# ---------------------------------------------------------
# 5. AI CONSULTANT (NEPHRO-GPT)
# ---------------------------------------------------------
st.divider()
st.subheader("🤖 Nephro-GPT: Management Assistant")

# Check if we have a patient in memory
if 'risk_prob' in st.session_state:
    # Get the saved data
    risk_p = st.session_state['risk_prob']
    ctx = st.session_state['patient_context']
    
    st.caption(f"Chatting about patient with Risk: {risk_p:.1%}")

    # 1. Setup API
    try:
        if "GOOGLE_API_KEY" in st.secrets:
            genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
            model_ai = genai.GenerativeModel('gemini-1.5-flash')
        else:
            st.error("⚠️ Google API Key missing. Please add it to Secrets.")
            model_ai = None
    except Exception as e:
        st.error(f"API Error: {e}")
        model_ai = None

    # 2. Initialize Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 3. Create the SBAR Context
    sbar_context = f"""
    ACT AS A SENIOR NEPHROLOGIST.
    PATIENT SBAR:
    - Situation: Patient requires risk assessment for dialysis.
    - Assessment:
      - Creatinine: {ctx['cr']}, BUN: {ctx['bun']}
      - Potassium: {ctx['k']}, pH: {ctx['ph']}
      - Fluid Grade: {ctx['fluid']}, Encephalopathy: {ctx['enceph']}
      - Urine Output: {ctx['uo']}ml/24h
      - Calculated Dialysis Probability: {risk_p:.1%}
    - Recommendation: Guide management based on KDIGO guidelines.
    """

    # 4. Display Chat Interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 5. Handle User Input
    if prompt := st.chat_input("Ask about management (e.g., 'Dose of Lasix?')..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if model_ai:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    full_prompt = sbar_context + "\n\nUser Question: " + prompt
                    try:
                        response = model_ai.generate_content(full_prompt)
                        st.markdown(response.text)
                        st.session_state.messages.append({"role": "assistant", "content": response.text})
                    except Exception as e:
                        st.error(f"AI Error: {e}")

else:
    st.info("👆 **Please run the analysis above** to activate the AI Consultant.")
