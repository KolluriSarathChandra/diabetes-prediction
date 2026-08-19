import streamlit as st
import streamlit.components.v1 as components
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from fpdf import FPDF
from datetime import datetime
import io

model = joblib.load("model/diabetes_model.pkl")

st.set_page_config(page_title="Diabetes Prediction System", page_icon="🩺", layout="wide")

# ===== BACKGROUND ANIMATION =====
components.html("""
<script>
const doc = window.parent.document;
if (!doc.getElementById('bg-anim-style')) {
    const style = doc.createElement('style');
    style.id = 'bg-anim-style';
    style.innerHTML = `
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%) !important;
            background-size: 400% 400% !important;
            animation: gradientShift 12s ease infinite !important;
        }
        [data-testid="stHeader"] { background: transparent !important; }
        @keyframes gradientShift { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }
        .bg-particle { position: fixed; width:9px; height:9px; background:#ef4444; border-radius:50%;
            box-shadow:0 0 14px 5px rgba(239,68,68,0.7); animation: floatUp linear infinite; opacity:0.85; z-index:0; pointer-events:none; }
        @keyframes floatUp { 0%{transform:translateY(110vh) scale(0.6);opacity:0} 10%{opacity:0.85} 90%{opacity:0.85} 100%{transform:translateY(-10vh) scale(1.2);opacity:0} }
        #ecg-container { position:fixed; top:0; left:0; width:100%; height:90px; z-index:0; overflow:hidden; opacity:0.6; pointer-events:none; }
        #ecg-line { stroke:#22d3ee; stroke-width:2.5; fill:none; filter:drop-shadow(0 0 6px #22d3ee);
            stroke-dasharray:1500; stroke-dashoffset:1500; animation: draw 4s linear infinite; }
        @keyframes draw { 0%{stroke-dashoffset:1500} 100%{stroke-dashoffset:-1500} }
        label, .stNumberInput label p, [data-testid="stWidgetLabel"] p { color:#f1f5f9 !important; font-weight:600 !important; text-shadow:0 1px 3px rgba(0,0,0,0.5); }
        [data-testid="stVerticalBlockBorderWrapper"] { background-color: rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.15); border-radius:16px; padding:10px 6px; }
        .stTabs [data-baseweb="tab"] { color: #e2e8f0; font-weight:600; }
        iframe { display:none; }
    `;
    doc.head.appendChild(style);
    const positions = [10,22,35,50,65,78,88];
    positions.forEach((left, i) => {
        const p = doc.createElement('div');
        p.className = 'bg-particle';
        p.style.left = left + '%';
        p.style.animationDuration = (8 + i*0.7) + 's';
        p.style.animationDelay = (i*0.5) + 's';
        doc.body.appendChild(p);
    });
    const ecgWrap = doc.createElement('div');
    ecgWrap.id = 'ecg-container';
    ecgWrap.innerHTML = `<svg width="100%" height="90" viewBox="0 0 1500 90" preserveAspectRatio="none">
        <path id="ecg-line" d="M0,45 L100,45 L120,45 L135,10 L150,80 L165,45 L200,45 L400,45 L420,45 L435,10 L450,80 L465,45 L500,45
        L700,45 L720,45 L735,10 L750,80 L765,45 L800,45 L1000,45 L1020,45 L1035,10 L1050,80 L1065,45 L1100,45
        L1300,45 L1320,45 L1335,10 L1350,80 L1365,45 L1500,45"/></svg>`;
    doc.body.appendChild(ecgWrap);
}
</script>
""", height=0, width=0)

# ===== STYLING =====
st.markdown("""
<style>
.title-text { font-size:44px; font-weight:800; color:#f8fafc; text-align:center; margin-bottom:4px; text-shadow:0 0 20px rgba(34,211,238,0.5); }
.subtitle-text { font-size:16px; color:#e2e8f0; text-align:center; margin-bottom:20px; }
.stButton>button { background: linear-gradient(90deg, #2563eb, #22d3ee); color:white; font-weight:700; border-radius:10px;
    padding:12px 24px; border:none; width:100%; font-size:16px; }
.result-box-positive { background-color:rgba(254,242,242,0.98); border-left:6px solid #dc2626; padding:22px 26px; border-radius:12px; box-shadow:0 8px 20px rgba(0,0,0,0.3); margin-top:20px; }
.result-box-negative { background-color:rgba(240,253,244,0.98); border-left:6px solid #16a34a; padding:22px 26px; border-radius:12px; box-shadow:0 8px 20px rgba(0,0,0,0.3); margin-top:20px; }
.result-box-positive h3, .result-box-negative h3 { color:#1f2937; font-size:22px; }
.result-box-positive p, .result-box-negative p { color:#374151; font-size:15px; line-height:1.5; }
.tips-card { background-color:rgba(255,255,255,0.98); border-radius:14px; padding:22px 28px; margin-top:16px; box-shadow:0 8px 20px rgba(0,0,0,0.3); }
.tips-title { font-size:20px; font-weight:800; color:#1f2937; margin-bottom:12px; }
.tip-item { padding:8px 0; font-size:15px; line-height:1.5; color:#1f2937; border-bottom:1px solid #f1f5f9; }
.tip-item:last-child { border-bottom:none; }
.consult-card { background-color:rgba(255,255,255,0.98); border-radius:14px; padding:22px 28px; margin-top:16px; box-shadow:0 8px 20px rgba(0,0,0,0.3); }
.disclaimer-box { background-color:rgba(255,255,255,0.9); border-radius:10px; padding:14px 20px; margin-top:22px; font-size:13px; color:#4b5563; text-align:center; }
</style>
""", unsafe_allow_html=True)

# ===== SESSION STATE FOR HISTORY =====
if "history" not in st.session_state:
    st.session_state.history = []

# ===== HEADER =====
st.markdown('<p class="title-text">🩺 Diabetes Prediction System</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">AI-powered risk assessment with personalized guidance</p>', unsafe_allow_html=True)

tab_home, tab_predict, tab_history, tab_about = st.tabs(["🏠 Home", "🔍 Predict", "📊 History", "ℹ️ About"])

# ===== HOME TAB =====
with tab_home:
    st.markdown("""
        <div class="tips-card">
            <div class="tips-title">Welcome</div>
            <p style="color:#374151; font-size:15px; line-height:1.6;">
            This tool uses a machine learning model trained on patient health data to estimate diabetes risk.
            Go to the <b>Predict</b> tab to enter patient details and get a risk assessment, personalized
            lifestyle guidance, and the option to request a doctor consultation. Your session's past predictions
            are saved in the <b>History</b> tab.
            </p>
        </div>
    """, unsafe_allow_html=True)

# ===== PREDICT TAB =====
with tab_predict:
    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
            blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
            insulin = st.number_input("Insulin", min_value=0, max_value=900, value=79)
            dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
        with col2:
            glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120)
            skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
            bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
            age = st.number_input("Age", min_value=1, max_value=120, value=30)

        predict_clicked = st.button("🔍 Predict")

    if predict_clicked:
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]  # probability of class 1 (diabetic)

        # Save to history
        st.session_state.history.append({
            "Time": datetime.now().strftime("%H:%M:%S"),
            "Glucose": glucose, "BMI": bmi, "Age": age,
            "Risk %": round(probability * 100, 1),
            "Result": "Diabetic" if prediction == 1 else "Not Diabetic"
        })

        colA, colB = st.columns([1, 1])

        with colA:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                title={'text': "Diabetes Risk %", 'font': {'color': 'white'}},
                number={'font': {'color': 'white'}, 'suffix': "%"},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': 'white'},
                    'bar': {'color': "#ef4444" if probability > 0.5 else "#16a34a"},
                    'steps': [
                        {'range': [0, 40], 'color': "#166534"},
                        {'range': [40, 70], 'color': "#ca8a04"},
                        {'range': [70, 100], 'color': "#7f1d1d"},
                    ],
                }
            ))
            gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=280, margin=dict(t=40, b=10))
            st.plotly_chart(gauge, use_container_width=True)

        with colB:
            ref_df = pd.DataFrame({
                "Metric": ["Glucose", "BMI", "Blood Pressure", "Age"],
                "Patient Value": [glucose, bmi, blood_pressure, age],
                "Healthy Reference": [100, 22, 80, 35]
            })
            fig = px.bar(ref_df, x="Metric", y=["Patient Value", "Healthy Reference"], barmode="group",
                         color_discrete_sequence=["#ef4444", "#22d3ee"])
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font=dict(color="white"), height=280, legend=dict(font=dict(color="white")))
            st.plotly_chart(fig, use_container_width=True)

        if prediction == 1:
            st.markdown(f"""
                <div class="result-box-positive">
                    <h3>⚠️ Result: Likely Diabetic ({probability*100:.1f}% risk)</h3>
                    <p>Based on the entered values, the model predicts a higher risk of diabetes.
                    This is not a medical diagnosis — please consult a doctor for proper testing and advice.</p>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("""
                <div class="tips-card">
                    <div class="tips-title">💡 General Lifestyle Guidance</div>
                    <div class="tip-item">🥗 <b>Diet:</b> Reduce refined sugar and processed carbs. Favor whole grains, vegetables, and lean protein.</div>
                    <div class="tip-item">🏃 <b>Exercise:</b> Aim for at least 30 minutes of moderate activity most days of the week.</div>
                    <div class="tip-item">⚖️ <b>Weight:</b> Gradual weight loss, if overweight, can meaningfully improve blood sugar control.</div>
                    <div class="tip-item">🩸 <b>Monitoring:</b> Regularly check blood glucose levels as advised by a healthcare provider.</div>
                    <div class="tip-item">😴 <b>Sleep:</b> Poor sleep affects insulin sensitivity — aim for 7-8 hours nightly.</div>
                    <div class="tip-item">👩‍⚕️ <b>Medical follow-up:</b> See a doctor for an HbA1c test and a personalized care plan.</div>
                </div>
            """, unsafe_allow_html=True)

            urgency = "within the next 1–2 weeks" if probability > 0.7 else "at your next convenient opportunity"
            st.markdown(f"""
                <div class="consult-card">
                    <div class="tips-title">👩‍⚕️ Talk to a Doctor</div>
                    <p style="color:#374151; font-size:15px;">Based on this risk level, we'd suggest booking a check-up
                    <b>{urgency}</b>. A doctor can order an HbA1c/fasting glucose test to confirm your actual status
                    and, if needed, prescribe appropriate treatment — medication decisions should always come from a
                    licensed physician, not an app.</p>
                </div>
            """, unsafe_allow_html=True)

            with st.expander("📅 Request a consultation (demo form)"):
                with st.form("consult_form"):
                    name = st.text_input("Full name")
                    preferred_time = st.selectbox("Preferred time", ["Morning", "Afternoon", "Evening"])
                    notes = st.text_area("Notes for the doctor (optional)")
                    submitted = st.form_submit_button("Request Consultation")
                    if submitted:
                        st.success(f"Request received, {name or 'patient'}! A doctor will reach out during your preferred time slot: {preferred_time}. (This is a demo — no real appointment was booked.)")

        else:
            st.markdown(f"""
                <div class="result-box-negative">
                    <h3>✅ Result: Likely Not Diabetic ({probability*100:.1f}% risk)</h3>
                    <p>Based on the entered values, the model predicts a lower risk of diabetes. Keep up healthy habits to maintain this.</p>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("""
                <div class="tips-card">
                    <div class="tips-title">💡 Tips to Stay Healthy</div>
                    <div class="tip-item">🥗 <b>Diet:</b> Maintain a balanced diet rich in fiber, vegetables, and whole grains.</div>
                    <div class="tip-item">🏃 <b>Exercise:</b> Stay active with regular physical activity.</div>
                    <div class="tip-item">🩸 <b>Check-ups:</b> Get periodic health screenings, especially if diabetes runs in your family.</div>
                    <div class="tip-item">😴 <b>Sleep & Stress:</b> Manage stress and prioritize good sleep.</div>
                </div>
            """, unsafe_allow_html=True)

        # PDF Report
        def generate_pdf():
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 16)
            pdf.cell(0, 10, "Diabetes Risk Assessment Report", ln=True)
            pdf.set_font("Helvetica", "", 11)
            pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
            pdf.ln(5)
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, f"Result: {'Likely Diabetic' if prediction == 1 else 'Likely Not Diabetic'}", ln=True)
            pdf.cell(0, 8, f"Risk Probability: {probability*100:.1f}%", ln=True)
            pdf.ln(5)
            pdf.set_font("Helvetica", "", 11)
            fields = [("Pregnancies", pregnancies), ("Glucose", glucose), ("Blood Pressure", blood_pressure),
                      ("Skin Thickness", skin_thickness), ("Insulin", insulin), ("BMI", bmi),
                      ("Diabetes Pedigree Function", dpf), ("Age", age)]
            for label, val in fields:
                pdf.cell(0, 7, f"{label}: {val}", ln=True)
            pdf.ln(5)
            pdf.set_font("Helvetica", "I", 9)
            pdf.multi_cell(0, 6, "Disclaimer: This is a statistical prediction from a machine learning model, not a medical diagnosis. Consult a licensed physician for medical advice.")
            return bytes(pdf.output())

        pdf_bytes = generate_pdf()
        st.download_button("📄 Download PDF Report", data=pdf_bytes,
                            file_name="diabetes_risk_report.pdf", mime="application/pdf")

# ===== HISTORY TAB =====
with tab_history:
    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        st.markdown('<div class="tips-card"><div class="tips-title">📊 Prediction History (this session)</div></div>', unsafe_allow_html=True)
        st.dataframe(hist_df, use_container_width=True)
        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.markdown('<div class="tips-card"><p style="color:#374151;">No predictions yet this session. Go to the Predict tab to get started.</p></div>', unsafe_allow_html=True)

# ===== ABOUT TAB =====
with tab_about:
    st.markdown("""
        <div class="tips-card">
            <div class="tips-title">About This Project</div>
            <p style="color:#374151; font-size:15px; line-height:1.6;">
            Built as an end-to-end MLOps pipeline: a Random Forest classifier trained on the PIMA Indians
            Diabetes dataset, version-controlled on GitHub with an automated CI pipeline (GitHub Actions)
            that retrains and validates the model on every push, and deployed via Streamlit Community Cloud.
            </p>
            <p style="color:#374151; font-size:15px; line-height:1.6;">
            <b>Note on accuracy:</b> This model achieves realistic real-world accuracy for this dataset
            (roughly 78–83%). Claims of near-100% accuracy on a dataset like this typically indicate
            data leakage or overfitting rather than genuine predictive power.
            </p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("""
    <div class="disclaimer-box">
        ⚠️ This tool provides a statistical prediction based on a machine learning model and is not a substitute for professional medical advice, diagnosis, or treatment.
    </div>
""", unsafe_allow_html=True)