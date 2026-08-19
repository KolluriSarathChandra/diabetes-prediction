import streamlit as st
import streamlit.components.v1 as components
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import json
import os
import time
from datetime import datetime
from fpdf import FPDF

# ============================================================
# DATA / MODEL LOADING (real values only — nothing invented)
# ============================================================
model = joblib.load("model/diabetes_model.pkl")

METRICS = None
if os.path.exists("model/metrics.json"):
    with open("model/metrics.json") as f:
        METRICS = json.load(f)

st.set_page_config(page_title="Diabetes Risk Prediction", page_icon="🩺", layout="wide")

# ============================================================
# PREMIUM BACKGROUND: gradient + glow blobs + grid + neural net
# injected directly into the parent document (bypasses Streamlit's
# internal containers so position:fixed layers actually work)
# ============================================================
components.html("""
<script>
const doc = window.parent.document;
if (!doc.getElementById('premium-bg-style')) {

    const style = doc.createElement('style');
    style.id = 'premium-bg-style';
    style.innerHTML = `
        :root {
            --navy: #071A2B;
            --blue: #2563EB;
            --electric: #38BDF8;
            --teal: #14B8A6;
            --cyan: #67E8F9;
            --white: #F8FAFC;
            --dark: #0F172A;
            --success: #22C55E;
            --warning: #F59E0B;
            --danger: #EF4444;
        }
        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(ellipse 900px 500px at 15% 10%, rgba(37,99,235,0.16), transparent 60%),
                radial-gradient(ellipse 800px 600px at 85% 25%, rgba(20,184,166,0.14), transparent 60%),
                radial-gradient(ellipse 700px 500px at 50% 90%, rgba(56,189,248,0.10), transparent 60%),
                linear-gradient(160deg, #071A2B 0%, #0B2138 45%, #071A2B 100%) !important;
            background-attachment: fixed !important;
        }
        [data-testid="stHeader"] { background: transparent !important; }

        /* subtle animated grid */
        #premium-grid {
            position: fixed; inset: 0; z-index: 0; pointer-events: none;
            background-image:
                linear-gradient(rgba(103,232,249,0.045) 1px, transparent 1px),
                linear-gradient(90deg, rgba(103,232,249,0.045) 1px, transparent 1px);
            background-size: 46px 46px;
            mask-image: radial-gradient(ellipse 80% 60% at 50% 20%, black, transparent 75%);
        }

        /* glowing blobs, slow float */
        .glow-blob {
            position: fixed; border-radius: 50%; filter: blur(70px);
            z-index: 0; pointer-events: none; opacity: 0.5;
            animation: floatBlob 18s ease-in-out infinite;
        }
        @keyframes floatBlob {
            0%, 100% { transform: translate(0,0) scale(1); }
            50% { transform: translate(30px,-25px) scale(1.08); }
        }

        /* neural network connective lines (hero) */
        #neural-net { position: fixed; top: 0; left: 0; width: 100%; height: 340px; z-index: 0; opacity: 0.35; pointer-events:none; }
        .nn-line { stroke: #38BDF8; stroke-width: 1; opacity: 0.35; }
        .nn-node { fill: #67E8F9; filter: drop-shadow(0 0 4px #38BDF8); }
        .nn-pulse { animation: nnPulse 3s ease-in-out infinite; }
        @keyframes nnPulse { 0%,100% { opacity: 0.3; } 50% { opacity: 0.9; } }

        /* Streamlit widget label visibility on dark bg */
        label, .stNumberInput label p, [data-testid="stWidgetLabel"] p {
            color: #F1F5F9 !important; font-weight: 600 !important;
        }

        [data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(255,255,255,0.045);
            border: 1px solid rgba(103,232,249,0.14);
            border-radius: 20px;
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            padding: 8px 4px;
        }

        /* Tabs styled as premium nav */
        .stTabs [data-baseweb="tab-list"] { gap: 6px; }
        .stTabs [data-baseweb="tab"] {
            color: #94A3B8; font-weight: 600; font-size: 14.5px;
            padding: 10px 18px; border-radius: 10px 10px 0 0;
        }
        .stTabs [aria-selected="true"] {
            color: #67E8F9 !important;
            border-bottom: 2px solid #38BDF8 !important;
        }

        iframe { display: none; }
    `;
    doc.head.appendChild(style);

    const grid = doc.createElement('div');
    grid.id = 'premium-grid';
    doc.body.appendChild(grid);

    const blobConfigs = [
        {top:'5%', left:'8%', size:'380px', color:'#2563EB', delay:'0s'},
        {top:'15%', left:'70%', size:'420px', color:'#14B8A6', delay:'4s'},
        {top:'55%', left:'40%', size:'320px', color:'#38BDF8', delay:'8s'},
    ];
    blobConfigs.forEach(b => {
        const el = doc.createElement('div');
        el.className = 'glow-blob';
        el.style.top = b.top; el.style.left = b.left;
        el.style.width = b.size; el.style.height = b.size;
        el.style.background = b.color;
        el.style.animationDelay = b.delay;
        doc.body.appendChild(el);
    });

    // Neural network graphic (hero background)
    const nn = doc.createElement('div');
    nn.id = 'neural-net';
    const nodes = [
        [120,60],[120,160],[120,260],
        [420,40],[420,120],[420,200],[420,280],
        [720,80],[720,180],[720,270],
        [1020,60],[1020,160],[1020,260],
        [1300,100],[1300,220]
    ];
    let lines = '';
    const layers = [[0,1,2],[3,4,5,6],[7,8,9],[10,11,12],[13,14]];
    for (let l = 0; l < layers.length - 1; l++) {
        layers[l].forEach(a => {
            layers[l+1].forEach(b => {
                lines += `<line class="nn-line" x1="${nodes[a][0]}" y1="${nodes[a][1]}" x2="${nodes[b][0]}" y2="${nodes[b][1]}"/>`;
            });
        });
    }
    let circles = '';
    nodes.forEach((n, i) => {
        circles += `<circle class="nn-node nn-pulse" cx="${n[0]}" cy="${n[1]}" r="4" style="animation-delay:${(i%5)*0.4}s"/>`;
    });
    nn.innerHTML = `<svg width="100%" height="340" viewBox="0 0 1400 340" preserveAspectRatio="xMidYMid slice">${lines}${circles}</svg>`;
    doc.body.appendChild(nn);
}
</script>
""", height=0, width=0)

# ============================================================
# GLOBAL CSS (cards, buttons, typography, results)
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero-wrap { position: relative; z-index: 2; text-align: center; padding: 38px 10px 18px 10px; }
.hero-eyebrow {
    display:inline-block; color:#67E8F9; background: rgba(56,189,248,0.10);
    border: 1px solid rgba(56,189,248,0.3); padding: 5px 16px; border-radius: 999px;
    font-size: 12.5px; font-weight:700; letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 18px;
}
.hero-title {
    font-size: 52px; font-weight: 900; color: #F8FAFC; margin: 0 0 10px 0; line-height: 1.1;
    background: linear-gradient(90deg, #F8FAFC 30%, #67E8F9 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-subtitle { font-size: 18px; color: #38BDF8; font-weight: 600; margin-bottom: 10px; }
.hero-desc { font-size: 15px; color: #94A3B8; max-width: 620px; margin: 0 auto; line-height: 1.6; }

.glass-card {
    background: rgba(255,255,255,0.045);
    border: 1px solid rgba(103,232,249,0.14);
    border-radius: 20px;
    backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
    padding: 26px 30px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.35);
    margin-bottom: 20px;
    transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
}
.glass-card:hover {
    transform: translateY(-3px);
    border-color: rgba(103,232,249,0.35);
    box-shadow: 0 14px 40px rgba(0,0,0,0.45);
}
.glass-card-light {
    background: rgba(248,250,252,0.97);
    border-radius: 18px; padding: 24px 28px; margin-top: 16px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
}

.section-title { color: #F8FAFC; font-size: 24px; font-weight: 800; margin-bottom: 4px; }
.section-sub { color: #94A3B8; font-size: 14px; margin-bottom: 22px; }

.step-num {
    font-size: 34px; font-weight: 900; color: transparent;
    -webkit-text-stroke: 1.4px #38BDF8; margin-bottom: 6px;
}
.step-title { color: #67E8F9; font-weight: 800; font-size: 15px; letter-spacing: 1px; margin-bottom: 6px; }
.step-desc { color: #CBD5E1; font-size: 14px; line-height: 1.5; }

.stat-value { font-size: 34px; font-weight: 900; color: #F8FAFC; margin: 2px 0; }
.stat-label { color: #94A3B8; font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
.stat-icon { font-size: 22px; margin-bottom: 6px; }

div.stButton > button {
    background: linear-gradient(90deg, #2563EB 0%, #38BDF8 100%);
    color: white; font-weight: 700; font-size: 16px;
    border: none; border-radius: 12px; padding: 14px 28px; width: 100%;
    box-shadow: 0 6px 20px rgba(37,99,235,0.4);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
div.stButton > button:hover {
    transform: translateY(-2px) scale(1.01);
    box-shadow: 0 10px 28px rgba(56,189,248,0.5);
}
div.stButton > button:active { transform: translateY(0) scale(0.99); }

.result-positive {
    background: linear-gradient(135deg, rgba(239,68,68,0.12), rgba(248,250,252,0.98));
    border-left: 5px solid #EF4444; border-radius: 16px; padding: 26px 30px; margin-top: 18px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
}
.result-moderate {
    background: linear-gradient(135deg, rgba(245,158,11,0.12), rgba(248,250,252,0.98));
    border-left: 5px solid #F59E0B; border-radius: 16px; padding: 26px 30px; margin-top: 18px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
}
.result-negative {
    background: linear-gradient(135deg, rgba(34,197,94,0.12), rgba(248,250,252,0.98));
    border-left: 5px solid #22C55E; border-radius: 16px; padding: 26px 30px; margin-top: 18px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
}
.result-heading { font-size: 22px; font-weight: 800; color: #0F172A; margin-bottom: 6px; }
.result-text { font-size: 15px; color: #334155; line-height: 1.6; }

.tip-item { padding: 9px 0; font-size: 14.5px; line-height: 1.5; color: #1F2937; border-bottom: 1px solid #E2E8F0; }
.tip-item:last-child { border-bottom: none; }

.footer-wrap {
    text-align: center; padding: 34px 10px 20px 10px; margin-top: 30px;
    border-top: 1px solid rgba(103,232,249,0.12);
}
.footer-title { color: #F8FAFC; font-weight: 800; font-size: 16px; }
.footer-sub { color: #64748B; font-size: 12.5px; margin: 6px 0 14px 0; }
.disclaimer-box {
    background: rgba(255,255,255,0.05); border: 1px solid rgba(245,158,11,0.25);
    border-radius: 10px; padding: 12px 20px; font-size: 12.5px; color: #CBD5E1; max-width: 700px; margin: 0 auto;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE
# ============================================================
if "history" not in st.session_state:
    st.session_state.history = []

def risk_tier(prob):
    if prob < 0.40:
        return "LOW RISK", "#22C55E", "result-negative"
    elif prob < 0.70:
        return "MODERATE RISK", "#F59E0B", "result-moderate"
    else:
        return "HIGH RISK", "#EF4444", "result-positive"

# ============================================================
# NAVIGATION
# ============================================================
tab_home, tab_predict, tab_insights, tab_history, tab_about = st.tabs(
    ["🏠  Home", "🔍  Predict", "📊  Insights", "🕓  History", "ℹ️  About"]
)

# ============================================================
# HOME TAB — hero + how it works + stat cards
# ============================================================
with tab_home:
    st.markdown("""
        <div class="hero-wrap">
            <div class="hero-eyebrow">AI · Machine Learning · Healthcare</div>
            <div class="hero-title">Diabetes Risk Prediction</div>
            <div class="hero-subtitle">Intelligent diabetes risk assessment powered by machine learning.</div>
            <div class="hero-desc">
                Enter a patient's clinical indicators and receive an instant, model-driven risk
                assessment — backed by a trained Random Forest classifier, validated with
                cross-validation, and presented with transparent, real performance metrics.
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title" style="text-align:center;">How It Works</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub" style="text-align:center;">Three simple steps from data to insight</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    steps = [
        ("01", "ENTER DATA", "Provide relevant health indicators such as glucose, BMI, blood pressure, and age."),
        ("02", "AI ANALYSIS", "A trained Random Forest model, validated with 5-fold cross-validation, analyzes the input."),
        ("03", "RISK ASSESSMENT", "Receive an instant, transparent estimate of diabetes risk with personalized guidance."),
    ]
    for col, (num, title, desc) in zip([c1, c2, c3], steps):
        with col:
            st.markdown(f"""
                <div class="glass-card">
                    <div class="step-num">{num}</div>
                    <div class="step-title">{title}</div>
                    <div class="step-desc">{desc}</div>
                </div>
            """, unsafe_allow_html=True)

    if METRICS:
        st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title" style="text-align:center;">Model Performance</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub" style="text-align:center;">Real metrics from held-out test data — nothing simulated</div>', unsafe_allow_html=True)

        s1, s2, s3, s4 = st.columns(4)
        stats = [
            ("🎯", "Accuracy", f"{METRICS['accuracy']*100:.1f}%"),
            ("📐", "Precision", f"{METRICS['precision']*100:.1f}%"),
            ("🔁", "Recall", f"{METRICS['recall']*100:.1f}%"),
            ("⚖️", "F1 Score", f"{METRICS['f1_score']*100:.1f}%"),
        ]
        for col, (icon, label, val) in zip([s1, s2, s3, s4], stats):
            with col:
                st.markdown(f"""
                    <div class="glass-card" style="text-align:center;">
                        <div class="stat-icon">{icon}</div>
                        <div class="stat-value">{val}</div>
                        <div class="stat-label">{label}</div>
                    </div>
                """, unsafe_allow_html=True)

# ============================================================
# PREDICT TAB
# ============================================================
with tab_predict:
    st.markdown('<div class="section-title">Patient Health Indicators</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">All fields are used directly by the trained model</div>', unsafe_allow_html=True)

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

        predict_clicked = st.button("✨  CALCULATE DIABETES RISK")

    if predict_clicked:
        status = st.empty()
        for msg in ["🔎 Analyzing health indicators...", "⚙️ Processing clinical features...", "📈 Generating risk assessment..."]:
            status.markdown(f'<div class="glass-card" style="text-align:center; color:#67E8F9; font-weight:600;">{msg}</div>', unsafe_allow_html=True)
            time.sleep(0.5)
        status.empty()

        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]  # real, model-derived probability

        tier_label, tier_color, tier_class = risk_tier(probability)

        st.session_state.history.append({
            "Time": datetime.now().strftime("%H:%M:%S"),
            "Glucose": glucose, "BMI": bmi, "Age": age,
            "Risk %": round(probability * 100, 1),
            "Tier": tier_label
        })

        colA, colB = st.columns([1, 1])
        with colA:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(probability * 100, 1),
                number={'suffix': "%", 'font': {'color': '#F8FAFC', 'size': 44}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#94A3B8', 'tickfont': {'color': '#94A3B8'}},
                    'bar': {'color': tier_color, 'thickness': 0.28},
                    'bgcolor': 'rgba(255,255,255,0.03)',
                    'borderwidth': 0,
                    'steps': [
                        {'range': [0, 40], 'color': 'rgba(34,197,94,0.18)'},
                        {'range': [40, 70], 'color': 'rgba(245,158,11,0.18)'},
                        {'range': [70, 100], 'color': 'rgba(239,68,68,0.18)'},
                    ],
                }
            ))
            gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=280, margin=dict(t=30, b=10, l=20, r=20))
            st.plotly_chart(gauge, use_container_width=True)

        with colB:
            st.markdown(f"""
                <div class="{tier_class}">
                    <div class="result-heading">DIABETES RISK ASSESSMENT</div>
                    <div style="font-size:48px; font-weight:900; color:{tier_color}; margin: 4px 0;">{probability*100:.1f}%</div>
                    <div style="font-size:18px; font-weight:800; color:{tier_color}; letter-spacing:1px; margin-bottom:10px;">{tier_label}</div>
                    <div class="result-text">
                        {"Based on the entered values, the model predicts a higher likelihood of diabetes. This is not a medical diagnosis — please consult a doctor for proper testing." if prediction == 1 else "Based on the entered values, the model predicts a lower likelihood of diabetes. Keep up healthy habits to maintain this."}
                    </div>
                </div>
            """, unsafe_allow_html=True)

        # Guidance
        if prediction == 1:
            st.markdown("""
                <div class="glass-card-light">
                    <div style="font-size:19px; font-weight:800; color:#0F172A; margin-bottom:12px;">💡 General Lifestyle Guidance</div>
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
                <div class="glass-card-light">
                    <div style="font-size:19px; font-weight:800; color:#0F172A; margin-bottom:10px;">👩‍⚕️ Talk to a Doctor</div>
                    <div class="result-text">Based on this risk level, consider booking a check-up <b>{urgency}</b>.
                    A doctor can order an HbA1c or fasting glucose test to confirm your actual status and determine
                    appropriate treatment — medication decisions should always come from a licensed physician, not an app.</div>
                </div>
            """, unsafe_allow_html=True)

            with st.expander("📅 Request a consultation (demo form)"):
                with st.form("consult_form"):
                    name = st.text_input("Full name")
                    preferred_time = st.selectbox("Preferred time", ["Morning", "Afternoon", "Evening"])
                    notes = st.text_area("Notes for the doctor (optional)")
                    submitted = st.form_submit_button("Request Consultation")
                    if submitted:
                        st.success(f"Request received, {name or 'patient'}! A doctor will reach out during your preferred slot: {preferred_time}. (Demo only — no real appointment booked.)")
        else:
            st.markdown("""
                <div class="glass-card-light">
                    <div style="font-size:19px; font-weight:800; color:#0F172A; margin-bottom:12px;">💡 Tips to Stay Healthy</div>
                    <div class="tip-item">🥗 <b>Diet:</b> Maintain a balanced diet rich in fiber, vegetables, and whole grains.</div>
                    <div class="tip-item">🏃 <b>Exercise:</b> Stay active with regular physical activity.</div>
                    <div class="tip-item">🩸 <b>Check-ups:</b> Get periodic health screenings, especially if diabetes runs in your family.</div>
                    <div class="tip-item">😴 <b>Sleep & Stress:</b> Manage stress and prioritize good sleep.</div>
                </div>
            """, unsafe_allow_html=True)

        # PDF report
        def generate_pdf():
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 16)
            pdf.cell(0, 10, "Diabetes Risk Assessment Report", ln=True)
            pdf.set_font("Helvetica", "", 11)
            pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
            pdf.ln(4)
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, f"Result: {tier_label} ({probability*100:.1f}% probability)", ln=True)
            pdf.ln(4)
            pdf.set_font("Helvetica", "", 11)
            for label, val in [("Pregnancies", pregnancies), ("Glucose", glucose), ("Blood Pressure", blood_pressure),
                                ("Skin Thickness", skin_thickness), ("Insulin", insulin), ("BMI", bmi),
                                ("Diabetes Pedigree Function", dpf), ("Age", age)]:
                pdf.cell(0, 7, f"{label}: {val}", ln=True)
            pdf.ln(4)
            pdf.set_font("Helvetica", "I", 9)
            pdf.multi_cell(0, 6, "Disclaimer: This is a statistical prediction from a machine learning model, not a medical diagnosis. Consult a licensed physician for medical advice.")
            return bytes(pdf.output())

        st.download_button("📄 Download PDF Report", data=generate_pdf(),
                            file_name="diabetes_risk_report.pdf", mime="application/pdf")

# ============================================================
# INSIGHTS TAB — feature importance + confusion matrix (real data)
# ============================================================
with tab_insights:
    st.markdown('<div class="section-title">Model Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Transparency into how the model makes decisions — real values from training</div>', unsafe_allow_html=True)

    if METRICS:
        fi_df = pd.DataFrame({
            "Feature": METRICS["feature_names"],
            "Importance": METRICS["feature_importance"]
        }).sort_values("Importance", ascending=True)

        fig = go.Figure(go.Bar(
            x=fi_df["Importance"], y=fi_df["Feature"], orientation='h',
            marker=dict(
                color=fi_df["Importance"],
                colorscale=[[0, "#14B8A6"], [0.5, "#2563EB"], [1, "#38BDF8"]],
            )
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E2E8F0"), height=380,
            margin=dict(t=20, b=20, l=10, r=20),
            xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
        )
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<b style="color:#F8FAFC;">Feature Importance</b>', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Reflects how much each feature influenced the model's decisions — not a claim of medical causation.")
        st.markdown('</div>', unsafe_allow_html=True)

        cm = METRICS["confusion_matrix"]
        cm_fig = go.Figure(data=go.Heatmap(
            z=cm, x=["Predicted: No", "Predicted: Yes"], y=["Actual: No", "Actual: Yes"],
            colorscale=[[0, "#0B2138"], [1, "#38BDF8"]], showscale=False,
            text=cm, texttemplate="%{text}", textfont={"color": "#F8FAFC", "size": 18}
        ))
        cm_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font=dict(color="#E2E8F0"), height=320, margin=dict(t=20, b=20, l=10, r=20))
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<b style="color:#F8FAFC;">Confusion Matrix (Test Set)</b>', unsafe_allow_html=True)
        st.plotly_chart(cm_fig, use_container_width=True)
        st.caption(f"Evaluated on {METRICS['test_set_size']} held-out test samples, trained on {METRICS['train_set_size']}.")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Run train.py to generate model/metrics.json — insights will appear here automatically.")

# ============================================================
# HISTORY TAB
# ============================================================
with tab_history:
    st.markdown('<div class="section-title">Prediction History</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">This session only</div>', unsafe_allow_html=True)
    if st.session_state.history:
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)
        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.markdown('<div class="glass-card">No predictions yet this session.</div>', unsafe_allow_html=True)

# ============================================================
# ABOUT TAB
# ============================================================
with tab_about:
    st.markdown('<div class="section-title">About This Project</div>', unsafe_allow_html=True)
    st.markdown(f"""
        <div class="glass-card">
            <div class="result-text" style="color:#CBD5E1;">
            Built as an end-to-end MLOps pipeline: a Random Forest classifier trained on the PIMA Indians
            Diabetes dataset, version-controlled on GitHub with an automated CI pipeline (GitHub Actions)
            that retrains and validates the model on every push, and deployed via Streamlit Community Cloud.
            </div>
            <div class="result-text" style="color:#CBD5E1; margin-top:10px;">
            <b style="color:#67E8F9;">On accuracy:</b> All performance numbers shown in this app are real,
            computed on a held-out test set — {f"currently {METRICS['accuracy']*100:.1f}% test accuracy" if METRICS else "see model/metrics.json"}.
            Claims of near-100% accuracy on a dataset like this typically indicate data leakage or
            overfitting rather than genuine predictive power.
            </div>
        </div>
    """, unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
    <div class="footer-wrap">
        <div class="footer-title">Diabetes Risk Prediction</div>
        <div class="footer-sub">Machine Learning · Healthcare Analytics · Streamlit</div>
        <div class="disclaimer-box">
            ⚠️ This tool provides a statistical prediction based on a machine learning model and is not a
            substitute for professional medical advice, diagnosis, or treatment.
        </div>
    </div>
""", unsafe_allow_html=True)
