import streamlit as st
import streamlit.components.v1 as components
import joblib
import numpy as np

# Load trained model
model = joblib.load("model/diabetes_model.pkl")

# Page config
st.set_page_config(page_title="Diabetes Prediction System", page_icon="🩺", layout="centered")

# ===== INJECT BACKGROUND ANIMATION DIRECTLY INTO THE PAGE =====
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
        @keyframes gradientShift {
            0%   { background-position: 0% 50%; }
            50%  { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        .bg-particle {
            position: fixed;
            width: 9px;
            height: 9px;
            background: #ef4444;
            border-radius: 50%;
            box-shadow: 0 0 14px 5px rgba(239,68,68,0.7);
            animation: floatUp linear infinite;
            opacity: 0.85;
            z-index: 0;
            pointer-events: none;
        }
        @keyframes floatUp {
            0%   { transform: translateY(110vh) scale(0.6); opacity: 0; }
            10%  { opacity: 0.85; }
            90%  { opacity: 0.85; }
            100% { transform: translateY(-10vh) scale(1.2); opacity: 0; }
        }
        #ecg-container {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 90px;
            z-index: 0;
            overflow: hidden;
            opacity: 0.6;
            pointer-events: none;
        }
        #ecg-line {
            stroke: #22d3ee;
            stroke-width: 2.5;
            fill: none;
            filter: drop-shadow(0 0 6px #22d3ee);
            stroke-dasharray: 1500;
            stroke-dashoffset: 1500;
            animation: draw 4s linear infinite;
        }
        @keyframes draw {
            0%   { stroke-dashoffset: 1500; }
            100% { stroke-dashoffset: -1500; }
        }
        iframe {
            display: none;
        }
    `;
    doc.head.appendChild(style);

    const positions = [10,22,35,50,65,78,88];
    positions.forEach((left, i) => {
        const p = doc.createElement('div');
        p.className = 'bg-particle';
        p.style.left = left + '%';
        p.style.animationDuration = (8 + i * 0.7) + 's';
        p.style.animationDelay = (i * 0.5) + 's';
        doc.body.appendChild(p);
    });

    const ecgWrap = doc.createElement('div');
    ecgWrap.id = 'ecg-container';
    ecgWrap.innerHTML = `
        <svg width="100%" height="90" viewBox="0 0 1500 90" preserveAspectRatio="none">
            <path id="ecg-line" d="
                M0,45 L100,45 L120,45 L135,10 L150,80 L165,45 L200,45
                L400,45 L420,45 L435,10 L450,80 L465,45 L500,45
                L700,45 L720,45 L735,10 L750,80 L765,45 L800,45
                L1000,45 L1020,45 L1035,10 L1050,80 L1065,45 L1100,45
                L1300,45 L1320,45 L1335,10 L1350,80 L1365,45 L1500,45
            "/>
        </svg>`;
    doc.body.appendChild(ecgWrap);
}
</script>
""", height=0, width=0)

# ===== GLOBAL STYLING =====
st.markdown("""
    <style>
    .title-text {
        font-size: 46px;
        font-weight: 800;
        color: #f8fafc;
        text-align: center;
        margin-bottom: 4px;
        text-shadow: 0 0 20px rgba(34,211,238,0.5);
    }
    .subtitle-text {
        font-size: 17px;
        color: #e2e8f0;
        text-align: center;
        margin-bottom: 30px;
    }

    /* Make Streamlit's own widget labels visible on the dark background */
    label, .stNumberInput label p, [data-testid="stWidgetLabel"] p {
        color: #f1f5f9 !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        text-shadow: 0 1px 3px rgba(0,0,0,0.5);
    }

    /* Group the input widgets visually using Streamlit's native container border */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 16px;
        padding: 10px 6px;
    }

    .stButton>button {
        background: linear-gradient(90deg, #2563eb, #22d3ee);
        color: white;
        font-weight: 700;
        border-radius: 10px;
        padding: 12px 24px;
        border: none;
        width: 100%;
        font-size: 17px;
        margin-top: 10px;
    }
    .stButton>button:hover {
        transform: scale(1.01);
    }

    .result-box-positive {
        background-color: rgba(254, 242, 242, 0.98);
        border-left: 6px solid #dc2626;
        padding: 22px 26px;
        border-radius: 12px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.3);
        margin-top: 20px;
    }
    .result-box-negative {
        background-color: rgba(240, 253, 244, 0.98);
        border-left: 6px solid #16a34a;
        padding: 22px 26px;
        border-radius: 12px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.3);
        margin-top: 20px;
    }
    .result-box-positive h3, .result-box-negative h3 {
        color: #1f2937;
        font-size: 24px;
        margin-bottom: 8px;
    }
    .result-box-positive p, .result-box-negative p {
        color: #374151;
        font-size: 16px;
        line-height: 1.5;
    }

    .tips-card {
        background-color: rgba(255,255,255,0.98);
        border-radius: 14px;
        padding: 22px 28px;
        margin-top: 18px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.3);
    }
    .tips-title {
        font-size: 22px;
        font-weight: 800;
        color: #1f2937;
        margin-bottom: 14px;
    }
    .tip-item {
        padding: 10px 0px;
        font-size: 16px;
        line-height: 1.5;
        color: #1f2937;
        border-bottom: 1px solid #f1f5f9;
    }
    .tip-item:last-child {
        border-bottom: none;
    }
    .tip-item b {
        color: #111827;
    }

    .disclaimer-box {
        background-color: rgba(255,255,255,0.9);
        border-radius: 10px;
        padding: 14px 20px;
        margin-top: 22px;
        font-size: 13.5px;
        color: #4b5563;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ===== HEADER =====
st.markdown('<p class="title-text">🩺 Diabetes Prediction System</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">Enter patient details below to check diabetes risk and get personalized guidance</p>', unsafe_allow_html=True)

# ===== INPUT SECTION (native Streamlit container with border — actually wraps the widgets) =====
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

# ===== PREDICTION + RECOMMENDATIONS =====
if predict_clicked:
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                             insulin, bmi, dpf, age]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.markdown("""
            <div class="result-box-positive">
                <h3>⚠️ Result: Likely Diabetic</h3>
                <p>Based on the entered values, the model predicts a higher risk of diabetes.
                This is not a medical diagnosis — please consult a doctor for proper testing and advice.</p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div class="tips-card">
                <div class="tips-title">💡 General Lifestyle Guidance</div>
                <div class="tip-item">🥗 <b>Diet:</b> Reduce refined sugar and processed carbs. Favor whole grains, vegetables, and lean protein.</div>
                <div class="tip-item">🏃 <b>Exercise:</b> Aim for at least 30 minutes of moderate activity (walking, cycling) most days of the week.</div>
                <div class="tip-item">⚖️ <b>Weight:</b> Gradual weight loss, if overweight, can meaningfully improve blood sugar control.</div>
                <div class="tip-item">💧 <b>Hydration:</b> Drink water instead of sugary beverages.</div>
                <div class="tip-item">🩸 <b>Monitoring:</b> Regularly check blood glucose levels as advised by a healthcare provider.</div>
                <div class="tip-item">🚭 <b>Habits:</b> Avoid smoking and limit alcohol intake.</div>
                <div class="tip-item">😴 <b>Sleep:</b> Poor sleep affects insulin sensitivity — aim for 7-8 hours nightly.</div>
                <div class="tip-item">👩‍⚕️ <b>Medical follow-up:</b> See a doctor for an HbA1c test and a personalized care plan.</div>
                <div class="tip-item">🧘 <b>Stress management:</b> Chronic stress raises blood sugar — try breathing exercises, meditation, or light yoga.</div>
                <div class="tip-item">👀 <b>Complication checks:</b> Get regular eye, kidney, and foot check-ups to catch complications early.</div>
            </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
            <div class="result-box-negative">
                <h3>✅ Result: Likely Not Diabetic</h3>
                <p>Based on the entered values, the model predicts a lower risk of diabetes. Keep up healthy habits to maintain this.</p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div class="tips-card">
                <div class="tips-title">💡 Tips to Stay Healthy</div>
                <div class="tip-item">🥗 <b>Diet:</b> Maintain a balanced diet rich in fiber, vegetables, and whole grains.</div>
                <div class="tip-item">🏃 <b>Exercise:</b> Stay active with regular physical activity — aim for 150 minutes a week.</div>
                <div class="tip-item">🩸 <b>Check-ups:</b> Get periodic health screenings, especially if diabetes runs in your family.</div>
                <div class="tip-item">😴 <b>Sleep & Stress:</b> Manage stress and prioritize good sleep — both affect long-term metabolic health.</div>
                <div class="tip-item">⚖️ <b>Weight:</b> Maintain a healthy BMI to lower long-term risk.</div>
                <div class="tip-item">💧 <b>Hydration:</b> Prioritize water over sugary drinks.</div>
            </div>
        """, unsafe_allow_html=True)

# ===== DISCLAIMER =====
st.markdown("""
    <div class="disclaimer-box">
        ⚠️ This tool provides a statistical prediction based on a machine learning model and is not a substitute for professional medical advice, diagnosis, or treatment.
    </div>
""", unsafe_allow_html=True)