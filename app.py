import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("model/diabetes_model.pkl")

# Page config
st.set_page_config(page_title="Diabetes Prediction System", page_icon="🩺", layout="centered")

# Custom CSS styling
st.markdown("""
    <style>
    .main {
        background-color: #f7f9fc;
    }
    .title-text {
        font-size: 42px;
        font-weight: 800;
        color: #1f2937;
        text-align: center;
        margin-bottom: 0px;
    }
    .subtitle-text {
        font-size: 16px;
        color: #6b7280;
        text-align: center;
        margin-bottom: 30px;
    }
    .card {
        background-color: white;
        padding: 25px 30px;
        border-radius: 16px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        margin-bottom: 20px;
    }
    .stButton>button {
        background-color: #2563eb;
        color: white;
        font-weight: 600;
        border-radius: 10px;
        padding: 10px 24px;
        border: none;
        width: 100%;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
    }
    .result-box-positive {
        background-color: #fef2f2;
        border-left: 6px solid #dc2626;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .result-box-negative {
        background-color: #f0fdf4;
        border-left: 6px solid #16a34a;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .tip-item {
        padding: 6px 0px;
        font-size: 15px;
        color: #374151;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="title-text">🩺 Diabetes Prediction System</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">Enter patient details below to check diabetes risk and get personalized guidance</p>', unsafe_allow_html=True)

# Input card
st.markdown('<div class="card">', unsafe_allow_html=True)

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
st.markdown('</div>', unsafe_allow_html=True)

# Prediction + recommendations
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

        st.markdown("### 💡 General Lifestyle Guidance")
        st.markdown("""
            <div class="tip-item">🥗 <b>Diet:</b> Reduce refined sugar and processed carbs. Favor whole grains, vegetables, and lean protein.</div>
            <div class="tip-item">🏃 <b>Exercise:</b> Aim for at least 30 minutes of moderate activity (walking, cycling) most days of the week.</div>
            <div class="tip-item">⚖️ <b>Weight:</b> Gradual weight loss, if overweight, can meaningfully improve blood sugar control.</div>
            <div class="tip-item">💧 <b>Hydration:</b> Drink water instead of sugary beverages.</div>
            <div class="tip-item">🩸 <b>Monitoring:</b> Regularly check blood glucose levels as advised by a healthcare provider.</div>
            <div class="tip-item">🚭 <b>Habits:</b> Avoid smoking and limit alcohol intake.</div>
            <div class="tip-item">😴 <b>Sleep:</b> Poor sleep affects insulin sensitivity — aim for 7-8 hours nightly.</div>
            <div class="tip-item">👩‍⚕️ <b>Medical follow-up:</b> See a doctor for an HbA1c test and a personalized care plan.</div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
            <div class="result-box-negative">
                <h3>✅ Result: Likely Not Diabetic</h3>
                <p>Based on the entered values, the model predicts a lower risk of diabetes. 
                Keep up healthy habits to maintain this.</p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("### 💡 Tips to Stay Healthy")
        st.markdown("""
            <div class="tip-item">🥗 <b>Diet:</b> Maintain a balanced diet rich in fiber, vegetables, and whole grains.</div>
            <div class="tip-item">🏃 <b>Exercise:</b> Stay active with regular physical activity.</div>
            <div class="tip-item">🩸 <b>Check-ups:</b> Get periodic health screenings, especially if diabetes runs in your family.</div>
            <div class="tip-item">😴 <b>Sleep & Stress:</b> Manage stress and prioritize good sleep — both affect long-term metabolic health.</div>
        """, unsafe_allow_html=True)

# Disclaimer
st.markdown("---")
st.caption("⚠️ This tool provides a statistical prediction based on a machine learning model and is not a substitute for professional medical advice, diagnosis, or treatment.")