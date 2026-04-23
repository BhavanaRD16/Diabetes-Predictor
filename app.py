# ============================================================
# DIABETES PREDICTION SYSTEM — Streamlit App
# Kongu Engineering College
# ============================================================

import streamlit as st
import numpy as np
import joblib
import os

# ── Page Configuration ─────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="centered"
)

# ── Custom CSS ─────────────────────────────────────────────
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }

    .header-box {
        background: linear-gradient(135deg, #1a6fc4, #0d4a8a);
        padding: 30px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 25px;
        color: white;
    }
    .header-box h1 { font-size: 2rem; margin: 0; }
    .header-box p  { font-size: 1rem; margin: 5px 0 0 0; opacity: 0.85; }

    .result-diabetic {
        background: linear-gradient(135deg, #c0392b, #e74c3c);
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        color: white;
        font-size: 1.6rem;
        font-weight: bold;
        margin-top: 20px;
    }
    .result-safe {
        background: linear-gradient(135deg, #1a8a4a, #27ae60);
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        color: white;
        font-size: 1.6rem;
        font-weight: bold;
        margin-top: 20px;
    }
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .section-title {
        color: #1a6fc4;
        font-size: 1.1rem;
        font-weight: bold;
        margin-bottom: 10px;
        border-bottom: 2px solid #1a6fc4;
        padding-bottom: 5px;
    }
    .disclaimer {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 12px 16px;
        border-radius: 6px;
        font-size: 0.85rem;
        color: #856404;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ── Load Models ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    model   = joblib.load('best_diabetes_model.pkl')
    scaler  = joblib.load('scaler.pkl')
    le_g    = joblib.load('le_gender.pkl')
    le_s    = joblib.load('le_smoking.pkl')
    return model, scaler, le_g, le_s

model, scaler, le_gender, le_smoking = load_models()

# ── Header ──────────────────────────────────────────────────
st.markdown("""
    <div class="header-box">
        <h1>🩺 Diabetes Prediction System</h1>
        <p>Enter patient details below to predict diabetes risk using Machine Learning</p>
    </div>
""", unsafe_allow_html=True)

# ── Input Form ──────────────────────────────────────────────
st.markdown('<div class="section-title">👤 Patient Information</div>',
            unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox(
        "Gender",
        options=["Female", "Male"],
        help="Patient's gender"
    )
    age = st.slider(
        "Age (years)",
        min_value=1, max_value=100, value=45,
        help="Patient's age in years"
    )
    hypertension = st.selectbox(
        "Hypertension",
        options=["No", "Yes"],
        help="Does the patient have hypertension?"
    )
    heart_disease = st.selectbox(
        "Heart Disease",
        options=["No", "Yes"],
        help="Does the patient have heart disease?"
    )

with col2:
    smoking_history = st.selectbox(
        "Smoking History",
        options=["never", "former", "current", "unknown"],
        help="Patient's smoking history"
    )
    bmi = st.slider(
        "BMI",
        min_value=10.0, max_value=60.0, value=27.0, step=0.1,
        help="Body Mass Index"
    )
    hba1c = st.slider(
        "HbA1c Level (%)",
        min_value=3.5, max_value=9.0, value=5.5, step=0.1,
        help="Glycated haemoglobin level — key diabetes marker"
    )
    blood_glucose = st.slider(
        "Blood Glucose Level (mg/dL)",
        min_value=80, max_value=300, value=100,
        help="Fasting blood glucose level"
    )

# ── Clinical Reference ──────────────────────────────────────
st.markdown('<br>', unsafe_allow_html=True)
with st.expander("📋 Clinical Reference Ranges"):
    ref_col1, ref_col2 = st.columns(2)
    with ref_col1:
        st.markdown("""
        **HbA1c Levels:**
        - 🟢 Normal : Below 5.7%
        - 🟡 Pre-diabetic : 5.7% – 6.4%
        - 🔴 Diabetic : 6.5% and above
        """)
    with ref_col2:
        st.markdown("""
        **Blood Glucose (Fasting):**
        - 🟢 Normal : Below 100 mg/dL
        - 🟡 Pre-diabetic : 100 – 125 mg/dL
        - 🔴 Diabetic : 126 mg/dL and above
        """)

# ── Predict Button ──────────────────────────────────────────
st.markdown('<br>', unsafe_allow_html=True)
predict_btn = st.button("🔍 Predict Diabetes Risk",
                         use_container_width=True,
                         type="primary")

if predict_btn:

    # Encode inputs
    gender_encoded  = le_gender.transform([gender])[0]
    smoking_encoded = le_smoking.transform([smoking_history])[0]
    hypertension_val = 1 if hypertension == "Yes" else 0
    heart_disease_val = 1 if heart_disease == "Yes" else 0

    # Build input array — same column order as training
    # Order: gender, age, hypertension, heart_disease,
    #        smoking_history, bmi, HbA1c_level, blood_glucose_level
    input_data = np.array([[
        gender_encoded,
        age,
        hypertension_val,
        heart_disease_val,
        smoking_encoded,
        bmi,
        hba1c,
        blood_glucose
    ]])

    # Scale using saved scaler
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction   = model.predict(input_scaled)[0]
    probability  = model.predict_proba(input_scaled)[0]
    prob_diabetic     = probability[1] * 100
    prob_non_diabetic = probability[0] * 100

    # ── Result Display ──────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Prediction Result")

    if prediction == 1:
        st.markdown(f"""
            <div class="result-diabetic">
                🔴 HIGH DIABETES RISK DETECTED<br>
                <span style="font-size:1rem; font-weight:normal;">
                Probability: {prob_diabetic:.1f}%
                </span>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="result-safe">
                🟢 LOW DIABETES RISK<br>
                <span style="font-size:1rem; font-weight:normal;">
                Probability of Diabetes: {prob_diabetic:.1f}%
                </span>
            </div>
        """, unsafe_allow_html=True)

    # ── Probability Breakdown ───────────────────────────────
    st.markdown('<br>', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)

    with m1:
        st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:0.85rem; color:#666;">Diabetes Risk</div>
                <div style="font-size:1.8rem; font-weight:bold;
                     color:{'#e74c3c' if prediction==1 else '#27ae60'};">
                     {prob_diabetic:.1f}%
                </div>
            </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:0.85rem; color:#666;">No Diabetes</div>
                <div style="font-size:1.8rem; font-weight:bold; color:#27ae60;">
                     {prob_non_diabetic:.1f}%
                </div>
            </div>
        """, unsafe_allow_html=True)

    with m3:
        risk_level = "High" if prob_diabetic >= 70 else \
                     "Moderate" if prob_diabetic >= 40 else "Low"
        risk_color = "#e74c3c" if risk_level == "High" else \
                     "#f39c12" if risk_level == "Moderate" else "#27ae60"
        st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:0.85rem; color:#666;">Risk Level</div>
                <div style="font-size:1.8rem; font-weight:bold;
                     color:{risk_color};">
                     {risk_level}
                </div>
            </div>
        """, unsafe_allow_html=True)

    # ── Input Summary ───────────────────────────────────────
    st.markdown('<br>', unsafe_allow_html=True)
    with st.expander("📝 Patient Input Summary"):
        summary_col1, summary_col2 = st.columns(2)
        with summary_col1:
            st.write(f"**Gender:** {gender}")
            st.write(f"**Age:** {age} years")
            st.write(f"**BMI:** {bmi}")
            st.write(f"**Hypertension:** {hypertension}")
        with summary_col2:
            st.write(f"**Heart Disease:** {heart_disease}")
            st.write(f"**Smoking History:** {smoking_history}")
            st.write(f"**HbA1c Level:** {hba1c}%")
            st.write(f"**Blood Glucose:** {blood_glucose} mg/dL")

    # ── Disclaimer ──────────────────────────────────────────
    st.markdown("""
        <div class="disclaimer">
            ⚠️ <strong>Medical Disclaimer:</strong> This tool is for educational
            and research purposes only. It is not a substitute for professional
            medical diagnosis. Always consult a qualified healthcare provider
            for medical advice.
        </div>
    """, unsafe_allow_html=True)

# ── Footer ───────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
    <div style="text-align:center; color:#888; font-size:0.8rem;">
        Diabetes Prediction System &nbsp;|&nbsp;
        Kongu Engineering College &nbsp;|&nbsp;
        XGBoost Model &nbsp;|&nbsp; ROC-AUC: 0.9757
    </div>
""", unsafe_allow_html=True)