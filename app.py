import streamlit as st
import numpy as np
import joblib
import google.generativeai as genai

# -------------------------
# 1. CONFIGURE GEMINI API
# -------------------------
# 💡 Replace with your actual Gemini API key (keep it secret in production!)
genai.configure(api_key="AIzaSyBnbf8kYdNXHwfvyeBgARgyh7KrUMK-U5w")

# Load Gemini model
g_model = genai.GenerativeModel("gemini-pro")

# -------------------------
# 2. LOAD MODEL & ENCODER
# -------------------------
model = joblib.load("model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# -------------------------
# 3. STREAMLIT UI
# -------------------------
st.set_page_config(page_title="PneumoBridge AI", layout="centered")
st.title("🧠 PneumoBridge AI")
st.markdown("**Early Pneumonia Detection from Vital Signs**")
st.divider()

# -------------------------
# 4. USER INPUT
# -------------------------
st.subheader("📋 Enter Patient Vitals")

col1, col2 = st.columns(2)
with col1:
    temp = st.slider("Temperature (°F)", 95.0, 105.0, 98.6)
    spo2 = st.slider("Oxygen Saturation (%)", 85, 100, 95)

with col2:
    hr = st.slider("Heart Rate (bpm)", 50, 150, 85)
    rr = st.slider("Respiratory Rate (breaths/min)", 10, 40, 20)

# -------------------------
# 5. PREDICT AND EXPLAIN
# -------------------------
if st.button("🔍 Predict Pneumonia Risk"):
    input_data = np.array([[temp, hr, spo2, rr]])
    pred_encoded = model.predict(input_data)[0]
    risk_level = label_encoder.inverse_transform([pred_encoded])[0]

    st.success(f"✅ Predicted Risk Level: **{risk_level}**")

    # Gemini prompt
    prompt = f"""
    A patient has the following vital signs:
    - Temperature: {temp} °F
    - Heart Rate: {hr} bpm
    - Oxygen Saturation: {spo2} %
    - Respiratory Rate: {rr} breaths per minute

    Based on these, the AI model predicts the pneumonia risk level as: {risk_level}.

    Explain what this risk level means in simple terms and what steps healthcare workers should take next.
    """

    try:
        response = g_model.generate_content(prompt)
        st.subheader("🤖 Gemini AI Explanation")
        st.write(response.text)
    except Exception as e:
        st.error(f"Gemini API Error: {e}")
