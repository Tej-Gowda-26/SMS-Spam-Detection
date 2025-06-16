import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from src.predict import predict_sms

st.set_page_config(page_title="SMS Spam Detection", page_icon="📱")
st.title("📱 SMS Spam Detection")

# 📘 Model Info Button
with st.expander("📘 Model Used"):
    st.markdown("A Logistic Regression model is used to learn patterns from SMS messages and classify them as spam or ham.")
    
    try:
        with open("models/evaluation.txt", "r") as f:
            report = f.read()
        st.code(report, language="text")
    except FileNotFoundError:
        st.warning("Evaluation report not found. Please run `evaluate.py` to generate it.")

# 📥 Message Input and Prediction
st.markdown("### ℹ️ Enter an SMS message to classify:")

message = st.text_area("✉️ Your message:")

if st.button("🔍 Predict"):
    if not message.strip():
        st.warning("⚠️ Please enter a message before predicting.")
    else:
        label, spam_prob, ham_prob = predict_sms(message)
        
        if label == "Spam":
            st.error("🚫 This message is likely **Spam**!")
        else:
            st.success("✅ This message is classified as **Ham** (Not Spam).")

        st.markdown("**🧠 Prediction Confidence:**")
        st.info(f"Spam: {spam_prob*100:.2f}% &nbsp;&nbsp;&nbsp; Ham: {ham_prob*100:.2f}%")
