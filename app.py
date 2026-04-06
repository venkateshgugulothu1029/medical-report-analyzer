import streamlit as st
from dotenv import load_dotenv
from utils import extract_text
from langchain.chat_models import ChatOpenAI

load_dotenv()

st.set_page_config(page_title="Medical AI Analyzer", layout="wide")

st.title("🩺 Medical Report Analyzer")
st.write("Upload a medical report and get AI-powered insights.")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    text = extract_text(uploaded_file)

    llm = ChatOpenAI(temperature=0)

    prompt = f"""
    You are a medical assistant AI.

    Analyze the following medical report and provide:

    1. Summary
    2. Key Observations
    3. Potential Health Risks (if any)

    Keep answers simple and clear.

    Report:
    {text}
    """

    if st.button("Analyze Report"):
        with st.spinner("Analyzing report..."):
            response = llm.predict(prompt)

        st.subheader("📊 Analysis Result")
        st.write(response)

    if "high sugar" in text.lower():
        st.warning("⚠️ Possible Diabetes Risk Detected")
