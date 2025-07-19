import streamlit as st
import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import requests
import re

# Streamlit UI config
st.set_page_config(page_title="AI HR Assistant", layout="wide")

st.markdown("""
    <style>
    body {
        background: linear-gradient(-45deg, #f5f7fa, #c3cfe2, #dfe9f3, #e2ebf0);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
    }
    @keyframes gradientBG {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    .stApp {
        background-color: #ffffffcc !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 World-Class AI HR Assistant")

# Job input
job_title = st.text_input("📌 Hiring for (Job Title)", value="Data Scientist")
job_description = st.text_area("📄 Job Description", value="Looking for a data scientist with experience in Python, ML, and data analysis.")
uploaded_files = st.file_uploader("📁 Upload CVs (PDF, scanned PDFs supported)", type=["pdf"], accept_multiple_files=True)
api_key = st.secrets.get("OPENROUTER_API_KEY", "")

# Skill mapping (optional preset)
skill_map = {
    "Data Scientist": ["Python", "Machine Learning", "Data Analysis", "Statistics"],
    "Frontend Developer": ["HTML", "CSS", "JavaScript", "React"],
    "HR Manager": ["Recruitment", "Onboarding", "HR Policies"]
}

# Extract text from PDFs (OCR fallback)
def extract_text(file):
    try:
        doc = fitz.open(stream=file.read(), filetype="pdf")
        full_text = ""
        for page in doc:
            text = page.get_text()
            if not text.strip():
                # Fallback: OCR image from scanned page
                pix = page.get_pixmap(dpi=300)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text = pytesseract.image_to_string(img)
            full_text += text + "\n"
        return full_text
    except Exception as e:
        return f"Error: {str(e)}"

# Send prompt to OpenRouter with GPT-4 Turbo or Mistral
def call_openrouter_api(prompt):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "openai/gpt-4-turbo",  # or "mistralai/mistral-large"
        "messages": [
            {"role": "system", "content": "You are a world-class AI HR assistant. Give structured, helpful candidate evaluation."},
            {"role": "user", "content": prompt}
        ]
    }
    try:
        res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
        return res.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error: {str(e)}"

# Create the analysis prompt
def generate_prompt(cv_text, job_title, job_description):
    skills = skill_map.get(job_title, [])
    skill_text = ", ".join(skills) if skills else "Let AI infer"
    return f"""
We are hiring for: {job_title}
Job Description: {job_description}
Expected Skills: {skill_text}

Candidate Resume:
{cv_text}

Instructions:
Evaluate this candidate based on the role.
Return the following in clear format:
🟢 AI Recommendation (Should be hired or not)
📈 Fit Score (0–100)
✅ Skill Match %
⏳ Years of Relevant Experience
💪 Top 3 Strengths
⚠️ Red Flags or Concerns
🧠 Why Selected or Not Selected
"""

# Button to run analysis
if st.button("🚀 Analyze CVs"):
    if not api_key:
        st.error("Missing OpenRouter API key. Add it to Streamlit secrets.")
    elif not uploaded_files:
        st.warning("Upload at least one CV.")
    else:
        for file in uploaded_files:
            with st.spinner(f"Analyzing {file.name}..."):
                text = extract_text(file)
                prompt = generate_prompt(text, job_title, job_description)
                result = call_openrouter_api(prompt)
                st.subheader(f"📄 {file.name}")
                st.markdown(result)
