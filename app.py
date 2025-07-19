import streamlit as st
import os
import zipfile
import tempfile
import fitz  # PyMuPDF
import pandas as pd
import requests
from io import BytesIO
import plotly.express as px
import re
import pytesseract
from PIL import Image

# Page Config
st.set_page_config(page_title="HR AI - Candidate Analyzer", layout="wide")
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
    </style>
""", unsafe_allow_html=True)

st.title("🧠 All-in-One AI HR Assistant")

# Input fields
job_title = st.text_input("🎯 Hiring For (Job Title / Role)")
job_description = st.text_area("📌 Job Description or Role Requirements", height=200)
uploaded_zip = st.file_uploader("📁 Upload ZIP of candidate CVs (PDF, DOCX, TXT, scanned PDF)", type=["zip"])
pasted_candidates = st.text_area("📝 Paste candidate data (separate candidates with ---)", height=300)
process_button = st.button("🚀 Analyze Candidates")

# Load OpenRouter API key from secrets
api_key = st.secrets.get("OPENROUTER_API_KEY", "")

# Utility Functions
def extract_pdf_text(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            txt = page.get_text()
            if not txt.strip():
                pix = page.get_pixmap(dpi=300)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                txt = pytesseract.image_to_string(img)
            text += txt + "\n"
        return text
    except Exception as e:
        return f"Error reading {os.path.basename(pdf_path)}: {str(e)}"

def extract_number(text):
    match = re.search(r"\d+", text)
    return int(match.group()) if match else 0

def call_openrouter_api(prompt):
    if not api_key:
        return "No API key configured."
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "openchat/openchat-3.5-0106",
        "messages": [
            {"role": "system", "content": "You are a world-class HR AI assistant. Provide structured insights and explain rejections clearly."},
            {"role": "user", "content": prompt}
        ]
    }
    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
        result = response.json()
        return result.get("choices", [{}])[0].get("message", {}).get("content", "No response")
    except Exception as e:
        return f"API Error: {str(e)}"

def generate_prompt(cv_text, job_title, job_description):
    return f"""
You are a world-class AI HR assistant.

We are hiring for the role: {job_title}

Job Description:
{job_description}

Candidate Resume:
{cv_text}

Provide a structured report:
---
Score: XX/100
Recommendation: Strong Fit / Moderate Fit / Not Recommended

Top 3 Strengths:
1. ...
2. ...
3. ...

Red Flags (if any):
- ...

Role Fit Justification:
- ...

Skill Match Percentage: XX%
Years of Experience: X years
Why Not Selected (if applicable): ...
---
"""

def extract_between(text, start_key, end_key=None):
    try:
        start = text.lower().index(start_key.lower()) + len(start_key)
        end = text.lower().index(end_key.lower(), start) if end_key else None
        return text[start:end].strip()
    except:
        return "N/A"

# Processing Logic
if process_button and job_description and (uploaded_zip or pasted_candidates):
    with st.spinner("🤖 AI analyzing candidates. Please wait..."):
        candidates = []

        if uploaded_zip:
            with tempfile.TemporaryDirectory() as tmpdir:
                zip_path = os.path.join(tmpdir, "uploaded.zip")
                with open(zip_path, "wb") as f:
                    f.write(uploaded_zip.read())
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(tmpdir)
                for file in os.listdir(tmpdir):
                    if file.lower().endswith(".pdf"):
                        path = os.path.join(tmpdir, file)
                        text = extract_pdf_text(path)
                        candidates.append((file, text))

        if pasted_candidates:
            for i, chunk in enumerate(pasted_candidates.split("---")):
                candidates.append((f"Pasted_Candidate_{i+1}.txt", chunk.strip()))

        results = []
        for name, cv_text in candidates:
            prompt = generate_prompt(cv_text, job_title, job_description)
            ai_response = call_openrouter_api(prompt)
            score = extract_number(extract_between(ai_response, "Score:", "\n"))
            rec = extract_between(ai_response, "Recommendation:", "\n")
            match_pct = extract_number(extract_between(ai_response, "Skill Match Percentage:", "\n"))
            exp_years = extract_between(ai_response, "Years of Experience:", "\n")
            strengths = extract_between(ai_response, "Top 3 Strengths:", "Red Flags")
            red_flags = extract_between(ai_response, "Red Flags", "Role Fit Justification")
            justification = extract_between(ai_response, "Role Fit Justification:", "Skill Match")
            why_not = extract_between(ai_response, "Why Not Selected", None)

            results.append({
                "Candidate": name,
                "Score": score,
                "Recommendation": rec,
                "Skill Match %": match_pct,
                "Experience (Years)": exp_years,
                "Top Strengths": strengths,
                "Red Flags": red_flags,
                "Fit Justification": justification,
                "Why Not Selected": why_not,
                "Full AI Analysis": ai_response
            })

        if results:
            df = pd.DataFrame(results)
            st.success("✅ AI Analysis Complete")
            st.subheader("📊 Candidate Insights Dashboard")

            min_score = st.slider("🔎 Filter candidates by minimum score", 0, 100, 50)
            filtered_df = df[df["Score"] >= min_score]

            st.markdown(f"**🧑‍💼 {len(filtered_df)} candidates meet the criteria.**")
            st.plotly_chart(px.bar(filtered_df, x="Candidate", y="Score", color="Recommendation", text="Score"), use_container_width=True)
            st.plotly_chart(px.pie(filtered_df, names="Recommendation"), use_container_width=True)
            st.plotly_chart(px.bar(filtered_df, x="Candidate", y="Skill Match %", color="Skill Match %"), use_container_width=True)

            for _, row in filtered_df.iterrows():
                with st.expander(f"📌 {row['Candidate']} — Score: {row['Score']} — {row['Recommendation']}"):
                    st.markdown(f"**Top Strengths**:\n{row['Top Strengths']}")
                    st.markdown(f"**Red Flags**:\n{row['Red Flags']}")
                    st.markdown(f"**Fit Justification**:\n{row['Fit Justification']}")
                    st.markdown(f"**Why Not Selected**: {row['Why Not Selected']}")
                    st.markdown(f"**Skill Match %**: {row['Skill Match %']} | **Experience**: {row['Experience (Years)']}")
                    with st.expander("📄 Full AI Response"):
                        st.code(row["Full AI Analysis"], language="markdown")

            st.download_button("📥 Download CSV Report", data=filtered_df.to_csv(index=False).encode("utf-8"), file_name="AI_Hiring_Report.csv", mime="text/csv")
else:
    if process_button:
        st.error("⚠️ Please fill in the job title, description, and candidate data.")
