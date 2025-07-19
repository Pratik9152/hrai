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
import datetime
import numpy as np

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
    .stApp {
        background-color: #ffffffcc !important;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 16px;
    }
    .stTextInput>div>input, .stTextArea>div>textarea {
        background-color: #f7fafd;
        border: 1px solid #ccc;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 All-in-One AI HR Assistant")

# Input fields
job_title = st.text_input("🎯 Hiring For (Job Title / Role)")
job_description = st.text_area("📌 Job Description or Role Requirements", height=200)
custom_threshold = st.slider("📈 Minimum Fit Score Required", 0, 100, 50)
uploaded_files = st.file_uploader("📁 Upload candidate CVs (PDF, DOCX, TXT, scanned PDF)", type=["pdf", "docx", "txt"], accept_multiple_files=True)
pasted_candidates = st.text_area("📝 Paste candidate data (separate candidates with ---)", height=300)
process_button = st.button("🚀 Analyze Candidates")

api_key = st.secrets.get("OPENROUTER_API_KEY", "")

skill_map = {
    "Data Scientist": ["Python", "Machine Learning", "Statistics", "Data Analysis"],
    "Frontend Developer": ["HTML", "CSS", "JavaScript", "React"],
    "HR Manager": ["Recruitment", "Onboarding", "HR Policies", "Employee Relations"],
}

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
    return int(match.group()) if match else np.nan

def extract_between(text, start_key, end_key=None):
    try:
        pattern = re.escape(start_key) + r"(.*?)(?=" + re.escape(end_key) + r"|$)" if end_key else re.escape(start_key) + r"(.*)"
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else "N/A"
    except Exception:
        return "N/A"

def call_openrouter_api(prompt):
    if not api_key:
        return "No API key configured."
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "mistralai/mistral-large",
        "messages": [
            {"role": "system", "content": "You are a world-class HR AI assistant. Provide structured insights and clear ranking for best-fit candidates."},
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
    role_skills = skill_map.get(job_title, [])
    skills_required = ", ".join(role_skills) if role_skills else "[Let AI infer skills]"
    return f"""
We are hiring for the role: {job_title}

Job Description:
{job_description}

Key Skills Expected:
{skills_required}

Resume:
{cv_text}

Evaluate the following:
1. Score out of 100 for fit.
2. Skill Match Percentage.
3. Experience Years.
4. Top 3 Strengths.
5. Red Flags or concerns.
6. Justify role fit.
7. If not recommended, explain why.
8. Final Verdict: Strong Fit / Moderate Fit / Not Recommended.
9. Provide a one-line recommendation: Should this candidate be hired or not with a reason.
10. Summarize key insights and data extracted from resume (e.g., education, certifications, locations, tools used, etc.)

Respond in structured markdown format.
"""

# Processing Logic
if process_button and job_description and (uploaded_files or pasted_candidates):
    with st.spinner("🤖 AI analyzing candidates. Please wait..."):
        candidates = []

        if uploaded_files:
            with tempfile.TemporaryDirectory() as tmpdir:
                for file in uploaded_files:
                    temp_path = os.path.join(tmpdir, file.name)
                    with open(temp_path, "wb") as f:
                        f.write(file.read())
                    if file.name.lower().endswith(".pdf"):
                        text = extract_pdf_text(temp_path)
                        candidates.append((file.name, text))

        if pasted_candidates:
            for i, chunk in enumerate(pasted_candidates.split("---")):
                candidates.append((f"Pasted_Candidate_{i+1}.txt", chunk.strip()))

        results = []
        for name, cv_text in candidates:
            prompt = generate_prompt(cv_text, job_title, job_description)
            ai_response = call_openrouter_api(prompt)
            score = extract_number(extract_between(ai_response, "Score:"))
            rec = extract_between(ai_response, "Final Verdict:", "\n")
            match_pct = extract_number(extract_between(ai_response, "Skill Match Percentage:"))
            exp_years = extract_between(ai_response, "Experience Years:", "\n")
            strengths = extract_between(ai_response, "Top 3 Strengths:", "Red Flags")
            red_flags = extract_between(ai_response, "Red Flags", "Justify")
            justification = extract_between(ai_response, "Justify role fit:", "If not recommended")
            why_not = extract_between(ai_response, "If not recommended, explain why:", "Final Verdict")
            hiring_line = extract_between(ai_response, "Provide a one-line recommendation:")
            summary_data = extract_between(ai_response, "Summarize key insights and data extracted from resume")

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
                "AI Recommendation": hiring_line,
                "Resume Summary": summary_data,
                "Full AI Analysis": ai_response
            })

        if results:
            df = pd.DataFrame(results)

            # ✅ Fix: ensure Score column is numeric
            df["Score"] = pd.to_numeric(df["Score"], errors="coerce")

            filtered_df = df[df["Score"] >= custom_threshold]
            best_df = df[df["Score"] == df["Score"].max()]

            st.success("✅ AI Analysis Complete")
            st.subheader("📊 Candidate Insights Dashboard")
            st.markdown(f"**🧑‍💼 {len(filtered_df)} candidates meet the criteria.**")

            st.plotly_chart(px.bar(filtered_df, x="Candidate", y="Score", color="Recommendation", text="Score"), use_container_width=True)
            st.plotly_chart(px.pie(filtered_df, names="Recommendation"), use_container_width=True)
            st.plotly_chart(px.bar(filtered_df, x="Candidate", y="Skill Match %", color="Skill Match %"), use_container_width=True)

            for _, row in filtered_df.iterrows():
                with st.expander(f"📌 {row['Candidate']} — Score: {row['Score']} — {row['Recommendation']}"):
                    st.markdown(f"### 🟢 AI Recommendation: {row['AI Recommendation']}")
                    st.markdown(f"**Top Strengths**:\n{row['Top Strengths']}")
                    st.markdown(f"**Red Flags**:\n{row['Red Flags']}")
                    st.markdown(f"**Fit Justification**:\n{row['Fit Justification']}")
                    st.markdown(f"**Why Not Selected**: {row['Why Not Selected']}")
                    st.markdown(f"**Skill Match %**: {row['Skill Match %']} | **Experience**: {row['Experience (Years)']}")
                    st.markdown(f"**📌 Resume Summary**:\n{row['Resume Summary']}")
                    with st.expander("📄 Full AI Response"):
                        st.code(row["Full AI Analysis"], language="markdown")

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
            st.download_button("📥 Download All Filtered Candidates", data=filtered_df.to_csv(index=False).encode("utf-8"), file_name=f"Filtered_Candidates_{timestamp}.csv", mime="text/csv")
            st.download_button("🌟 Download Best Candidate(s)", data=best_df.to_csv(index=False).encode("utf-8"), file_name=f"Best_Candidate_{timestamp}.csv", mime="text/csv")
else:
    if process_button:
        st.error("⚠️ Please fill in the job title, description, and candidate data.")
