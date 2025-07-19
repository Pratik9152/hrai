# Paste this entire code into your app.py

import streamlit as st
import os
import tempfile
import fitz  # PyMuPDF
import pandas as pd
import requests
from io import BytesIO
import re
import pytesseract
from PIL import Image
import datetime

# Optional: for Windows
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

st.set_page_config(page_title="AI HR Assistant", layout="wide")

# Styling
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
    .best {
        background-color: #d4edda;
        padding: 8px;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 AI HR Assistant - Resume Analyzer")

job_title = st.text_input("🎯 Role Hiring For")
job_description = st.text_area("📌 Job Description / Requirements", height=150)
custom_threshold = st.slider("📈 Fit Score Threshold", 0, 100, 50)
uploaded_files = st.file_uploader("📁 Upload CVs (PDF)", type=["pdf"], accept_multiple_files=True)
pasted_data = st.text_area("📝 Paste CV text(s) here (separate multiple with ---)", height=200)
process = st.button("🚀 Analyze Candidates")

api_key = "sk-..."  # ← replace with your actual OpenRouter key

skill_map = {
    "Data Scientist": ["Python", "Machine Learning", "Pandas", "Statistics"],
    "Frontend Developer": ["HTML", "CSS", "JavaScript", "React"],
    "HR Manager": ["Recruitment", "Onboarding", "Compliance", "Policies"],
}

def extract_text_from_pdf(path):
    try:
        doc = fitz.open(path)
        text = ""
        for page in doc:
            txt = page.get_text().strip()
            if not txt:
                img = Image.frombytes("RGB", [page.get_pixmap(dpi=300).width, page.get_pixmap(dpi=300).height], page.get_pixmap().samples)
                txt = pytesseract.image_to_string(img)
            text += txt + "\n"
        return text.strip()
    except Exception as e:
        return f"[ERROR reading PDF] {e}"

def extract_value(text, label, fallback="N/A", is_number=False):
    pattern = re.escape(label) + r"[\s:]*([^\n]+)"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        val = match.group(1).strip()
        if is_number:
            digits = re.findall(r"\d+", val)
            return int(digits[0]) if digits else fallback
        return val
    return fallback

def generate_prompt(cv, title, desc):
    skills = skill_map.get(title, [])
    skill_str = ", ".join(skills) if skills else "Infer from JD"
    return f"""
You are an expert AI HR evaluator.

We are hiring for: {title}
Job Description: {desc}
Expected Skills: {skill_str}

Candidate CV:
{cv}

Give me:
1. Score out of 100
2. Skill Match %
3. Years of Experience
4. Top Strengths
5. Red Flags
6. Final Verdict: Strong Fit / Moderate Fit / Not Recommended
7. One-line hire recommendation
8. Summary of education, certifications, tools, etc.
"""

def call_openrouter(prompt):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {
        "model": "openchat/openchat-3.5-0106",  # Use gpt-4 or mistral if needed
        "messages": [
            {"role": "system", "content": "You are a skilled HR AI analyzing candidate resumes."},
            {"role": "user", "content": prompt}
        ]
    }
    try:
        r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[API Error] {e}"

if process and job_description and (uploaded_files or pasted_data):
    with st.spinner("Analyzing..."):
        candidates = []

        if uploaded_files:
            with tempfile.TemporaryDirectory() as temp:
                for f in uploaded_files:
                    path = os.path.join(temp, f.name)
                    with open(path, "wb") as out:
                        out.write(f.read())
                    text = extract_text_from_pdf(path)
                    candidates.append((f.name, text))

        if pasted_data:
            for i, chunk in enumerate(pasted_data.split("---")):
                candidates.append((f"Pasted_{i+1}", chunk.strip()))

        results = []
        for name, cv_text in candidates:
            prompt = generate_prompt(cv_text, job_title, job_description)
            reply = call_openrouter(prompt)

            score = extract_value(reply, "Score", is_number=True)
            match = extract_value(reply, "Skill Match", is_number=True)
            experience = extract_value(reply, "Years of Experience")
            strengths = extract_value(reply, "Top Strengths")
            red_flags = extract_value(reply, "Red Flags")
            verdict = extract_value(reply, "Final Verdict")
            hire_rec = extract_value(reply, "hire recommendation")
            summary = extract_value(reply, "Summary")

            results.append({
                "Candidate": name,
                "Score": score,
                "Match %": match,
                "Experience": experience,
                "Verdict": verdict,
                "Hire": hire_rec,
                "Strengths": strengths,
                "Red Flags": red_flags,
                "Summary": summary,
                "Full Reply": reply
            })

        df = pd.DataFrame(results)
        filtered = df[df["Score"] >= custom_threshold]
        best = df[df["Score"] == df["Score"].max()]

        st.success(f"✅ {len(filtered)} candidate(s) passed the threshold of {custom_threshold}")

        for _, row in df.iterrows():
            top = "best" if row["Candidate"] in best["Candidate"].values else ""
            with st.expander(f"🧾 {row['Candidate']} — Score: {row['Score']}"):
                st.markdown(f"<div class='{top}'>", unsafe_allow_html=True)
                st.markdown(f"**✔ AI Recommendation:** {row['Hire']}")
                st.markdown(f"**✅ Top Strengths:** {row['Strengths']}")
                st.markdown(f"**🚫 Red Flags:** {row['Red Flags']}")
                st.markdown(f"**📌 Summary:** {row['Summary']}")
                st.markdown(f"**📈 Skill Match:** {row['Match %']}% | **Experience:** {row['Experience']}")
                st.markdown("</div>", unsafe_allow_html=True)
                with st.expander("📄 Full AI Response"):
                    st.code(row["Full Reply"])

        csv = df.to_csv(index=False).encode("utf-8")
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        st.download_button("📥 Download Candidate Report", csv, file_name=f"HR_Report_{now}.csv", mime="text/csv")

else:
    if process:
        st.error("Please enter job description and upload or paste at least one CV.")
