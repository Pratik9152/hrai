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

st.title("🧠 AI-Powered HR Assistant")

# Tabs for input
tabs = st.tabs(["🔍 Job Description", "📁 Upload CVs", "📝 Paste Candidate Info"])

# Load OpenRouter API key from secrets
api_key = st.secrets["OPENROUTER_API_KEY"]

# Tab 1: Job Description
with tabs[0]:
    job_description = st.text_area("📌 Enter Job Description / Role Requirements", height=200)

# Tab 2: Upload ZIP
with tabs[1]:
    uploaded_zip = st.file_uploader("📁 Upload ZIP of candidate CVs (PDF, DOCX, TXT, scanned PDF)", type=["zip"])

# Tab 3: Paste Candidate Info
with tabs[2]:
    pasted_candidates = st.text_area("📝 Paste multiple candidate resumes below (separate with ---)", height=300)

# Process button
process_button = st.button("🚀 Analyze Candidates")

# Utility functions
def extract_pdf_text(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            txt = page.get_text()
            if not txt.strip():
                # OCR fallback
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

def call_openrouter_api(prompt, api_key):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "openchat/openchat-3.5-0106",
        "messages": [
            {"role": "system", "content": "You are a world-class HR agent. Provide structured and detailed insights."},
            {"role": "user", "content": prompt}
        ]
    }
    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
        result = response.json()
        if "choices" in result:
            return result["choices"][0]["message"]["content"]
        else:
            return "API Error: No valid response"
    except Exception as e:
        return f"API Error: {str(e)}"

def generate_prompt(cv_text, job_description):
    return f"""
You are a world-class AI HR assistant.

Given the following Job Description:

{job_description}

And this Candidate Resume:

{cv_text}

Please analyze and return a structured report:
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
- ...
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

# Process logic
if process_button and job_description and api_key:
    with st.spinner("🤖 Processing candidates with AI... Please wait."):
        candidates = []

        # If ZIP uploaded
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

        # If pasted
        if pasted_candidates:
            for i, chunk in enumerate(pasted_candidates.split("---")):
                candidates.append((f"Pasted_Candidate_{i+1}.txt", chunk.strip()))

        results = []
        for name, cv_text in candidates:
            prompt = generate_prompt(cv_text, job_description)
            ai_response = call_openrouter_api(prompt, api_key)
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

            # Visuals
            score_chart = px.bar(filtered_df, x="Candidate", y="Score", color="Recommendation", text="Score")
            st.plotly_chart(score_chart, use_container_width=True)

            rec_pie = px.pie(filtered_df, names="Recommendation")
            st.plotly_chart(rec_pie, use_container_width=True)

            skill_chart = px.bar(filtered_df, x="Candidate", y="Skill Match %", color="Skill Match %")
            st.plotly_chart(skill_chart, use_container_width=True)

            for _, row in filtered_df.iterrows():
                with st.expander(f"📌 {row['Candidate']} — Score: {row['Score']} — {row['Recommendation']}"):
                    st.markdown(f"**Top Strengths**:\n{row['Top Strengths']}")
                    st.markdown(f"**Red Flags**:\n{row['Red Flags']}")
                    st.markdown(f"**Fit Justification**:\n{row['Fit Justification']}")
                    st.markdown(f"**Why Not Selected**: {row['Why Not Selected']}")
                    st.markdown(f"**Skill Match %**: {row['Skill Match %']} | **Experience**: {row['Experience (Years)']}")
                    with st.expander("📄 Full AI Response"):
                        st.code(row["Full AI Analysis"], language="markdown")

            st.markdown("## 🏆 Top Recommended Candidates")
            top_candidates = df[
                (df["Recommendation"].str.lower().str.contains("strong")) &
                (df["Score"] >= 70)
            ].sort_values(by="Score", ascending=False)

            if not top_candidates.empty:
                st.dataframe(top_candidates[["Candidate", "Score", "Skill Match %", "Experience (Years)", "Recommendation"]])
            else:
                st.info("No strong recommendations found yet. Try adjusting filters or reviewing moderate fits.")

            csv_data = filtered_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download CSV Report", data=csv_data, file_name="AI_Hiring_Insights_Report.csv", mime="text/csv")
else:
    if process_button:
        st.error("⚠️ Please provide job description to proceed.")
