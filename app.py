import streamlit as st
import os
import zipfile
import tempfile
import fitz  # PyMuPDF
import pandas as pd
import requests
from io import BytesIO
import plotly.express as px

# --------------- PAGE CONFIG --------------- #
st.set_page_config(page_title="HR AI - Candidate Analyzer", layout="wide")
st.title("🧠 AI-Powered HR Assistant")
st.markdown("Upload CVs (ZIP of PDFs) + Job Description → Get smart AI-driven candidate analysis.")

# --------------- INPUT FIELDS --------------- #
openrouter_key = st.text_input("🔐 Enter your OpenRouter API Key", type="password")
job_description = st.text_area("📌 Enter Job Description / Role Requirements", height=200)
uploaded_zip = st.file_uploader("📁 Upload ZIP file of candidate CVs (PDF only)", type="zip")
process_button = st.button("🚀 Analyze Candidates")

# --------------- UTILITY FUNCTIONS --------------- #
def extract_pdf_text(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        return "\n".join([page.get_text() for page in doc])
    except Exception as e:
        return f"Error reading {os.path.basename(pdf_path)}: {str(e)}"

def call_openrouter_api(prompt, api_key):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "openchat/openchat-3.5-0106",  # You can change this to another model
        "messages": [
            {"role": "system", "content": "You are an expert HR assistant. Analyze candidate resumes."},
            {"role": "user", "content": prompt}
        ]
    }
    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"API Error: {str(e)}"

def generate_prompt(cv_text, job_description):
    return f"""
You are a world-class AI recruitment assistant.

Given the following Job Description:

{job_description}

And this Candidate Resume:

{cv_text}

Please analyze and return a structured report in this format:

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
---
Only return in this exact format.
"""

def extract_between(text, start_key, end_key=None):
    try:
        start = text.index(start_key) + len(start_key)
        end = text.index(end_key, start) if end_key else None
        return text[start:end].strip()
    except:
        return "N/A"

# --------------- MAIN ANALYSIS SECTION --------------- #
if process_button and uploaded_zip and job_description and openrouter_key:
    with st.spinner("🤖 Processing CVs with AI... Please wait."):
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "uploaded.zip")
            with open(zip_path, "wb") as f:
                f.write(uploaded_zip.read())

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(tmpdir)

            results = []
            for file in os.listdir(tmpdir):
                if file.lower().endswith(".pdf"):
                    file_path = os.path.join(tmpdir, file)
                    candidate_text = extract_pdf_text(file_path)
                    prompt = generate_prompt(candidate_text, job_description)
                    ai_response = call_openrouter_api(prompt, openrouter_key)

                    score = extract_between(ai_response, "Score:", "\n")
                    rec = extract_between(ai_response, "Recommendation:", "\n")
                    match_pct = extract_between(ai_response, "Skill Match Percentage:", "\n")
                    exp_years = extract_between(ai_response, "Years of Experience:", "\n")
                    strengths = extract_between(ai_response, "Top 3 Strengths:", "Red Flags").strip()
                    red_flags = extract_between(ai_response, "Red Flags", "Role Fit Justification").strip()
                    justification = extract_between(ai_response, "Role Fit Justification:", "Skill Match").strip()

                    results.append({
                        "Candidate": file,
                        "Score": int(score) if score.isdigit() else 0,
                        "Recommendation": rec,
                        "Skill Match %": int(match_pct.replace('%', '').strip()) if match_pct.replace('%', '').strip().isdigit() else 0,
                        "Experience (Years)": exp_years,
                        "Top Strengths": strengths,
                        "Red Flags": red_flags,
                        "Fit Justification": justification,
                        "Full AI Analysis": ai_response
                    })

            if not results:
                st.warning("❌ No valid PDF resumes found in the ZIP.")
                st.stop()

            df = pd.DataFrame(results)

# --------------- DASHBOARD SECTION --------------- #
            st.success("✅ AI Analysis Complete")
            st.subheader("📊 Candidate Insights Dashboard")

            min_score = st.slider("🔎 Filter candidates by minimum score", 0, 100, 50)
            filtered_df = df[df["Score"] >= min_score]

            # Score Chart
            st.markdown("### 📈 Candidate Score Comparison")
            score_chart = px.bar(
                filtered_df,
                x="Candidate",
                y="Score",
                color="Recommendation",
                text="Score",
                title="Candidate AI Score by Recommendation",
                color_discrete_map={
                    "Strong Fit": "green",
                    "Moderate Fit": "orange",
                    "Not Recommended": "red"
                }
            )
            st.plotly_chart(score_chart, use_container_width=True)

            # Recommendation Pie
            st.markdown("### 🥧 Recommendation Breakdown")
            rec_pie = px.pie(
                filtered_df,
                names="Recommendation",
                title="AI Recommendation Summary",
                color_discrete_map={
                    "Strong Fit": "green",
                    "Moderate Fit": "orange",
                    "Not Recommended": "red"
                }
            )
            st.plotly_chart(rec_pie, use_container_width=True)

            # Skill Match %
            st.markdown("### 🧠 Skill Match Percentage")
            skill_chart = px.bar(
                filtered_df,
                x="Candidate",
                y="Skill Match %",
                title="Skill Alignment with Job Description",
                color="Skill Match %",
                color_continuous_scale="blues"
            )
            st.plotly_chart(skill_chart, use_container_width=True)

            # Expandable Details
            st.markdown("### 🔍 Candidate Deep Dive")
            for _, row in filtered_df.iterrows():
                with st.expander(f"📌 {row['Candidate']} — Score: {row['Score']} — {row['Recommendation']}"):
                    st.markdown(f"**Top Strengths**:\n{row['Top Strengths']}")
                    st.markdown(f"**Red Flags**:\n{row['Red Flags']}")
                    st.markdown(f"**Role Fit Justification**:\n{row['Fit Justification']}")
                    st.markdown(f"**Skill Match %**: {row['Skill Match %']} | **Experience**: {row['Experience (Years)']}")

            # CSV Export
            csv_data = filtered_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download CSV Report",
                data=csv_data,
                file_name="AI_Hiring_Insights_Report.csv",
                mime="text/csv"
            )
elif process_button:
    st.error("⚠️ Please upload a ZIP, enter job description, and provide your API key.")
