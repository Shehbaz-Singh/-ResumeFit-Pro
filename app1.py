import streamlit as st
import google.generativeai as genai
import pymupdf
import os
from dotenv import load_dotenv
import re
from fpdf import FPDF
import tempfile
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from io import BytesIO

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PROMPT = os.getenv("PROMPT")

# Configure Gemini API
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found in .env. Please ensure it is set.")
    st.stop()

try:
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    st.error(f"Error configuring Gemini API: {e}")
    st.stop()

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    try:
        with pymupdf.open(stream=pdf_file.read(), filetype="pdf") as doc:
            text = ""
            for page in doc:
                text += page.get_text("text")
            return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return None

# Function to analyze resume (using Gemini)
def analyze_resume(resume_text, job_desc):
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        prompt = f"""
        {PROMPT}

        Resume:
        {resume_text}

        Job Description:
        {job_desc}

        Perform a highly focused and practical analysis of the resume against the job description.
        1. Identify and list specific similarities and differences in skills, experience, and keywords.
        2. For each key skill or requirement mentioned in the job description, provide direct, currently working and relevant online reference links to books, notes, or educational websites.
        3. Based on the calculated match percentage, provide clear and actionable advice:
           - If the match is below 60%, recommend specific new topics to study, accompanied by currently working and relevant learning links to books, notes, or educational websites.
           - If the match is 60% or higher, focus on interview preparation tips and suggest targeted resume improvements to better align with the job description.
        4. Ensure all links provided are directly relevant, currently functional, and are available to the user immediately.
        5. Provide a cover letter based on the job description and resume, with proper indentation.
        6. Provide an ATS formatting check and highlight any issues.
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error analyzing resume: {e}")
        return None

# Extract match % from response
def extract_match_percentage(text):
    if text:
        match = re.search(r'(\d{1,3})%', text)
        return min(int(match.group(1)), 100) if match else None
    return None

# Create PDF from analysis
def generate_pdf_report(content):
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        for line in content.split("\n"):
            pdf.multi_cell(0, 10, line)
        output = pdf.output(dest='S').encode('latin-1')
        return output
    except Exception as e:
        st.error(f"Error generating PDF report: {e}")
        return None

# Generate cover letter PDF
def generate_cover_letter_pdf(cover_letter_text):
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        for line in cover_letter_text.split("\n"):
            pdf.multi_cell(0, 10, line)
        output = pdf.output(dest='S').encode('latin-1')
        return output
    except Exception as e:
        st.error(f"Error generating cover letter PDF: {e}")
        return None

# --- New Functions for Plotting ---

def create_radar_chart(skills_data, job_title="Job Skills"):
    """
    Creates a radar chart to visualize skill proficiency.

    Args:
        skills_data (dict): A dictionary where keys are skills and values are proficiency scores (0-100).
        job_title (str): The title of the job for which the skills are being compared.

    Returns:
        matplotlib.figure.Figure: The generated radar chart.
    """
    # Number of variables
    num_vars = len(skills_data)
    if num_vars == 0:
        return None  # Return None if there are no skills

    # Create the angles for each variable
    angles = [n / float(num_vars) * 2 * 3.14159 for n in range(num_vars)]
    angles += angles[:1]  # Close the circle

    # Create the radar chart figure
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # Add the data to the plot
    values = list(skills_data.values())
    values += values[:1]  # Close the circle
    ax.plot(angles, values, linewidth=2, linestyle='solid', color="#4CAF50")  # Green color
    ax.fill(angles, values, 'g', alpha=0.2)  # Fill with transparency

    # Set the axis labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(list(skills_data.keys()), color="#333333")  # Dark grey labels

    # Set the y-axis limits
    ax.set_ylim(0, 100)

    # Add a title
    ax.set_title(f"{job_title} - Skill Proficiency", color="#212121", fontsize=14, pad=20)  # Darker title

    # Add grid lines
    ax.grid(color="#BDBDBD", linestyle='--', linewidth=0.5)  # Lighter grid

    return fig


def create_bar_chart(match_score, title="Resume Match Score"):
    """
    Creates a simple bar chart to display the resume match score.

    Args:
        match_score (int): The resume match percentage (0-100).
        title (str): The title of the chart.

    Returns:
        matplotlib.figure.Figure: The generated bar chart.
    """
    fig, ax = plt.subplots(figsize=(6, 2.5))  # Adjust figure size
    ax.barh([title], [match_score], color="#880E4F")  # Deep purple
    ax.set_xlim(0, 100)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticks([])  # Remove x-ticks
    ax.set_yticks([])
    for i, v in enumerate([match_score]):
        ax.text(v + 3, i, str(v) + "%", color="#311B92", fontweight='bold')  # Darker text
    return fig

# --- UI ---

st.set_page_config(page_title="ResumeFit Pro", layout="wide")
# Apply a custom CSS style for a more modern look
st.markdown(
    """
    <style>
        body {
            font-family: 'Arial', sans-serif;
            color: #212121;  /* Darker default text */
            background-color: #F5F5F5; /* Off-white background */
        }
        .sidebar .sidebar-content {
            background-color: #E0F7FA; /* Light cyan sidebar */
            border-right: 1px solid #B2EBF2;
        }
        .stButton>button {
            color: #FFFFFF;
            background-color: #00897B; /* Teal button */
            border: none;
            border-radius: 6px;
            padding: 10px 20px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            transition: background-color 0.3s ease, transform 0.2s ease;
        }
        .stButton>button:hover {
            background-color: #00695C;  /* Darker teal on hover */
            transform: translateY(-2px);
        }
        h1 {
            color: #2C3E50; /* Dark blue for main heading */
            text-align: center;
            margin-bottom: 20px;
        }
        h2 {
            color: #34495E; /* Darker blue for subheadings */
            margin-top: 30px;
            margin-bottom: 15px;
        }
        h3 {
            color: #424242; /* Very dark grey */
            margin-top: 20px;
            margin-bottom: 10px;
        }
        p {
            color: #555555;  /* Medium grey body text */
            line-height: 1.7;
        }
        .reportview-container .main .streamlit-main-content {
            max-width: 100%;
            padding-left: 100px;
            padding-right: 100px;
        }

    </style>
    """,
    unsafe_allow_html=True,
)

# Main App Title
st.markdown("<h1 style='text-align: center; color: #3366cc;'>ResumeFit Pro: AI Resume Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enhance your resume with AI-driven insights.</p>", unsafe_allow_html=True)

# Input Section
st.sidebar.markdown("<h2 style='color: #3366cc;'>Input</h2>", unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader("📄 Upload Resume (PDF)", type=["pdf"])
job_description = st.sidebar.text_area("📋 Job Description", height=200)
analyze_button = st.sidebar.button("🚀 Analyze", use_container_width=True)

# Main Analysis Area
if analyze_button and uploaded_file and job_description:
    with st.spinner("Analyzing..."):
        resume_text = extract_text_from_pdf(uploaded_file)
        if resume_text:
            analysis = analyze_resume(resume_text, job_description)
            if analysis:
                match_score = extract_match_percentage(analysis)
                pdf_report_bytes = generate_pdf_report(analysis)

                st.markdown("<hr>", unsafe_allow_html=True)
                st.markdown("<h2 style='color: #3366cc;'>Analysis Report</h2>", unsafe_allow_html=True)

                # Display Match Score
                if match_score is not None:
                    st.markdown(f"<p style='font-size: 1.8em; font-weight: bold; color:#2E7D32;'>Match Score: {match_score}%</p>", unsafe_allow_html=True)  # Strong green
                    st.progress(match_score / 100)

                    # Bar Chart
                    match_bar_chart = create_bar_chart(match_score, title="Resume Match")
                    if match_bar_chart:
                        st.pyplot(match_bar_chart)
                else:
                    st.warning("Match score could not be calculated.")

                # Extract skills and create radar chart
                skills_match = re.search(r"Skills Match:\s*({.*?})", analysis)  # Regex to find skills
                if skills_match:
                    try:
                        skills_dict = eval(skills_match.group(1))  # Safely convert string to dict
                        radar_chart = create_radar_chart(skills_dict, job_title="Skills Overview")
                        if radar_chart:
                            st.pyplot(radar_chart)
                    except Exception as e:
                        st.error(f"Error processing skills data: {e}")
                else:
                    st.info("No specific skills match found to visualize.")

                st.markdown("<h3 style='color: #3366cc;'>Key Insights</h3>", unsafe_allow_html=True)
                st.markdown(analysis, unsafe_allow_html=True)

                # Cover Letter Download
                cover_letter_match = re.search(r"Cover Letter:\s*(.*?)(?:ATS Formatting Check:|Interview Questions:|$)", analysis, re.DOTALL)
                if cover_letter_match:
                    cover_letter = cover_letter_match.group(1).strip()
                    cover_letter_pdf_data = generate_cover_letter_pdf(cover_letter)
                    if cover_letter_pdf_data:
                        st.download_button(
                            label="📄 Download Cover Letter (PDF)",
                            data=cover_letter_pdf_data,
                            file_name="cover_letter.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                        )
                    else:
                        st.error("Failed to generate Cover Letter PDF")

                # Full Report Download
                if pdf_report_bytes:
                    st.download_button(
                        label="📥 Download Full Report (PDF)",
                        data=pdf_report_bytes,
                        file_name="resume_analysis_report.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                else:
                    st.warning("Failed to generate PDF report.")
            else:
                st.warning("Failed to analyze resume.")
        else:
            st.warning("Failed to extract text from the uploaded PDF.")
