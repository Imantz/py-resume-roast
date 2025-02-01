import os
from dotenv import load_dotenv
import gradio as gr
import pdfplumber
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

# Load AI models
model = SentenceTransformer("all-MiniLM-L6-v2")

# Configure Google Gemini API
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
gemini_model_name = "gemini-2.0-flash-thinking-exp-01-21"

# Define Gemini model instance
gemini_model = genai.GenerativeModel(model_name=gemini_model_name)

def extract_text_from_pdf(pdf_file):
    """Extract text from an uploaded PDF resume."""
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

def generate_embedding(text):
    """Generate an embedding for a given text."""
    return model.encode([text])[0]

def calculate_relevance(resume_pdf, job_description, skills=""):
    """Compare a resume against a job description and send data to Gemini for ATS improvements."""
    
    yield "⏳ **Processing resume...**", "⏳ **Generating AI suggestions...**"

    # Extract resume text
    resume_text = extract_text_from_pdf(resume_pdf)
    
    if not resume_text:
        yield "❌ No text found in the resume. Please upload a valid PDF.", ""
        return

    if not job_description.strip():
        yield "❌ Please paste a job description.", ""
        return

    # Combine job description with optional skills
    job_text = job_description
    if skills.strip():
        job_text += "\n\n" + "Skills: " + skills.strip()

    # Generate embeddings
    resume_embedding = generate_embedding(resume_text)
    job_embedding = generate_embedding(job_text)

    # Compute similarity score
    similarity_score = cosine_similarity([resume_embedding], [job_embedding])[0][0]
    match_percentage = similarity_score * 100

    # Result message
    result = f"🔍 **Relevance Score: {similarity_score:.4f}**\n\n"
    result += f"✅ **Match Percentage: {match_percentage:.2f}%**\n\n"
    
    # Provide feedback
    if match_percentage > 80:
        result += "🔥 **Great match! Your resume is highly relevant.**"
    elif match_percentage > 60:
        result += "✅ **Good match. You might need some tweaks.**"
    else:
        result += "⚠️ **Low match. Consider adjusting your resume to match the job better.**"

    yield result, "⏳ **Waiting for AI response...**"

    # Send data to Gemini for ATS optimization suggestions
    gemini_prompt = f"""
    You are an expert in resume optimization for Applicant Tracking Systems (ATS) and job applications.

    ### Task:
    Analyze the following resume **in comparison to the job description** and provide feedback on how to improve it for **better ATS ranking and recruiter visibility**.

    ---

    ### **🔍 Resume Text:**
    {resume_text}

    ---

    ### **📌 Job Description:**
    {job_description}

    ---

    ### **🎯 Key Skills from Job Posting:**
    {skills if skills.strip() else "No specific skills provided"}

    ---

    ### **💡 Instructions for Optimization Feedback:**
    1️⃣ **ATS Keyword Matching**:
    - Identify missing **keywords, skills, or industry terms** that should be added.
    - Highlight **any unnecessary or weak keywords** that can be removed.

    2️⃣ **Resume Formatting & Structure**:
    - Suggest ways to improve **section organization (e.g., skills, experience, summary)**.
    - Highlight **common ATS formatting mistakes** (e.g., tables, images, excessive styling).

    3️⃣ **Bullet Point Optimization**:
    - Improve the **clarity and impact** of resume bullet points.
    - Suggest **quantifiable metrics** (e.g., “Increased efficiency by 30%” instead of “Improved processes”).

    4️⃣ **Soft Skills & Achievements**:
    - Identify **key soft skills** that could make the resume stronger.
    - Suggest **measurable achievements** that enhance credibility.

    5️⃣ **Final ATS Score & Recommendations**:
    - Give an estimated **ATS relevance score (out of 100%)**.
    - Provide a **short action plan** for improving the resume before applying.

    ---

    ### 🎯 **Output Format (Example Response):**
    - **Missing Keywords:** [List of important keywords to add]
    - **Formatting Issues:** [Any structural issues detected]
    - **Bullet Point Enhancements:** [Examples of improved bullet points]
    - **Soft Skills to Include:** [Suggestions for making the resume more attractive]
    - **Final ATS Score:** [Estimated relevance score]
    - **Action Plan:** [Step-by-step suggestions for fixing the resume]

    Provide a **concise, structured response** so the user can quickly make improvements.

    ### **Output style:**
    - **Cachy, funny, engaging, and informative. Make sure to keep the user engaged and provide valuable feedback.
    - **Don't call real name, call it Pikachu or something fun.**
    """


    # Query Gemini (takes time)
    gemini_response = gemini_model.generate_content(gemini_prompt)

    yield result, gemini_response.text  # Return both match score and improvement suggestions

# Gradio UI with two outputs (Relevance Score + Gemini ATS Suggestions)
iface = gr.Interface(
    fn=calculate_relevance,
    inputs=[
        gr.File(label="Upload Resume (PDF)"),
        gr.TextArea(label="Paste Job Description (Markdown Supported)", placeholder="Copy and paste the job description here..."),
        gr.Textbox(label="Optional: Skills (Comma-separated)", placeholder="e.g. React, JavaScript, Node.js")
    ],
    outputs=[
        "markdown",  # Match score output
        "markdown"   # Gemini ATS optimization response
    ],
    title="AI Resume Relevance & ATS Optimizer",
    description="Upload your resume and a job description to check how well they match. AI will also suggest improvements to increase your ATS score!"
)

# Launch the app
iface.launch()
