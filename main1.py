import time  # For temp filename
from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename
import os
import fitz  # PyMuPDF
import docx2txt  # For DOCX files
import re  # For name extraction
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

load_dotenv()

# Allowed file extensions
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def format_html_output(text):
    """Format and sanitize HTML output from LLM"""
    # First escape all HTML
    text = text.replace('<', '&lt;').replace('>', '&gt;')
    
    # Then selectively allow specific tags
    allowed_tags = {
        '&lt;strong&gt;': '<strong>',
        '&lt;/strong&gt;': '</strong>',
        '&lt;em&gt;': '<em>',
        '&lt;/em&gt;': '</em>',
        '&lt;span class="highlight"&gt;': '<span class="highlight">',
        '&lt;span class="percentage"&gt;': '<span class="percentage">',
        '&lt;span class="keyword"&gt;': '<span class="keyword">',
        '&lt;span class="recommendation"&gt;': '<span class="recommendation">',
        '&lt;/span&gt;': '</span>',
        '&lt;ul&gt;': '<ul>',
        '&lt;/ul&gt;': '</ul>',
        '&lt;li&gt;': '<li>',
        '&lt;/li&gt;': '</li>',
        '&lt;h2&gt;': '<h2>',
        '&lt;/h2&gt;': '</h2>',
        '&lt;h3&gt;': '<h3>',
        '&lt;/h3&gt;': '</h3>',
        '&lt;h4&gt;': '<h4>',
        '&lt;/h4&gt;': '</h4>'
    }
    
    for escaped, tag in allowed_tags.items():
        text = text.replace(escaped, tag)
    
    return text

def extract_text_from_file(filepath, extension):
    """Extract text from PDF or DOC/DOCX files"""
    text = ""
    try:
        if extension == 'pdf':
            doc = fitz.open(filepath)
            for page in doc:
                text += page.get_text()
        elif extension in ['doc', 'docx']:
            text = docx2txt.process(filepath)
    except Exception as e:
        print(f"Error extracting text: {str(e)}")
    return text

def extract_name_from_resume(text):
    patterns = [
        # Added capturing group () around the name pattern
        r"^([A-Z][a-z]+ [A-Z][a-z]+)$",
        r"\b([A-Z][a-z]+ [A-Z][a-z]+)\b",
        r"Name:\s*([A-Z][a-z]+ [A-Z][a-z]+)",
        r"Resume of ([A-Z][a-z]+ [A-Z][a-z]+)",
        r"Curriculum Vitae of ([A-Z][a-z]+ [A-Z][a-z]+)"
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.MULTILINE)
        if match:
            # Return group 1 which now always exists
            return match.group(1)
    return "The candidate"

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "supersecretkey")
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize Groq LLM
groq_api_key = os.getenv("GROQ_API_KEY")
if groq_api_key:
    llm = ChatGroq(
        temperature=0.5,
        model_name="meta-llama/llama-4-scout-17b-16e-instruct",
        groq_api_key=groq_api_key
    )
else:
    llm = None

# Prompt templates
input_prompt1 = """You are an experienced Technical HR Manager evaluating a resume against job requirements. 
Mandatory Skills: {mandatory_skills}
Required Experience: {required_experience}

Provide a concise bullet-point and new line for each break line evaluation with this structure:
Please format your response using HTML tags for better presentation:
- Wrap section headers in <strong> tags
- Wrap strengths in <em> tags
- Wrap weaknesses in <span class="highlight"> tags
- Use <ul> and <li> for lists

• Key Strengths:
  - Strength 1
  - Strength 2
• Gaps:
  - Missing skill 1
  - Missing skill 2
• Experience Analysis: {experience_analysis}
"""
# Updated prompt for match button
input_prompt3 = """You are an skilled ATS (Applicant Tracking System) scanner. Analyze this resume against:
Mandatory Skills: {mandatory_skills}
Required Experience: {required_experience}

Format your response using HTML with this structure:


  <h2>Match Percentage: <span class="percentage">{percentage}%</span></h2>



  <h3>Missing Skills:</h3>
  <ul>
    {missing_skills}
  </ul>



  <h3>Final Recommendation:</h3>
  <span class="recommendation">{recommendation}</span>


Instructions:
- The percentage should be wrapped in <span class="percentage">
- Each missing skill should be wrapped in <span class="keyword">
- The recommendation should address the candidate by name from the resume
- Be concise ,short and direct
- Use proper HTML formatting throughout
- If no mandatory skills are provided, base analysis on job description match"""

prompt = PromptTemplate(
    input_variables=["role_prompt", "job_description", "resume_text", "mandatory_skills", 
                    "required_experience", "percentage", "missing_skills", "recommendation"],
    template="""
{role_prompt}

Job Description:
{job_description}

Resume:
{resume_text}
"""
)

chain = LLMChain(llm=llm, prompt=prompt) if llm else None

@app.route('/', methods=['GET', 'POST'])
def index():
    # Initialize variables from session
    job_description_text = session.get('job_description_text', '')
    mandatory_skills = session.get('mandatory_skills', '')
    required_experience = session.get('required_experience', '')
    job_link = session.get('job_link', '')
    filename = session.get('filename', '')
    evaluation_result = session.get('evaluation_result', '')
    match_result = session.get('match_result', '')
    error = ''
    
    # This will hold the final JD for LLM processing
    job_description_for_llm = ""
    
    if request.method == 'POST':
        # Get form data
        job_description_text = request.form.get('job_description', '')
        mandatory_skills = request.form.get('mandatory_skills', '')
        required_experience = request.form.get('required_experience', '')
        job_link = request.form.get('job_link', '')
        action = request.form.get('action')
        
        # Reset the combined JD
        job_description_for_llm = ""
        sources_used = []
        
        # Process job link FIRST (if provided)
        if job_link:
            try:
                jd_from_url = extract_jd_from_url(job_link)
                if jd_from_url:
                    job_description_for_llm += f"Job Posting from URL: {job_link}\n\n{jd_from_url}\n\n"
                    sources_used.append("URL")
                else:
                    error = "Could not extract job description from the provided link"
            except Exception as e:
                error = f"Error processing job link: {str(e)}"
        
        # Process JD file SECOND (if provided and no error)
        if not error and 'jd_file' in request.files:
            jd_file = request.files['jd_file']
            if jd_file.filename != '':
                if jd_file and allowed_file(jd_file.filename):
                    try:
                        extension = jd_file.filename.rsplit('.', 1)[1].lower()
                        temp_filename = f"temp_jd_{int(time.time())}.{extension}"
                        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
                        jd_file.save(temp_path)
                        jd_text = extract_text_from_file(temp_path, extension)
                        job_description_for_llm += f"Job Description from File Upload: {jd_file.filename}\n\n{jd_text}\n\n"
                        sources_used.append("File")
                        os.remove(temp_path)
                    except Exception as e:
                        error = error or f"Error processing JD file: {str(e)}"
                else:
                    error = error or "Invalid JD file type"
        
        # Process text area THIRD (if provided)
        if job_description_text.strip():
            job_description_for_llm += f"Job Description from Text Input:\n\n{job_description_text}\n\n"
            sources_used.append("Text")
        
        # Store in session
        session['job_description_text'] = job_description_text
        session['mandatory_skills'] = mandatory_skills
        session['required_experience'] = required_experience
        session['job_link'] = job_link
        
        # Resume upload handling
        if 'resume' in request.files and not error:
            file = request.files['resume']
            if file.filename != '':
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(filepath)
                    session['filename'] = filename
                    # Clear previous results when new file is uploaded
                    session.pop('evaluation_result', None)
                    session.pop('match_result', None)
                else:
                    error = "Allowed file types are PDF, DOC, DOCX"
        
        # Process if we have inputs
        if not error and (session.get('filename') or job_description_for_llm.strip()):
            resume_text = ""
            if session.get('filename'):
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], session['filename'])
                extension = session['filename'].rsplit('.', 1)[1].lower()
                resume_text = extract_text_from_file(filepath, extension)
            
            # If we didn't get any JD source, use the text area as fallback
            if not job_description_for_llm.strip() and job_description_text.strip():
                job_description_for_llm = job_description_text
            
            try:
                if not llm:
                    error = "LLM service not configured"
                elif not job_description_for_llm.strip():
                    error = "Please provide job description through text, file, or URL"
                else:
                    # Extract candidate name
                    extracted_name = extract_name_from_resume(resume_text)
                    
                    # Simple experience extraction
                    exp_analysis = f"Candidate experience vs required {required_experience}"
                    
                    # Simple fit determination based on skills
                    is_fit = True  # Default to fit if no mandatory skills
                    if mandatory_skills:
                        is_fit = any(skill.strip().lower() in resume_text.lower() 
                                   for skill in mandatory_skills.split(','))
                    
                    if action == "evaluate":
                        evaluation_result = chain.run(
                            role_prompt=input_prompt1,
                            job_description=job_description_for_llm,  # Use combined JD
                            resume_text=resume_text,
                            mandatory_skills=mandatory_skills,
                            required_experience=required_experience,
                            experience_analysis=exp_analysis,
                            percentage="",
                            missing_skills="",
                            recommendation=""
                        )
                        evaluation_result = format_html_output(evaluation_result)
                        session['evaluation_result'] = evaluation_result
                    elif action == "match":
                        # Generate dynamic recommendation based on fit
                        fit_status = "a strong fit" if is_fit else "not a fit"
                        recommendation = (
                            f"{extracted_name}, you are {fit_status} for this position. "
                            f"{'We recommend applying!' if is_fit else 'Consider other roles that match your skills.'}"
                        )
                        
                        # Calculate a simple match percentage (this should be improved)
                        match_percentage = 0
                        if mandatory_skills:
                            skills_list = [skill.strip().lower() for skill in mandatory_skills.split(',')]
                            matched_skills = sum(1 for skill in skills_list if skill in resume_text.lower())
                            match_percentage = min(100, int((matched_skills / len(skills_list)) * 100))
                        else:
                            # Simple fallback if no skills provided
                            match_percentage = 80 if is_fit else 40
                        
                        # Generate missing skills list
                        missing_skills_html = ""
                        if mandatory_skills:
                            skills_list = [skill.strip() for skill in mandatory_skills.split(',')]
                            missing_skills = [skill for skill in skills_list if skill.lower() not in resume_text.lower()]
                            missing_skills_html = "".join(
                                f"<li><span class='keyword'>{skill}</span></li>" 
                                for skill in missing_skills
                            )
                        
                        match_result = chain.run(
                            role_prompt=input_prompt3,
                            job_description=job_description_for_llm,  # Use combined JD
                            resume_text=resume_text,
                            mandatory_skills=mandatory_skills,
                            required_experience=required_experience,
                            percentage=str(match_percentage),
                            missing_skills=missing_skills_html,
                            recommendation=recommendation
                        )
                        match_result = format_html_output(match_result)
                        session['match_result'] = match_result
            except Exception as e:
                error = f"Error processing: {str(e)}"
        elif not session.get('filename') and not job_description_for_llm.strip():
            error = "Please upload a resume and provide job description"
    
    return render_template('home.html',
                         job_description=job_description_text,  # Only show text input
                         mandatory_skills=mandatory_skills,
                         required_experience=required_experience,
                         job_link=job_link,
                         filename=filename,
                         evaluation_result=evaluation_result,
                         match_result=match_result,
                         error=error)

@app.route('/clear', methods=['POST'])
def clear():
    # Clear uploaded file if exists
    if 'filename' in session:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], session['filename'])
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception as e:
            print(f"Error deleting file {filepath}: {e}")
    
    # Clear session data
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # Default for local
    app.run(host="0.0.0.0", port=port, debug=False)
