from flask import Flask, render_template, request, jsonify
from main import extract_technical_terms, find_required_skills, match_resume_skills, read_pdf, read_docx
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'pdf', 'docx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Get job description
        job_description = request.form.get('job_description', '').strip()
        if not job_description:
            return jsonify({'error': 'Please enter a job description'}), 400

        # Get resume file
        if 'resume' not in request.files:
            return jsonify({'error': 'No resume file uploaded'}), 400
            
        file = request.files['resume']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        if not allowed_file(file.filename):
            return jsonify({'error': 'Unsupported file format. Please use PDF or DOCX files.'}), 400

        # Save file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Read resume
        if filename.lower().endswith('.pdf'):
            resume_text = read_pdf(filepath)
        else:
            resume_text = read_docx(filepath)

        # Clean up
        os.remove(filepath)

        if not resume_text:
            return jsonify({'error': 'Failed to read the resume file'}), 400

        # Analyze job posting
        required_skills = find_required_skills(job_description)
        
        # Match skills
        matched_skills = match_resume_skills(resume_text, required_skills)
        
        # Calculate missing skills
        missing_skills = [skill for skill in required_skills if skill not in matched_skills]
        
        # Calculate match percentage
        if required_skills:
            match_percentage = (len(matched_skills) / len(required_skills)) * 100
        else:
            match_percentage = 0

        return jsonify({
            'matched_skills': matched_skills,
            'missing_skills': missing_skills,
            'match_percentage': round(match_percentage, 1)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)