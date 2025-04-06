from flask import Flask, render_template, request, jsonify
from main import (
    extract_technical_terms, 
    find_required_skills, 
    match_resume_skills, 
    read_pdf, 
    read_docx,
    calculate_similarity
)
import os
from werkzeug.utils import secure_filename
import traceback
import logging
import sys

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists with proper permissions
try:
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    logger.info(f"Upload folder created/verified at: {os.path.abspath(app.config['UPLOAD_FOLDER'])}")
except Exception as e:
    logger.error(f"Error creating upload folder: {str(e)}")
    raise

ALLOWED_EXTENSIONS = {'pdf', 'docx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        logger.debug("Starting resume analysis")
        
        # Check for required files and data
        if 'resume' not in request.files:
            logger.error("No resume file in request")
            return jsonify({'error': 'Missing resume file'}), 400
            
        if 'job_description' not in request.form:
            logger.error("No job description in request")
            return jsonify({'error': 'Missing job description'}), 400
        
        resume_file = request.files['resume']
        job_description = request.form['job_description']
        
        if resume_file.filename == '':
            logger.error("Empty filename")
            return jsonify({'error': 'No selected file'}), 400
            
        if not allowed_file(resume_file.filename):
            logger.error(f"Invalid file type: {resume_file.filename}")
            return jsonify({'error': 'Invalid file type. Please upload PDF or DOCX files only.'}), 400
        
        # Save and read resume
        filename = secure_filename(resume_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        logger.debug(f"Saving file to: {filepath}")
        
        try:
            resume_file.save(filepath)
            logger.debug(f"File saved successfully at: {filepath}")
        except Exception as e:
            logger.error(f"Error saving file: {str(e)}")
            return jsonify({'error': f'Error saving file: {str(e)}'}), 500
        
        # Read resume text
        text = None
        try:
            if filename.lower().endswith('.pdf'):
                text = read_pdf(filepath)
            elif filename.lower().endswith('.docx'):
                text = read_docx(filepath)
        except Exception as e:
            logger.error(f"Error reading file: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({'error': f'Error reading file: {str(e)}'}), 500
        
        if not text:
            logger.error("No text extracted from file")
            return jsonify({'error': 'Could not extract text from the file'}), 400
        
        logger.debug(f"Successfully extracted text. Length: {len(text)}")
        
        # Clean up uploaded file
        try:
            os.remove(filepath)
            logger.debug("Cleaned up uploaded file")
        except Exception as e:
            logger.warning(f"Could not remove temporary file: {str(e)}")
        
        # Analyze resume
        try:
            logger.debug("Calculating semantic similarity")
            semantic_similarity = calculate_similarity(text, job_description)
            
            logger.debug("Finding required skills")
            required_skills = find_required_skills(job_description)
            
            logger.debug("Extracting technical terms")
            resume_skills = extract_technical_terms(text)
            
            logger.debug("Matching skills")
            matched_skills, categorized_missing, category_importance = match_resume_skills(resume_skills, required_skills)
            
            # Calculate match percentage
            total_weight = sum(required_skills.values())
            matched_weight = sum(matched_skills.values())
            semantic_boost = semantic_similarity * 0.2
            match_percentage = ((matched_weight / total_weight) * 0.8 + semantic_boost) * 100
            
            logger.debug(f"Analysis complete. Match percentage: {match_percentage}")
            logger.debug(f"Matched skills: {matched_skills}")
            logger.debug(f"Categorized missing: {categorized_missing}")
            logger.debug(f"Category importance: {category_importance}")
            
            response_data = {
                'match_percentage': round(match_percentage, 1),
                'semantic_similarity': round(semantic_similarity * 100, 1),
                'matched_skills': matched_skills,
                'missing_skills': categorized_missing,
                'category_importance': category_importance
            }
            logger.debug(f"Sending response data: {response_data}")
            
            return jsonify(response_data)
            
        except Exception as e:
            logger.error(f"Error during analysis: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({'error': f'Error during analysis: {str(e)}'}), 500
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)