from flask import Flask, render_template, request, jsonify
from main import (
    extract_technical_terms, 
    find_required_skills, 
    match_resume_skills, 
    read_pdf, 
    read_docx,
    calculate_similarity,
    identify_sections,
    analyze_job_and_resume,
    initialize_nltk
)
import os
from werkzeug.utils import secure_filename
import traceback
import logging
import sys
import json
import requests
from dotenv import load_dotenv
import signal
import psutil
import socket
import time

# Initialize NLTK at startup
initialize_nltk()

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

# Load environment variables
load_dotenv()

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

# OpenRouter configuration
OPENROUTER_API_KEY = "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"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_chat_response(message, resume_data):
    """Get response from OpenRouter API."""
    try:
        print(f"\nProcessing chat message: {message}")
        print(f"Resume data available: {bool(resume_data)}")
        
        headers = {
            'Authorization': f'Bearer {OPENROUTER_API_KEY}',
            'HTTP-Referer': 'http://localhost:8080',
            'Content-Type': 'application/json'
        }
        
        # Create system message with resume context
        system_message = f"""You are an AI resume assistant analyzing the following resume data:
        Match Percentage: {resume_data.get('match_percentage')}%
        Matched Skills: {', '.join(str(k) for k in resume_data.get('matched_skills', {}).keys())}
        Missing Skills: {', '.join(str(k) for k in resume_data.get('missing_skills', {}).keys())}
        
        Provide specific, actionable advice about how the resume can be improved."""
        
        print("Sending request to OpenRouter API...")
        
        payload = {
            'model': 'openai/gpt-3.5-turbo',
            'messages': [
                {'role': 'system', 'content': system_message},
                {'role': 'user', 'content': message}
            ]
        }
        
        response = requests.post(
            OPENROUTER_API_URL,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        print(f"API Response status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"API Error Response: {response.text}")
            raise Exception(f"API request failed with status {response.status_code}")
        
        response_data = response.json()
        if 'choices' not in response_data or not response_data['choices']:
            print(f"Unexpected API response: {response_data}")
            raise ValueError("Invalid API response format")
            
        return response_data['choices'][0]['message']['content']
        
    except Exception as e:
        print(f"Error in get_chat_response: {str(e)}")
        raise

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
            # Get detailed analysis including section recommendations
            analysis_results = analyze_job_and_resume(job_description, text)
            
            # Calculate semantic similarity
            semantic_similarity = calculate_similarity(text, job_description)
            
            # Add semantic similarity to results
            analysis_results['semantic_similarity'] = round(semantic_similarity * 100, 1)
            
            logger.debug(f"Analysis complete. Results: {analysis_results}")
            
            return jsonify(analysis_results)
            
        except Exception as e:
            logger.error(f"Error during analysis: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({'error': f'Error during analysis: {str(e)}'}), 500
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        print("\n=== Chat Request ===")
        
        data = request.json
        if not data:
            print("Error: No data provided in request")
            return jsonify({'error': 'No data provided'}), 400
            
        message = data.get('message')
        resume_data = data.get('resume_data')
        
        print(f"Message received: {message}")
        print(f"Resume data present: {bool(resume_data)}")
        
        if not message:
            print("Error: No message provided")
            return jsonify({'error': 'No message provided'}), 400
        if not resume_data:
            print("Error: No resume data provided")
            return jsonify({'error': 'No resume data provided'}), 400
        
        # Get response from OpenRouter
        try:
            response = get_chat_response(message, resume_data)
            print("Successfully got chat response")
            return jsonify({'response': response})
        except Exception as e:
            print(f"Error getting chat response: {str(e)}")
            return jsonify({'error': str(e)}), 500
            
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

def is_port_in_use(port):
    """Check if a port is in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('127.0.0.1', port))
            return False
        except OSError:
            return True

def force_kill_process(pid):
    """Force kill a process and wait for it to terminate."""
    try:
        process = psutil.Process(pid)
        # Try graceful termination first
        process.terminate()
        
        # Wait up to 3 seconds for the process to terminate
        try:
            process.wait(timeout=3)
        except psutil.TimeoutExpired:
            # If process doesn't terminate gracefully, force kill it
            process.kill()
            process.wait(timeout=3)
            
        # Additional verification
        time.sleep(0.5)  # Give OS time to free up the port
        return True
    except (psutil.NoSuchProcess, psutil.AccessDenied, Exception) as e:
        logger.error(f"Error killing process {pid}: {str(e)}")
        return False

def cleanup_port(port):
    """Kill any process using the specified port."""
    killed_something = False
    try:
        for proc in psutil.process_iter(['pid', 'name', 'connections']):
            try:
                for conn in proc.connections():
                    if conn.laddr.port == port:
                        logger.info(f"Attempting to kill process {proc.pid} using port {port}")
                        if force_kill_process(proc.pid):
                            killed_something = True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
                
        # Verify port is actually free
        time.sleep(1)  # Wait for OS to fully release port
        if not is_port_in_use(port):
            return True
        elif killed_something:
            logger.warning(f"Killed processes but port {port} still in use")
            return False
        else:
            logger.warning(f"No processes found using port {port}")
            return False
            
    except Exception as e:
        logger.error(f"Error cleaning up port: {str(e)}")
        return False

def signal_handler(sig, frame):
    """Handle cleanup when server is shutting down."""
    logger.info("Shutting down server...")
    cleanup_port(8080)
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == '__main__':
    print("\n=== Starting Resume Analysis Server ===")
    print(f"Server running on port {8080}")
    app.run(host='0.0.0.0', port=8080, debug=True)