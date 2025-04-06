import PyPDF2
import docx
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import requests
from bs4 import BeautifulSoup
import time
from collections import Counter
import os
from difflib import SequenceMatcher

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

"""
Desired_Terminal_Output:
    (Print) Analyzing job posting from: https://www.linkedin.com/jobs/view/4193717139/?alternateChannel=search&refId=sY8swhpNxLDD7k6seEKuug%3D%3D&trackingId=py1Iv3LI9%2FD8PxzrnBpKhA%3D%3D
    (Print) Analyzing resume from: /Users/revantpatel/Downloads/TestResume.pdf
    (Print) Job and Resume Match Percentage: 100.0%
    (Print) Matched Skills in Resume:
    (Print) Skills Missing from Resume:
"""

def read_pdf(file_path):
    """Read text content from a PDF file."""
    try:
        if not os.path.exists(file_path):
            print(f"DEBUG: PDF file not found at {file_path}")
            return None
            
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            print(f"DEBUG: Successfully read PDF. Text length: {len(text)} characters")
            return text
            
    except Exception as e:
        print(f"DEBUG: Error reading PDF: {str(e)}")
        return None

def read_docx(file_path):
    """Read text content from a DOCX file."""
    try:
        if not os.path.exists(file_path):
            print(f"DEBUG: DOCX file not found at {file_path}")
            return None
            
        doc = docx.Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        print(f"DEBUG: Successfully read DOCX. Text length: {len(text)} characters")
        return text
            
    except Exception as e:
        print(f"DEBUG: Error reading DOCX: {str(e)}")
        return None

def scrape_job_posting(url):
    """Scrape job posting content from a given URL."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Common class names and tags for job descriptions
        job_description = None
        possible_classes = [
            # General job description classes
            'job-description', 'job-details', 'description', 'job-content',
            'Core Responsibilities', 'job-overview', 'job-summary',
            
            # LinkedIn specific
            'description__text', 'jobs-description__content',
            'jobs-unified-top-card__job-insight',
            
            # Indeed specific
            'jobsearch-JobComponent-description',
            'jobsearch-jobDescriptionText',
            
            # Glassdoor specific
            'jobDescriptionContent',
            'jobDescription',
            
            # Company website specific
            'careers-job-description',
            'position-description',
            'role-description',
            'job-requirements',
            'job-responsibilities',
            
            # Common content containers
            'content', 'main-content', 'article-content',
            'rich-text', 'rich-text-content',
            
            # Common section headers
            'job-requirements-section',
            'responsibilities-section',
            'qualifications-section',
            'about-the-role',
            'role-overview'

            # Handshake specific
            'Key Responsibilities',
            'Key Requirements',
            'Key Qualifications',
            'Key Skills',
            'Key Competencies',
            'Key Responsibilities',
            'Key Requirements',
        ]
        
        # First try to find by class name
        for class_name in possible_classes:
            job_description = soup.find('div', class_=class_name)
            if job_description:
                break
        
        # If not found by class, try other common selectors
        if not job_description:
            # Try finding by common HTML structure
            job_description = (
                soup.find('div', {'id': 'job-description'}) or
                soup.find('div', {'id': 'description'}) or
                soup.find('div', {'id': 'job-content'}) or
                soup.find('main') or
                soup.find('article') or
                soup.find('div', class_='content') or
                # Look for content within common job posting containers
                soup.find('div', {'class': 'job-posting'}) or
                soup.find('div', {'class': 'careers-content'})
            )
        
        # If still not found, try to find the largest text block
        if not job_description:
            text_blocks = soup.find_all(['div', 'section', 'article'])
            if text_blocks:
                # Find the block with the most text content
                job_description = max(text_blocks, key=lambda x: len(x.get_text()))
        
        if job_description:
            return job_description.get_text(separator=' ', strip=True)
        else:
            return "Could not find job description in the page."
            
    except requests.RequestException as e:
        return f"Error fetching the job posting: {str(e)}"
    except Exception as e:
        return f"Error processing the job posting: {str(e)}"

def extract_technical_terms(text):
    """Extract technical terms from text using NLP techniques."""
    # Tokenize and clean text
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    
    # Remove stopwords and punctuation
    tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    
    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Common technical skills and qualifications
    common_skills = {
        'communication', 'leadership', 'management', 'analysis', 'research',
        'project management', 'teamwork', 'problem solving', 'organization',
        'planning', 'coordination', 'development', 'implementation',
        'technical', 'professional', 'administrative', 'strategic',
        'analytical', 'critical thinking', 'data analysis', 'reporting',
        'documentation', 'training', 'mentoring', 'supervision',
        'budgeting', 'forecasting', 'evaluation', 'assessment',
        'collaboration', 'innovation', 'creativity', 'adaptability',
        'flexibility', 'initiative', 'independence', 'reliability',
        'attention to detail', 'time management', 'multitasking',
        'presentation', 'negotiation', 'decision making'
    }
    
    # Extract potential technical terms
    technical_terms = set()
    
    # Add single word terms
    for token in tokens:
        if len(token) > 2:  # Skip very short words
            technical_terms.add(token)
    
    # Add multi-word terms
    text_lower = text.lower()
    for skill in common_skills:
        if skill in text_lower:
            technical_terms.add(skill)
    
    return list(technical_terms)

def find_required_skills(text):
    """Find required skills from job description with importance weights."""
    # Define sections and their weights
    sections = {
        "CRITICAL KNOWLEDGE AND SKILLS": 1.0,
        "Required Qualifications": 1.0,
        "Preferred Qualifications": 0.7,
        "KEY RESPONSIBILITIES": 0.8,
        "EDUCATION and EXPERIENCE": 0.6
    }
    
    # Extract skills from each section
    skills = {}  # Dictionary to store skills and their weights
    text_lower = text.lower()
    
    # Common skill indicators and their weights
    skill_indicators = {
        "must have": 1.0,
        "required": 1.0,
        "critical": 1.0,
        "essential": 1.0,
        "knowledge of": 0.8,
        "experience with": 0.8,
        "proficiency in": 0.8,
        "ability to": 0.7,
        "skills in": 0.7,
        "expertise in": 0.7,
        "familiarity with": 0.6,
        "understanding of": 0.6,
        "competency in": 0.6,
        "background in": 0.6,
        "nice to have": 0.5,
        "preferred": 0.5
    }
    
    # Technical and professional skills with base weights
    common_skills = {
        # Software and Technical Skills
        'microsoft office': 0.8, 'excel': 0.8, 'word': 0.8, 'powerpoint': 0.8,
        'database': 0.8, 'gps': 0.8, 'fleet management': 0.9, 'data analysis': 0.9,
        'reporting': 0.8, 'spreadsheet': 0.8,
        
        # Business and Administrative Skills
        'administration': 0.8, 'management': 0.9, 'coordination': 0.8,
        'organization': 0.8, 'documentation': 0.7, 'communication': 0.9,
        'interpersonal': 0.8, 'leadership': 0.9, 'problem solving': 0.9,
        'critical thinking': 0.9, 'time management': 0.8, 'project management': 0.9,
        'customer service': 0.8, 'analysis': 0.8,
        
        # Specific Domain Knowledge
        'vehicle maintenance': 0.9, 'fleet operations': 0.9, 'transportation': 0.8,
        'fuel systems': 0.8, 'cost analysis': 0.8, 'inventory management': 0.8,
        'compliance': 0.9, 'regulations': 0.8, 'safety': 0.9, 'insurance': 0.8,
        
        # Soft Skills
        'teamwork': 0.8, 'collaboration': 0.8, 'attention to detail': 0.9,
        'multi-tasking': 0.7, 'verbal communication': 0.8, 'written communication': 0.8,
        'active listening': 0.7, 'independent work': 0.7
    }
    
    # Extract skills from text
    for section, section_weight in sections.items():
        section_pos = text.find(section)
        if section_pos != -1:
            # Get text after section header until next section or end
            next_section_pos = len(text)  # Default to end of text
            for next_section in sections:
                pos = text.find(next_section, section_pos + len(section))
                if pos != -1 and pos < next_section_pos:
                    next_section_pos = pos
            
            section_text = text[section_pos:next_section_pos].lower()
            
            # Look for skills after skill indicators
            for indicator, indicator_weight in skill_indicators.items():
                pos = 0
                while True:
                    pos = section_text.find(indicator, pos)
                    if pos == -1:
                        break
                    
                    # Extract the text after the indicator until the next punctuation
                    start = pos + len(indicator)
                    end = start
                    while end < len(section_text) and section_text[end] not in '.,:;':
                        end += 1
                    
                    skill = section_text[start:end].strip()
                    if skill:
                        # Calculate weight based on section and indicator
                        weight = section_weight * indicator_weight
                        # If skill already exists, take the higher weight
                        if skill in skills:
                            skills[skill] = max(skills[skill], weight)
                        else:
                            skills[skill] = weight
                    
                    pos = end
    
    # Add common skills if they appear in the text
    for skill, base_weight in common_skills.items():
        if skill in text_lower:
            # Count occurrences to adjust weight
            occurrences = text_lower.count(skill)
            weight = base_weight * (1 + 0.1 * min(occurrences - 1, 3))  # Cap at 4x base weight
            if skill in skills:
                skills[skill] = max(skills[skill], weight)
            else:
                skills[skill] = weight
    
    # Clean up skills
    cleaned_skills = {}
    for skill, weight in skills.items():
        # Remove common words and clean up the skill
        words = skill.split()
        words = [w for w in words if w not in {'a', 'an', 'the', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}]
        if words:
            cleaned_skill = ' '.join(words)
            if cleaned_skill in cleaned_skills:
                cleaned_skills[cleaned_skill] = max(cleaned_skills[cleaned_skill], weight)
            else:
                cleaned_skills[cleaned_skill] = weight
    
    return cleaned_skills

def similarity_score(a, b):
    return SequenceMatcher(None, a, b).ratio()

SKILL_SYNONYMS = {
    'python': ['py', 'python3'],
    'javascript': ['js', 'ecmascript'],
    'machine learning': ['ml', 'deep learning', 'ai'],
    'artificial intelligence': ['ai', 'machine learning'],
    'amazon web services': ['aws', 'amazon cloud'],
    'data science': ['data analysis', 'data analytics'],
    'web development': ['web dev', 'frontend', 'backend'],
    'software engineering': ['software development', 'programming'],
    'cloud computing': ['cloud', 'cloud services'],
    'database': ['db', 'databases'],
    'sql': ['mysql', 'postgresql', 'mongodb'],
    'react': ['reactjs', 'react.js'],
    'node.js': ['nodejs', 'node'],
    'git': ['github', 'gitlab'],
    'docker': ['container', 'containers'],
    'kubernetes': ['k8s', 'container orchestration'],
    'agile': ['scrum', 'kanban'],
    'rest api': ['rest', 'api', 'apis'],
    'microservices': ['microservice', 'microservice architecture'],
    'ci/cd': ['continuous integration', 'continuous deployment', 'devops']
}

def match_resume_skills(resume_skills, required_skills):
    """Match skills from resume against required skills from job posting with weights."""
    matched_skills = {}  # Dictionary to store matched skills and their weights
    missing_skills = {}  # Dictionary to store missing skills and their weights
    
    # Convert all skills to lowercase for comparison
    resume_skills = [skill.lower() for skill in resume_skills]
    
    # Direct matches and weighted scoring
    for skill, weight in required_skills.items():
        skill_lower = skill.lower()
        if skill_lower in resume_skills:
            matched_skills[skill] = weight
        else:
            # Check for partial matches
            found_match = False
            for resume_skill in resume_skills:
                if skill_lower in resume_skill or resume_skill in skill_lower:
                    # Partial match gets 80% of the weight
                    matched_skills[skill] = weight * 0.8
                    found_match = True
                    break
            
            # Check synonyms
            if not found_match and skill_lower in SKILL_SYNONYMS:
                for synonym in SKILL_SYNONYMS[skill_lower]:
                    if synonym in resume_skills:
                        # Synonym match gets 90% of the weight
                        matched_skills[skill] = weight * 0.9
                        found_match = True
                        break
            
            if not found_match:
                missing_skills[skill] = weight
    
    return matched_skills, missing_skills

def analyze_job_posting(url):
    """Analyze a job posting and extract required skills."""
    print(f"\nAnalyzing job posting from: {url}")
    job_description = scrape_job_posting(url)
    
    if job_description.startswith("Error"):
        print(job_description)
        return None
    
    required_skills = find_required_skills(job_description)
    print("\nRequired Skills (extracted from job posting):")
    for skill in required_skills:
        print(f"- {skill}")
    print(f"\nAmount of skills extracted: {len(required_skills)}")
    return required_skills

def analyze_resume(resume_path, job_description_path):
    """Analyze resume against job description."""
    print(f"\nAnalyzing resume: {os.path.basename(resume_path)}")
    
    # Read job description
    try:
        with open(job_description_path, 'r', encoding='utf-8') as file:
            job_description = file.read()
    except FileNotFoundError:
        print(f"Error: Job description file not found at {job_description_path}")
        return
    
    # Extract required skills from job description
    required_skills = find_required_skills(job_description)
    print("\nRequired Skills from Job Description (with importance weights):")
    for skill, weight in required_skills.items():
        print(f"- {skill} (Weight: {weight:.2f})")
    
    # Read and analyze resume
    text = None
    if resume_path.lower().endswith('.pdf'):
        text = read_pdf(resume_path)
    elif resume_path.lower().endswith('.docx'):
        text = read_docx(resume_path)
    
    if text:
        # Extract skills from resume
        resume_skills = extract_technical_terms(text)
        
        # Match skills
        matched_skills, missing_skills = match_resume_skills(resume_skills, required_skills)
        
        # Calculate weighted match percentage
        total_weight = sum(required_skills.values())
        matched_weight = sum(matched_skills.values())
        match_percentage = (matched_weight / total_weight * 100) if total_weight > 0 else 0
        
        # Display results
        print(f"\nSkill Match Score: {match_percentage:.1f}%")
        
        print("\nMatched Skills in Resume:")
        for skill, weight in matched_skills.items():
            print(f"✓ {skill} (Weight: {weight:.2f})")
        
        print("\nSkills Missing from Resume:")
        for skill, weight in missing_skills.items():
            print(f"✗ {skill} (Weight: {weight:.2f})")
    else:
        print("Error: Could not read resume file")

def extract_skill_context(text, skill):
    """Extract sentences containing the skill"""
    sentences = nltk.sent_tokenize(text.lower())
    skill_contexts = []
    
    for sentence in sentences:
        if skill in sentence:
            # Look for experience indicators
            experience_words = ['experience', 'worked', 'developed', 'built', 'created']
            has_experience = any(word in sentence for word in experience_words)
            skill_contexts.append({
                'skill': skill,
                'context': sentence,
                'indicates_experience': has_experience
            })
    
    return skill_contexts

SKILL_LEVELS = {
    'beginner': ['basic', 'familiar', 'learning'],
    'intermediate': ['proficient', 'experienced', 'worked with'],
    'expert': ['expert', 'advanced', 'mastery', 'led', 'architected']
}

def detect_skill_level(context):
    for level, indicators in SKILL_LEVELS.items():
        if any(indicator in context.lower() for indicator in indicators):
            return level
    return 'mentioned'

def main():
    # File paths
    resume_path = "C:/Users/epicg/Downloads/Mikey_Mubako_-_Part_time_Resume_updated.docx-220240722-23-ebn98120240722-23-447pch.pdf"
    job_description_path = "job_description.txt"
    
    # Analyze resume against job description
    analyze_resume(resume_path, job_description_path)

if __name__ == "__main__":
    main()

"""
Desired_Terminal_Output:
    (Print) Analyzing job posting from: https://www.linkedin.com/jobs/view/4193717139/?alternateChannel=search&refId=sY8swhpNxLDD7k6seEKuug%3D%3D&trackingId=py1Iv3LI9%2FD8PxzrnBpKhA%3D%3D
    (Print) Analyzing resume from: /Users/revantpatel/Downloads/TestResume.pdf
    (Print) Job and Resume Match Percentage: 100.0%
    (Print) Matched Skills in Resume:
    (Print) Skills Missing from Resume:
"""