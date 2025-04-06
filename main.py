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
    # nltk.download('punkt')
    # nltk.download('stopwords')
    # nltk.download('averaged_perceptron_tagger')

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
    if not text:
        print("DEBUG: No text provided to extract_technical_terms")
        return []
    
    # Tokenize and clean the text
    tokens = word_tokenize(text.lower())
    print(f"DEBUG: Initial token count: {len(tokens)}")
    
    stop_words = set(stopwords.words('english'))
    
    # Remove stopwords and non-alphabetic tokens
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    print(f"DEBUG: Token count after removing stopwords: {len(tokens)}")
    
    # Create a set to store both original and lemmatized forms
    technical_terms = set()
    lemmatizer = WordNetLemmatizer()
    
    # Add both original and lemmatized forms
    for token in tokens:
        technical_terms.add(token)  # Add original form
        lemmatized = lemmatizer.lemmatize(token)
        if lemmatized != token:  # Only add lemmatized form if it's different
            technical_terms.add(lemmatized)
    
    print(f"DEBUG: Unique technical terms before POS tagging: {len(technical_terms)}")
    
    # Get parts of speech
    pos_tags = nltk.pos_tag(list(technical_terms))
    
    # Extract potential technical terms
    final_terms = []
    
    # Look for noun phrases and technical terms
    for i, (word, tag) in enumerate(pos_tags):
        # Single technical terms (nouns and adjectives)
        if tag.startswith('NN') or tag.startswith('JJ'):
            final_terms.append(word)
        
        # Compound technical terms (e.g., "machine learning")
        if i < len(pos_tags) - 1:
            next_word, next_tag = pos_tags[i + 1]
            if (tag.startswith('NN') or tag.startswith('JJ')) and (next_tag.startswith('NN') or next_tag.startswith('JJ')):
                final_terms.append(f"{word} {next_word}")
    
    print(f"DEBUG: Final technical terms count: {len(final_terms)}")
    print(f"DEBUG: Sample of technical terms: {final_terms[:5]}")
    return final_terms

def find_required_skills(job_description):
    """Extract required skills from job description."""
    print(f"DEBUG: Job description length: {len(job_description)} characters")
    
    # Extract technical terms
    technical_terms = extract_technical_terms(job_description)
    
    # Count frequency of terms
    term_frequency = Counter(technical_terms)
    
    # Filter and sort by frequency - now including terms that appear at least once
    required_skills = [(term, freq) for term, freq in term_frequency.items() if freq >= 1]
    required_skills.sort(key=lambda x: x[1], reverse=True)
    
    print(f"DEBUG: Required skills count: {len(required_skills)}")
    print(f"DEBUG: Required skills with frequencies: {required_skills[:5]}")
    
    return [skill for skill, _ in required_skills]

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

def match_resume_skills(resume_text, required_skills):
    """Match resume skills against required skills from job posting."""
    if not resume_text:
        print("DEBUG: No resume text provided")
        return []
    
    print(f"DEBUG: Resume text length: {len(resume_text)} characters")
    
    # Extract technical terms from resume
    resume_terms = extract_technical_terms(resume_text)
    
    print(f"DEBUG: Resume technical terms count: {len(resume_terms)}")
    print(f"DEBUG: Sample of resume technical terms: {resume_terms[:5]}")
    
    # Find matches
    matched_skills = []
    for skill in required_skills:
        # Direct match
        if skill in resume_terms:
            matched_skills.append(skill)
            print(f"DEBUG: Direct match found: {skill}")
            continue
            
        # Synonym match
        if skill in SKILL_SYNONYMS:
            for synonym in SKILL_SYNONYMS[skill]:
                if synonym in resume_terms:
                    matched_skills.append(skill)
                    print(f"DEBUG: Synonym match found: {skill} (synonym: {synonym})")
                    break
        # Partial match (e.g., "python" in "python programming")
        else:
            for term in resume_terms:
                if skill in term or term in skill:
                    matched_skills.append(skill)
                    print(f"DEBUG: Partial match found: {skill} (in term: {term})")
                    break
    
    print(f"DEBUG: Total matched skills: {len(matched_skills)}")
    print(f"DEBUG: Matched skills: {matched_skills}")
    return matched_skills

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

def analyze_resume(resume_path, required_skills):
    """Analyze a resume against required skills from job posting."""
    # Determine file type and read content
    if resume_path.lower().endswith('.pdf'):
        text = read_pdf(resume_path)
    elif resume_path.lower().endswith('.docx'):
        text = read_docx(resume_path)
    else:
        print("Unsupported file format. Please use PDF or DOCX files.")
        return
    
    if text:
        matched_skills = match_resume_skills(text, required_skills)
        print("\nMatched Skills in Resume:")
        for skill in matched_skills:
            print(f"- {skill}")
        
        # Calculate match percentage
        if required_skills:
            match_percentage = (len(matched_skills) / len(required_skills)) * 100
            print(f"\nMatch Percentage: {match_percentage:.1f}%")
    else:
        print("Failed to extract text from the resume.")

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
    # Example usage
    job_url = "https://app.joinhandshake.com/stu/postings"  # Replace with actual job posting URL
    resume_path = "/Users/revantpatel/Downloads/TestResume.pdf"  # Replace with actual resume path
    
    # First analyze the job posting to get required skills
    required_skills = analyze_job_posting(job_url)
    
    if required_skills:
        # Then analyze the resume against these skills
        analyze_resume(resume_path, required_skills)

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