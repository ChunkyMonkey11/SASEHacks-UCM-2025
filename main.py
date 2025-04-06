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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from skill_data import (
    SKILL_WEIGHTS, SECTION_WEIGHTS, SKILL_SYNONYMS,
    RELATED_TERMS, SKILL_PHRASES, SKILL_LEVELS,
    EXPERIENCE_INDICATORS, SKILL_CATEGORIES, CRITICAL_SKILLS
)

# Check if NLTK data is already downloaded
def initialize_nltk():
    """Initialize NLTK components required for text analysis."""
    try:
        # Only download if not already present
        nltk_data_path = os.path.expanduser('~/nltk_data')
        required_packages = ['punkt', 'averaged_perceptron_tagger', 'wordnet', 'stopwords']
        
        for package in required_packages:
            if not os.path.exists(os.path.join(nltk_data_path, package)):
                print(f"Downloading NLTK package: {package}")
                nltk.download(package, quiet=True)
        
        # Initialize stopwords
        global STOPWORDS
        STOPWORDS = set(stopwords.words('english'))
        
    except Exception as e:
        print(f"Error initializing NLTK: {str(e)}")
        raise

# Initialize NLTK when module is loaded
initialize_nltk()

# API KEY FOR OPENROUTER "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"
def calculate_similarity(text1, text2):
    """Calculate semantic similarity between two texts using TF-IDF."""
    # Create a custom tokenizer that preserves technical terms
    def custom_tokenizer(text):
        # First, extract technical terms from SKILL_WEIGHTS
        technical_terms = set()
        text_lower = text.lower()
        for term in SKILL_WEIGHTS.keys():
            if term in text_lower:
                technical_terms.add(term)
        
        # Then use NLTK tokenizer for the rest
        tokens = word_tokenize(text.lower())
        # Add technical terms back
        tokens.extend(list(technical_terms))
        return tokens
    
    # Create vectorizer with custom tokenizer and stop words
    vectorizer = TfidfVectorizer(
        tokenizer=custom_tokenizer,
        stop_words='english',
        ngram_range=(1, 2)  # Consider both single words and bigrams
    )
    
    # Transform texts
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    
    # Calculate cosine similarity
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    
    # Boost similarity if technical terms match
    technical_terms1 = set(term for term in SKILL_WEIGHTS.keys() if term in text1.lower())
    technical_terms2 = set(term for term in SKILL_WEIGHTS.keys() if term in text2.lower())
    technical_overlap = len(technical_terms1.intersection(technical_terms2))
    if technical_terms1 or technical_terms2:
        technical_boost = technical_overlap / max(len(technical_terms1), len(technical_terms2))
        similarity = 0.7 * similarity + 0.3 * technical_boost
    
    return similarity

def read_pdf(file_path):
    """Read text content from a PDF file."""
    try:
        if not os.path.exists(file_path):
            print(f"DEBUG: PDF file not found at {file_path}")
            return None
            
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            
            # Extract text from each page
            for page in pdf_reader.pages:
                # Get text with layout preservation
                text += page.extract_text() + "\n"
                
                # Try to extract text from any form fields
                if '/AcroForm' in page:
                    for field in page['/AcroForm']:
                        if field.get('/V'):
                            text += str(field['/V']) + "\n"
            
            # Clean up the text while preserving important formatting
            text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
            text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters
            text = text.strip()
            
            print(f"DEBUG: Successfully read PDF. Text length: {len(text)} characters")
            print(f"DEBUG: First 500 characters of extracted text: {text[:500]}")
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
    
    # Remove stopwords and punctuation
    tokens = [token for token in tokens if token.isalnum() and token not in STOPWORDS]
    
    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Extract potential technical terms
    technical_terms = set()
    
    # Add single word terms
    for token in tokens:
        if len(token) > 2:  # Skip very short words
            technical_terms.add(token)
    
    # Add multi-word terms
    text_lower = text.lower()
    for skill, weight in SKILL_WEIGHTS.items():
        if skill in text_lower:
            technical_terms.add(skill)
    
    return list(technical_terms)

def find_required_skills(text):
    """Find required skills from job description with importance weights."""
    skills = {}  # Dictionary to store skills and their weights
    text_lower = text.lower()
    
    # First pass: Identify all skills mentioned in the text
    all_skills = set()
    for skill in SKILL_WEIGHTS.keys():
        if skill in text_lower:
            all_skills.add(skill)
    
    # Calculate base weights based on frequency and context
    for skill in all_skills:
        # Count occurrences
        occurrences = text_lower.count(skill)
        
        # Check if skill is in required/preferred sections
        section_weight = 1.0
        for section, weight in SECTION_WEIGHTS.items():
            if section.lower() in text_lower:
                section_pos = text_lower.find(section.lower())
                section_end = len(text_lower)
                # Find end of section
                for next_section in SECTION_WEIGHTS:
                    if next_section.lower() in text_lower[section_pos + len(section):]:
                        pos = text_lower.find(next_section.lower(), section_pos + len(section))
                        if pos < section_end:
                            section_end = pos
                
                if skill in text_lower[section_pos:section_end]:
                    section_weight = weight
                    break
        
        # Check for emphasis words near the skill
        emphasis_words = ['required', 'must have', 'essential', 'necessary', 'expert', 'proficient', 'strong']
        emphasis_count = 0
        for word in emphasis_words:
            if word in text_lower:
                # Look for emphasis words near the skill
                skill_pos = text_lower.find(skill)
                context_start = max(0, skill_pos - 50)
                context_end = min(len(text_lower), skill_pos + 50)
                context = text_lower[context_start:context_end]
                if word in context:
                    emphasis_count += 1
        
        # Calculate final weight
        base_weight = 0.7  # Base weight for any mentioned skill
        frequency_bonus = min(0.3, occurrences * 0.1)  # Up to 0.3 bonus for frequency
        emphasis_bonus = min(0.3, emphasis_count * 0.1)  # Up to 0.3 bonus for emphasis
        final_weight = base_weight + frequency_bonus + emphasis_bonus
        final_weight *= section_weight
        
        skills[skill] = final_weight
    
    # Add related skills based on context
    for skill, related in RELATED_TERMS.items():
        if skill in text_lower and skill not in skills:
            # If main skill is mentioned, add related skills with slightly lower weight
            for related_skill in related:
                if related_skill in text_lower and related_skill not in skills:
                    skills[related_skill] = skills[skill] * 0.8
    
    # Clean up skills
    cleaned_skills = {}
    for skill, weight in skills.items():
        # Remove common words and clean up the skill
        words = skill.split()
        words = [w for w in words if w not in {'a', 'an', 'the', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'e', 'g'}]
        if words:
            cleaned_skill = ' '.join(words)
            if cleaned_skill in cleaned_skills:
                cleaned_skills[cleaned_skill] = max(cleaned_skills[cleaned_skill], weight)
            else:
                cleaned_skills[cleaned_skill] = weight
    
    return cleaned_skills

def similarity_score(a, b):
    return SequenceMatcher(None, a, b).ratio()

def categorize_missing_skills(missing_skills):
    """Categorize missing skills and calculate category importance."""
    categorized_skills = {}
    category_importance = {}
    
    # Initialize categories
    for category in SKILL_CATEGORIES:
        categorized_skills[category] = []
        category_importance[category] = 0
    
    # Categorize skills and calculate importance
    for skill, weight in missing_skills.items():
        skill_lower = skill.lower()
        categorized = False
        
        # Check if it's a critical skill
        if skill_lower in CRITICAL_SKILLS:
            weight *= 1.5  # Boost weight for critical skills
        
        # Categorize based on keywords
        for category, info in SKILL_CATEGORIES.items():
            if any(keyword in skill_lower for keyword in info['keywords']):
                categorized_skills[category].append({
                    'skill': skill,
                    'weight': weight,
                    'is_critical': skill_lower in CRITICAL_SKILLS
                })
                category_importance[category] += weight * info['weight']
                categorized = True
                break
        
        # If not categorized, put in most relevant category based on related terms
        if not categorized:
            for category, info in SKILL_CATEGORIES.items():
                if any(keyword in skill_lower for keyword in info['keywords']):
                    categorized_skills[category].append({
                        'skill': skill,
                        'weight': weight,
                        'is_critical': skill_lower in CRITICAL_SKILLS
                    })
                    category_importance[category] += weight * info['weight']
                    break
    
    # Sort skills within each category by weight
    for category in categorized_skills:
        categorized_skills[category].sort(key=lambda x: x['weight'], reverse=True)
    
    return categorized_skills, category_importance

def match_resume_skills(resume_skills, required_skills):
    """Match skills from resume against required skills from job posting with weights."""
    matched_skills = {}  # Dictionary to store matched skills and their weights
    missing_skills = {}  # Dictionary to store missing skills and their weights
    skill_feedback = {}  # Dictionary to store feedback for each skill
    
    # Convert all skills to lowercase for comparison
    resume_skills = [skill.lower() for skill in resume_skills]
    
    # Direct matches and weighted scoring
    for skill, weight in required_skills.items():
        skill_lower = skill.lower()
        found_match = False
        best_match = None
        best_match_score = 0
        
        # Direct match
        if skill_lower in resume_skills:
            matched_skills[skill] = weight
            found_match = True
            skill_feedback[skill] = "Strong match - Skill directly found in resume"
        else:
            # Check for partial matches with strict threshold
            for resume_skill in resume_skills:
                # Calculate similarity score
                similarity = similarity_score(skill_lower, resume_skill)
                
                # Check if either skill contains the other
                if skill_lower in resume_skill or resume_skill in skill_lower:
                    match_score = 0.9  # High but not perfect match for containing skills
                    feedback = f"Partial match - Found related skill: {resume_skill}"
                # Check for similar terms using word overlap
                else:
                    skill_words = set(skill_lower.split())
                    resume_words = set(resume_skill.split())
                    overlap = len(skill_words.intersection(resume_words))
                    match_score = min(0.7, overlap / len(skill_words))
                    feedback = f"Partial match - Found related skill: {resume_skill}"
                
                # Combine similarity and match score with adjusted weights
                combined_score = (similarity * 0.8 + match_score * 0.2)  # Favor exact matches
                
                if combined_score > best_match_score:
                    best_match_score = combined_score
                    best_match = (resume_skill, combined_score, feedback)
            
            # Check synonyms with strict matching
            if not found_match and skill_lower in SKILL_SYNONYMS:
                for synonym in SKILL_SYNONYMS[skill_lower]:
                    if synonym in resume_skills:
                        matched_skills[skill] = weight * 0.9
                        found_match = True
                        skill_feedback[skill] = f"Match through synonym - Found: {synonym}"
                        break
            
            # Check for related skills with context
            if not found_match:
                for term, related in RELATED_TERMS.items():
                    if skill_lower == term:
                        # Check if any related skills are present
                        related_matches = [r for r in related if r in resume_skills]
                        if related_matches:
                            # Use the best matching related skill
                            best_related = max(related_matches, key=lambda x: similarity_score(skill_lower, x))
                            matched_skills[skill] = weight * 0.8
                            found_match = True
                            skill_feedback[skill] = f"Match through related skill - Found: {best_related}"
                    break
            
            # Use best match if found with strict threshold
            if best_match and best_match[1] > 0.7:  # Increased threshold for partial matches
                matched_skills[skill] = weight * best_match[1]
                found_match = True
                skill_feedback[skill] = best_match[2]
            
            if not found_match:
                missing_skills[skill] = weight
                # Provide specific feedback for missing skills
                if skill_lower in CRITICAL_SKILLS:
                    skill_feedback[skill] = "Critical skill missing - Consider adding this to your resume"
                else:
                    skill_feedback[skill] = "Skill not found in resume - Consider adding if you have this experience"
    
    # Categorize missing skills
    categorized_missing, category_importance = categorize_missing_skills(missing_skills)
    
    # Add feedback to categorized missing skills
    for category in categorized_missing:
        for skill_info in categorized_missing[category]:
            skill_info['feedback'] = skill_feedback.get(skill_info['skill'], "Skill not found in resume")
    
    return matched_skills, categorized_missing, category_importance

def analyze_job_posting(url):
    """Analyze a job posting and extract required skills."""
    print(f"\nAnalyzing job posting from: {url}")
    job_description = scrape_job_posting(url)
    
    if job_description.startswith("Error"):
        print(job_description)
        return None
    
    required_skills = find_required_skills(job_description)
    print("\nRequired Skills (extracted from job posting):")
    for skill, weight in required_skills.items():
        print(f"- {skill} (Weight: {weight:.2f})")
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
    
    # Read and analyze resume
    text = None
    if resume_path.lower().endswith('.pdf'):
        text = read_pdf(resume_path)
    elif resume_path.lower().endswith('.docx'):
        text = read_docx(resume_path)
    
    if text:
        # Calculate overall semantic similarity
        semantic_similarity = calculate_similarity(text, job_description)
        print(f"\nOverall Semantic Similarity: {semantic_similarity * 100:.1f}%")
        
        # Extract required skills from job description
        required_skills = find_required_skills(job_description)
        print("\nRequired Skills from Job Description (with importance weights):")
        for skill, weight in required_skills.items():
            print(f"- {skill} (Weight: {weight:.2f})")
        
        # Extract skills from resume
        resume_skills = extract_technical_terms(text)
        
        # Match skills using both traditional and semantic matching
        matched_skills, categorized_missing, category_importance = match_resume_skills(resume_skills, required_skills)
        
        # Calculate final match percentage
        total_weight = sum(required_skills.values())
        matched_weight = sum(matched_skills.values())
        semantic_boost = semantic_similarity * 0.2  # Add a small boost based on overall semantic similarity
        match_percentage = ((matched_weight / total_weight) * 0.8 + semantic_boost) * 100
        
        # Display results with detailed feedback
        print(f"\nOverall Match Score: {match_percentage:.1f}%")
        
        # Analyze strengths
        print("\nYour Strengths:")
        critical_matches = [skill for skill in matched_skills.keys() if skill.lower() in CRITICAL_SKILLS]
        if critical_matches:
            print("✓ Strong match with critical required skills:")
            for skill in critical_matches:
                print(f"  - {skill} (Weight: {matched_skills[skill]:.2f})")
        
        # Analyze areas for improvement
        print("\nAreas for Improvement:")
        critical_missing = []
        for category, skills in categorized_missing.items():
            for skill_info in skills:
                if skill_info['is_critical']:
                    critical_missing.append((category, skill_info))
        
        if critical_missing:
            print("⚠️ Critical skills missing from your resume:")
            for category, skill_info in critical_missing:
                print(f"  - {skill_info['skill']} (Category: {category})")
                print(f"    Feedback: {skill_info['feedback']}")
        
        # Provide category-specific feedback
        print("\nCategory Analysis:")
        for category, importance in category_importance.items():
            if importance > 0:
                print(f"\n{category} (Importance: {importance:.2f}):")
                category_skills = categorized_missing.get(category, [])
                if category_skills:
                    print("  Missing skills:")
                    for skill_info in category_skills:
                        print(f"  - {skill_info['skill']}")
                        print(f"    Feedback: {skill_info['feedback']}")
                else:
                    print("  ✓ All required skills in this category are present")
        
        # Provide actionable recommendations
        print("\nRecommendations:")
        if match_percentage < 70:
            print("1. Consider adding more details about your experience with:")
            for category, skills in categorized_missing.items():
                if skills:
                    print(f"   - {category}: {', '.join(skill_info['skill'] for skill_info in skills[:3])}")
        
        if semantic_similarity < 0.3:
            print("2. Your resume's content could better align with the job description:")
            print("   - Review the job description's key terms and concepts")
            print("   - Add more relevant technical details to your experience")
            print("   - Include specific examples of your achievements")
        
        if critical_missing:
            print("3. Prioritize adding these critical missing skills:")
            for category, skill_info in critical_missing:
                print(f"   - {skill_info['skill']}")
        
        print("\nNext Steps:")
        print("1. Review the missing skills and add relevant experience if available")
        print("2. Strengthen your resume by adding specific examples and achievements")
        print("3. Consider adding a skills section that highlights your technical expertise")
        print("4. Use industry-standard terminology when describing your experience")
    else:
        print("Error: Could not read resume file")

def extract_skill_context(text, skill):
    """Extract sentences containing the skill"""
    sentences = nltk.sent_tokenize(text.lower())
    skill_contexts = []
    
    for sentence in sentences:
        if skill in sentence:
            # Look for experience indicators
            has_experience = any(word in sentence for word in EXPERIENCE_INDICATORS)
            skill_contexts.append({
                'skill': skill,
                'context': sentence,
                'indicates_experience': has_experience
            })
    
    return skill_contexts

def detect_skill_level(context):
    for level, indicators in SKILL_LEVELS.items():
        if any(indicator in context.lower() for indicator in indicators):
            return level
    return 'mentioned'

def identify_sections(text):
    """Identify different sections in the resume."""
    sections = {}
    lines = text.split('\n')
    current_section = None
    
    # Common section headers in resumes
    RESUME_SECTIONS = {
        'experience': ['experience', 'work experience', 'employment', 'work history', 'professional experience'],
        'education': ['education', 'academic background', 'qualifications', 'academic qualifications'],
        'skills': ['skills', 'technical skills', 'core competencies', 'expertise', 'technologies'],
        'projects': ['projects', 'personal projects', 'portfolio', 'technical projects'],
        'certifications': ['certifications', 'certificates', 'professional certifications']
    }
    
    # Process each line
    section_content = []
    for line in lines:
        line = line.strip().lower()
        if not line:
            continue
            
        # Check if line is a section header
        found_section = None
        for section_type, keywords in RESUME_SECTIONS.items():
            if any(keyword in line for keyword in keywords):
                found_section = section_type
                break
                
        if found_section:
            # Save previous section content
            if current_section:
                sections[current_section] = section_content
            # Start new section
            current_section = found_section
            section_content = []
        elif current_section:
            section_content.append(line)
    
    # Save last section
    if current_section and section_content:
        sections[current_section] = section_content
    
    return sections

def analyze_job_and_resume(job_description, resume_text):
    """Analyze job description and resume, return matching results with detailed recommendations."""
    # Get required skills from job description
    required_skills = find_required_skills(job_description)
    
    # Extract skills from resume
    resume_skills = extract_technical_terms(resume_text)
    
    # Match skills
    matched_skills, missing_skills, category_importance = match_resume_skills(resume_skills, required_skills)
    
    # Calculate match percentage
    total_weight = sum(required_skills.values())
    matched_weight = sum(matched_skills.values())
    match_percentage = (matched_weight / total_weight * 100) if total_weight > 0 else 0
    
    # Calculate semantic similarity for content match
    semantic_similarity = calculate_similarity(resume_text, job_description) * 100
    
    # Identify resume sections
    sections = identify_sections(resume_text)
    
    # Generate detailed recommendations
    recommendations = []
    
    # Define skill categories with more specific groupings
    skill_categories = {
        'cloud_devops': {
            'skills': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'terraform', 'ci/cd', 'devops'],
            'context': 'infrastructure and deployment automation',
            'project_ideas': [
                'Set up a CI/CD pipeline using Jenkins and Docker',
                'Deploy a microservices application on Kubernetes',
                'Create infrastructure as code using Terraform'
            ]
        },
        'software_development': {
            'skills': ['python', 'java', 'javascript', 'c++', 'typescript', 'react', 'angular', 'node'],
            'context': 'software development and programming',
            'project_ideas': [
                'Build a full-stack web application',
                'Develop a RESTful API service',
                'Create a responsive front-end interface'
            ]
        },
        'databases': {
            'skills': ['mysql', 'postgresql', 'mongodb', 'redis', 'sql', 'nosql'],
            'context': 'database management and data storage',
            'project_ideas': [
                'Design and implement a database schema',
                'Build a data pipeline',
                'Create a caching layer with Redis'
            ]
        },
        'testing_qa': {
            'skills': ['unit testing', 'integration testing', 'test automation', 'selenium', 'jest', 'pytest'],
            'context': 'quality assurance and testing',
            'project_ideas': [
                'Implement automated test suites',
                'Set up end-to-end testing',
                'Create test documentation'
            ]
        }
    }
    
    # Analyze missing skills by category
    for category_name, category_info in skill_categories.items():
        missing_in_category = []
        for skill in missing_skills:
            if any(tech in skill.lower() for tech in category_info['skills']):
                missing_in_category.append(skill)
        
        if missing_in_category:
            # Experience section recommendations
            if 'experience' in sections:
                recommendations.append({
                    'section': 'Experience',
                    'suggestion': f"Highlight your experience with {category_info['context']} using {', '.join(missing_in_category)}",
                    'skills': missing_in_category,
                    'examples': [
                        f"• Implemented {missing_in_category[0]} to improve {category_info['context']}",
                        f"• Developed solutions using {', '.join(missing_in_category[:2])}",
                        f"• Collaborated on projects involving {category_info['context']}"
                    ]
                })
            
            # Skills section recommendations
            if 'skills' in sections:
                recommendations.append({
                    'section': 'Skills',
                    'suggestion': f"Add these {category_name.replace('_', ' ')} skills to your technical skills section",
                    'skills': missing_in_category,
                    'examples': [
                        f"• List skills under a '{category_name.replace('_', ' ').title()}' subsection",
                        f"• Include proficiency levels (e.g., Proficient in {missing_in_category[0]})",
                        "• Group related technologies together"
                    ]
                })
            
            # Projects section recommendations
            if 'projects' in sections or len(missing_in_category) > 2:
                recommendations.append({
                    'section': 'Projects',
                    'suggestion': f"Create projects showcasing {category_info['context']} skills",
                    'skills': missing_in_category,
                    'examples': category_info['project_ideas']
                })
    
    # Add general recommendations if match percentage is low
    if match_percentage < 70:
        recommendations.append({
            'section': 'General',
            'suggestion': "Improve overall resume alignment with the job description",
            'skills': list(missing_skills.keys())[:3],
            'examples': [
                "• Use industry-standard terminology from the job description",
                "• Quantify your achievements with metrics and numbers",
                "• Focus on relevant experience and projects"
            ]
        })
    
    # Add keyword optimization if semantic similarity is low
    if semantic_similarity < 60:
        recommendations.append({
            'section': 'Keywords',
            'suggestion': "Optimize your resume with relevant keywords",
            'skills': list(required_skills.keys())[:5],
            'examples': [
                "• Include key terms from the job description naturally in your bullet points",
                "• Use similar language and terminology as the job posting",
                "• Highlight technologies and tools mentioned in the requirements"
            ]
        })
    
    return {
        'match_percentage': round(match_percentage, 1),
        'semantic_similarity': round(semantic_similarity, 1),
        'matched_skills': matched_skills,
        'missing_skills': missing_skills,
        'recommendations': recommendations
    }

def get_chat_response(message, resume_data):
    """Get response from OpenRouter API."""
    try:
        print(f"\nProcessing chat message: {message}")
        print(f"Resume data available: {bool(resume_data)}")
        
        headers = {
            'Authorization': f'Bearer {os.getenv("OPENROUTER_API_KEY")}',
            'HTTP-Referer': 'http://localhost:8080',
            'Content-Type': 'application/json'
        }
        
        # Create system message with resume context
        system_message = f"""You are an AI resume assistant analyzing the following resume data:
        Match Percentage: {resume_data.get('match_percentage')}%
        Matched Skills: {', '.join(resume_data.get('matched_skills', {}).keys())}
        Missing Skills: {resume_data.get('missing_skills', {})}
        
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
            'https://openrouter.ai/api/v1/chat/completions',
            headers=headers,
            json=payload
        )
        
        print(f"API Response status: {response.status_code}")
        
        response_data = response.json()
        if 'choices' not in response_data:
            print(f"Unexpected API response: {response_data}")
            raise ValueError("Invalid API response format")
            
        return response_data['choices'][0]['message']['content']
        
    except Exception as e:
        print(f"Error in get_chat_response: {str(e)}")
        raise

def main():
    # File paths
    resume_path = r"C:\Users\Revant\SASEHacks-UCM-2025\Revant_Patel_Resume.pdf"
    job_description_path = r"C:\Users\Revant\SASEHacks-UCM-2025\job_description.txt"
    
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