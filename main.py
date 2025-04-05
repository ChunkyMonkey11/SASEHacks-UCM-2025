import PyPDF2
import docx
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import os

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def read_pdf(file_path):
    """
    Reads a PDF file and returns its text content.
    
    Args:
        file_path (str): Path to the PDF file
        
    Returns:
        str: Text content of the PDF
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"Error: File '{file_path}' does not exist")
            return None
            
        # Open and read the PDF
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Get number of pages
            num_pages = len(pdf_reader.pages)
            print(f"Number of pages: {num_pages}")
            
            # Extract text from all pages
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
                
            return text
            
    except Exception as e:
        print(f"Error reading PDF: {str(e)}")
        return None

def read_docx(file_path):
    """
    Reads a DOCX file and returns its text content.
    
    Args:
        file_path (str): Path to the DOCX file
        
    Returns:
        str: Text content of the DOCX
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"Error: File '{file_path}' does not exist")
            return None
            
        # Open and read the DOCX
        doc = docx.Document(file_path)
        
        # Extract text from all paragraphs
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
            
        return text
            
    except Exception as e:
        print(f"Error reading DOCX: {str(e)}")
        return None

# Extract skills from text using NLTK
def extract_skills(text):
    """Extract skills from text using NLTK."""
    if not text:
        return []
    
    # Tokenize the text
    tokens = word_tokenize(text.lower())
    
    # Remove stopwords and non-alphabetic tokens
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    
    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Common technical skills to look for
    common_skills = {
        'python', 'java', 'javascript', 'html', 'css', 'sql', 'git', 'linux',
        'aws', 'docker', 'kubernetes', 'react', 'angular', 'node.js', 'django',
        'flask', 'spring', 'hibernate', 'machine learning', 'data science',
        'agile', 'scrum', 'jira', 'confluence', 'excel', 'powerpoint', 'word'
    }
    
    # Find matches between tokens and common skills
    found_skills = set()
    for token in tokens:
        if token in common_skills:
            found_skills.add(token)
    
    return list(found_skills)

def analyze_resume(resume_path):
    """Analyze a resume and extract relevant information."""
    # Determine file type and read content
    if resume_path.lower().endswith('.pdf'):
        text = read_pdf(resume_path)
    elif resume_path.lower().endswith('.docx'):
        text = read_docx(resume_path)
    else:
        print("Unsupported file format. Please use PDF or DOCX files.")
        return
    
    if text:
        # Extract skills
        skills = extract_skills(text)
        print("\nExtracted Skills:")
        for skill in skills:
            print(f"- {skill}")
    else:
        print("Failed to extract text from the resume.")

def main():
    # Example usage for PDF
    pdf_path = "/Users/revantpatel/Downloads/TestResume.pdf"
    print("\nReading PDF file:")
    print("----------------")
    pdf_text = read_pdf(pdf_path)
    if pdf_text:
        print("\nPDF Content:")
        print("------------")
        print(pdf_text)
    
    # # Example usage for DOCX
    # docx_path = "/Users/revantpatel/Downloads/TestResume.docx"  # Update this path to your DOCX file
    # print("\nReading DOCX file:")
    # print("-----------------")
    # docx_text = read_docx(docx_path)
    # if docx_text:
    #     print("\nDOCX Content:")
    #     print("-------------")
    #     print(docx_text)

    # Example usage
    resume_path = "path/to/your/resume.pdf"  # Replace with actual resume path
    analyze_resume(resume_path)

if __name__ == "__main__":
    main()