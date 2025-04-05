import nltk
# import spacy
import PyPDF2
from docx import Document
from sklearn import *
import os

import PyPDF2
import os

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
            reader = PyPDF2.PdfReader(file)
            
            # Get number of pages
            num_pages = len(reader.pages)
            print(f"Number of pages: {num_pages}")
            
            # Extract text from all pages
            text = ""
            for page_num in range(num_pages):
                page = reader.pages[page_num]
                text += page.extract_text()
                
            return text
            
    except Exception as e:
        print(f"Error reading PDF: {str(e)}")
        return None

def main():
    # Example usage
    # Replace this path with your PDF file path
    pdf_path = "/Users/revantpatel/Downloads/TestResume.pdf"
    
    # Read the PDF
    text = read_pdf(pdf_path)
    
    if text:
        print("\nPDF Content:")
        print("------------")
        print(text)

if __name__ == "__main__":
    main()