import nltk
import spacy
import PyPDF2
from docx import Document
from sklearn import *

# Function to extract text from a PDF file
# extract_pdf_text() -> 
def extract_pdf_text(file_path):
    with open(file_path, 'rb') as file:# file is a variable to the file object.
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

print(extract_pdf_text(""))