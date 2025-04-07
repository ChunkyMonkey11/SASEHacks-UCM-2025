# SASEHacks-UCM-2025
This repo contains Revant Patel's and Mikey Mubako's hackathon submission. 

## Project Overview
A tool that takes a resume and a job posting as input, calculates how well they match based on keywords, job titles, and skills, and outputs:
- A percentage match
- Top 3 strengths
- Top 3 weaknesses
- Detailed recommendations for improvement
- Interactive chat interface for resume guidance

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/SASEHacks-UCM-2025.git
cd SASEHacks-UCM-2025
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Set up OpenRouter API Key:
   - Create an account at [OpenRouter](https://openrouter.ai/)
   - Get your API key from the dashboard
   - Create a `.env` file in the project root
   - Add your API key to the `.env` file:
     ```
     OPENROUTER_API_KEY=your_api_key_here
     ```

### Running the Application
1. Start the server:
```bash
python server_manager.py
```

2. Open your browser and navigate to:
```
http://localhost:8080
```

## Features
- Resume and job description analysis
- Skill matching and gap analysis
- Detailed recommendations for improvement
- Interactive chat interface for resume guidance
- Real-time feedback and suggestions

## Requirements
```
PyPDF2==3.0.1
nltk==3.8.1
python-dotenv==1.0.0
requests==2.31.0
```

## Note
Make sure to keep your `.env` file secure and never commit it to version control. The `.env` file is already included in `.gitignore` to prevent accidental commits.

