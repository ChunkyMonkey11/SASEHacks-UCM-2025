"""
This module contains all skill-related dictionaries and constants used for resume analysis.
"""

# Base weights for different types of skills
SKILL_WEIGHTS = {
    # Programming Languages (0.9)
    'python': 0.9, 'javascript': 0.9, 'java': 0.9, 'c++': 0.9, 'c#': 0.9,
    'php': 0.8, 'ruby': 0.8, 'swift': 0.8, 'kotlin': 0.8, 'go': 0.8,
    
    # Web Development (0.8-0.9)
    'html': 0.8, 'css': 0.8, 'react': 0.9, 'angular': 0.8, 'vue': 0.8,
    'node.js': 0.9, 'express': 0.8, 'django': 0.8, 'flask': 0.8,
    'frontend': 0.8, 'front end': 0.8, 'front-end': 0.8,
    'backend': 0.8, 'back end': 0.8, 'back-end': 0.8,
    
    # Databases (0.7-0.9)
    'sql': 0.9, 'mysql': 0.8, 'postgresql': 0.8, 'mongodb': 0.8,
    'redis': 0.7, 'oracle': 0.8, 'database': 0.8,
    
    # Tools and Technologies (0.7-0.9)
    'git': 0.9, 'github': 0.8, 'docker': 0.8, 'kubernetes': 0.8,
    'aws': 0.9, 'azure': 0.8, 'gcp': 0.8, 'jenkins': 0.7,
    
    # Software Development Concepts (0.7-0.9)
    'agile': 0.8, 'scrum': 0.7, 'kanban': 0.7, 'ci/cd': 0.8,
    'microservices': 0.8, 'rest api': 0.9, 'graphql': 0.7,
    'debugging': 0.9, 'testing': 0.8, 'code review': 0.8,
    'version control': 0.8, 'source control': 0.8,
    
    # General Skills (0.7-0.9)
    'problem solving': 0.9, 'problem-solving': 0.9,
    'teamwork': 0.8, 'team work': 0.8, 'team-work': 0.8,
    'communication': 0.8, 'collaboration': 0.8,
    'analytical': 0.8, 'analysis': 0.8,
    'development': 0.8, 'software development': 0.9,
    'programming': 0.9, 'coding': 0.9,
    'software engineering': 0.9, 'engineering': 0.8,
    
    # Additional Soft Skills
    'leadership': 0.9, 'management': 0.8, 'project management': 0.9,
    'organization': 0.8, 'planning': 0.8, 'coordination': 0.8,
    'implementation': 0.8, 'technical': 0.9, 'professional': 0.8,
    'administrative': 0.7, 'strategic': 0.8, 'critical thinking': 0.9,
    'data analysis': 0.8, 'reporting': 0.7, 'documentation': 0.8,
    'training': 0.7, 'mentoring': 0.7, 'supervision': 0.7,
    'budgeting': 0.7, 'forecasting': 0.7, 'evaluation': 0.7,
    'assessment': 0.7, 'innovation': 0.8, 'creativity': 0.8,
    'adaptability': 0.8, 'flexibility': 0.7, 'initiative': 0.7,
    'independence': 0.7, 'reliability': 0.7, 'attention to detail': 0.9,
    'time management': 0.8, 'multitasking': 0.7, 'presentation': 0.7,
    'negotiation': 0.7, 'decision making': 0.8
}

# Section weights for job descriptions
SECTION_WEIGHTS = {
    "Requirements": 1.0,
    "Required Skills": 1.0,
    "Required Qualifications": 1.0,
    "Preferred Skills": 0.8,
    "Preferred Skills/Qualifications": 0.8,
    "Preferred Qualifications": 0.8,
    "Responsibilities": 0.7,
    "Nice to Have": 0.6
}

# Skill synonyms for matching variations
SKILL_SYNONYMS = {
    # Programming Languages
    'python': ['py', 'python3', 'python programming'],
    'javascript': ['js', 'ecmascript', 'javascript programming'],
    'java': ['java programming', 'java development'],
    'c++': ['cpp', 'c plus plus'],
    'c#': ['csharp', 'c sharp'],
    
    # Web Development
    'html': ['html5', 'hypertext markup language'],
    'css': ['css3', 'cascading style sheets'],
    'react': ['reactjs', 'react.js', 'react development'],
    'angular': ['angularjs', 'angular.js'],
    'vue': ['vuejs', 'vue.js'],
    'node.js': ['nodejs', 'node', 'node development'],
    
    # Databases
    'sql': ['mysql', 'postgresql', 'mongodb', 'database querying'],
    'database': ['db', 'databases', 'data storage'],
    
    # Cloud & DevOps
    'aws': ['amazon web services', 'amazon cloud', 'aws cloud'],
    'azure': ['microsoft azure', 'azure cloud'],
    'gcp': ['google cloud platform', 'google cloud'],
    'docker': ['container', 'containers', 'containerization'],
    'kubernetes': ['k8s', 'container orchestration', 'kubectl'],
    'git': ['github', 'gitlab', 'bitbucket', 'version control'],
    
    # Development Methodologies
    'agile': ['scrum', 'kanban', 'agile development'],
    'ci/cd': ['continuous integration', 'continuous deployment', 'devops'],
    
    # AI & Data
    'machine learning': ['ml', 'deep learning', 'ai', 'artificial intelligence'],
    'data science': ['data analysis', 'data analytics', 'big data'],
    
    # General Terms
    'web development': ['web dev', 'frontend', 'backend', 'full stack'],
    'software engineering': ['software development', 'programming', 'coding'],
    'cloud computing': ['cloud', 'cloud services', 'cloud platforms'],
    'rest api': ['rest', 'api', 'apis', 'web services'],
    'microservices': ['microservice', 'microservice architecture', 'distributed systems']
}

# Related terms for skill matching
RELATED_TERMS = {
    # Technical Skills
    'programming': ['coding', 'development', 'software', 'engineering', 'implementation'],
    'problem solving': ['debugging', 'troubleshooting', 'analytical', 'critical thinking', 'optimization'],
    'database': ['sql', 'mysql', 'postgresql', 'mongodb', 'data storage', 'data management'],
    'git': ['github', 'version control', 'source control', 'repository management'],
    'agile': ['scrum', 'kanban', 'sprint', 'iterative development', 'agile methodology'],
    'rest api': ['api', 'web services', 'http', 'endpoints', 'api development'],
    'frontend': ['react', 'angular', 'vue', 'javascript', 'ui development', 'client-side'],
    'backend': ['server', 'api', 'database', 'server-side', 'backend development'],
    
    # Soft Skills
    'teamwork': ['collaboration', 'cooperation', 'interpersonal', 'team player', 'group work'],
    'communication': ['verbal', 'written', 'interpersonal', 'presentation', 'documentation'],
    'leadership': ['management', 'supervision', 'mentoring', 'guidance', 'direction'],
    'project management': ['planning', 'coordination', 'organization', 'resource management'],
    'analytical': ['analysis', 'research', 'data analysis', 'problem solving', 'critical thinking'],
    'technical': ['engineering', 'development', 'programming', 'implementation', 'architecture']
}

# Skill phrases to look for in job descriptions
SKILL_PHRASES = {
    "programming language": ["python", "javascript", "java", "c++", "c#", "php", "ruby", "swift", "kotlin", "go"],
    "front end framework": ["react", "angular", "vue", "svelte", "next.js", "nuxt.js"],
    "database": ["mysql", "postgresql", "mongodb", "sql", "redis", "oracle"],
    "version control": ["git", "github", "gitlab", "bitbucket"],
    "development methodology": ["agile", "scrum", "kanban", "waterfall", "lean"],
    "api": ["rest", "graphql", "web services", "soap", "microservices"],
    "soft skill": ["problem solving", "teamwork", "communication", "leadership", "analytical"],
    "cloud platform": ["aws", "azure", "gcp", "digital ocean", "heroku"],
    "container": ["docker", "kubernetes", "containerization", "container orchestration"],
    "testing": ["unit testing", "integration testing", "test automation", "qa", "quality assurance"]
}

# Experience level indicators
SKILL_LEVELS = {
    'beginner': ['basic', 'familiar', 'learning', 'introduction', 'fundamentals'],
    'intermediate': ['proficient', 'experienced', 'worked with', 'developed', 'implemented'],
    'expert': ['expert', 'advanced', 'mastery', 'led', 'architected', 'designed', 'optimized']
}

# Experience indicators
EXPERIENCE_INDICATORS = [
    'experience', 'worked', 'developed', 'built', 'created',
    'implemented', 'designed', 'architected', 'led', 'managed',
    'coordinated', 'organized', 'planned', 'executed', 'delivered'
]

# Skill categories for better organization
SKILL_CATEGORIES = {
    'Programming Languages': {
        'keywords': ['python', 'javascript', 'java', 'c++', 'c#', 'php', 'ruby', 'swift', 'kotlin', 'go'],
        'weight': 1.0
    },
    'Web Development': {
        'keywords': ['html', 'css', 'react', 'angular', 'vue', 'node.js', 'express', 'django', 'flask', 'frontend', 'backend'],
        'weight': 0.9
    },
    'Databases': {
        'keywords': ['sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'oracle', 'database'],
        'weight': 0.9
    },
    'Cloud & DevOps': {
        'keywords': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'git', 'ci/cd', 'jenkins'],
        'weight': 0.9
    },
    'Software Development': {
        'keywords': ['agile', 'scrum', 'microservices', 'rest api', 'graphql', 'debugging', 'testing', 'code review'],
        'weight': 0.8
    },
    'AI & Data': {
        'keywords': ['machine learning', 'ai', 'data science', 'data analysis', 'big data'],
        'weight': 0.9
    },
    'Soft Skills': {
        'keywords': ['communication', 'leadership', 'teamwork', 'problem solving', 'project management'],
        'weight': 0.7
    }
}

# Critical skills that should always be highlighted if missing
CRITICAL_SKILLS = {
    'python': 1.0,
    'javascript': 1.0,
    'java': 1.0,
    'sql': 1.0,
    'git': 1.0,
    'aws': 1.0,
    'react': 1.0,
    'node.js': 1.0,
    'docker': 1.0,
    'agile': 0.9,
    'rest api': 0.9,
    'testing': 0.9,
    'problem solving': 0.9,
    'communication': 0.8
} 