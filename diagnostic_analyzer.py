"""
Diagnostic tool to analyze resume screening system
Run: python diagnostic_analyzer.py
"""

import os
from modules.ner.ner_phrase_matcher import PhraseSkillMatcher
from modules.ner.ner_entity_extractor import EntityExtractor

# Sample job description
JOB_DESC = """
Senior Software Engineer - Machine Learning
5+ years experience in Python, TensorFlow, PyTorch, scikit-learn, 
Pandas, NumPy, AWS, Docker, Kubernetes, SQL, Git, REST APIs, 
Machine Learning, Deep Learning, Natural Language Processing
"""

def analyze_skills_list():
    """Check the skills list file"""
    skills_file = "data/skills/skills_list.txt"
    
    print("="*60)
    print(" SKILLS LIST ANALYSIS")
    print("="*60)
    
    if not os.path.exists(skills_file):
        print(f" Skills file not found: {skills_file}")
        return
    
    with open(skills_file, 'r', encoding='utf-8') as f:
        skills = [line.strip().lower() for line in f if line.strip()]
    
    print(f"\n Total skills in database: {len(skills)}")
    print(f"\n Sample skills (first 20):")
    for i, skill in enumerate(skills[:20], 1):
        print(f"   {i}. {skill}")
    
    # Check for common ML/AI skills
    ml_keywords = ['python', 'tensorflow', 'pytorch', 'machine learning', 
                   'deep learning', 'aws', 'docker', 'kubernetes', 'sql', 
                   'pandas', 'numpy', 'scikit-learn', 'git']
    
    print(f"\n Common ML/Tech skills coverage:")
    for keyword in ml_keywords:
        found = any(keyword in skill for skill in skills)
        status = "good" if found else "shit"
        print(f"   {status} {keyword}")
    
    return skills

def test_extraction():
    """Test skill extraction on sample job description"""
    print("\n" + "="*60)
    print(" EXTRACTION TEST")
    print("="*60)
    
    print(f"\n Test Job Description:")
    print(JOB_DESC)
    
    # Test phrase matcher
    print("\n Testing PhraseSkillMatcher...")
    matcher = PhraseSkillMatcher()
    extracted_skills = matcher.extract(JOB_DESC)
    
    print(f"\n Extracted {len(extracted_skills)} skills:")
    for i, skill in enumerate(extracted_skills, 1):
        print(f"   {i}. {skill}")
    
    # Test entity extractor
    print("\n Testing EntityExtractor...")
    extractor = EntityExtractor()
    entities = extractor.extract(JOB_DESC)
    
    print(f"\n Extracted Entities:")
    print(f"   Skills: {entities['skills']}")
    print(f"   Titles: {entities['titles']}")
    print(f"   Organizations: {entities['organizations']}")
    print(f"   Locations: {entities['locations']}")
    print(f"   Experience: {entities['experience_years']} years")

def suggest_improvements(skills):
    """Suggest missing skills to add"""
    print("\n" + "="*60)
    print(" IMPROVEMENT SUGGESTIONS")
    print("="*60)
    
    # Common tech skills that might be missing
    suggested_additions = [
        'rest api', 'rest apis', 'restful api',
        'microservices', 'microservice architecture',
        'nlp', 'natural language processing',
        'computer vision', 'cv',
        'mlops', 'ml ops',
        'data pipelines', 'data pipeline',
        'scikit learn', 'sklearn',
        'ci/cd', 'continuous integration',
        'agile', 'scrum',
        'nosql', 'mongodb', 'redis',
        'spark', 'dask',
        'gcp', 'google cloud', 'azure'
    ]
    
    missing = []
    for suggestion in suggested_additions:
        if not any(suggestion in skill for skill in skills):
            missing.append(suggestion)
    
    if missing:
        print(f"\n Consider adding these skills ({len(missing)} missing):")
        for skill in missing[:15]:
            print(f"   • {skill}")
    else:
        print("\n Your skills list looks comprehensive!")

if __name__ == "__main__":
    print("\n RESUME SCREENING SYSTEM DIAGNOSTIC")
    print("="*60)
    
    skills = analyze_skills_list()
    test_extraction()
    
    if skills:
        suggest_improvements(skills)
    
    print("\n" + "="*60)
    print(" DIAGNOSTIC COMPLETE")
    print("="*60)