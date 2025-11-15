"""
Smart Weight Selector
---------------------
Automatically selects optimal weights based on job description analysis.

Analyzes:
- Seniority level (junior/mid/senior)
- Technical vs non-technical role
- Required skills count
- Experience requirements

Returns appropriate weights for HybridRanker.
"""

import re


class SmartWeightSelector:
    
    def __init__(self):
        # Predefined weight configurations
        self.configs = {
            "junior_technical": {
                'embedding': 0.55,
                'skill': 0.35,
                'ner': 0.10,
                'description': "Fresh grads/juniors - skills + potential"
            },
            "mid_technical": {
                'embedding': 0.50,
                'skill': 0.35,
                'ner': 0.15,
                'description': "Mid-level - balanced approach"
            },
            "senior_technical": {
                'embedding': 0.45,
                'skill': 0.30,
                'ner': 0.25,
                'description': "Senior roles - experience matters"
            },
            "highly_technical": {
                'embedding': 0.40,
                'skill': 0.45,
                'ner': 0.15,
                'description': "Specialized tech - exact skills critical"
            },
            "creative_soft_skills": {
                'embedding': 0.60,
                'skill': 0.25,
                'ner': 0.15,
                'description': "Creative/soft skill roles - overall fit"
            },
            "management": {
                'embedding': 0.50,
                'skill': 0.25,
                'ner': 0.25,
                'description': "Leadership roles - experience + fit"
            }
        }
    
    def detect_seniority(self, text: str) -> str:
        """
        Detect seniority level from job description.
        Returns: 'junior', 'mid', 'senior', 'lead'
        """
        text_lower = text.lower()
        
        # Senior level indicators
        senior_terms = ['senior', 'sr.', 'lead', 'principal', 'staff', 
                       'architect', 'head of', 'chief', 'director', 'vp']
        
        # Junior level indicators
        junior_terms = ['junior', 'jr.', 'entry level', 'entry-level', 
                       'intern', 'graduate', 'trainee', 'associate']
        
        # Manager/Lead indicators
        manager_terms = ['manager', 'team lead', 'engineering manager', 
                        'project manager', 'product manager', 'scrum master']
        
        if any(term in text_lower for term in senior_terms):
            return 'senior'
        elif any(term in text_lower for term in manager_terms):
            return 'manager'
        elif any(term in text_lower for term in junior_terms):
            return 'junior'
        else:
            return 'mid'
    
    def detect_role_type(self, text: str) -> str:
        """
        Detect if role is technical, creative, or management.
        Returns: 'technical', 'creative', 'management'
        """
        text_lower = text.lower()
        
        # Technical indicators
        tech_terms = ['engineer', 'developer', 'programmer', 'architect',
                     'devops', 'data scientist', 'analyst', 'ml', 'ai',
                     'backend', 'frontend', 'fullstack', 'software']
        
        # Creative indicators
        creative_terms = ['designer', 'ux', 'ui', 'creative', 'writer',
                         'content', 'marketing', 'brand', 'artist']
        
        # Management indicators
        mgmt_terms = ['manager', 'director', 'head of', 'chief', 'vp',
                     'coordinator', 'lead', 'supervisor']
        
        tech_count = sum(1 for term in tech_terms if term in text_lower)
        creative_count = sum(1 for term in creative_terms if term in text_lower)
        mgmt_count = sum(1 for term in mgmt_terms if term in text_lower)
        
        if mgmt_count > tech_count and mgmt_count > creative_count:
            return 'management'
        elif creative_count > tech_count:
            return 'creative'
        else:
            return 'technical'
    
    def extract_experience_requirement(self, text: str) -> int:
        """
        Extract years of experience required.
        Returns: number of years (0 if not found)
        """
        # Look for patterns like "5+ years", "3-5 years", etc.
        patterns = [
            r'(\d+)\+?\s*(?:years|yrs)',
            r'(\d+)\s*[-–to]\s*\d+\s*(?:years|yrs)',
        ]
        
        years = []
        for pattern in patterns:
            matches = re.findall(pattern, text.lower())
            for match in matches:
                try:
                    years.append(int(match))
                except:
                    pass
        
        return max(years) if years else 0
    
    def count_technical_skills(self, text: str) -> int:
        """
        Count technical skills mentioned (rough estimate).
        More skills = more technical role
        """
        # Common technical skill indicators
        tech_keywords = [
            'python', 'java', 'javascript', 'c++', 'sql', 'aws', 'azure',
            'docker', 'kubernetes', 'git', 'react', 'node', 'api',
            'tensorflow', 'pytorch', 'machine learning', 'deep learning',
            'database', 'cloud', 'microservices', 'ci/cd'
        ]
        
        text_lower = text.lower()
        return sum(1 for skill in tech_keywords if skill in text_lower)
    
    def analyze_job(self, job_description: str) -> dict:
        """
        Analyze job description and return characteristics.
        """
        seniority = self.detect_seniority(job_description)
        role_type = self.detect_role_type(job_description)
        experience_years = self.extract_experience_requirement(job_description)
        skill_count = self.count_technical_skills(job_description)
        
        return {
            'seniority': seniority,
            'role_type': role_type,
            'experience_years': experience_years,
            'skill_count': skill_count
        }
    
    def select_weights(self, job_description: str) -> dict:
        """
        Main function: Analyze job and return optimal weights.
        
        Returns:
            dict with keys: embedding, skill, ner, config_name, reasoning
        """
        analysis = self.analyze_job(job_description)
        
        # Decision logic
        seniority = analysis['seniority']
        role_type = analysis['role_type']
        exp_years = analysis['experience_years']
        skill_count = analysis['skill_count']
        
        # Select configuration based on analysis
        if role_type == 'management':
            config_name = 'management'
        elif role_type == 'creative':
            config_name = 'creative_soft_skills'
        elif seniority == 'junior':
            config_name = 'junior_technical'
        elif seniority == 'senior' or exp_years >= 5:
            config_name = 'senior_technical'
        elif skill_count >= 10:  # Highly technical with many skills
            config_name = 'highly_technical'
        else:
            config_name = 'mid_technical'
        
        weights = self.configs[config_name].copy()
        
        # Build reasoning
        reasoning = f"""
Job Analysis:
   • Seniority: {seniority.upper()}
   • Role Type: {role_type.upper()}
   • Experience Required: {exp_years} years
   • Technical Skills Mentioned: {skill_count}

Selected Configuration: {config_name.upper().replace('_', ' ')}
   • {weights['description']}
   
Weights Applied:
   • Embedding (semantic similarity): {weights['embedding']:.0%}
   • Skills (technical match): {weights['skill']:.0%}
   • NER (experience/entities): {weights['ner']:.0%}
"""
        
        return {
            'embedding': weights['embedding'],
            'skill': weights['skill'],
            'ner': weights['ner'],
            'config_name': config_name,
            'description': weights['description'],
            'reasoning': reasoning,
            'analysis': analysis
        }


# Convenience function
def get_smart_weights(job_description: str) -> dict:
    """
    Quick function to get weights for a job description.
    
    Usage:
        weights = get_smart_weights(job_desc)
        ranker = HybridRanker(
            w_embedding=weights['embedding'],
            w_skill=weights['skill'],
            w_ner_bonus=weights['ner']
        )
    """
    selector = SmartWeightSelector()
    return selector.select_weights(job_description)