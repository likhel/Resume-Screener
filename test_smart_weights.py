"""
Test Smart Weight Selector with Different Job Types
"""

from smart_weight_selector import SmartWeightSelector


def test_job_descriptions():
    selector = SmartWeightSelector()
    
    # Different job types to test
    test_jobs = {
        "Senior ML Engineer": """
            Senior Software Engineer - Machine Learning
            5+ years experience in Python, TensorFlow, PyTorch, AWS
            Lead ML projects and mentor junior engineers
        """,
        
        "Junior Developer": """
            Junior Frontend Developer
            Entry-level position for recent graduates
            Knowledge of HTML, CSS, JavaScript, React required
        """,
        
        "Product Manager": """
            Senior Product Manager
            10+ years experience in product strategy
            Lead cross-functional teams, define product roadmap
        """,
        
        "Data Analyst": """
            Data Analyst
            3-4 years experience with SQL, Python, Excel
            Analyze business metrics and create dashboards
        """,
        
        "UI/UX Designer": """
            UI/UX Designer
            Design user interfaces and experiences
            Proficiency in Figma, Adobe XD, user research
        """,
        
        "DevOps Engineer": """
            DevOps Engineer
            Strong expertise in Docker, Kubernetes, CI/CD, AWS, Terraform
            2+ years experience automating deployments
        """
    }
    
    print("\n" + "="*70)
    print("TESTING SMART WEIGHT SELECTOR")
    print("="*70)
    
    for job_title, job_desc in test_jobs.items():
        print(f"\n{'─'*70}")
        print(f"Job: {job_title}")
        print("─"*70)
        
        result = selector.select_weights(job_desc)
        print(result['reasoning'])


if __name__ == "__main__":
    test_job_descriptions()