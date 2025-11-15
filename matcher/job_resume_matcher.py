import pandas as pd
import torch
import os
import json

from modules.scorer.embedding_scorer import EmbeddingScorer
from modules.scorer.hybrid_ranker import HybridRanker
from modules.ner.ner_phrase_matcher import PhraseSkillMatcher
from modules.ner.ner_entity_extractor import EntityExtractor

# Import smart weight selector (optional - falls back gracefully if not available)
try:
    from smart_weight_selector import SmartWeightSelector
    SMART_WEIGHTS_AVAILABLE = True
except ImportError:
    SMART_WEIGHTS_AVAILABLE = False
    print("  Smart weight selector not available, using default weights")


# ------------------------------------------------------
# Load all resources
# ------------------------------------------------------
def load_resume_data():
    meta_path = "data/embeddings/resume_metadata.csv"
    embed_path = "data/embeddings/resume_embeddings.pt"

    if not os.path.exists(meta_path) or not os.path.exists(embed_path):
        raise FileNotFoundError(" Resume metadata or embeddings not found.")

    resumes = pd.read_csv(meta_path)
    resume_embeddings = torch.load(embed_path)

    return resumes, resume_embeddings


# ------------------------------------------------------
# JOB → RESUME MATCHING PIPELINE
# ------------------------------------------------------
def match_resumes(job_description, top_k=5, weight_mode='smart', custom_weights=None):
    """
    Match resumes to job description with flexible weight selection.
    
    Args:
        job_description: Job description text
        top_k: Number of top matches to return
        weight_mode: 'smart' (auto-detect), 'balanced', 'skills', 'experience', 'custom'
        custom_weights: Dict with 'embedding', 'skill', 'ner' keys (only for weight_mode='custom')
    
    Weight Modes:
        - 'smart': Auto-detect based on job characteristics (recommended)
        - 'balanced': 0.50/0.35/0.15 (good all-around)
        - 'skills': 0.40/0.45/0.15 (technical roles)
        - 'experience': 0.45/0.30/0.25 (senior roles)
        - 'embeddings': 0.60/0.30/0.10 (semantic fit)
        - 'custom': Use custom_weights dict
    """

    # ---- Load resume data ----
    resumes, resume_embeddings = load_resume_data()

    # ---- Weight Selection Logic ----
    print("\n" + "="*60)
    print("  WEIGHT SELECTION")
    print("="*60)
    
    weight_configs = {
        'balanced': {'embedding': 0.50, 'skill': 0.35, 'ner': 0.15},
        'skills': {'embedding': 0.40, 'skill': 0.45, 'ner': 0.15},
        'experience': {'embedding': 0.45, 'skill': 0.30, 'ner': 0.25},
        'embeddings': {'embedding': 0.60, 'skill': 0.30, 'ner': 0.10}
    }
    
    if weight_mode == 'smart' and SMART_WEIGHTS_AVAILABLE:
        # Smart auto-detection
        weight_selector = SmartWeightSelector()
        smart_weights = weight_selector.select_weights(job_description)
        
        print(smart_weights['reasoning'])
        
        ranker = HybridRanker(
            w_embedding=smart_weights['embedding'],
            w_skill=smart_weights['skill'],
            w_ner_bonus=smart_weights['ner']
        )
        selected_weights = smart_weights
        
    elif weight_mode == 'custom' and custom_weights:
        # Manual custom weights
        print("  Mode: CUSTOM WEIGHTS")
        print(f"   • Embedding: {custom_weights['embedding']:.0%}")
        print(f"   • Skills: {custom_weights['skill']:.0%}")
        print(f"   • NER/Experience: {custom_weights['ner']:.0%}\n")
        
        ranker = HybridRanker(
            w_embedding=custom_weights['embedding'],
            w_skill=custom_weights['skill'],
            w_ner_bonus=custom_weights['ner']
        )
        selected_weights = custom_weights
        
    elif weight_mode in weight_configs:
        # Predefined modes
        weights = weight_configs[weight_mode]
        
        mode_descriptions = {
            'balanced': 'Balanced approach - good for most roles',
            'skills': 'Skills-focused - for highly technical positions',
            'experience': 'Experience-focused - for senior roles',
            'embeddings': 'Semantic fit - for creative/soft skill roles'
        }
        
        print(f" Mode: {weight_mode.upper()}")
        print(f"   • {mode_descriptions[weight_mode]}")
        print(f"\n   Weights:")
        print(f"   • Embedding: {weights['embedding']:.0%}")
        print(f"   • Skills: {weights['skill']:.0%}")
        print(f"   • NER/Experience: {weights['ner']:.0%}\n")
        
        ranker = HybridRanker(
            w_embedding=weights['embedding'],
            w_skill=weights['skill'],
            w_ner_bonus=weights['ner']
        )
        selected_weights = weights
        
    else:
        # Fallback to balanced
        print("  Invalid mode or smart weights unavailable")
        print(" Using: BALANCED (default)")
        print("   • Embedding: 50% | Skills: 35% | NER: 15%\n")
        
        ranker = HybridRanker()
        selected_weights = weight_configs['balanced']

    # ---- Initialize modules ----
    embedder = EmbeddingScorer()
    phrase_matcher = PhraseSkillMatcher()
    entity_extractor = EntityExtractor()

    # ---- Encode job ----
    job_embedding = embedder.encode(job_description)

    # ---- Extract job skills ----
    job_skill_phrases = phrase_matcher.extract(job_description)
    job_entities = entity_extractor.extract(job_description)

    print(f" Extracted {len(job_skill_phrases)} skills from job description")
    print(f" Skills found: {job_skill_phrases[:15]}")
    if len(job_skill_phrases) > 15:
        print(f"   ... and {len(job_skill_phrases) - 15} more")
    
    print(f"\n Entities extracted:")
    print(f"   • Organizations: {job_entities.get('organizations', [])[:5]}")
    print(f"   • Locations: {job_entities.get('locations', [])}")
    print(f"   • Titles: {job_entities.get('titles', [])}")
    print(f"   • Experience: {job_entities.get('experience_years', 0)} years")

    # ---- Score all resumes ----
    results = []

    print("\n Matching resumes...")

    # Pre-compute embedding similarities
    embedding_scores = embedder.batch_similarity(job_embedding, resume_embeddings)

    for idx, row in resumes.iterrows():

        resume_text = row["cleaned_resume"]

        # Extract resume skills + NER entities
        resume_skill_phrases = phrase_matcher.extract(resume_text)
        resume_entities = entity_extractor.extract(resume_text)

        # Hybrid ranking
        scores = ranker.hybrid_score(
            embedding_score=float(embedding_scores[idx]),
            job_skills=job_skill_phrases,
            resume_skills=resume_skill_phrases,
            job_entities=job_entities,
            resume_entities=resume_entities
        )

        results.append({
            "filename": row["filename"],
            "embedding_score": scores["embedding"],
            "skill_overlap": scores["skill_overlap"],
            "ner_bonus": scores["ner_bonus"],
            "final_score": scores["final_score"],
            "matched_skills": len(set(job_skill_phrases) & set(resume_skill_phrases))
        })

    # ---- Sort by final score ----
    df = pd.DataFrame(results)
    df = df.sort_values(by="final_score", ascending=False).reset_index(drop=True)

    # Save results
    os.makedirs("data/results", exist_ok=True)
    df.to_csv("data/results/match_results_hybrid.csv", index=False)

    # Save weight configuration used
    with open("data/results/weight_config.json", "w") as f:
        json.dump({
            "mode": weight_mode,
            "weights": selected_weights,
            "timestamp": pd.Timestamp.now().isoformat()
        }, f, indent=2)

    print("\n Saved results to data/results/match_results_hybrid.csv")
    print(" Saved weight config to data/results/weight_config.json")

    return df.head(top_k)


# ------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------
if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    weight_mode = 'smart'  # default
    job_file = None
    
    if len(sys.argv) > 1:
        job_file = sys.argv[1]
        
    if len(sys.argv) > 2:
        # Allow specifying weight mode as second argument
        mode_arg = sys.argv[2].lower()
        if mode_arg in ['smart', 'balanced', 'skills', 'experience', 'embeddings']:
            weight_mode = mode_arg
    
    # Check if job description file is provided
    if job_file and os.path.exists(job_file):
        with open(job_file, 'r', encoding='utf-8') as f:
            job_description = f.read()
        print(f"\n Loaded job description from: {job_file}")
    elif job_file:
        print(f" File not found: {job_file}")
        print(f"\nUsage: python -m matcher.job_resume_matcher <job_file.txt> [mode]")
        print(f"Modes: smart (default), balanced, skills, experience, embeddings")
        exit(1)
    else:
        # Fallback to simple single-line input
        print("\n Enter Job Title/Description (single line):")
        job_description = input("> ").strip()
        
        if not job_description:
            print(" No job description provided!")
            print("\nTip: For full job descriptions, save to a file and run:")
            print("     python -m matcher.job_resume_matcher job.txt [mode]")
            exit(1)

    print("\n=============================================")
    print(" TOP MATCHING RESUMES (HYBRID MODEL) ")
    print("=============================================")

    output = match_resumes(job_description, top_k=5, weight_mode=weight_mode)
    
    print("\n" + "="*60)
    print("TOP 5 MATCHES:")
    print("="*60)
    print(output.to_string(index=False))