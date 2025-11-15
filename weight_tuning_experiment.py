"""
Weight Tuning Experiment
-------------------------
Tests different weight combinations to find optimal balance.
Run: python weight_tuning_experiment.py job_ml_engineer.txt
"""

import pandas as pd
import torch
import os
import sys

from modules.scorer.embedding_scorer import EmbeddingScorer
from modules.scorer.hybrid_ranker import HybridRanker
from modules.ner.ner_phrase_matcher import PhraseSkillMatcher
from modules.ner.ner_entity_extractor import EntityExtractor


def load_resume_data():
    """Load resume metadata and embeddings"""
    meta_path = "data/embeddings/resume_metadata.csv"
    embed_path = "data/embeddings/resume_embeddings.pt"

    if not os.path.exists(meta_path) or not os.path.exists(embed_path):
        raise FileNotFoundError("Resume metadata or embeddings not found.")

    resumes = pd.read_csv(meta_path)
    resume_embeddings = torch.load(embed_path)
    return resumes, resume_embeddings


def test_weights(job_description, weight_config, config_name):
    """Test a specific weight configuration"""
    
    # Load data
    resumes, resume_embeddings = load_resume_data()
    
    # Initialize modules
    embedder = EmbeddingScorer()
    phrase_matcher = PhraseSkillMatcher()
    entity_extractor = EntityExtractor()
    ranker = HybridRanker(
        w_embedding=weight_config['embedding'],
        w_skill=weight_config['skill'],
        w_ner_bonus=weight_config['ner']
    )
    
    # Extract job features
    job_embedding = embedder.encode(job_description)
    job_skill_phrases = phrase_matcher.extract(job_description)
    job_entities = entity_extractor.extract(job_description)
    
    # Score all resumes
    results = []
    embedding_scores = embedder.batch_similarity(job_embedding, resume_embeddings)
    
    for idx, row in resumes.iterrows():
        resume_text = row["cleaned_resume"]
        resume_skill_phrases = phrase_matcher.extract(resume_text)
        resume_entities = entity_extractor.extract(resume_text)
        
        scores = ranker.hybrid_score(
            embedding_score=float(embedding_scores[idx]),
            job_skills=job_skill_phrases,
            resume_skills=resume_skill_phrases,
            job_entities=job_entities,
            resume_entities=resume_entities
        )
        
        results.append({
            "filename": row["filename"],
            "final_score": scores["final_score"],
            "embedding": scores["embedding"],
            "skill_overlap": scores["skill_overlap"],
            "ner_bonus": scores["ner_bonus"]
        })
    
    # Sort by score
    df = pd.DataFrame(results)
    df = df.sort_values(by="final_score", ascending=False).reset_index(drop=True)
    
    return df.head(10)


def main():
    if len(sys.argv) < 2:
        print("Usage: python weight_tuning_experiment.py <job_description_file>")
        sys.exit(1)
    
    job_file = sys.argv[1]
    if not os.path.exists(job_file):
        print(f"File not found: {job_file}")
        sys.exit(1)
    
    with open(job_file, 'r', encoding='utf-8') as f:
        job_description = f.read()
    
    print("\n" + "="*70)
    print("WEIGHT TUNING EXPERIMENT")
    print("="*70)
    print(f"Job Description: {job_file}")
    
    # Define weight configurations to test
    configs = {
        "Current (Balanced)": {
            'embedding': 0.50,
            'skill': 0.35,
            'ner': 0.15
        },
        "Skills First": {
            'embedding': 0.40,
            'skill': 0.45,
            'ner': 0.15
        },
        "Trust Embeddings": {
            'embedding': 0.60,
            'skill': 0.30,
            'ner': 0.10
        },
        "Experience Matters": {
            'embedding': 0.45,
            'skill': 0.30,
            'ner': 0.25
        },
        "Strict Skills Match": {
            'embedding': 0.35,
            'skill': 0.50,
            'ner': 0.15
        },
        "Equal Balance": {
            'embedding': 0.45,
            'skill': 0.40,
            'ner': 0.15
        }
    }
    
    # Test each configuration
    all_results = {}
    
    for config_name, weights in configs.items():
        print(f"\n{'─'*70}")
        print(f"Testing: {config_name}")
        print(f"   Weights → Embedding: {weights['embedding']:.2f} | "
              f"Skills: {weights['skill']:.2f} | NER: {weights['ner']:.2f}")
        
        results = test_weights(job_description, weights, config_name)
        all_results[config_name] = results
        
        # Show top 3
        print(f"\n   Top 3 Candidates:")
        for idx, row in results.head(3).iterrows():
            print(f"   {idx+1}. {row['filename'][:30]:30} → Score: {row['final_score']:.4f}")
    
    # Summary comparison
    print("\n" + "="*70)
    print("RANKING COMPARISON ACROSS ALL CONFIGURATIONS")
    print("="*70)
    
    # Get unique candidates
    all_candidates = set()
    for results in all_results.values():
        all_candidates.update(results['filename'].tolist())
    
    # Create comparison table
    comparison = []
    for candidate in list(all_candidates)[:10]:  # Top 10 unique
        row = {'Candidate': candidate[:35]}
        for config_name, results in all_results.items():
            matching = results[results['filename'] == candidate]
            if not matching.empty:
                rank = matching.index[0] + 1
                score = matching.iloc[0]['final_score']
                row[config_name] = f"#{rank} ({score:.3f})"
            else:
                row[config_name] = "—"
        comparison.append(row)
    
    df_comparison = pd.DataFrame(comparison)
    print("\n" + df_comparison.to_string(index=False))
    
    # Recommendation
    print("\n" + "="*70)
    print("💡 RECOMMENDATIONS")
    print("="*70)
    
    print("\n1️⃣  'Current (Balanced)' - Good for most cases")
    print("    → Use when you want balance between semantic match and hard skills")
    
    print("\n2️⃣  'Skills First' - For technical positions")
    print("    → Use when specific technical skills are mandatory")
    
    print("\n3️⃣  'Trust Embeddings' - For creative/soft skill roles")
    print("    → Use when overall fit matters more than exact skill matches")
    
    print("\n4️⃣  'Experience Matters' - For senior positions")
    print("    → Use when years of experience and title match are critical")
    
    print("\n" + "="*70)
    print("Experiment Complete!")
    print("="*70)


if __name__ == "__main__":
    main()