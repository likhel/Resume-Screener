# ===========================================
# 🤖 Resume–Job Matching Script
# Author: [Your Name]
# Description:
#   Compares resumes and job descriptions using
#   Sentence-BERT embeddings and cosine similarity.
# ===========================================

import os
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

# ---------- Step 1. Load Data ----------
resumes = pd.read_csv("data/cleaned/resumes_extracted.csv")
jobs = pd.read_csv("data/job_description/job_data_cleaned.csv")

print(f"✅ Loaded {len(resumes)} resumes and {len(jobs)} job descriptions.")

# ---------- Step 2. Load Model ----------
print("🧠 Loading Sentence-BERT model (this may take a few seconds)...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# ---------- Step 3. Prepare Output ----------
os.makedirs("data/results", exist_ok=True)
results = []

# ---------- Step 4. Compute Similarities ----------
for _, job in tqdm(jobs.iterrows(), total=len(jobs), desc="🔍 Matching Jobs"):
    job_title = str(job['Job Title'])
    job_desc = str(job['Job Description'])
    
    # Encode job description once
    job_embed = model.encode(job_desc, convert_to_tensor=True)

    # Compare with every resume
    for _, res in resumes.iterrows():
        resume_file = str(res['filename'])
        resume_text = str(res['cleaned_resume'])
        
        res_embed = model.encode(resume_text, convert_to_tensor=True)
        score = util.cos_sim(res_embed, job_embed).item()
        
        results.append({
            "Job Title": job_title,
            "Resume File": resume_file,
            "Similarity Score": round(score, 3)
        })

# ---------- Step 5. Save Results ----------
results_df = pd.DataFrame(results)
output_path = "data/results/job_resume_matches.csv"
results_df.to_csv(output_path, index=False)
print(f"\n✅ Matching complete! Saved to: {output_path}")

# ---------- Step 6. Optional: Show Top Matches ----------
topN = 5
print("\n🏆 Top 5 matches for a random job:")
sample_job = results_df.sample(1)['Job Title'].iloc[0]
top_matches = results_df[results_df['Job Title'] == sample_job].sort_values(by='Similarity Score', ascending=False).head(topN)
print(top_matches)
