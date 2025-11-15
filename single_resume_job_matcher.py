# ===========================================
# Single Job–Resume Matching Script (Optimized + Visualization)
# Author: [Your Name]
# Description:
#   Uses precomputed resume embeddings for instant matching.
#   Finds top matching resumes for a single job description.
#   Includes visualizations: Top-5 bar chart + score distribution.
# ===========================================

import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import os
import matplotlib.pyplot as plt

# ---------- Step 1. Load Precomputed Resume Data ----------
meta_path = "data/embeddings/resume_metadata.csv"
embed_path = "data/embeddings/resume_embeddings.pt"

resumes = pd.read_csv(meta_path)
resume_embeddings = torch.load(embed_path)

print(f"✅ Loaded {len(resumes)} resumes and precomputed embeddings.")

# ---------- Step 2. Load SBERT Model ----------
print(" Loading Sentence-BERT model (all-MiniLM-L6-v2)...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# ---------- Step 3. Input Job Description ----------
job_description = input("\n Enter job description: ")

# Optional: use a job from dataset
# jobs = pd.read_csv("data/job_descriptions/job_data_cleaned.csv")
# job_description = jobs.iloc[0]['Job Description']

print("\n Matching resumes for job:")
print(job_description)

# ---------- Step 4. Encode the Job Description ----------
job_embedding = model.encode(job_description, convert_to_tensor=True)

# ---------- Step 5. Compute Similarity (Instantly) ----------
cosine_scores = util.cos_sim(job_embedding, resume_embeddings)[0]
resumes['Similarity'] = cosine_scores.cpu().numpy()

# ---------- Step 6. Show Top 5 Matches ----------
top_matches = resumes.sort_values(by='Similarity', ascending=False).head(5)

print("\nTop 5 matching resumes:")
print(top_matches[['filename', 'Similarity']])

# ---------- Step 7. Save Results ----------
os.makedirs("data/results", exist_ok=True)
output_path = "data/results/top_matches_single_job.csv"
top_matches.to_csv(output_path, index=False)
print(f"\nResults saved to: {output_path}")

# ---------- Step 8. Visualization 1: Top-5 Bar Chart ----------
plt.figure(figsize=(8, 5))
plt.barh(top_matches['filename'], top_matches['Similarity'], color='#4C9AFF')
plt.gca().invert_yaxis()  # Highest score at top
plt.xlabel("Similarity Score")
plt.title("Top 5 Resume Matches for Job Description")
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# ---------- Step 9. Visualization 2: Full Score Distribution ----------
plt.figure(figsize=(8, 5))
plt.hist(resumes['Similarity'], bins=20, color='#FFB347', edgecolor='black', alpha=0.8)
plt.title("Distribution of Similarity Scores (All Resumes)")
plt.xlabel("Similarity Score")
plt.ylabel("Number of Resumes")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

print("\nVisualization complete! Two charts displayed: Top-5 and Score Distribution.")