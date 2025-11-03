# ===========================================
# 🤖 Single Job–Resume Matching Script (with Visualization)
# Author: [Your Name]
# Description:
#   Finds the top matching resumes for a single job
#   using Sentence-BERT embeddings and cosine similarity.
#   Includes visualizations: top 5 bar chart + full score distribution.
# ===========================================

import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import os
import matplotlib.pyplot as plt

# ---------- Step 1. Load Resumes ----------
resume_path = "data/cleaned/resumes_extracted.csv"
resumes = pd.read_csv(resume_path)
print(f"✅ Loaded {len(resumes)} resumes.")

# ---------- Step 2. Load Sentence-BERT Model ----------
print("🧠 Loading Sentence-BERT model (all-MiniLM-L6-v2)...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# ---------- Step 3. Encode All Resumes Once ----------
print("⚙️ Encoding all resumes (this happens once)...")
resume_embeddings = model.encode(
    resumes['cleaned_resume'].tolist(),
    convert_to_tensor=True,
    show_progress_bar=True
)

# ---------- Step 4. Input Job Description ----------
job_description = input("\n📝 Enter job description: ")

# Optional: load one automatically from cleaned job dataset
# jobs = pd.read_csv("data/job_description/job_data_cleaned.csv")
# job_description = jobs.iloc[0]['Job Description']

print("\n📋 Matching resumes for job:")
print(job_description)

# ---------- Step 5. Encode Job Description ----------
job_embedding = model.encode(job_description, convert_to_tensor=True)

# ---------- Step 6. Compute Similarities ----------
cosine_scores = util.cos_sim(job_embedding, resume_embeddings)[0]
resumes['Similarity'] = cosine_scores.cpu().numpy()

# ---------- Step 7. Sort and Show Top Matches ----------
top_matches = resumes.sort_values(by='Similarity', ascending=False).head(5)
print("\n🏆 Top 5 matching resumes:")
print(top_matches[['filename', 'Similarity']])

# ---------- Step 8. Save Results ----------
os.makedirs("data/results", exist_ok=True)
output_path = "data/results/top_matches_single_job.csv"
top_matches.to_csv(output_path, index=False)
print(f"\n✅ Results saved to: {output_path}")

# ---------- Step 9. Visualization 1: Top-5 Bar Chart ----------
plt.figure(figsize=(8, 5))
plt.barh(top_matches['filename'], top_matches['Similarity'], color='#4C9AFF')
plt.gca().invert_yaxis()  # Highest score at top
plt.xlabel("Similarity Score")
plt.title("Top 5 Resume Matches for Job Description")
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# ---------- Step 10. Visualization 2: Full Score Distribution ----------
plt.figure(figsize=(8, 5))
plt.hist(resumes['Similarity'], bins=20, color='#FFB347', edgecolor='black', alpha=0.8)
plt.title("Distribution of Similarity Scores (All Resumes)")
plt.xlabel("Similarity Score")
plt.ylabel("Number of Resumes")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

print("\n📊 Visualization complete! Two charts displayed: Top-5 and Score Distribution.")

