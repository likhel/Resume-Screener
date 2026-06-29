

import pandas as pd
# pyrefly: ignore [missing-import]
from sentence_transformers import SentenceTransformer, util
import torch
import os
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt

# ---------- Step 1. Load Precomputed Resume Data ----------
meta_path = "data/embeddings/resume_metadata.csv"
embed_path = "data/embeddings/resume_embeddings.pt"

resumes = pd.read_csv(meta_path)
resume_embeddings = torch.load(embed_path)

print(f"✅ Loaded {len(resumes)} resumes and precomputed embeddings.")

print(" Loading Sentence-BERT model (all-MiniLM-L6-v2)...")
model = SentenceTransformer('all-MiniLM-L6-v2')


job_description = input("\n Enter job description: ")



print("\n Matching resumes for job:")
print(job_description)


job_embedding = model.encode(job_description, convert_to_tensor=True)


cosine_scores = util.cos_sim(job_embedding, resume_embeddings)[0]
resumes['Similarity'] = cosine_scores.cpu().numpy()


top_matches = resumes.sort_values(by='Similarity', ascending=False).head(5)

print("\nTop 5 matching resumes:")
print(top_matches[['filename', 'Similarity']])


os.makedirs("data/results", exist_ok=True)
output_path = "data/results/top_matches_single_job.csv"
top_matches.to_csv(output_path, index=False)
print(f"\nResults saved to: {output_path}")


plt.figure(figsize=(8, 5))
plt.barh(top_matches['filename'], top_matches['Similarity'], color='#4C9AFF')
plt.gca().invert_yaxis()  
plt.xlabel("Similarity Score")
plt.title("Top 5 Resume Matches for Job Description")
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 5))
plt.hist(resumes['Similarity'], bins=20, color='#FFB347', edgecolor='black', alpha=0.8)
plt.title("Distribution of Similarity Scores (All Resumes)")
plt.xlabel("Similarity Score")
plt.ylabel("Number of Resumes")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

print("\nVisualization complete! Two charts displayed: Top-5 and Score Distribution.")