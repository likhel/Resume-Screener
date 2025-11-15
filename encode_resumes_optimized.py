# ===========================================
# Optimized Resume Embedding Encoder
# Author: [Your Name]
# Description:
#   Encodes all cleaned resumes safely, in batches,
#   handles memory limits and saves reusable embeddings.
# ===========================================

import os
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ---------- Step 1. Load Resumes ----------
resume_path = "data/cleaned/resumes_extracted.csv"
resumes = pd.read_csv(resume_path)
print(f" Loaded {len(resumes)} resumes.")

# Truncate very long resumes (avoid overlength inefficiency)
resumes['cleaned_resume'] = resumes['cleaned_resume'].astype(str).str.slice(0, 2000)

# ---------- Step 2. Create Output Folder ----------
os.makedirs("data/embeddings", exist_ok=True)

# ---------- Step 3. Load Model ----------
print(" Loading Sentence-BERT model (all-MiniLM-L6-v2)...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# ---------- Step 4. Encode in Batches ----------
embeddings = []
batch_size = 16

print("Encoding resumes in safe batches...")
for i in tqdm(range(0, len(resumes), batch_size)):
    batch_texts = resumes['cleaned_resume'][i:i+batch_size].tolist()
    try:
        batch_embeds = model.encode(
            batch_texts,
            batch_size=batch_size,
            convert_to_tensor=True,
            show_progress_bar=False
        )
        embeddings.append(batch_embeds)
    except Exception as e:
        print(f"Error encoding batch {i}-{i+batch_size}: {e}")
        continue

# Concatenate all batches into one tensor
resume_embeddings = torch.cat(embeddings, dim=0)

# ---------- Step 5. Save for Reuse ----------
torch.save(resume_embeddings, "data/embeddings/resume_embeddings.pt")
resumes['embedding_index'] = range(len(resumes))
resumes.to_csv("data/embeddings/resume_metadata.csv", index=False)

print("\n All embeddings saved:")
print("data/embeddings/resume_embeddings.pt")
print("data/embeddings/resume_metadata.csv")

print("\nTip: You can now load instantly in future scripts using:")
print("resume_embeddings = torch.load('data/embeddings/resume_embeddings.pt')")