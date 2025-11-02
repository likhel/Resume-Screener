# ===========================================
# 📄 Resume Text Extraction Script
# Author: [Your Name]
# Project: AI Resume Screener
# Description:
#   Reads all .docx resumes from 'data/resumes_docx/',
#   extracts text, cleans it, and saves results to CSV.
# ===========================================

import os
from docx import Document
import pandas as pd
from tqdm import tqdm
import re
import string

# ---------- Step 1. Folder Paths ----------
RESUME_FOLDER = "data/resumes_docx"
OUTPUT_FOLDER = "data/cleaned"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ---------- Step 2. Helper Functions ----------

def extract_text_from_docx(file_path):
    """Extract text from a .docx file."""
    try:
        doc = Document(file_path)
        text = [para.text for para in doc.paragraphs]
        return '\n'.join(text)
    except Exception as e:
        print(f"⚠️ Error reading {file_path}: {e}")
        return ""

def clean_resume_text(text):
    """Clean text: remove URLs, punctuation, extra spaces, etc."""
    text = re.sub('http\\S+', ' ', text)              # remove urls
    text = re.sub('RT|cc', ' ', text)                 # remove RT and cc
    text = re.sub('#\\S+', '', text)                  # remove hashtags
    text = re.sub('@\\S+', ' ', text)                 # remove mentions
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)  # punctuation
    text = re.sub(r'\s+', ' ', text)                  # extra whitespace
    return text.strip().lower()                       # lowercase and trim

# ---------- Step 3. Process All Resumes ----------

resumes_data = []

print(f"📂 Extracting text from .docx resumes in '{RESUME_FOLDER}'...\n")

for filename in tqdm(os.listdir(RESUME_FOLDER)):
    if filename.endswith(".docx"):
        file_path = os.path.join(RESUME_FOLDER, filename)
        raw_text = extract_text_from_docx(file_path)
        cleaned_text = clean_resume_text(raw_text)
        resumes_data.append({
            "filename": filename,
            "resume_text": raw_text,
            "cleaned_resume": cleaned_text
        })

# ---------- Step 4. Save as CSV ----------

df = pd.DataFrame(resumes_data)
output_path = os.path.join(OUTPUT_FOLDER, "resumes_extracted.csv")
df.to_csv(output_path, index=False, encoding='utf-8')

print(f"\n✅ Extraction complete! {len(df)} resumes processed.")
print(f"💾 Saved to: {output_path}")
