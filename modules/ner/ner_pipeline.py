"""
modules/ner/ner_pipeline.py

High-level pipeline that loads cleaned resumes, processes them with EntityExtractor,
and writes structured entities to CSV.

Run this from the project root (resume-screener/) as:
    python -m modules.ner.ner_pipeline
or
    python modules/ner/ner_pipeline.py
(Prefer the -m form for package-relative imports.)
"""

import os
import json
import pandas as pd

from modules.ner.ner_entity_extractor import EntityExtractor

RESUME_CSV = "data/cleaned/resumes_extracted.csv"
OUTPUT_CSV = "data/embeddings/resume_entities.csv"
SKILL_FILE = "data/skills/skills_list.txt"


def load_skill_list(path: str):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip().lower() for line in f if line.strip()]
    return []


def run():
    print(" Loading skills...")
    skills = load_skill_list(SKILL_FILE)

    print(" Initializing EntityExtractor...")
    extractor = EntityExtractor(spacy_model="en_core_web_sm", skills_list_file=SKILL_FILE)

    if not os.path.exists(RESUME_CSV):
        raise FileNotFoundError(f"Resumes CSV not found: {RESUME_CSV}. Run your resume extraction step first.")

    print(" Loading resumes...")
    df = pd.read_csv(RESUME_CSV)

    rows = []
    print(f" Processing {len(df)} resumes...")
    for idx, row in df.iterrows():
        text = row.get("cleaned_resume", "") or ""
        ents = extractor.extract(text)
        rows.append({
            "filename": row.get("filename"),
            "skills": json.dumps(ents.get("skills", []), ensure_ascii=False),
            "titles": json.dumps(ents.get("titles", []), ensure_ascii=False),
            "organizations": json.dumps(ents.get("organizations", []), ensure_ascii=False),
            "locations": json.dumps(ents.get("locations", []), ensure_ascii=False),
            "dates": json.dumps(ents.get("dates", []), ensure_ascii=False),
            "experience_years": ents.get("experience_years", 0)
        })

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f" Saved extracted entities to {OUTPUT_CSV}")


if __name__ == "__main__":
    run()

