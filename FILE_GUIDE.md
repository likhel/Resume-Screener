# 📁 Resume Screener - File Guide

## 🎯 Quick Start: Which Files to Run?

### **Main Entry Point (START HERE!)**

```bash
# Run the complete resume screening system
python -m matcher.job_resume_matcher job_ml_engineer.txt
```

This is the **primary file** you should run to see the project in action!

---

## 🚀 Executable Files (Scripts You Can Run)

### **1. Main Matcher (Primary)**

#### `matcher/job_resume_matcher.py` ⭐ **START HERE**
**Purpose:** Main resume screening engine - matches resumes against a job description

**How to run:**
```bash
python -m matcher.job_resume_matcher job_ml_engineer.txt
# or with specific weight mode:
python -m matcher.job_resume_matcher job_ml_engineer.txt skills
```

**What it does:**
- Loads all resumes from `data/` directory
- Analyzes the job description
- Automatically selects optimal weights (Smart Mode)
- Scores each resume using 3 methods:
  - Semantic embeddings (overall fit)
  - Skill matching (exact skills)
  - NER matching (experience, companies, titles)
- Outputs ranked results to `data/results/match_results_hybrid.csv`

**Output example:**
```
📊 Job Analysis:
   • Seniority: SENIOR
   • Role Type: TECHNICAL
   
🎯 Selected: SENIOR_TECHNICAL
   • Embedding: 45% | Skills: 30% | NER: 25%

Top 5 Matches:
1. john_doe.txt - Score: 87.5
2. jane_smith.txt - Score: 82.3
...
```

---

### **2. Single Resume Matcher**

#### `single_resume_job_matcher.py`
**Purpose:** Match ONE specific resume against a job description

**How to run:**
```bash
python single_resume_job_matcher.py job_ml_engineer.txt data/resume_john_doe.txt
```

**Use case:** Quick testing or evaluating a specific candidate

---

### **3. Batch Processor**

#### `batch_resume_job_matcher.py`
**Purpose:** Process multiple resumes in batches (same as main matcher but with batch optimization)

**How to run:**
```bash
python batch_resume_job_matcher.py job_ml_engineer.txt
```

**Use case:** Large-scale resume screening with progress tracking

---

### **4. Demo Scripts**

#### `demo_weight_modes.py` 🎬 **GREAT FOR PRESENTATIONS**
**Purpose:** Demonstrates how different weight modes affect ranking

**How to run:**
```bash
# Quick demo (shows smart mode)
python demo_weight_modes.py job_ml_engineer.txt

# Full demo (compares all modes side-by-side)
python demo_weight_modes.py job_ml_engineer.txt full
```

**What it shows:**
- How Smart Mode analyzes jobs
- How different weight configurations change rankings
- Side-by-side comparison of all modes

**Perfect for:** Presentations, demos, understanding the system

---

### **5. Testing & Diagnostics**

#### `test_smart_weights.py`
**Purpose:** Unit tests for smart weight selection logic

**How to run:**
```bash
python test_smart_weights.py
```

**What it tests:**
- Senior technical job detection
- Junior role detection
- Creative role detection
- Weight selection accuracy

---

#### `diagnostic_analyzer.py`
**Purpose:** Diagnostic tool to verify system components are working

**How to run:**
```bash
python diagnostic_analyzer.py
```

**What it checks:**
- Skill extraction working correctly
- NER pipeline functioning
- Model loading properly
- Sample job description analysis

---

### **6. Utility Scripts**

#### `extract_resumes_docx.py`
**Purpose:** Converts `.docx` resume files to `.txt` format

**How to run:**
```bash
python extract_resumes_docx.py
```

**When to use:** When you have Word document resumes that need to be processed

---

#### `encode_resumes_optimized.py`
**Purpose:** Pre-computes embeddings for all resumes (speeds up future matching)

**How to run:**
```bash
python encode_resumes_optimized.py
```

**What it does:**
- Loads all resumes
- Generates embeddings using Sentence-BERT
- Saves to `data/embeddings/resume_embeddings.pt`

**Benefit:** Faster matching (embeddings are cached)

---

#### `clean_job_description.py`
**Purpose:** Cleans and normalizes job description text

**How to run:**
```bash
python clean_job_description.py
```

---

#### `weight_tuning_experiment.py`
**Purpose:** Experimental script for tuning weight configurations

**How to run:**
```bash
python weight_tuning_experiment.py
```

**Use case:** Research and optimization of weight parameters

---

## 📚 Core Modules (Library Files - Not Directly Runnable)

### **Matcher Module**

#### `matcher/job_resume_matcher.py`
- Main matching logic
- Hybrid scoring system
- Weight mode selection
- Result generation

---

### **NER Module** (`modules/ner/`)

#### `ner_pipeline.py`
**Purpose:** High-level NER pipeline orchestrator

**Can run standalone:**
```bash
python -m modules.ner.ner_pipeline
```

**What it does:**
- Processes all resumes
- Extracts entities (skills, companies, titles, etc.)
- Saves to `data/embeddings/resume_entities.csv`

---

#### `ner_entity_extractor.py`
**Purpose:** Core entity extraction logic
- Extracts skills, titles, organizations, locations, dates
- Calculates experience years
- Filters tech terms from being tagged as organizations

---

#### `ner_model_loader.py`
**Purpose:** Loads and caches spaCy NER model

---

#### `ner_phrase_matcher.py`
**Purpose:** Phrase-based skill matching using spaCy's PhraseMatcher

---

### **Scorer Module** (`modules/scorer/`)

Contains scoring algorithms for different matching methods:
- Embedding similarity scoring
- Skill matching scoring
- NER entity overlap scoring
- Hybrid score calculation

---

### **Utils Module** (`utils/`)

#### `skills_extractor.py`
**Purpose:** Utility functions for extracting skills from text

---

#### `text_cleaner.py`
**Purpose:** Text preprocessing and cleaning utilities

---

## 📄 Configuration & Data Files

### **Configuration**

#### `requirements.txt`
**Purpose:** Python package dependencies
```bash
pip install -r requirements.txt
```

---

#### `environment.yml`
**Purpose:** Conda environment specification (alternative to requirements.txt)
```bash
conda env create -f environment.yml
```

---

### **Documentation**

#### `HOW_TO_RUN.md` 📖
**Purpose:** Comprehensive setup and usage guide (the file I created for you!)

---

#### `WEIGHT_MODES_GUIDE.md` 📖
**Purpose:** Detailed explanation of all weight modes
- Smart Mode
- Balanced Mode
- Skills Mode
- Experience Mode
- Embeddings Mode

---

### **Sample Data**

#### `job_ml_engineer.txt`
**Purpose:** Sample job description for testing
- Use this for your first run!

---

## 🗂️ Data Directory Structure

```
data/
├── cleaned/              # Cleaned resume text
│   └── resumes_extracted.csv
├── embeddings/           # Pre-computed embeddings
│   ├── resume_embeddings.pt
│   ├── resume_entities.csv
│   └── resume_metadata.csv
├── results/              # Matching results
│   ├── match_results_hybrid.csv
│   └── weight_config.json
└── skills/               # Skills database
    └── skills_list.txt
```

---

## 🎓 Recommended Workflow for First-Time Users

### **Step 1: Install Dependencies**
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### **Step 2: Prepare Data (if needed)**
```bash
# If you have .docx resumes:
python extract_resumes_docx.py
```

### **Step 3: Run the Main Matcher** ⭐
```bash
python -m matcher.job_resume_matcher job_ml_engineer.txt
```

### **Step 4: Check Results**
Open `data/results/match_results_hybrid.csv` to see ranked candidates!

### **Step 5: Try the Demo (Optional)**
```bash
python demo_weight_modes.py job_ml_engineer.txt full
```

---

## 🔍 File Categories Summary

| Category | Files | Purpose |
|----------|-------|---------|
| **Main Entry** | `matcher/job_resume_matcher.py` | Primary screening engine |
| **Demos** | `demo_weight_modes.py` | Presentations & understanding |
| **Testing** | `test_smart_weights.py`, `diagnostic_analyzer.py` | Verification & debugging |
| **Utilities** | `extract_resumes_docx.py`, `encode_resumes_optimized.py` | Data preparation |
| **Core Modules** | `modules/ner/*`, `modules/scorer/*` | Library code |
| **Documentation** | `HOW_TO_RUN.md`, `WEIGHT_MODES_GUIDE.md` | Guides |
| **Config** | `requirements.txt`, `environment.yml` | Setup |

---

## 💡 Quick Reference

**Want to see the project work?**
```bash
python -m matcher.job_resume_matcher job_ml_engineer.txt
```

**Want to understand how it works?**
```bash
python demo_weight_modes.py job_ml_engineer.txt full
```

**Want to test a single resume?**
```bash
python single_resume_job_matcher.py job_ml_engineer.txt path/to/resume.txt
```

**Want to extract entities from resumes?**
```bash
python -m modules.ner.ner_pipeline
```

---

**Happy Screening! 🎯**
