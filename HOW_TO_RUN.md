# How to Run the Resume Screener Project

## 📋 Overview

This is a **Resume Screening System** that uses machine learning to match resumes with job descriptions. The system uses:
- **Sentence Transformers** for semantic embeddings
- **spaCy** for Named Entity Recognition (NER)
- **Custom skill matching** algorithms
- **Smart weight selection** that adapts to different job types

---

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+** installed on your system
2. **pip** package manager
3. At least **4GB RAM** (for ML models)

### Step 1: Install Dependencies

Open a terminal in the project directory and run:

```bash
pip install -r requirements.txt
```

This will install all required packages including:
- PyTorch (deep learning framework)
- Sentence Transformers (for embeddings)
- spaCy (for NLP)
- scikit-learn (for ML utilities)
- And many more...

> ⚠️ **Note**: Installation may take 5-10 minutes depending on your internet speed, as it downloads large ML models.

### Step 2: Download spaCy Language Model

After installing requirements, download the English language model:

```bash
python -m spacy download en_core_web_sm
```

### Step 3: Prepare Your Data

The system expects:
- **Resumes** in the `data/` directory (as `.txt` or `.docx` files)
- **Job description** as a text file

If you have `.docx` resumes, extract them first:

```bash
python extract_resumes_docx.py
```

### Step 4: Run the Matcher

**Basic usage (recommended - uses Smart Mode):**

```bash
python -m matcher.job_resume_matcher job_ml_engineer.txt
```

This will:
1. Analyze the job description
2. Automatically select optimal weights
3. Match against all resumes in `data/`
4. Show top matches with scores

---

## 🎯 Usage Modes

### 1. Smart Mode (Default - Recommended)

```bash
python -m matcher.job_resume_matcher your_job_description.txt
```

**What it does:**
- Automatically detects job seniority (junior/mid/senior)
- Identifies role type (technical/creative/management)
- Selects optimal matching weights

**Example output:**
```
📊 Job Analysis:
   • Seniority: SENIOR
   • Role Type: TECHNICAL
   • Experience Required: 5 years
   • Technical Skills Mentioned: 18

🎯 Selected: SENIOR_TECHNICAL
   • Embedding: 45% | Skills: 30% | NER: 25%
```

### 2. Manual Modes

You can override the smart mode with specific weight configurations:

**Skills-focused** (for technical roles):
```bash
python -m matcher.job_resume_matcher job_description.txt skills
```

**Experience-focused** (for senior positions):
```bash
python -m matcher.job_resume_matcher job_description.txt experience
```

**Balanced** (general purpose):
```bash
python -m matcher.job_resume_matcher job_description.txt balanced
```

**Embeddings-focused** (for creative/soft-skill roles):
```bash
python -m matcher.job_resume_matcher job_description.txt embeddings
```

See [WEIGHT_MODES_GUIDE.md](WEIGHT_MODES_GUIDE.md) for detailed information on each mode.

---

## 📁 Project Structure

```
resume-screener/
├── data/                          # Resume files and results
│   └── results/                   # Output files
│       ├── match_results_hybrid.csv
│       └── weight_config.json
├── matcher/                       # Core matching module
├── modules/                       # ML modules (embeddings, NER, skills)
├── utils/                         # Utility functions
├── requirements.txt               # Python dependencies
├── job_ml_engineer.txt           # Sample job description
├── single_resume_job_matcher.py  # Match single resume
├── batch_resume_job_matcher.py   # Batch processing
├── demo_weight_modes.py          # Demo script
└── WEIGHT_MODES_GUIDE.md         # Detailed weight modes guide
```

---

## 🔧 Advanced Usage

### Match a Single Resume

```bash
python single_resume_job_matcher.py job_description.txt path/to/resume.txt
```

### Batch Processing

```bash
python batch_resume_job_matcher.py job_description.txt
```

### Demo All Weight Modes

```bash
python demo_weight_modes.py job_description.txt full
```

This compares all weight modes side-by-side.

### Encode Resumes (Optimization)

Pre-encode all resumes for faster matching:

```bash
python encode_resumes_optimized.py
```

---

## 🧠 Understanding the NER Pipeline

### What is NER?

**NER (Named Entity Recognition)** is a Natural Language Processing technique that identifies and extracts structured information from unstructured text. In this resume screening system, the NER pipeline extracts key entities from resumes and job descriptions.

### What Does the NER Pipeline Extract?

The NER pipeline (`modules/ner/ner_pipeline.py`) extracts the following entities:

1. **Skills** 🛠️
   - Technical skills (Python, Java, Docker, AWS, etc.)
   - Uses phrase matching against a curated skills list
   - Example: `["python", "machine learning", "docker", "kubernetes"]`

2. **Job Titles/Roles** 💼
   - Current and past positions
   - Example: `["senior software engineer", "data scientist"]`

3. **Organizations** 🏢
   - Companies where the candidate worked
   - Filters out tech terms (e.g., "Python" won't be tagged as an organization)
   - Example: `["Google", "Microsoft", "Amazon"]`

4. **Locations** 📍
   - Geographic locations (cities, countries)
   - Example: `["San Francisco", "New York", "Remote"]`

5. **Dates** 📅
   - Employment dates and time periods
   - Example: `["2020-2023", "January 2021"]`

6. **Experience Years** ⏱️
   - Calculated from date ranges or explicit mentions
   - Example: `5` (years)

### How NER Contributes to Matching

The NER pipeline is **one of three scoring components** in the hybrid matching system:

```
Final Score = (Embedding Score × W1) + (Skill Score × W2) + (NER Score × W3)
```

Where:
- **Embedding Score**: Semantic similarity (overall fit)
- **Skill Score**: Exact skill matching
- **NER Score**: Entity overlap (organizations, titles, experience)

**Example weights in Smart Mode:**
- Senior Technical Role: `Embedding 45% | Skills 30% | NER 25%`
- Junior Role: `Embedding 50% | Skills 35% | NER 15%`

### Why NER Matters

1. **Experience Validation** ✅
   - Verifies claimed years of experience
   - Matches seniority requirements

2. **Company Prestige** 🏆
   - Identifies candidates from top-tier companies
   - Useful for competitive positions

3. **Location Matching** 🌍
   - Filters for geographic preferences
   - Remote vs. on-site candidates

4. **Career Progression** 📈
   - Analyzes title progression over time
   - Identifies leadership experience

### Running the NER Pipeline Standalone

You can run the NER pipeline independently to extract entities from resumes:

```bash
python -m modules.ner.ner_pipeline
```

**Prerequisites:**
1. Resumes must be in `data/cleaned/resumes_extracted.csv`
2. Skills list must be in `data/skills/skills_list.txt`

**Output:**
- Creates `data/embeddings/resume_entities.csv` with extracted entities
- Each row contains JSON-formatted lists of entities per resume

**Example output structure:**
```csv
filename,skills,titles,organizations,locations,dates,experience_years
john_doe.txt,"[""python"", ""docker"", ""aws""]","[""senior engineer""]","[""Google"", ""Meta""]","[""San Francisco""]","[""2018-2023""]",5
```

### NER Pipeline Architecture

```
Resume Text
    ↓
┌─────────────────────────────────────┐
│  EntityExtractor                    │
│  ├─ spaCy NER Model                 │
│  │  └─ Extracts: ORG, GPE, LOC, DATE│
│  ├─ PhraseSkillMatcher              │
│  │  └─ Matches skills from list     │
│  └─ Heuristic Extractors            │
│     ├─ Title extraction             │
│     └─ Experience calculation       │
└─────────────────────────────────────┘
    ↓
Structured Entities (JSON)
```

### Tech Term Filtering

The NER pipeline includes **smart filtering** to prevent tech terms from being misclassified:

**Problem:** spaCy might tag "Python" or "Docker" as organizations
**Solution:** Maintains a blacklist of 40+ tech terms

```python
# These won't be tagged as organizations:
'python', 'java', 'docker', 'kubernetes', 'aws', 'tensorflow', etc.
```

This ensures accurate organization extraction!

---

## 📊 Understanding Results

Results are saved in `data/results/match_results_hybrid.csv` with columns:
- **resume_name**: Candidate's resume filename
- **hybrid_score**: Overall match score (0-100)
- **embedding_score**: Semantic similarity score
- **skill_score**: Skill matching score
- **ner_score**: Named entity matching score
- **rank**: Ranking position

Higher scores = better match!

---

## 🐛 Troubleshooting

### Issue: "No module named 'torch'"

**Solution:** Install PyTorch:
```bash
pip install torch
```

### Issue: "Can't find model 'en_core_web_sm'"

**Solution:** Download spaCy model:
```bash
python -m spacy download en_core_web_sm
```

### Issue: "No resumes found"

**Solution:** 
1. Check that resume files are in the `data/` directory
2. Ensure files are `.txt` format
3. If you have `.docx` files, run: `python extract_resumes_docx.py`

### Issue: Out of memory errors

**Solution:**
- Close other applications
- Process fewer resumes at once
- Use a machine with more RAM

---

## 💡 Tips for Best Results

1. **Job Descriptions**: Be specific and detailed in your job descriptions
2. **Resume Format**: Plain text works best; extract from `.docx` if needed
3. **Smart Mode**: Use smart mode for most cases - it adapts automatically
4. **Review Results**: Check the CSV file for detailed scoring breakdown
5. **Tune Weights**: If results aren't satisfactory, try different weight modes

---

## 📚 Additional Resources

- [WEIGHT_MODES_GUIDE.md](WEIGHT_MODES_GUIDE.md) - Comprehensive guide on weight modes
- `data/results/weight_config.json` - See what weights were used in your last run

---

## 🎓 Example Workflow

Here's a complete example workflow:

```bash
# 1. Install dependencies (first time only)
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Extract resumes from DOCX (if needed)
python extract_resumes_docx.py

# 3. Run the matcher with smart mode
python -m matcher.job_resume_matcher job_ml_engineer.txt

# 4. Check results
# Open data/results/match_results_hybrid.csv
```

---

## ❓ Need Help?

- Check [WEIGHT_MODES_GUIDE.md](WEIGHT_MODES_GUIDE.md) for mode selection
- Review the sample job description: `job_ml_engineer.txt`
- Run the demo: `python demo_weight_modes.py job_ml_engineer.txt`

---

**Happy Screening! 🎯**
