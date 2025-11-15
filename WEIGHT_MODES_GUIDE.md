# Resume Screening System - Weight Modes Guide

##  Quick Start

### Basic Usage (Smart Mode - Recommended)
```bash
python -m matcher.job_resume_matcher job_description.txt
```
The system automatically detects job characteristics and selects optimal weights.

---

##  Available Modes

### 1. **SMART Mode** (Default) 
```bash
python -m matcher.job_resume_matcher job_description.txt smart
```

**What it does:**
- Automatically analyzes the job description
- Detects seniority level (junior/mid/senior)
- Identifies role type (technical/creative/management)
- Counts technical skills mentioned
- Extracts experience requirements
- **Selects optimal weights automatically**

**Best for:**
-  General use (works for any job)
-  When you don't know the best weights
-  Demos and presentations (shows intelligence)

**Example Output:**
```
 Job Analysis:
   • Seniority: SENIOR
   • Role Type: TECHNICAL
   • Experience Required: 5 years
   • Technical Skills Mentioned: 18

  Selected: SENIOR_TECHNICAL
   • Embedding: 45% | Skills: 30% | NER: 25%
```

---

### 2. **BALANCED Mode** 
```bash
python -m matcher.job_resume_matcher job_description.txt balanced
```

**Weights:** Embedding 50% | Skills 35% | NER 15%

**Best for:**
-  General-purpose matching
-  When job requirements are mixed
-  Mid-level positions
-  Fallback/default option

---

### 3. **SKILLS Mode** 
```bash
python -m matcher.job_resume_matcher job_description.txt skills
```

**Weights:** Embedding 40% | Skills 45% | NER 15%

**Best for:**
- Highly technical positions
- When specific skills are mandatory
- DevOps, Backend, Frontend roles
- Positions with 10+ required skills

---

### 4. **EXPERIENCE Mode** 
```bash
python -m matcher.job_resume_matcher job_description.txt experience
```

**Weights:** Embedding 45% | Skills 30% | NER 25%

**Best for:**
- Senior positions (5+ years)
- Leadership roles
- Principal/Staff level
- When experience is critical

---

### 5. **EMBEDDINGS Mode** 🧠
```bash
python -m matcher.job_resume_matcher job_description.txt embeddings
```

**Weights:** Embedding 60% | Skills 30% | NER 10%

**Best for:**
- Creative roles (Designer, Writer)
- Soft skill positions
- Product Management
- When overall fit matters more than exact skills

---

## Custom Weights (Programmatic)

For advanced use, you can set custom weights programmatically:

```python
from matcher.job_resume_matcher import match_resumes

# Custom weights
results = match_resumes(
    job_description,
    top_k=5,
    weight_mode='custom',
    custom_weights={
        'embedding': 0.55,
        'skill': 0.30,
        'ner': 0.15
    }
)
```

---

## Demo Scripts

### Quick Demo (2 minutes)
```bash
python demo_weight_modes.py job_description.txt
```
Shows smart weight selection in action.

### Full Demo (5-10 minutes)
```bash
python demo_weight_modes.py job_description.txt full
```
Compares all weight modes side-by-side.

---

## Presentation Flow

### Recommended Demo Sequence:

1. **Show Smart Mode (30 sec)**
   ```bash
   python -m matcher.job_resume_matcher senior_ml_job.txt
   ```
   Highlight how it detects "senior technical" automatically.

2. **Show Different Job Type (30 sec)**
   ```bash
   python -m matcher.job_resume_matcher junior_dev_job.txt
   ```
   Show how weights change for junior roles.

3. **Compare Modes (1 min)**
   ```bash
   python demo_weight_modes.py job_description.txt full
   ```
   Show how rankings change with different weights.

---

## Configuration Files

After running, check these files:

- `data/results/match_results_hybrid.csv` - Detailed match results
- `data/results/weight_config.json` - Weights used for the run

Example `weight_config.json`:
```json
{
  "mode": "smart",
  "weights": {
    "embedding": 0.45,
    "skill": 0.30,
    "ner": 0.25
  },
  "timestamp": "2024-01-15T10:30:00"
}
```

---

## Tips for Presentation

### Key Talking Points:

1. **"The system adapts automatically"**
   - Different roles need different evaluation criteria
   - Junior = potential, Senior = experience
   - Technical = exact skills, Creative = overall fit

2. **"But we keep manual control"**
   - Can override with specific modes
   - Flexible for edge cases
   - Best of both worlds

3. **"Real-world applicability"**
   - HR systems do this in practice
   - Shows production-ready thinking
   - Not just a toy project

### Potential Questions & Answers:

**Q: Why not always use smart mode?**
A: Smart mode is default, but manual modes let HR tune for specific needs or edge cases.

**Q: How does smart detection work?**
A: It analyzes seniority keywords, technical terms, and experience requirements using NLP patterns.

**Q: Can it be wrong?**
A: Yes, edge cases exist. That's why we provide manual override options.

---

## Quick Reference

| Job Type | Recommended Mode | Why |
|----------|-----------------|-----|
| Senior ML Engineer | `experience` or `smart` | 5+ years, technical |
| Junior Developer | `smart` (auto: junior_technical) | Fresh grad, potential |
| DevOps Engineer | `skills` or `smart` | Many tools required |
| Product Manager | `embeddings` or `smart` | Soft skills + experience |
| UI/UX Designer | `embeddings` | Creative fit |
| Data Analyst | `balanced` or `skills` | Mix of technical + analysis |

---
## Summary

**Default:** Use `smart` mode - it works for 90% of cases

**Override:** Use specific modes when you know the job type well

**Custom:** Use programmatic API for integration with other systems

**Demo:** Use demo scripts for presentations