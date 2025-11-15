"""
Hybrid Ranker
-------------
Combines:
- Embedding similarity
- Phrase-level skill matches
- NER entity bonuses (organizations, locations, titles, experience)
- Weighted scoring system

Final Score = (0.50 * embedding_score)
            + (0.35 * skill_match_score)
            + (0.15 * ner_bonus_score)
"""


class HybridRanker:

    def __init__(self,
                 w_embedding=0.50,
                 w_skill=0.35,
                 w_ner_bonus=0.15):
        """
        Initialize with adjustable weights.
        Default: More balanced between embedding and skills.
        """
        self.w_embedding = w_embedding
        self.w_skill = w_skill
        self.w_ner_bonus = w_ner_bonus
        
        # Normalize weights
        total = w_embedding + w_skill + w_ner_bonus
        if abs(total - 1.0) > 0.01:
            self.w_embedding /= total
            self.w_skill /= total
            self.w_ner_bonus /= total

    # ----------------------------------------------------
    def compute_skill_overlap(self, job_skills, resume_skills):
        """
        Returns weighted score combining:
        - Jaccard similarity (intersection / union)
        - Coverage (what % of job requirements met)
        
        Higher weight on coverage since meeting requirements matters more.
        """
        if not job_skills:
            return 0.0
            
        job_set = set(s.lower().strip() for s in job_skills)
        res_set = set(s.lower().strip() for s in resume_skills)

        if len(job_set) == 0:
            return 0.0

        intersection = len(job_set & res_set)
        union = len(job_set | res_set)
        
        # Coverage: what % of job requirements are met
        coverage = intersection / len(job_set)
        
        # Jaccard: intersection / union (handles over-qualified candidates)
        jaccard = intersection / union if union > 0 else 0.0
        
        # Weighted combination (favor coverage)
        return 0.7 * coverage + 0.3 * jaccard

    # ----------------------------------------------------
    def compute_ner_bonus(self, job_entities, resume_entities):
        """
        Reward resumes with matching entities.
        
        Entities from EntityExtractor:
        - organizations: list of companies/orgs
        - locations: list of places
        - titles: list of job titles
        - skills: list of extracted skills
        - experience_years: int
        
        Scoring breakdown:
        - Experience match: 40%
        - Title keywords: 30%
        - Location match: 15%
        - Organization match: 15%
        """
        bonus = 0.0
        
        # 1. Experience matching (40% of bonus)
        job_exp = job_entities.get('experience_years', 0)
        resume_exp = resume_entities.get('experience_years', 0)
        
        if job_exp > 0 and resume_exp > 0:
            if resume_exp >= job_exp:
                bonus += 0.40  # Meets or exceeds requirement
            elif resume_exp >= job_exp - 2:
                bonus += 0.20  # Close enough (within 2 years)
            elif resume_exp >= job_exp - 4:
                bonus += 0.10  # Somewhat close
        
        # 2. Title matching (30% of bonus)
        job_titles = [t.lower() for t in job_entities.get('titles', [])]
        resume_titles = [t.lower() for t in resume_entities.get('titles', [])]
        
        if job_titles and resume_titles:
            # Extract keywords from titles (remove common words)
            stop_words = {'the', 'a', 'an', 'and', 'or', 'for', 'with', 'at', 'in', 'on'}
            
            title_match = False
            for job_title in job_titles:
                job_words = set(job_title.split()) - stop_words
                for resume_title in resume_titles:
                    resume_words = set(resume_title.split()) - stop_words
                    overlap = job_words & resume_words
                    
                    # If 2+ keywords match, consider it a match
                    if len(overlap) >= 2:
                        bonus += 0.30
                        title_match = True
                        break
                if title_match:
                    break
        
        # 3. Location matching (15% of bonus)
        job_locs = set(loc.lower().strip() for loc in job_entities.get('locations', []))
        resume_locs = set(loc.lower().strip() for loc in resume_entities.get('locations', []))
        
        if job_locs and resume_locs and (job_locs & resume_locs):
            bonus += 0.15
        
        # 4. Organization matching (15% of bonus)
        # Bonus if candidate worked at companies mentioned in job description
        job_orgs = set(org.lower().strip() for org in job_entities.get('organizations', []))
        resume_orgs = set(org.lower().strip() for org in resume_entities.get('organizations', []))
        
        if job_orgs and resume_orgs and (job_orgs & resume_orgs):
            bonus += 0.15

        return min(bonus, 1.0)  # Cap at 1.0

    # ----------------------------------------------------
    def hybrid_score(self, embedding_score,
                     job_skills, resume_skills,
                     job_entities, resume_entities):
        """
        Compute final hybrid score.
        
        Returns dict with all component scores + final score.
        """
        skill_overlap = self.compute_skill_overlap(job_skills, resume_skills)
        ner_bonus = self.compute_ner_bonus(job_entities, resume_entities)

        final_score = (
            self.w_embedding * embedding_score +
            self.w_skill * skill_overlap +
            self.w_ner_bonus * ner_bonus
        )

        return {
            "embedding": round(float(embedding_score), 6),
            "skill_overlap": round(float(skill_overlap), 6),
            "ner_bonus": round(float(ner_bonus), 6),
            "final_score": round(float(final_score), 6)
        }