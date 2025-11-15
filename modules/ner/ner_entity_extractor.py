"""
modules/ner/ner_entity_extractor.py

Advanced EntityExtractor with tech term filtering.
Prevents tech terms from being tagged as organizations/locations.
"""

import re
from collections import OrderedDict
from typing import List

from modules.ner.ner_model_loader import NERModelLoader
from modules.ner.ner_phrase_matcher import PhraseSkillMatcher


class EntityExtractor:
    def __init__(self, spacy_model: str = "en_core_web_sm", skills_list_file: str = "data/skills/skills_list.txt"):
        # load spaCy model (with NER, tokenizer, sentence segmentation)
        loader = NERModelLoader(model_name=spacy_model)
        self.nlp = loader.get_model()

        # phrase matcher for skills
        self.phrase_matcher = PhraseSkillMatcher(skills_file=skills_list_file, spacy_model=spacy_model)

        # role/title heuristics tokens
        self._role_terms = ["engineer", "developer", "scientist", "analyst", "manager", "consultant", 
                           "architect", "intern", "lead", "principal", "director", "officer", "specialist",
                           "coordinator", "administrator", "designer", "programmer"]
        
        # Tech terms that shouldn't be tagged as ORG/LOC
        self._tech_blacklist = {
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#',
            'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'sklearn',
            'pandas', 'numpy', 'scipy', 'matplotlib',
            'aws', 'azure', 'gcp', 'google cloud',
            'docker', 'kubernetes', 'k8s',
            'mongodb', 'postgresql', 'mysql', 'redis',
            'react', 'angular', 'vue', 'node.js', 'nodejs',
            'sql', 'nosql', 'git', 'github', 'gitlab',
            'linux', 'unix', 'windows',
            'html', 'css', 'json', 'xml', 'yaml',
            'api', 'rest', 'graphql',
            'ml', 'ai', 'nlp', 'cv', 'mlops', 'devops',
            'spark', 'hadoop', 'kafka', 'airflow',
            'jupyter', 'anaconda', 'conda'
        }

    # ------------------------------
    def _is_tech_term(self, text: str) -> bool:
        """Check if text is a tech term that shouldn't be tagged as ORG/LOC"""
        text_lower = text.lower().strip()
        return text_lower in self._tech_blacklist

    # ------------------------------
    def _extract_experience_years(self, text: str) -> int:
        """
        Heuristic extraction:
        - looks for 'X years', 'X+ years', 'X yrs' patterns (returns max found)
        - looks for date ranges like 2018-2020 (returns max span)
        """
        if not isinstance(text, str):
            return 0
        text_l = text.lower()
        yrs = []
        for m in re.findall(r"(\d{1,2})\s*\+?\s*(?:years|yrs|year)", text_l):
            try:
                yrs.append(int(m))
            except:
                pass
        if yrs:
            return max(yrs)
        # date ranges like 2018-2020 or 2018 – 2020
        range_matches = re.findall(r"(20\d{2})\D{0,6}(20\d{2})", text_l)
        spans = []
        for a, b in range_matches:
            try:
                spans.append(int(b) - int(a))
            except:
                pass
        if spans:
            return max(spans)
        return 0

    # ------------------------------
    def _extract_title(self, doc) -> List[str]:
        """
        Heuristic title extraction:
         - first short line (<=8 tokens) often contains name or title
         - otherwise search sentences for role terms
        """
        text = doc.text.strip()
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        titles = []
        if lines:
            first = lines[0]
            if len(first.split()) <= 8 and any(rt in first.lower() for rt in self._role_terms):
                titles.append(first.strip().lower())
            elif len(first.split()) <= 6 and len(first.split()) > 1:
                # sometimes first line is "john doe\nbackend engineer"
                titles.append(first.strip().lower())
        # if not found, look for sentence containing role terms
        if not titles:
            for sent in doc.sents:
                s = sent.text.strip().lower()
                for rt in self._role_terms:
                    if rt in s:
                        titles.append(s)
                        break
                if titles:
                    break
        # dedupe preserve order
        return list(OrderedDict.fromkeys([t for t in titles if t]))

    # ------------------------------
    def extract(self, text: str) -> dict:
        """
        Main extraction function. Returns:
        {
            'skills': [...],
            'titles': [...],
            'organizations': [...],
            'locations': [...],
            'dates': [...],
            'experience_years': int
        }
        """
        if not isinstance(text, str) or not text.strip():
            return {
                "skills": [],
                "titles": [],
                "organizations": [],
                "locations": [],
                "dates": [],
                "experience_years": 0
            }

        # run spaCy pipeline
        doc = self.nlp(text)

        # extract spaCy NER entities (with filtering)
        orgs = []
        locs = []
        dates = []
        for ent in doc.ents:
            if ent.label_ == "ORG":
                # Filter out tech terms
                if not self._is_tech_term(ent.text):
                    orgs.append(ent.text)
            elif ent.label_ in ("GPE", "LOC"):
                # Filter out tech terms
                if not self._is_tech_term(ent.text):
                    locs.append(ent.text)
            elif ent.label_ == "DATE":
                dates.append(ent.text)

        # phrase-match skills (high precision)
        skills = self.phrase_matcher.extract(text)

        # title extraction heuristics
        titles = self._extract_title(doc)

        # experience heuristics
        exp_years = self._extract_experience_years(text)

        # normalize: dedupe while preserving order
        def _dedupe_keep_order(lst):
            return list(OrderedDict.fromkeys([s.strip() for s in lst if s and str(s).strip()]))

        return {
            "skills": _dedupe_keep_order([s.lower() for s in skills]),
            "titles": _dedupe_keep_order(titles),
            "organizations": _dedupe_keep_order(orgs),
            "locations": _dedupe_keep_order(locs),
            "dates": _dedupe_keep_order(dates),
            "experience_years": int(exp_years or 0)
        }