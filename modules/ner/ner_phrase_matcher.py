"""
modules/ner/ner_phrase_matcher.py

Phrase-level skill extractor using spaCy PhraseMatcher.
Loads skills from a skills text file (one skill per line).
"""

import os
import re
from spacy.matcher import PhraseMatcher
from spacy.lang.en import English


class PhraseSkillMatcher:
    def __init__(self, skills_file: str = "data/skills/skills_list.txt", spacy_model: str = "en_core_web_sm"):
        self.skills_file = skills_file
        self.skills = self._load_skills(skills_file)
        # Use a lightweight tokenizer for phrase patterns
        self.nlp = English()  # tokenizer only
        self.matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        if self.skills:
            patterns = [self.nlp.make_doc(skill) for skill in self.skills]
            self.matcher.add("SKILL", patterns)
        print(f" PhraseSkillMatcher loaded {len(self.skills)} skills from {skills_file}")

    def _load_skills(self, path: str):
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            skills = [line.strip().lower() for line in f if line.strip()]
        # simple normalization: remove duplicates while preserving order
        seen = set()
        out = []
        for s in skills:
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out

    def normalize(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def extract(self, text: str):
        """
        Returns a list of matched skills (lowercased, deduped).
        """
        if not self.skills:
            return []
        doc = self.nlp(self.normalize(text))
        matches = self.matcher(doc)
        found = []
        for _match_id, start, end in matches:
            span = doc[start:end].text.strip().lower()
            if span not in found:
                found.append(span)
        return found
