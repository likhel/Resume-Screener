"""
Skill extraction helper using keyword/phrase matching.
Used by the NER pipeline and hybrid ranking system.
"""

import spacy
from spacy.matcher import PhraseMatcher


class SkillsExtractor:
    def __init__(self, skills_list=None, spacy_model="en_core_web_sm"):
        """
        skills_list: list of lowercase skill strings
        """
        self.nlp = spacy.load(spacy_model, disable=["ner"])
        self.matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")

        self.skills_list = skills_list or []

        # Convert skills list to spaCy docs and add to matcher
        patterns = [self.nlp.make_doc(skill) for skill in self.skills_list]
        if patterns:
            self.matcher.add("SKILL", patterns)

    def extract(self, text: str):
        """
        Extract skills found in text using phrase matching.

        Returns:
            A list of unique matched skills.
        """
        doc = self.nlp(text.lower())
        matches = self.matcher(doc)

        extracted = []
        for match_id, start, end in matches:
            span = doc[start:end].text.strip()
            extracted.append(span)

        # Deduplicate while preserving order
        seen = set()
        final_list = []
        for s in extracted:
            if s not in seen:
                final_list.append(s)
                seen.add(s)

        return final_list
