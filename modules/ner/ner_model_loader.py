"""
modules/ner/ner_model_loader.py

Loads a spaCy model (with fallback to download if missing).
"""

import spacy
import spacy.cli


class NERModelLoader:
    def __init__(self, model_name: str = "en_core_web_sm"):
        self.model_name = model_name
        print(f" Loading spaCy model: {self.model_name}")
        try:
            self.nlp = spacy.load(self.model_name)
        except OSError:
            # attempt automatic download then load
            print(f"Model {self.model_name} not found. Downloading...")
            
            spacy.cli.download(self.model_name)
            self.nlp = spacy.load(self.model_name)

    def get_model(self):
        return self.nlp
