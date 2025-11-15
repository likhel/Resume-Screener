"""
Text cleaner utility used across the resume processing pipeline.
Applies normalization, whitespace cleanup, URL/email removal, etc.
"""

import re


class TextCleaner:

    @staticmethod
    def clean_text(text: str) -> str:
        if not isinstance(text, str):
            return ""

        # Remove URLs
        text = re.sub(r"http\S+|www\.\S+", " ", text)

        # Remove emails
        text = re.sub(r"\S+@\S+", " ", text)

        # Remove phone numbers
        text = re.sub(r"\b\d{10,}\b", " ", text)

        # Remove HTML tags
        text = re.sub(r"<.*?>", " ", text)

        # Remove special characters except basic punctuation
        text = re.sub(r"[^a-zA-Z0-9,.?!()\-\s]", " ", text)

        # Normalize multiple spaces
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    @staticmethod
    def clean_for_ner(text: str) -> str:
        """
        Lighter cleaning specifically for NER (preserves capitalization & structure).
        """
        if not isinstance(text, str):
            return ""

        # Remove URLs & emails — these confuse NER
        text = re.sub(r"http\S+|www\.\S+", " ", text)
        text = re.sub(r"\S+@\S+", " ", text)

        # Keep punctuation, remove junk
        text = re.sub(r"[^a-zA-Z0-9,.?!()&/\-\s]", " ", text)

        # Normalize spacing
        text = re.sub(r"\s+", " ", text)

        return text.strip()
