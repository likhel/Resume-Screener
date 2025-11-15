"""
Embedding Scorer
----------------
Handles:
- SentenceTransformer model loading
- Embedding generation
- Cosine similarity scoring
"""

import torch
from sentence_transformers import SentenceTransformer, util


class EmbeddingScorer:

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Load sentence embedding model.
        """
        print(f" Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

    # ----------------------------------------------------
    def encode(self, text):
        """
        Convert text → embedding tensor
        """
        return self.model.encode(text, convert_to_tensor=True)

    # ----------------------------------------------------
    def compute_similarity(self, job_embedding, resume_embedding):
        """
        Returns cosine similarity (float).
        """
        return float(util.cos_sim(job_embedding, resume_embedding)[0][0])

    # ----------------------------------------------------
    def batch_similarity(self, job_embedding, resume_embeddings):
        """
        Fast scoring for all resumes at once.
        """
        scores = util.cos_sim(job_embedding, resume_embeddings)[0]
        return scores.cpu().numpy()
