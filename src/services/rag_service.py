import os
import logging
import numpy as np
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)

class RAGService:
    """
    Retrieval-Augmented Generation service.
    Uses semantic vector search (sentence-transformers) to find
    historically similar cases from the dataset.
    """
    EMBED_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, dataset_path: str, top_k: int = 3):
        self.dataset_path = dataset_path
        self.top_k = top_k
        self.df: Optional[pd.DataFrame] = None
        self.embeddings: Optional[np.ndarray] = None
        self._encoder = None

    def load(self):
        """Load dataset và pre-compute embeddings cho tất cả prompts."""
        if not os.path.exists(self.dataset_path):
            logger.warning(f"RAG dataset not found at '{self.dataset_path}'. RAG disabled.")
            return

        try:
            # Lazy import để không cần cài nếu không dùng
            from sentence_transformers import SentenceTransformer
            from sklearn.metrics.pairwise import cosine_similarity as _cs
            self._cosine_similarity = _cs

            self.df = pd.read_csv(self.dataset_path)
            self._encoder = SentenceTransformer(self.EMBED_MODEL)

            # Pre-compute embeddings 1 lần duy nhất khi startup
            prompts = self.df['prompt'].fillna("").tolist()
            self.embeddings = self._encoder.encode(prompts, show_progress_bar=False)

            logger.info(f"RAG loaded: {len(self.df)} cases | model={self.EMBED_MODEL}")
            print(f"✓ RAG Service loaded: {len(self.df)} cases (Vector Search enabled)")

        except ImportError:
            logger.warning("sentence-transformers not installed. Falling back to keyword search.")
            self._encoder = None
            if self.df is None:
                self.df = pd.read_csv(self.dataset_path)
        except Exception as e:
            logger.error(f"RAG load error: {e}")

    def query(self, user_prompt: str) -> str:
        """
        Tìm top-K trường hợp tương tự bằng semantic vector search.
        Fallback sang keyword search nếu encoder chưa sẵn sàng.
        """
        if self.df is None or self.df.empty:
            return ""

        if self._encoder is not None and self.embeddings is not None:
            similar_cases = self._vector_search(user_prompt)
        else:
            similar_cases = self._keyword_search(user_prompt)

        if similar_cases.empty:
            similar_cases = self.df.sample(min(2, len(self.df)))

        context = "### HISTORICAL REFERENCE CASES (RAG)\n"
        for _, row in similar_cases.iterrows():
            context += f"- PROMPT: {row['prompt']}\n"
            context += f"  ISSUES: {row['issues']}\n"
            context += f"  ACTIONS: {row['actions']}\n"
            context += f"  REFINED: {row['refined_prompt']}\n\n"

        return context

    def _vector_search(self, user_prompt: str) -> pd.DataFrame:
        """Semantic search bằng cosine similarity trên embedding vectors."""
        query_vec = self._encoder.encode([user_prompt])
        scores = self._cosine_similarity(query_vec, self.embeddings)[0]
        top_k_idx = scores.argsort()[::-1][:self.top_k]
        return self.df.iloc[top_k_idx]

    def _keyword_search(self, user_prompt: str) -> pd.DataFrame:
        """Fallback: keyword matching nếu không có vector encoder."""
        keywords = user_prompt.lower().split()
        mask = self.df['prompt'].str.contains('|'.join(keywords), case=False, na=False)
        return self.df[mask].head(self.top_k)
