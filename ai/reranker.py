"""
DevMentor AI — Document Reranker
==================================
Production-grade cross-encoder reranking with:

- Cross-encoder model for precise relevance scoring
- Lazy model loading (no startup delay)
- Batch processing for efficiency
- Multiple model support with fallback
- Score normalization and thresholding
- Reranking statistics tracking
- Graceful degradation on model failure
- BM25 score fusion (Reciprocal Rank Fusion)

How it works:
    After initial vector search retrieves candidate documents,
    the reranker re-scores each (query, document) pair using a
    cross-encoder model that reads both together — much more
    accurate than vector similarity alone.

    Vector search: fast, approximate, scales to millions of docs
    Reranker:      slow, precise, runs on top-k candidates only

Usage:
    from ai.reranker import Reranker

    reranker = Reranker()
    results = reranker.rerank(query, documents, top_k=5)

    for doc, score in results:
        print(f"Score: {score:.3f} | {doc[:100]}")
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from config import get_settings

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
settings = get_settings()

# ---------------------------------------------------------------------------
# Available reranker models (in order of preference)
# ---------------------------------------------------------------------------
RERANKER_MODELS = [
    "BAAI/bge-reranker-base",           # Fast, good quality (default)
    "cross-encoder/ms-marco-MiniLM-L-6-v2",  # Slightly smaller
    "cross-encoder/ms-marco-MiniLM-L-12-v2", # Better quality, slower
]


# ===========================================================================
# Data Classes
# ===========================================================================

@dataclass
class RankedDocument:
    """A document with its reranking score and metadata."""
    content: str
    score: float
    original_index: int
    original_score: Optional[float] = None   # Score before reranking
    normalized_score: Optional[float] = None  # Score normalized to 0-1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "score": self.score,
            "original_index": self.original_index,
            "original_score": self.original_score,
            "normalized_score": self.normalized_score,
            "metadata": self.metadata,
        }


@dataclass
class RerankResult:
    """Full reranking result with stats."""
    documents: List[RankedDocument]
    query: str
    model_used: str
    duration_ms: float
    total_candidates: int
    returned_count: int
    fallback_used: bool = False


# ===========================================================================
# Reranker
# ===========================================================================

class Reranker:
    """
    Production-grade document reranker using cross-encoder models.

    Cross-encoders jointly encode (query, document) pairs, making
    them significantly more accurate than bi-encoder similarity
    scores alone — at the cost of being slower.

    Best used on the top-k results from a fast initial retrieval
    (vector search or BM25), not on the full document corpus.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        max_length: int = 512,
        batch_size: int = 32,
    ):
        """
        Args:
            model_name:  Cross-encoder model name (HuggingFace)
            max_length:  Max token length per (query, doc) pair
            batch_size:  Batch size for model inference
        """
        self.model_name = model_name or settings.rag_reranker_model
        self.max_length = max_length
        self.batch_size = batch_size
        self._model = None
        self._model_loaded = False
        self._load_failed = False

    def _load_model(self) -> bool:
        """
        Lazy-load cross-encoder model on first use.
        Tries primary model first, falls back to alternatives.

        Returns:
            True if model loaded successfully
        """
        if self._model_loaded:
            return self._model is not None

        if self._load_failed:
            return False

        # Try primary model first
        models_to_try = [self.model_name] + [
            m for m in RERANKER_MODELS if m != self.model_name
        ]

        for model_name in models_to_try:
            try:
                from sentence_transformers import CrossEncoder
                logger.info("Loading reranker model: %s", model_name)
                start = time.time()

                self._model = CrossEncoder(
                    model_name,
                    max_length=self.max_length,
                )
                self.model_name = model_name
                load_time = (time.time() - start) * 1000

                logger.info(
                    "Reranker model loaded in %.0fms: %s",
                    load_time, model_name,
                )
                self._model_loaded = True
                return True

            except ImportError:
                logger.error(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                )
                self._load_failed = True
                return False

            except Exception as exc:
                logger.warning(
                    "Failed to load reranker model %s: %s — trying next",
                    model_name, exc,
                )
                continue

        logger.error(
            "All reranker models failed to load. "
            "Reranking will use fallback score passthrough."
        )
        self._load_failed = True
        self._model_loaded = True
        return False

    def rerank(
        self,
        query: str,
        documents: List[Union[str, Dict]],
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        return_scores: bool = True,
    ) -> List[Tuple[str, float]]:
        """
        Rerank documents by relevance to query.

        Simple interface — returns list of (document, score) tuples.

        Args:
            query:           Search query
            documents:       List of document strings or dicts with 'content' key
            top_k:           Number of top results to return (default: all)
            score_threshold: Minimum score to include (default: no threshold)
            return_scores:   Whether to include scores in output

        Returns:
            List of (document_text, score) tuples, sorted by relevance desc

        Example:
            results = reranker.rerank("What is RAG?", docs, top_k=3)
            for doc, score in results:
                print(f"{score:.3f}: {doc[:80]}")
        """
        result = self.rerank_detailed(query, documents, top_k, score_threshold)
        return [(doc.content, doc.score) for doc in result.documents]

    def rerank_detailed(
        self,
        query: str,
        documents: List[Union[str, Dict]],
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> RerankResult:
        """
        Rerank documents and return full result with stats.

        Args:
            query:           Search query
            documents:       List of document strings or dicts
            top_k:           Number of top results to return
            score_threshold: Minimum score threshold

        Returns:
            RerankResult with ranked documents and performance stats
        """
        start_time = time.time()

        if not documents:
            return RerankResult(
                documents=[],
                query=query,
                model_used=self.model_name,
                duration_ms=0,
                total_candidates=0,
                returned_count=0,
            )

        # Normalize documents to strings
        doc_texts, doc_metadata = self._normalize_documents(documents)
        total_candidates = len(doc_texts)

        # Determine top_k
        if top_k is None:
            top_k = settings.rag_top_k
        top_k = min(top_k, total_candidates)

        # Load model
        model_available = self._load_model()

        if model_available and self._model is not None:
            ranked_docs, fallback_used = self._rerank_with_model(
                query, doc_texts, doc_metadata, top_k, score_threshold
            )
        else:
            # Fallback: return documents in original order
            ranked_docs, fallback_used = self._fallback_rank(
                doc_texts, doc_metadata, top_k, score_threshold
            )

        duration_ms = (time.time() - start_time) * 1000

        logger.info(
            "Reranking complete: %d → %d docs in %.0fms (fallback=%s)",
            total_candidates, len(ranked_docs), duration_ms, fallback_used,
        )

        return RerankResult(
            documents=ranked_docs,
            query=query,
            model_used=self.model_name if not fallback_used else "fallback",
            duration_ms=duration_ms,
            total_candidates=total_candidates,
            returned_count=len(ranked_docs),
            fallback_used=fallback_used,
        )

    def _rerank_with_model(
        self,
        query: str,
        doc_texts: List[str],
        doc_metadata: List[Dict],
        top_k: int,
        score_threshold: Optional[float],
    ) -> Tuple[List[RankedDocument], bool]:
        """Run cross-encoder inference on (query, doc) pairs."""
        try:
            # Build (query, document) pairs
            pairs = [(query, doc) for doc in doc_texts]

            # Score in batches
            all_scores = []
            for i in range(0, len(pairs), self.batch_size):
                batch = pairs[i:i + self.batch_size]
                batch_scores = self._model.predict(
                    batch,
                    show_progress_bar=False,
                )
                all_scores.extend(batch_scores.tolist() if hasattr(batch_scores, 'tolist') else batch_scores)

            # Normalize scores to 0-1 range
            normalized = self._normalize_scores(all_scores)

            # Build RankedDocument list
            ranked = []
            for idx, (text, score, norm_score, meta) in enumerate(
                zip(doc_texts, all_scores, normalized, doc_metadata)
            ):
                if score_threshold is not None and norm_score < score_threshold:
                    continue
                ranked.append(RankedDocument(
                    content=text,
                    score=float(score),
                    original_index=idx,
                    normalized_score=float(norm_score),
                    metadata=meta,
                ))

            # Sort by score descending
            ranked.sort(key=lambda x: x.score, reverse=True)
            return ranked[:top_k], False

        except Exception as exc:
            logger.error("Cross-encoder inference failed: %s", exc)
            return self._fallback_rank(doc_texts, doc_metadata, top_k, score_threshold)

    def _fallback_rank(
        self,
        doc_texts: List[str],
        doc_metadata: List[Dict],
        top_k: int,
        score_threshold: Optional[float],
    ) -> Tuple[List[RankedDocument], bool]:
        """
        Fallback when model is unavailable.
        Returns documents in original order with uniform scores.
        """
        logger.warning("Using fallback reranking — documents returned in original order")
        ranked = [
            RankedDocument(
                content=text,
                score=1.0 - (idx * 0.01),  # Slight score decrease per position
                original_index=idx,
                normalized_score=1.0 - (idx * 0.01),
                metadata=meta,
            )
            for idx, (text, meta) in enumerate(zip(doc_texts, doc_metadata))
        ]
        return ranked[:top_k], True

    def _normalize_documents(
        self,
        documents: List[Union[str, Dict]],
    ) -> Tuple[List[str], List[Dict]]:
        """
        Normalize mixed document input to (texts, metadata) lists.

        Accepts:
        - List of strings
        - List of dicts with 'content', 'text', or 'page_content' key
        - Mixed list
        """
        texts = []
        metadata = []

        for doc in documents:
            if isinstance(doc, str):
                texts.append(doc)
                metadata.append({})
            elif isinstance(doc, dict):
                content = (
                    doc.get("content")
                    or doc.get("text")
                    or doc.get("page_content")
                    or str(doc)
                )
                texts.append(content)
                meta = {k: v for k, v in doc.items()
                        if k not in {"content", "text", "page_content"}}
                metadata.append(meta)
            else:
                texts.append(str(doc))
                metadata.append({})

        return texts, metadata

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """
        Normalize raw cross-encoder scores to 0-1 range using sigmoid.

        Cross-encoder scores are logits (unbounded). Sigmoid maps them
        to probabilities, making thresholding meaningful.
        """
        import math
        return [1 / (1 + math.exp(-s)) for s in scores]

    def rerank_with_fusion(
        self,
        query: str,
        documents: List[str],
        initial_scores: Optional[List[float]] = None,
        top_k: Optional[int] = None,
        fusion_weight: float = 0.7,
    ) -> List[Tuple[str, float]]:
        """
        Rerank using Reciprocal Rank Fusion (RRF) to combine:
        - Cross-encoder reranking scores
        - Initial retrieval scores (BM25 or vector similarity)

        RRF is robust and doesn't require score normalization.

        Args:
            query:          Search query
            documents:      Candidate documents
            initial_scores: Original retrieval scores (optional)
            top_k:          Number of results to return
            fusion_weight:  Weight for reranker vs initial scores (0-1)
                           1.0 = reranker only, 0.0 = initial scores only

        Returns:
            List of (document, fused_score) tuples
        """
        if not documents:
            return []

        top_k = top_k or settings.rag_top_k
        reranked = self.rerank(query, documents, top_k=len(documents))

        if initial_scores is None or len(initial_scores) != len(documents):
            return reranked[:top_k]

        # Build rank lookup from reranker
        reranker_ranks = {doc: rank for rank, (doc, _) in enumerate(reranked)}

        # Build rank lookup from initial scores
        initial_ranked = sorted(
            enumerate(documents),
            key=lambda x: initial_scores[x[0]] if x[0] < len(initial_scores) else 0,
            reverse=True,
        )
        initial_ranks = {doc: rank for rank, (_, doc) in enumerate(initial_ranked)}

        # Reciprocal Rank Fusion
        k = 60  # RRF constant
        fused_scores = {}
        for doc in documents:
            rr_score = 1 / (k + reranker_ranks.get(doc, len(documents)))
            initial_rr = 1 / (k + initial_ranks.get(doc, len(documents)))
            fused_scores[doc] = (
                fusion_weight * rr_score +
                (1 - fusion_weight) * initial_rr
            )

        sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs[:top_k]


# ===========================================================================
# Module-level singleton
# ===========================================================================
_reranker_instance: Optional[Reranker] = None


def get_reranker() -> Reranker:
    """Get or create the global Reranker singleton."""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = Reranker()
    return _reranker_instance


# ===========================================================================
# Convenience function (backward compatible)
# ===========================================================================

def rerank_documents(
    query: str,
    documents: List[str],
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    """
    Simple rerank function — backward compatible with original API.

    Args:
        query:     Search query
        documents: List of document strings
        top_k:     Number of top results to return

    Returns:
        List of (document, score) tuples sorted by relevance
    """
    return get_reranker().rerank(query, documents, top_k=top_k)