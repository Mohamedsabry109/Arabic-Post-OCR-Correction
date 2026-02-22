"""RAG retrieval module for Phase 5 (OpenITI corpus grounding).

Provides RAGRetriever which embeds query texts and retrieves the most
similar sentences from a pre-built FAISS index of OpenITI sentences.

The index is built once locally (run_phase5.py --mode build), then used
during the export step to inject retrieved context into each JSONL record.
The Kaggle/Colab inference step requires no changes — context is already
embedded in the JSONL.

Graceful degradation: if sentence-transformers or faiss are not installed,
RAGRetriever.enabled is False and retrieve() returns [].
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Optional heavy dependencies — loaded lazily with graceful fallback
_SentenceTransformer = None
_faiss = None
_np = None


def _try_import_deps() -> bool:
    """Attempt to import numpy, sentence_transformers, and faiss. Returns True if all available."""
    global _SentenceTransformer, _faiss, _np  # noqa: PLW0603
    if _SentenceTransformer is not None and _faiss is not None and _np is not None:
        return True
    try:
        import numpy as _numpy
        from sentence_transformers import SentenceTransformer as _ST
        import faiss as _f

        _np = _numpy
        _SentenceTransformer = _ST
        _faiss = _f
        return True
    except ImportError as exc:
        logger.warning(
            "RAG disabled: %s. Install numpy, sentence-transformers, and faiss-cpu.", exc
        )
        return False


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RetrievedChunk:
    """A single retrieved sentence with its similarity score."""

    text: str
    source_uri: str
    score: float    # cosine similarity in [0, 1]
    rank: int       # 1 = most similar


# ---------------------------------------------------------------------------
# RAGRetriever
# ---------------------------------------------------------------------------


class RAGRetriever:
    """Retrieve similar Arabic sentences from OpenITI using dense retrieval.

    Embeds a query text and returns top-K similar sentences from a pre-built
    FAISS index. Index is built once from corpus.jsonl and reused across runs.

    Usage::

        retriever = RAGRetriever(config)
        retriever.build_index(corpus_path, index_path)    # one-time
        chunks = retriever.retrieve(ocr_text, k=3)
        context = retriever.format_for_prompt(chunks)

    Graceful degradation: if sentence-transformers or faiss are not installed,
    RAGRetriever.enabled is False and retrieve() returns [].
    """

    def __init__(self, config: dict) -> None:
        """Initialise retriever with config settings.

        Attempts to load a pre-built FAISS index if index_path exists.
        Does NOT build the index — call build_index() or run
        ``python pipelines/run_phase5.py --mode build`` first.

        Args:
            config: Parsed config dict. Reads from config['rag'] and config['phase5'].
        """
        self._config = config
        rag_cfg = config.get("rag", {})
        p5_cfg = config.get("phase5", {}).get("retrieval", {})

        self._model_name: str = rag_cfg.get(
            "embedding_model",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
        self._default_k: int = p5_cfg.get("top_k", rag_cfg.get("top_k", 3))

        self._model = None
        self._index = None
        self._sentences: list[dict] = []   # ordered list matching FAISS row positions
        self._enabled = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """True if the FAISS index is loaded and retrieval is available."""
        return self._enabled

    @property
    def corpus_size(self) -> int:
        """Number of sentences in the loaded index."""
        return len(self._sentences)

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def build_index(
        self,
        corpus_path: Path,
        index_path: Path,
        batch_size: int = 256,
        show_progress: bool = True,
    ) -> None:
        """Build and save a FAISS IndexFlatIP from corpus.jsonl.

        Embeds all sentences in corpus_path in batches, builds a FAISS
        IndexFlatIP (inner product after L2 normalization = cosine similarity),
        and saves both the index and the sentence list to disk.

        Args:
            corpus_path: Path to corpus.jsonl (produced by OpenITILoader).
            index_path: Path to save the FAISS index (.faiss extension).
            batch_size: Embedding batch size (tune for available memory).
            show_progress: If True, show tqdm progress bar.

        Saves:
            {index_path}                  -- FAISS binary index
            {index_path}.sentences.jsonl  -- Ordered sentence records (for lookup)
        """
        if not _try_import_deps():
            raise RuntimeError(
                "Cannot build index: sentence-transformers and/or faiss not installed.\n"
                "Run: pip install sentence-transformers faiss-cpu"
            )

        if not corpus_path.exists():
            raise FileNotFoundError(
                f"Corpus file not found: {corpus_path}\n"
                "Run: python pipelines/run_phase5.py --mode build"
            )

        # Load corpus
        sentences: list[dict] = []
        with open(corpus_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        sentences.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

        if not sentences:
            raise ValueError(f"Corpus file is empty: {corpus_path}")

        logger.info(
            "Building FAISS index: %d sentences, model=%s",
            len(sentences), self._model_name,
        )

        # Load embedding model
        logger.info("Loading embedding model: %s", self._model_name)
        model = _SentenceTransformer(self._model_name)

        # Embed in batches
        texts = [s["text"] for s in sentences]
        logger.info("Embedding %d sentences (batch_size=%d) ...", len(texts), batch_size)

        try:
            from tqdm import tqdm as _tqdm
            disable_tqdm = not show_progress
        except ImportError:
            disable_tqdm = True

        embeddings_list = []
        for start in (
            range(0, len(texts), batch_size)
            if disable_tqdm
            else __import__("tqdm").tqdm(
                range(0, len(texts), batch_size),
                desc="Embedding",
                unit="batch",
                disable=not show_progress,
            )
        ):
            batch = texts[start: start + batch_size]
            emb = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
            embeddings_list.append(emb.astype(_np.float32))

        embeddings = _np.vstack(embeddings_list)

        # L2-normalize for cosine similarity via inner product
        norms = _np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = _np.where(norms == 0, 1.0, norms)
        embeddings = embeddings / norms

        # Build FAISS index
        dim = embeddings.shape[1]
        index = _faiss.IndexFlatIP(dim)
        index.add(embeddings)

        # Save index
        index_path.parent.mkdir(parents=True, exist_ok=True)
        _faiss.write_index(index, str(index_path))
        logger.info("FAISS index saved: %s (%d vectors, dim=%d)", index_path, len(sentences), dim)

        # Save sentence list alongside the index
        sentences_path = Path(str(index_path) + ".sentences.jsonl")
        with open(sentences_path, "w", encoding="utf-8") as f:
            for s in sentences:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        logger.info("Sentence list saved: %s", sentences_path)

        # Load into memory
        self._model = model
        self._index = index
        self._sentences = sentences
        self._enabled = True
        logger.info("Index built and loaded. RAGRetriever ready.")

    def load_index(self, index_path: Path) -> None:
        """Load a pre-built FAISS index from disk.

        Args:
            index_path: Path to the .faiss index file.

        Raises:
            FileNotFoundError: If index_path or its sentence list are missing.
            RuntimeError: If faiss / sentence-transformers are not installed.
        """
        if not _try_import_deps():
            raise RuntimeError(
                "Cannot load index: sentence-transformers and/or faiss not installed.\n"
                "Run: pip install sentence-transformers faiss-cpu"
            )

        sentences_path = Path(str(index_path) + ".sentences.jsonl")

        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {index_path}\n"
                "Run: python pipelines/run_phase5.py --mode build"
            )
        if not sentences_path.exists():
            raise FileNotFoundError(
                f"Sentence list not found: {sentences_path}\n"
                "Run: python pipelines/run_phase5.py --mode build"
            )

        logger.info("Loading FAISS index from %s ...", index_path)
        index = _faiss.read_index(str(index_path))

        sentences: list[dict] = []
        with open(sentences_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        sentences.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

        logger.info("Loading embedding model: %s", self._model_name)
        model = _SentenceTransformer(self._model_name)

        self._model = model
        self._index = index
        self._sentences = sentences
        self._enabled = True
        logger.info(
            "RAGRetriever ready: %d sentences in index.", len(sentences)
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query_text: str,
        k: Optional[int] = None,
        min_score: float = 0.0,
    ) -> list[RetrievedChunk]:
        """Retrieve top-K most similar sentences for query_text.

        If the retriever is disabled (index not loaded), returns [].

        Args:
            query_text: OCR text to use as query.
            k: Number of results. Defaults to config phase5.retrieval.top_k.
            min_score: Minimum cosine similarity to include (default: 0.0).

        Returns:
            List of RetrievedChunk sorted by score descending.
        """
        if not self._enabled:
            return []

        if not query_text or not query_text.strip():
            return []

        k = k if k is not None else self._default_k
        k = min(k, len(self._sentences))

        # Embed query
        emb = self._model.encode([query_text], convert_to_numpy=True).astype(_np.float32)
        norm = _np.linalg.norm(emb, axis=1, keepdims=True)
        if norm[0, 0] == 0:
            return []
        emb = emb / norm

        # Search
        scores, indices = self._index.search(emb, k)
        scores = scores[0]
        indices = indices[0]

        chunks: list[RetrievedChunk] = []
        rank = 1
        for score, idx in zip(scores, indices):
            if idx < 0 or idx >= len(self._sentences):
                continue
            score_f = float(score)
            if score_f < min_score:
                continue
            sentence = self._sentences[idx]
            chunks.append(RetrievedChunk(
                text=sentence.get("text", ""),
                source_uri=sentence.get("source_uri", ""),
                score=round(score_f, 4),
                rank=rank,
            ))
            rank += 1

        return chunks

    # ------------------------------------------------------------------
    # Prompt formatting
    # ------------------------------------------------------------------

    def format_for_prompt(
        self,
        chunks: list[RetrievedChunk],
        style: str = "numbered_arabic",
        max_chars_per_chunk: int = 150,
    ) -> str:
        """Format retrieved sentences as Arabic context for prompt injection.

        Args:
            chunks: Retrieved chunks from retrieve().
            style: Formatting style:
                - "numbered_arabic" (default): numbered list with header.
                - "plain": newline-separated sentences only.
            max_chars_per_chunk: Truncate each sentence to this length.

        Returns:
            Multi-line Arabic string. Empty string if chunks is [].
        """
        if not chunks:
            return ""

        truncated = [
            (c.text[:max_chars_per_chunk] if len(c.text) > max_chars_per_chunk else c.text)
            for c in chunks
        ]

        if style == "plain":
            return "\n".join(truncated)

        # numbered_arabic
        header = "نصوص مرجعية صحيحة:"
        lines = [header]
        for i, text in enumerate(truncated, 1):
            lines.append(f"{i}. {text}")
        return "\n".join(lines)
