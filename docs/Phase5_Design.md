# Phase 5: Retrieval-Augmented Generation (RAG) — Design Document

## 1. Overview

### 1.1 Purpose

Phase 5 tests whether grounding the LLM in a large Arabic corpus (OpenITI) improves OCR
correction. It is an isolated experiment, comparing only against Phase 2 (zero-shot baseline).

| Aspect | Detail |
|--------|--------|
| **Research Question** | Does retrieving similar correct Arabic sentences help the LLM correct OCR errors? |
| **Knowledge Source** | OpenITI corpus (`data/OpenITI/`) — classical/medieval Arabic texts |
| **Mechanism** | For each OCR text, retrieve top-K similar sentences; inject as context into system prompt |
| **Comparison** | Phase 5 vs Phase 2 (isolated) |
| **Prompt Version** | `p5v1` |
| **New Module** | `src/core/rag_retriever.py` — `RAGRetriever` class |
| **Modified Module** | `src/data/knowledge_base.py` — add `OpenITILoader` class |
| **New Pipeline** | `pipelines/run_phase5.py` |

### 1.2 Hypothesis

If the LLM sees correct Arabic sentences that are lexically similar to the OCR input, it has
richer vocabulary context to guide its corrections. This should help with rare words and
domain-specific terminology.

**Counter-hypothesis to test**: OpenITI is classical/medieval Arabic, while PATS-A01 is
typewritten modern Arabic and KHATT is handwritten modern Arabic. The genre mismatch may
limit retrieval quality — this is itself a research finding worth reporting.

### 1.3 Isolated Comparison Design

> **CRITICAL**: Phase 5 compares **only against Phase 2** (zero-shot baseline).
> No comparison with Phase 1, Phase 3, or Phase 4.
> This isolates the contribution of corpus grounding over a vanilla LLM.
>
> Key metric: Δ CER = Phase 5 CER − Phase 2 CER (negative = improvement)

### 1.4 Downstream Use

- **Phase 6**: RAG context is one component in combination experiments
- **Paper**: Row "5: +RAG (OpenITI)" in the main results table
- **Ablation (Phase 6)**: `−RAG` condition removes corpus grounding from the full system

---

## 2. Data Source: OpenITI

### 2.1 Corpus Overview

```
data/OpenITI/
├── OpenITI_metadata_2023-1-8.csv  # TSV file with metadata for all texts
├── README.md
├── release_notes/
└── data/                          # 3,353 author directories
    └── {AuthorID}{AuthorName}/
        └── {AuthorID}{AuthorName}.{BookTitle}/
            ├── {version_uri}-ara1         # Plain text (no extension)
            ├── {version_uri}-ara1.mARkdown  # Annotated markdown
            └── README.md
```

**Scale**:
- 13,158 text files (8,600 `pri` status + 4,558 `sec` status)
- ~21.6 GB total
- ~1.05 billion tokens in `pri`-status texts alone
- Spans classical Arabic literature, hadith, tafsir, poetry, history, science

### 2.2 File Format: OpenITI Plain Text

All text files share a common format:

```
######OpenITI#

#META# 000.SortField  :: JK_007501
#META# 010.AuthorNAME :: أبو طالب
#META# 020.BookTITLE  :: ديوان أبو طالب
...
#META#Header#End#

# Chapter heading or section title
# First line of content
# Second line of content...
PageV01P001
# More content after page marker...
### | 1          ← chapter/section marker
# المتن / verse / prose content
~~كلمة 1        ← poem line marker (~~)
```

**Key parsing rules**:
- Lines starting with `#META#` → metadata (skip until `#META#Header#End#`)
- Lines matching `PageV{n}P{n}` → page markers (skip)
- Lines starting with `# ` → content lines (strip `# ` prefix, keep text)
- Lines starting with `### ` → section headers (skip)
- Lines starting with `~~` → poem/verse markers (strip `~~`, keep text)
- Lines containing only numbers or `%` separators → noise (skip)
- Verse/line numbers at end of content lines → strip

### 2.3 Metadata CSV

The file `OpenITI_metadata_2023-1-8.csv` is tab-separated with columns:

| Column | Description |
|--------|-------------|
| `version_uri` | Unique text version identifier |
| `status` | `pri` (primary) or `sec` (secondary/duplicate) |
| `tok_length` | Token count |
| `char_length` | Character count |
| `local_path` | Relative path from CSV location to text file |
| `tags` | Genre tags (e.g., `GAL@literature-arabic`, `_SHICR`) |
| `title_ar` | Arabic title |
| `author_lat` | Author name in Latin transliteration |
| `date` | Approximate author death year (hijri) |

Only `pri`-status texts are used (avoids duplicates).

### 2.4 Scope: Why We Cannot Use All of OpenITI

OpenITI is a 21 GB corpus with 1B+ tokens. Using it entirely is infeasible:

| Constraint | Detail |
|------------|--------|
| **Embedding time** | At 1,000 sentences/sec, 10M sentences takes ~3 hours |
| **Index size** | 10M × 384 dims × 4 bytes = ~15 GB FAISS index |
| **Kaggle GPU RAM** | 13 GB — index must fit in RAM for inference |
| **Export latency** | Querying at export time: ~1M OCR samples × Ks retrieval |

**Solution**: Build a **size-capped corpus** from OpenITI using stratified sampling.

---

## 3. Corpus Building Strategy

### 3.1 Size Targets

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Target sentences | 200,000 | Sufficient diversity; index < 300 MB |
| Min sentence length | 30 chars | Avoids fragments |
| Max sentence length | 300 chars | Avoids overly long context |
| Chunk size | 200 chars | Matches `config.yaml rag.chunk_size` |
| Sampling strategy | Stratified by date | Cover different Arabic periods |
| Status filter | `pri` only | Avoid duplicate texts |

### 3.2 Stratified Sampling by Date

OpenITI dates are hijri year of author's death. To ensure broad coverage:

| Era | Hijri date range | Approx CE | Target fraction |
|-----|-----------------|-----------|-----------------|
| Early classical | 0–300 | 622–912 | 10% |
| Classical | 300–600 | 912–1203 | 30% |
| Late classical | 600–900 | 1203–1494 | 30% |
| Post-classical | 900–1400 | 1494–1979 | 30% |

This avoids over-representing any single prolific author (e.g., Ibn Kathir with hundreds
of thousands of sentences).

### 3.3 Text Extraction from OpenITI Files

**`OpenITILoader`** handles:
1. Reads metadata CSV → filters to `pri`-status texts
2. Stratified sampling of file paths by date
3. Parses each file: extracts content lines, strips markers, splits into sentences
4. Filters by length, deduplicates within a file
5. Returns list of `CorpusSentence` objects
6. Saves to `results/phase5/corpus.jsonl` (one sentence per line)

**`CorpusSentence` dataclass**:
```python
@dataclass
class CorpusSentence:
    text: str           # Clean Arabic text (stripped of markers)
    source_uri: str     # OpenITI version URI (for citation)
    date: int           # Author's approximate date (hijri)
    char_len: int       # Character count
```

**`corpus.jsonl` line schema**:
```json
{"text": "...", "source_uri": "0001AbuTalibCabdManaf.Diwan.JK007501-ara1", "date": 1, "char_len": 145, "idx": 0}
```

---

## 4. RAG Architecture

### 4.1 Design Choice: Pre-Retrieve During Export

Two possible retrieval architectures:

| Architecture | Pros | Cons |
|-------------|------|------|
| **A: Pre-retrieve at export** (chosen) | infer.py unchanged; no index on Kaggle | Export step slower; context fixed at export |
| **B: Online retrieval at inference** | Dynamic retrieval | Must upload FAISS index to Kaggle (~300 MB) |

**Decision**: Architecture A — retrieve during the local export step and embed the context
in each JSONL record. This keeps `scripts/infer.py` unchanged (just a new `prompt_type`).

### 4.2 `RAGRetriever` — Module Design

**Location**: `src/core/rag_retriever.py` (new file)

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class RetrievedChunk:
    """A single retrieved sentence with its similarity score."""
    text: str
    source_uri: str
    score: float       # cosine similarity in [0, 1]
    rank: int          # 1 = most similar


class RAGRetriever:
    """Retrieve similar Arabic sentences from OpenITI using dense retrieval.

    Embeds a query text and returns top-K similar sentences from a pre-built
    FAISS index. Index is built once from corpus.jsonl and reused across runs.

    Usage::

        retriever = RAGRetriever(config)
        retriever.build_index(corpus_path)              # one-time
        chunks = retriever.retrieve(ocr_text, k=3)
        context = retriever.format_for_prompt(chunks)

    Graceful degradation: if sentence-transformers or faiss are not installed,
    RAGRetriever.enabled is False and retrieve() returns [].
    """

    def __init__(self, config: dict) -> None:
        """Initialise retriever with config settings.

        Loads the pre-built index if it exists. Does NOT build the index —
        call build_index() or run_phase5.py --mode build first.

        Args:
            config: Parsed config dict. Reads from config['rag']:
                embedding_model: model name for SentenceTransformer
                top_k: default number of results to return
                index_path: path to saved FAISS index
                corpus_path: path to corpus.jsonl (for text lookup)
        """

    @property
    def enabled(self) -> bool:
        """True if index is loaded and retrieval is available."""

    def build_index(
        self,
        corpus_path: Path,
        index_path: Path,
        batch_size: int = 256,
        show_progress: bool = True,
    ) -> None:
        """Build and save FAISS index from corpus.jsonl.

        Embeds all sentences in corpus.jsonl in batches, builds a FAISS
        IndexFlatIP (inner product, equivalent to cosine after normalization),
        and saves both the index and the sentence ID list to disk.

        Args:
            corpus_path: Path to corpus.jsonl (produced by OpenITILoader).
            index_path: Path to save the FAISS index file (.faiss).
            batch_size: Embedding batch size (tune for GPU memory).
            show_progress: If True, show tqdm progress bar.

        Saves:
            {index_path}               -- FAISS index binary
            {index_path}.sentences.jsonl -- Ordered sentence list (for lookup)
        """

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
            k: Number of results. Defaults to config['rag']['top_k'].
            min_score: Minimum cosine similarity to include (default: 0.0,
                i.e., no filtering). Useful for excluding irrelevant results.

        Returns:
            List of RetrievedChunk sorted by score descending.
        """

    def format_for_prompt(
        self,
        chunks: list[RetrievedChunk],
        style: str = "numbered_arabic",
    ) -> str:
        """Format retrieved sentences as Arabic context for prompt injection.

        Args:
            chunks: Retrieved chunks from retrieve().
            style: Formatting style:
                - "numbered_arabic" (default): numbered list
                - "plain": just the sentences, newline-separated

        Returns:
            Multi-line Arabic string. Empty string if chunks is [].
        """
```

### 4.3 Embedding Model

**Config**: `config.rag.embedding_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"`

**Properties**:
- Multilingual (supports Arabic)
- 384-dimensional embeddings
- ~120 MB model size
- ~1,000 sentences/second on CPU, ~10,000/second on GPU
- Available on HuggingFace without authentication

**Alternative** (to evaluate in analysis): `intfloat/multilingual-e5-small` — Arabic-specific tuning, 384 dims, similar speed.

### 4.4 FAISS Index Type

- **IndexFlatIP**: Exact cosine similarity (after L2 normalization). No quantization.
- Chosen for correctness over speed — with 200K sentences, exact search is fast enough
  (~5 ms per query on CPU for 200K × 384 float32).
- If corpus scales to 1M+, switch to `IndexIVFFlat` with quantization.

---

## 5. Prompt Design

### 5.1 System Prompt

**Version**: `RAG_PROMPT_VERSION = "p5v1"`

```python
RAG_SYSTEM_V1: str = (
    "أنت مصحح نصوص عربية متخصص. "
    "فيما يلي نصوص عربية صحيحة مشابهة للنص المراد تصحيحه، استخدمها كمرجع:\n\n"
    "{retrieval_context}\n\n"
    "صحح النص التالي مستعيناً بهذه النصوص المرجعية. "
    "أعد النص المصحح فقط بدون أي شرح أو تعليق إضافي."
)
```

**`PromptBuilder` addition**:
```python
def build_rag(self, ocr_text: str, retrieval_context: str) -> list[dict]:
    """Build RAG-augmented correction prompt (Phase 5).
    Falls back to zero-shot if retrieval_context is empty.
    """

@property
def rag_prompt_version(self) -> str:
    return self.RAG_PROMPT_VERSION
```

### 5.2 Formatted Retrieval Context (numbered_arabic style)

```
نصوص مرجعية صحيحة:
1. قال الوزير إن الحكومة ستنظر في هذا الأمر بجدية تامة خلال الأيام القليلة المقبلة.
2. أعلنت وزارة التربية والتعليم عن فتح باب التسجيل للعام الدراسي الجديد.
3. يتضمن الكتاب دراسة مفصلة حول تطور اللغة العربية عبر العصور المختلفة.
```

**Design choices**:
- Top-3 sentences (config: `rag.top_k = 3`)
- No source attribution (too verbose; citation not needed for correction task)
- Sentences truncated to 150 characters if very long (avoid inflating context window)

### 5.3 Prompt Version Comparison

| Aspect | Phase 2 | Phase 3 | Phase 4A | Phase 4B | Phase 5 |
|--------|---------|---------|----------|----------|---------|
| Injects confusion pairs | No | Yes | No | No | No |
| Injects rules | No | No | Yes | No | No |
| Injects examples | No | No | No | Yes | No |
| Injects similar sentences | No | No | No | No | Yes |
| Context source | — | Phase 1 matrix | CORE_RULES | QALB | OpenITI |
| Context is per-sample | No | No | No | No | **Yes** |
| Prompt version | `v1` | `p3v1` | `p4av1` | `p4bv1` | `p5v1` |
| Approx. prompt tokens | ~25 | ~150 | ~100 | ~200 | ~150–250 |

**Phase 5 is unique**: context is **per-sample** (different retrieved sentences for each OCR text),
whereas Phases 3/4A/4B inject the same context for all samples.

---

## 6. `OpenITILoader` — Module Design

**Location**: `src/data/knowledge_base.py` (new class added to existing module)

```python
@dataclass
class CorpusSentence:
    """A single extracted sentence from the OpenITI corpus."""
    text: str           # Clean Arabic sentence text
    source_uri: str     # OpenITI version URI (e.g. "0001AbuTalibCabdManaf.Diwan.JK007501-ara1")
    date: int           # Author's approximate hijri death year
    char_len: int       # Character count
    idx: int            # Sequential index in the corpus


class OpenITILoader:
    """Load and extract clean Arabic sentences from the OpenITI corpus.

    Handles the two OpenITI file formats (plain text and .mARkdown),
    parses out content lines, strips structural markers, and returns
    clean Arabic sentences suitable for embedding.

    Usage::

        loader = OpenITILoader(config)
        sentences = loader.load(
            max_sentences=200_000,
            min_char_len=30,
            max_char_len=300,
            seed=42,
        )
        loader.save_corpus(sentences, Path("results/phase5/corpus.jsonl"))
        # Or load a previously saved corpus:
        sentences = loader.load_corpus(Path("results/phase5/corpus.jsonl"))
    """

    def __init__(self, config: dict) -> None:
        """Initialise with config.

        Reads config['data']['openiti'] for the corpus root path.
        Reads config['rag'] for chunk_size and sampling parameters.
        """

    def load(
        self,
        max_sentences: int = 200_000,
        min_char_len: int = 30,
        max_char_len: int = 300,
        status_filter: str = "pri",
        seed: int = 42,
        show_progress: bool = True,
    ) -> list[CorpusSentence]:
        """Load and return clean Arabic sentences from OpenITI.

        Reads the metadata CSV for file paths, applies stratified sampling
        by date, parses each file, and returns sentences filtered by length.

        Args:
            max_sentences: Maximum total sentences to return.
            min_char_len: Minimum character count (filters fragments).
            max_char_len: Maximum character count (filters very long passages).
            status_filter: Only use texts with this status ("pri" or "sec").
            seed: Random seed for reproducible stratified sampling.
            show_progress: Show tqdm progress bar.

        Returns:
            List of CorpusSentence, deduplicated within each source file.
        """

    def save_corpus(self, sentences: list[CorpusSentence], path: Path) -> None:
        """Save corpus to a JSONL file."""

    def load_corpus(self, path: Path) -> list[CorpusSentence]:
        """Load a previously saved corpus from JSONL."""

    @staticmethod
    def parse_file(file_path: Path) -> list[str]:
        """Extract clean Arabic text lines from one OpenITI file.

        Handles both plain text (no extension) and .mARkdown formats.
        Returns a list of clean Arabic lines (no structural markers).

        Parsing rules:
        - Skip lines before #META#Header#End#
        - Skip lines matching ^PageV\\d+P\\d+
        - Lines starting with "# ": strip "# " prefix; further strip
          verse numbers (trailing digits), % separators
        - Lines starting with "~~": strip "~~ " prefix (poem lines)
        - Skip section markers (### ..., ##, #)
        - Skip lines with fewer than 10 Arabic characters after stripping
        """
```

---

## 7. Pipeline: `pipelines/run_phase5.py`

### 7.1 Modes

Phase 5 has **four** modes (vs two for Phases 3/4A/4B):

| Mode | Stage | Where | Description |
|------|-------|-------|-------------|
| `build` | 0 (one-time) | Local | Build corpus + FAISS index from OpenITI |
| `export` | 1 | Local | Retrieve for each OCR sample; write inference_input.jsonl |
| *(inference)* | 2 | Kaggle/Colab | Run `scripts/infer.py` (unchanged) |
| `analyze` | 3 | Local | Load corrections.jsonl, compute metrics |

### 7.2 Typical Workflow

```bash
# Stage 0: Build index (one-time, local — takes 20–60 min depending on corpus size)
python pipelines/run_phase5.py --mode build

# Stage 1: Export (local — retrieve for each OCR sample)
python pipelines/run_phase5.py --mode export

# Stage 2: Inference (Kaggle/Colab — same as all other phases)
python scripts/infer.py \
    --input  results/phase5/inference_input.jsonl \
    --output results/phase5/corrections.jsonl

# Stage 3: Analyze (local)
python pipelines/run_phase5.py --mode analyze
```

### 7.3 Build Mode Details

```
run_phase5.py --mode build
│
├─ 1. Load config; check data/OpenITI/ exists
│
├─ 2. OpenITILoader.load()
│       Read metadata CSV → filter pri-status texts
│       Stratified sampling by date → select ~N files
│       Parse each file → extract clean sentences
│       Filter by length → deduplicate within files
│       Return list[CorpusSentence]
│
├─ 3. Save corpus to results/phase5/corpus.jsonl
│       Log: N sentences, size breakdown by date era
│
├─ 4. RAGRetriever.build_index()
│       Load SentenceTransformer model (downloads if needed)
│       Embed sentences in batches (GPU if available)
│       Build FAISS IndexFlatIP
│       Save: results/phase5/faiss.index
│              results/phase5/faiss.index.sentences.jsonl
│
└─ 5. Log summary: index built, retrieval ready
```

### 7.4 Export Mode Details

```
run_phase5.py --mode export
│
├─ 1. Load config; initialise RAGRetriever (loads pre-built index)
│       If index missing: error + "Run --mode build first"
│
├─ 2. Load DataLoader; resolve active datasets
│
├─ 3. For each dataset in active_datasets:
│     │   Resume: skip dataset if already in inference_input.jsonl
│     │
│     └─ For each OCR sample:
│             chunks = retriever.retrieve(ocr_text, k=top_k)
│             retrieval_context = retriever.format_for_prompt(chunks)
│             Write to inference_input.jsonl:
│             {
│               "sample_id": ...,
│               "dataset": ...,
│               "ocr_text": ...,
│               "gt_text": ...,
│               "prompt_type": "rag",
│               "retrieval_context": "...",
│               "retrieved_k": 3,
│               "retrieval_scores": [0.82, 0.79, 0.71],
│             }
```

### 7.5 CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--mode` | str | required | `build` \| `export` \| `analyze` |
| `--limit` | int | None | Max samples per dataset |
| `--datasets` | str+ | None | Subset of dataset keys |
| `--force` | flag | False | Re-run even if output exists |
| `--no-error-analysis` | flag | False | Skip error_changes.json |
| `--config` | path | `configs/config.yaml` | Config file |
| `--results-dir` | path | `results/phase5` | Output directory |
| `--phase2-dir` | path | `results/phase2` | Phase 2 baseline directory |
| `--top-k` | int | 3 | Number of sentences to retrieve per sample |
| `--min-score` | float | 0.0 | Minimum cosine similarity to include |
| `--max-sentences` | int | 200000 | Corpus size cap (for `--mode build`) |
| `--corpus-path` | path | `results/phase5/corpus.jsonl` | Corpus JSONL (override) |
| `--index-path` | path | `results/phase5/faiss.index` | FAISS index (override) |

---

## 8. `scripts/infer.py` Addition

One new `prompt_type` dispatch case:

```python
elif prompt_type == "rag":
    retrieval_context = record.get("retrieval_context", "")
    messages = builder.build_rag(record["ocr_text"], retrieval_context)
    prompt_ver = builder.rag_prompt_version
    if not retrieval_context.strip():
        prompt_type = "zero_shot_fallback"
```

Fallback to zero-shot is triggered when retrieval returned no results (e.g., if the index was
empty or the query had no Arabic tokens). This matches the pattern in Phases 3/4A/4B.

---

## 9. Output Structure

```
results/phase5/
├── corpus.jsonl                  # Built corpus (one sentence per line)
├── faiss.index                   # FAISS binary index
├── faiss.index.sentences.jsonl   # Ordered sentence list for lookup
├── inference_input.jsonl         # Export: one record per OCR sample
├── corrections.jsonl             # Inference output (place here before analyze)
├── {dataset_name}/
│   ├── corrections.jsonl         # Per-dataset split (auto-generated)
│   ├── metrics.json
│   ├── comparison_vs_phase2.json
│   ├── error_changes.json
│   └── retrieval_analysis.json   # Per-dataset retrieval statistics
├── metrics.json                  # Aggregated across all datasets
├── comparison.json               # Aggregated comparison vs Phase 2
├── report.md
└── phase5.log
```

---

## 10. Output Schemas

### 10.1 `metrics.json` (per dataset)

Same structure as Phases 3/4, with updated `phase` and `prompt_type`:

```json
{
  "meta": {
    "phase": "phase5",
    "dataset": "KHATT-train",
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "prompt_type": "rag",
    "prompt_version": "p5v1",
    "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "top_k": 3,
    "corpus_size": 200000,
    "generated_at": "...",
    "num_samples": 1400
  },
  "corrected": {
    "cer": 0.071,
    "wer": 0.195,
    ...
  }
}
```

### 10.2 `retrieval_analysis.json` (per dataset)

Measures retrieval quality and its relationship to correction quality:

```json
{
  "meta": {
    "dataset": "KHATT-train",
    "top_k": 3,
    "embedding_model": "...",
    "generated_at": "..."
  },
  "retrieval_stats": {
    "samples_with_any_retrieval":  1380,
    "samples_zero_results":          20,
    "avg_top1_score":             0.743,
    "avg_top3_score":             0.701,
    "score_distribution": {
      "0.8+":  312,
      "0.7–0.8": 681,
      "0.6–0.7": 387,
      "<0.6":   0
    }
  },
  "retrieval_by_error_type": {
    "note": "Correlation between retrieval score and per-sample CER improvement",
    "high_score_samples": {
      "cer_improvement": 0.031,
      "n_samples": 345
    },
    "low_score_samples": {
      "cer_improvement": 0.008,
      "n_samples": 345
    }
  },
  "corpus_date_distribution": {
    "0-300": 0.10,
    "300-600": 0.30,
    "600-900": 0.30,
    "900+": 0.30
  }
}
```

### 10.3 `comparison_vs_phase2.json` (per dataset)

Identical schema to Phase 3/4 comparison files:

```json
{
  "meta": {"comparison": "phase5_vs_phase2", "dataset": "KHATT-train"},
  "phase2_baseline": {"cer": 0.089, "wer": 0.234},
  "phase5_corrected": {"cer": 0.071, "wer": 0.195},
  "delta": {
    "cer_absolute": -0.018,
    "wer_absolute": -0.039,
    "cer_relative_pct": -20.2,
    "wer_relative_pct": -16.7
  },
  "interpretation": "CER reduced by 20.2% vs Phase 2."
}
```

---

## 11. Configuration (`configs/config.yaml` Additions)

```yaml
# ---------------------------------------------------------------------------
# Phase 5 specific
# ---------------------------------------------------------------------------
phase5:
  corpus:
    max_sentences: 200000        # Sentences to extract from OpenITI
    min_char_len: 30             # Minimum sentence character length
    max_char_len: 300            # Maximum sentence character length
    status_filter: "pri"         # Only use primary-status OpenITI texts
    seed: 42
  index:
    type: "FlatIP"               # FAISS index type: "FlatIP" | "IVFFlat" (for large corpora)
    batch_size: 256              # Embedding batch size
  retrieval:
    top_k: 3                     # Sentences to retrieve per OCR sample
    min_score: 0.0               # Minimum cosine similarity (0 = no filter)
    format_style: "numbered_arabic"
  analyze_errors: true
  max_retries: 2
```

---

## 12. New and Modified Files

### 12.1 New Files

| File | Purpose |
|------|---------|
| `src/core/rag_retriever.py` | `RAGRetriever`, `RetrievedChunk` |
| `pipelines/run_phase5.py` | Full pipeline: build / export / analyze |

### 12.2 Modified Files

| File | Changes |
|------|---------|
| `src/data/knowledge_base.py` | Add `CorpusSentence` dataclass + `OpenITILoader` class |
| `src/core/prompt_builder.py` | Add `build_rag()`, `RAG_SYSTEM_V1`, `RAG_PROMPT_VERSION = "p5v1"`, `rag_prompt_version` property |
| `scripts/infer.py` | Add `elif prompt_type == "rag":` dispatch branch |
| `configs/config.yaml` | Add `phase5:` block |
| `HOW_TO_RUN.md` | Add Phase 5 documentation |

### 12.3 No Modifications to Phases 1–4

Existing pipeline files and output files are untouched.

---

## 13. Dependencies

### 13.1 New Packages Required

```bash
pip install sentence-transformers faiss-cpu
# For GPU-accelerated embedding (optional but recommended for build step):
pip install faiss-gpu  # instead of faiss-cpu
```

These are NOT in `requirements.txt` yet — add them:

```
sentence-transformers>=2.2.0
faiss-cpu>=1.7.4
```

### 13.2 Graceful Degradation

`RAGRetriever.__init__()` wraps the import in a try/except:

```python
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    self._enabled = True
except ImportError as exc:
    logger.warning("RAG disabled: %s. Install sentence-transformers and faiss-cpu.", exc)
    self._enabled = False
```

If disabled:
- `retrieve()` returns `[]`
- `build_rag()` falls back to `build_zero_shot()` (same as Phases 3/4)
- `run_phase5.py --mode build` exits with a clear error message
- `run_phase5.py --mode export` still works but writes `prompt_type: "zero_shot"` for all records

---

## 14. Testing

### 14.1 New Test Files

```
tests/
├── test_openiti_loader.py
├── test_rag_retriever.py
├── test_prompt_builder_phase5.py
└── fixtures/
    ├── sample_openiti_plain.txt    # 5-page excerpt from one OpenITI file
    └── sample_openiti_markdown.mARkdown  # 5-page .mARkdown excerpt
```

### 14.2 `test_openiti_loader.py`

- `test_parse_file_extracts_arabic_lines`
- `test_parse_file_strips_meta_header`
- `test_parse_file_skips_page_markers`
- `test_parse_file_strips_verse_numbers`
- `test_parse_file_handles_poem_lines`
- `test_parse_file_skips_short_lines`
- `test_load_corpus_filters_by_length`
- `test_save_and_load_corpus_roundtrip`

### 14.3 `test_rag_retriever.py`

- `test_retrieve_returns_empty_when_disabled`
- `test_retrieve_returns_top_k_results`
- `test_retrieve_scores_are_descending`
- `test_retrieve_empty_query_returns_empty`
- `test_format_numbered_arabic_produces_header`
- `test_format_plain_returns_newline_separated`
- `test_build_index_creates_files` (requires faiss)
- `test_retrieve_after_build_returns_similar_sentences` (requires faiss + sentence-transformers)

### 14.4 `test_prompt_builder_phase5.py`

- `test_build_rag_returns_two_messages`
- `test_build_rag_contains_retrieval_in_system`
- `test_build_rag_empty_context_falls_back_to_zero_shot`
- `test_rag_prompt_version_is_p5v1`
- `test_phase4_build_rule_augmented_unchanged` (regression)
- `test_phase3_build_ocr_aware_unchanged` (regression)
- `test_phase2_build_zero_shot_unchanged` (regression)

---

## 15. Known Risks & Mitigations

| Risk | Probability | Mitigation |
|------|-------------|-----------|
| **Genre mismatch**: OpenITI is classical, PATS/KHATT are modern | High | This is a research finding. Report similarity score distributions. Consider adding a small modern Arabic news corpus as a supplementary index (e.g., 50K sentences from Arabic Wikipedia). |
| **Low retrieval relevance**: cosine similarity scores all < 0.6 | Medium | Log `avg_top1_score` in retrieval_analysis.json. If median score < 0.6, try `min_score=0.0` (always retrieve) vs filtering out low scores. |
| **LLM ignores retrieved context** | Possible | Classic RAG challenge. Log correlation between retrieval score and CER improvement. If no correlation, report as negative finding. |
| **Build step OOM on CPU** (embedding 200K sentences) | Low | Use `batch_size=64` and `faiss-cpu`. 200K × 384 float32 = ~300 MB; embedding fits in ~4 GB RAM. |
| **FAISS not available on Windows (pip install issues)** | Medium | Fallback to `sklearn.metrics.pairwise.cosine_similarity` with numpy (slower but no C++ dependency). Add to graceful degradation. |
| **OpenITI text files not properly parsed** (unknown encoding) | Low | All OpenITI files are UTF-8. Add `errors='replace'` as fallback. Log parse failures per file. |
| **Very long inference_input.jsonl** (context per sample) | Low | Each record adds ~500–800 chars of retrieval context. For 40K samples × 700 chars ≈ 28 MB increase. Manageable. |
| **Index not built before export** | Medium | `run_phase5.py --mode export` checks for index file existence and exits with clear error + instructions. |

---

## 16. Research Questions Answered by `retrieval_analysis.json`

Phase 5 produces unique analysis outputs beyond CER/WER:

| Question | Output Field |
|----------|-------------|
| How similar is OpenITI to the OCR inputs? | `avg_top1_score`, `score_distribution` |
| Do higher-similarity retrievals lead to better corrections? | `retrieval_by_error_type` (high vs low score groups) |
| Which error types benefit most from corpus grounding? | `retrieval_by_error_type` per type |
| Does date/era of retrieved text matter? | `corpus_date_distribution` of top-1 retrievals |

The correlation between retrieval quality and correction quality directly answers the research
question: "Does corpus grounding help, and when?"

---

## 17. Implementation Order

| Step | Action | Notes |
|------|--------|-------|
| 1 | `src/data/knowledge_base.py` — add `OpenITILoader` | Parse OpenITI files |
| 2 | `tests/test_openiti_loader.py` | Verify with fixture files |
| 3 | `src/core/rag_retriever.py` — full `RAGRetriever` | FAISS + embedding |
| 4 | `tests/test_rag_retriever.py` | Unit tests (mock for fast CI) |
| 5 | `src/core/prompt_builder.py` — add `build_rag()` | Minimal, additive |
| 6 | `tests/test_prompt_builder_phase5.py` | Including regression tests |
| 7 | `scripts/infer.py` — add `elif prompt_type == "rag":` | One new elif branch |
| 8 | `configs/config.yaml` — add `phase5:` block | |
| 9 | `pipelines/run_phase5.py` — `build` mode | Corpus + index build |
| 10 | Smoke test build: `--mode build --max-sentences 1000` | Quick sanity check |
| 11 | `pipelines/run_phase5.py` — `export` mode | Retrieval + JSONL |
| 12 | Smoke test export: `--mode export --limit 5 --datasets KHATT-train` | |
| 13 | Run inference on Kaggle/Colab | `scripts/infer.py --input results/phase5/...` |
| 14 | `pipelines/run_phase5.py` — `analyze` mode | Metrics + retrieval_analysis |
| 15 | Smoke test analyze: `--mode analyze --datasets KHATT-train` | |
| 16 | Full run: all datasets, corpus_size=200K | Paper numbers |

---

## 18. Appendix: OpenITI Text Parsing Rules (Full Specification)

### Content Line Extraction

```python
def _is_content_line(line: str) -> bool:
    """True if line is content (not structural marker)."""
    stripped = line.strip()
    if not stripped:
        return False
    # Page markers: PageV01P001
    if re.match(r'^PageV\d+P\d+', stripped):
        return False
    # Section markers: ### | N, ###, ##, #
    if re.match(r'^#{2,}', stripped):
        return False
    # Metadata lines
    if stripped.startswith('#META#'):
        return False
    # Content lines start with "# " or "~~"
    if stripped.startswith('# ') or stripped.startswith('~~'):
        return True
    # Some files have bare content lines (post-header text without # prefix)
    # These are included if they contain Arabic characters
    arabic_chars = sum(1 for c in stripped if '\u0600' <= c <= '\u06ff')
    return arabic_chars >= 10


def _clean_content_line(line: str) -> str:
    """Strip structural markers from a content line."""
    text = line.strip()
    # Remove "# " prefix
    if text.startswith('# '):
        text = text[2:]
    # Remove "~~ " or "~~" prefix (poem lines)
    if text.startswith('~~'):
        text = text.lstrip('~').strip()
    # Remove page markers embedded in line: PageV01P001
    text = re.sub(r'PageV\d+P\d+', '', text)
    # Remove trailing verse/line numbers: "... word 42" or "% word % 5"
    text = re.sub(r'\s+\d+\s*$', '', text)
    # Remove % separators (poetry format): "% word1 word2 % % word3 %"
    text = text.replace('%', ' ')
    # Normalise whitespace
    text = ' '.join(text.split())
    return text
```

### Sentence Splitting

OpenITI content lines are typically 1–3 sentences already (verse lines or prose sentences).
**No sentence splitting is applied** — each extracted line is treated as one corpus unit.
This avoids mid-sentence splits and matches the natural chunking of the corpus.

For the Quran (`.mARkdown` format), each verse (`# surah|verse| text`) is one unit.

---

## 19. Comparison: Phase 4B vs Phase 5 (Few-Shot vs RAG)

Both phases inject prior Arabic text into the prompt, but differ fundamentally:

| Aspect | Phase 4B (Few-shot) | Phase 5 (RAG) |
|--------|---------------------|---------------|
| Source | QALB (human typos) | OpenITI (classical corpus) |
| Selection | Global (same for all samples) | **Per-sample** (retrieved by similarity) |
| Content type | Error→correction pairs | Correct sentences only |
| Demonstrates correction | Yes (shows before/after) | No (shows target domain only) |
| Error type coverage | Hamza, taa marbuta, alef | Vocabulary, context |
| Corpus size | ~4,000 filtered pairs | 200,000 sentences |
| Build step needed | No | Yes |
| Inference input size | ~300 chars/sample | ~500 chars/sample |

The two approaches are complementary and both are tested individually before combining in Phase 6.
