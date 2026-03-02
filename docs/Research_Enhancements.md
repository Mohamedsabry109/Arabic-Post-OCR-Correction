# Research Enhancements, Innovations & Paper Writing Guide

> **Audience**: Author (thesis self-reference) and advisor review
> **Purpose**: What to do if LLM results are disappointing, what to modify, novel ideas, and how to write the paper
> **Based on**: Deep analysis of the full 8-phase pipeline (Phases 1–6 implemented)

---

## Table of Contents

1. [Diagnosing Poor LLM Results](#1-diagnosing-poor-llm-results)
2. [Modifications to the Current Pipeline](#2-modifications-to-the-current-pipeline)
3. [Innovative Alternative Ideas](#3-innovative-alternative-ideas)
4. [Prompt Engineering Deep Dive](#4-prompt-engineering-deep-dive)
5. [How to Write the Results in the Paper](#5-how-to-write-the-results-in-the-paper)

---

## 1. Diagnosing Poor LLM Results

Before changing anything, identify *which* failure mode you are facing. The treatments differ radically depending on root cause.

### 1.1 The Five Root Causes

#### A. Model Capability Ceiling

**Symptom**: LLM corrections barely move CER/WER vs OCR baseline, or even make it worse, regardless of prompt type.

**Most likely on**: KHATT (baseline CER 60–66%). When OCR destroys 65% of characters, the LLM sees mostly noise. There is no recoverable signal in heavily damaged text.

**Indicators**:
- Phase 2 (zero-shot) shows large CER decrease on PATS but near-zero or negative gain on KHATT
- Error analysis shows LLM "introduced errors" > "fixed errors" on KHATT
- LLM output is semantically plausible Arabic but bears little resemblance to the ground truth

**What it means for the paper**: KHATT is an extreme case. Frame it honestly as "beyond the scope of prompt-based correction at this damage level." This is a valid finding — it defines the boundary of the approach, which is itself a contribution.

**Do not**:
- Spend time engineering more prompts for KHATT if the signal is gone
- Report PATS and KHATT aggregated together as a single number

---

#### B. Over-Correction (LLM Changes Correct Text)

**Symptom**: CER *increases* after LLM correction on clean fonts (Akhbar: baseline 1.94% CER — the LLM cannot improve much but can easily hurt).

**Indicators**:
- Error analysis: "introduced errors" is high, "fixed errors" is low
- Specific pattern: LLM over-applies ة/ه, ي/ى, hamza corrections where OCR was already correct
- Texts with low baseline error rate (Akhbar, Simplified) fare worse than Andalus and Thuluth

**Why it happens**: The LLM is biased by its training data to "normalize" Arabic spelling. It applies orthographic corrections even when the OCR output was accurate. This is a type of hallucination — the model corrects based on its priors, not on actual errors.

**Treatment**:
- Add explicit conservative instruction: "إذا لم تكن متأكداً من وجود خطأ، أبقِ الكلمة كما هي"
- Add the CAMeL revert strategy (Phase 4C) post-hoc even for phases 3–5
- Try the "detect-then-correct" two-pass approach (see Section 3.3)
- Report a "net improvement rate" = (fixed errors − introduced errors) / total errors

---

#### C. Context Ignored (Instruction Dilution)

**Symptom**: Phases 3–5 show no benefit over Phase 2 (zero-shot), even though knowledge is injected.

**Why it happens**: When the system prompt is long (1000+ chars of confusion pairs + rules + examples) and the user message is also long Arabic text, smaller models (4B) often ignore the system prompt and generate fluent Arabic without applying the specific instructions.

**Indicators**:
- Phase 3 (OCR-aware) ≈ Phase 2 (zero-shot) on CER/WER
- Changing the confusion matrix format or top-N pairs makes no difference
- The LLM makes the same category of errors despite them being listed in the prompt

**Treatment**:
- Move key instructions to the **user message** instead of system message (some models attend more to user role)
- Reduce system prompt length — inject only the top 5 confusion pairs, not 10
- Try the output prefix trick: end user message with "النص المصحح:" so model continues from there
- Try `enable_thinking=True` (Qwen3 reasoning mode) to see if explicit reasoning helps
- Segment long texts into shorter chunks (≤50 chars) so the ratio of instruction-to-content improves

---

#### D. Hallucination (Plausible but Wrong)

**Symptom**: LLM output is grammatically correct Arabic but semantically wrong or replaced words that don't match the ground truth at all.

**Indicators**:
- WER is high even when CER is moderate (word-level replacements, not character substitutions)
- Error analysis shows "other substitution" category spikes after LLM correction
- LLM silently deletes or merges words in the OCR text

**Why it happens**: The model "fills in" gaps in damaged OCR text with plausible Arabic continuations rather than faithfully recovering the original. This is particularly likely when the OCR text is fragmented (KHATT) or the sentence domain is specialized (religious texts, classical Arabic in OpenITI).

**Treatment**:
- Add length constraint: "أعد النص بنفس الطول تقريباً. لا تحذف أي كلمة ولا تضف كلمات جديدة"
- Add explicit word-preservation instruction: "صحّح الأحرف فقط، ولا تغيّر بنية الجملة"
- Use the edit-distance gating approach (Section 3.6): reject corrections that differ from OCR by more than K characters
- Two-pass approach (Section 3.3): first localize errors, then correct only those positions

---

#### E. Dataset/Domain Mismatch (QALB vs OCR Errors)

**Symptom**: Phase 4B (few-shot) does not improve or degrades performance vs Phase 2.

**Why it happens**: QALB contains human grammar/spelling errors (e.g., missing hamza in casual typing, wrong letter due to keyboard proximity). OCR errors are caused by visual confusion (ب↔ت due to dot position). These are different error distributions. The few-shot examples may actually confuse the model by showing grammar corrections when it should be doing OCR corrections.

**Indicators**:
- Phase 4B underperforms Phase 4A (rules) despite having 5 concrete examples
- The few-shot examples shown involve ة↔ه but the OCR errors are mainly dot confusions (ف↔ق, ص↔ض)

**Treatment**:
- Generate synthetic OCR error pairs directly from PATS data: take Phase 1 confusion pairs and manufacture (ocr_erroneous, ground_truth) examples from the training set
- Use Phase 1 ground-truth + OCR pairs as few-shot training data (they are *actual* OCR errors, not human typos)
- Filter QALB more aggressively: only keep pairs where error type matches the top-5 confusion pairs from Phase 1

---

### 1.2 Per-Phase Failure Signatures

| Phase | Typical Failure Signature | Primary Suspect |
|-------|--------------------------|-----------------|
| **2 (Zero-Shot)** | Gets worse on clean fonts (Akhbar CER rises) | Over-correction bias |
| **2 (Zero-Shot)** | Near-zero improvement on KHATT | Capability ceiling |
| **3 (OCR-Aware)** | No improvement over Phase 2 | Context ignored |
| **3 (OCR-Aware)** | Improvement on PATS but regression on KHATT | Confusion matrix too noisy for handwriting |
| **4A (Rules)** | No improvement over Phase 2 | Rules already internalized by LLM |
| **4A (Rules)** | Improvement on some fonts, regression on others | Rule-text misalignment |
| **4B (Few-Shot)** | No improvement, or Phase 4A > Phase 4B | QALB domain mismatch |
| **4B (Few-Shot)** | Works on ة/ه but not on dot confusions | Selection diversity issue |
| **5 (RAG)** | No improvement | Embedding space mismatch (classical Arabic ≠ modern OCR text) |
| **5 (RAG)** | High retrieval similarity scores but no CER improvement | Retrieved sentences not providing useful correction signal |
| **6 (Combined)** | Full prompt worse than individual components | Context overload / instruction dilution |
| **6 (Ablation)** | Removing a component *helps* performance | That component is actively hurting results |

---

### 1.3 Decision Tree: What to Investigate First

```
Results bad overall?
├── Is KHATT >> PATS in terms of CER gap?
│   └── YES → Capability ceiling. Report PATS/KHATT separately. Don't tune for KHATT.
└── NO → Continue...
    ├── Is CER worse after LLM than baseline on clean fonts (Akhbar/Simplified)?
    │   └── YES → Over-correction. Add conservative constraint. Apply CAMeL revert.
    └── NO → Continue...
        ├── Do phases 3-5 all perform the same as phase 2?
        │   └── YES → Context ignored. Shorten prompt. Move context to user message.
        └── NO → Continue...
            ├── Is WER high even with moderate CER?
            │   └── YES → Hallucination (word-level changes). Add length/word constraints.
            └── NO →
                ├── Is phase 4B worse than 4A?
                │   └── YES → QALB mismatch. Use synthetic OCR pairs from Phase 1 data.
                └── Otherwise → Phase-specific tuning (see Section 2).
```

---

## 2. Modifications to the Current Pipeline

These are targeted changes to existing parameters, format choices, and strategies — requiring minimal code changes.

### 2.1 Model Generation Settings

Current settings in `configs/config.yaml`:
```yaml
model:
  temperature: 0.1
  max_tokens: 1024
```

**Recommended experiments:**

| Setting | Current | Try | Effect |
|---------|---------|-----|--------|
| `temperature` | 0.1 | 0.0 | Fully greedy — most deterministic, good for short correction |
| `temperature` | 0.1 | 0.3 | More creative, may help with heavily damaged text |
| `do_sample` | True | False (greedy) | Deterministic, reproducible — use with temp=0 |
| `max_new_tokens` | 1024 | ~2× input length | Prevent truncated outputs on long texts |
| `repetition_penalty` | not set | 1.1–1.3 | Helps with Qaari's repetition bug bleeding through |
| `enable_thinking` | False | True | Qwen3 reasoning scratchpad — costs 2× tokens but may improve quality |

**To test greedy decoding in `src/core/llm_corrector.py`:**
```python
generate_kwargs = dict(
    input_ids=input_ids,
    max_new_tokens=max_tokens,
    do_sample=False,          # greedy
    # temperature not used with do_sample=False
    repetition_penalty=1.1,
    pad_token_id=tokenizer.eos_token_id,
)
```

---

### 2.2 Confusion Matrix Parameters (Phase 3 / Phase 6)

```yaml
phase3:
  top_n: 10          # Try: 5, 3, 15, 20
  format_style: "flat_arabic"   # Try: "grouped_arabic"
  min_substitutions: 200        # Try: 100 (include more datasets)
```

**Key experiments:**

- **top_n = 3**: Fewer, higher-confidence pairs — reduces noise, better focus
- **top_n = 20**: More pairs — may help if model uses them, may overwhelm
- **grouped_arabic format**: Groups by character category (all hamza errors together) — may be clearer for model to apply rules consistently
- **Weighted formatting**: Show pairs proportional to their frequency, not just top-N
- **Per-position stats**: "يستبدل (ة) بـ (ه) في نهاية الكلمة" — adding position context (word-end ة errors) may help

---

### 2.3 Few-Shot Examples (Phase 4B)

```yaml
phase4:
  few_shot:
    num_examples: 5      # Try: 3, 7, 10
    selection: "diverse"  # Try: "most_common", "random"
    max_length: 300      # Try: 200 (shorter = more token budget for other context)
    max_words_changed: 15  # Try: 5 (tighter OCR-relevance filter)
```

**Key experiments:**

- **num_examples = 3**: Less context, cleaner signal — better if model has context overload
- **num_examples = 10**: More diverse examples — better if few-shot learning is actually helping
- **selection = "most_common"**: Show the most frequent error type → focuses LLM on what Qaari does most
- **Use PATS Phase 1 pairs as few-shot**: Instead of QALB, use actual (ocr_text, gt_text) pairs from the training split. These are true OCR errors. Implementation: Sample from training split, filter by edit distance ≤5, format as few-shot. This requires one additional KnowledgeBase loader (e.g., `PATSPairLoader`).

---

### 2.4 RAG Parameters (Phase 5)

```yaml
phase5:
  retrieval:
    top_k: 3          # Try: 1, 5, 7
    min_score: 0.0    # Try: 0.5, 0.6 (filter low-relevance retrievals)
```

**Key experiments:**

- **top_k = 1**: Only the best match — reduces noise, uses fewer tokens
- **min_score = 0.5**: Only inject when cosine similarity ≥ 0.5 — avoids injecting irrelevant sentences that confuse the model
- **Chunk size**: Current is 200 chars for corpus sentence. Try 100 chars — shorter but more semantically focused chunks
- **Better embedding model**: `intfloat/multilingual-e5-large` or `CAMeL-Lab/bert-base-arabic-camelbert-msa` may give better Arabic-language similarity scores
- **Hybrid retrieval**: Combine BM25 lexical matching with dense retrieval — exact character matches help with OCR-specific noise (see Section 3.8)

---

### 2.5 CAMeL Validation Strategy (Phase 4C)

Current in `config.yaml`:
```yaml
camel:
  validation:
    strategy: "revert"
    min_confidence: 0.5
```

**Key experiments:**

- **Apply CAMeL post-processing to ALL phases**: The revert strategy is currently only tested as an isolated Phase 4C experiment. As a post-processing step, apply it on top of Phase 2, 3, 4A, 4B, and 5 outputs. This does not require new inference — just a new `--mode validate` run.
- **Confidence thresholds**: Change `min_confidence` to 0.7 (stricter) or 0.3 (looser). Higher = more LLM words are reverted.
- **Soft revert**: Instead of binary revert/keep, score each LLM word by morphological frequency and revert only words that are both (a) invalid AND (b) edit-distance > 1 from OCR word. This avoids reverting valid-but-uncommon words that the LLM correctly introduced.

---

### 2.6 Text Segmentation

**Problem**: Long OCR texts (>100 chars) dilute the instruction-to-content ratio. On PATS, texts average ~45 chars, but some samples are 200+ chars.

**Strategy**: Split texts longer than a threshold into overlapping segments, correct each segment, then stitch back.

```python
# Pseudocode: segment-correct-stitch
def correct_long_text(ocr_text, threshold=80):
    if len(ocr_text) <= threshold:
        return llm_correct(ocr_text)

    words = ocr_text.split()
    segments = create_overlapping_segments(words, max_words=10, overlap=2)
    corrected_segments = [llm_correct(seg) for seg in segments]
    return stitch_segments(corrected_segments, overlap=2)
```

**Caution**: Stitching is non-trivial with Arabic (words split by overlap). A simpler approach: split only on sentence boundaries (؟ . ! ،).

---

## 3. Innovative Alternative Ideas

These require more implementation effort but could yield significant improvements.

### 3.1 Synthetic OCR Pair Extraction for Few-Shot

**Concept**: Instead of QALB (human grammar errors), use *actual Qaari errors* from the PATS training split as few-shot examples.

**Why it's better**: The error distribution is identical to the test set — same OCR engine, same domain, same character confusion patterns.

**Implementation**:
1. In Phase 1, also save `(ocr_sample, gt_sample)` pairs with small edit distance (1–5 chars) from the training split
2. Create `PATSPairLoader` in `src/data/knowledge_base.py` that loads these pairs
3. Filter by edit distance threshold and error type (using `ErrorAnalyzer.classify_error()`)
4. Use as drop-in replacement for QALB in Phase 4B

**Expected benefit**: Higher relevance (same OCR engine), better calibration (actual confusion patterns), potential for much stronger few-shot signal.

**Note**: Use ONLY train split pairs to avoid data leakage.

---

### 3.2 Self-Consistency Majority Voting

**Concept**: Run the LLM 3–5 times at temperature=0.3, then take a word-level majority vote.

**Why it works**: For ambiguous corrections (e.g., should this ه be ة?), the model may choose differently each run. Majority voting picks the most confident answer.

**Implementation**:
```python
# In TransformersCorrector, add n_samples parameter
corrections = [model.generate(prompt, temperature=0.3) for _ in range(3)]
# Word-level majority vote
words_per_run = [c.split() for c in corrections]
final_words = [Counter(w[i] for w in words_per_run).most_common(1)[0][0]
               for i in range(max_len)]
result = " ".join(final_words)
```

**Cost**: 3× inference time per sample. Feasible on Kaggle P100/T4 with the 4B model.

**Expected benefit**: Reduces variance, particularly for ambiguous characters (ة/ه, ى/ي).

---

### 3.3 Two-Pass Correction (Detect, Then Fix)

**Concept**: Use two LLM calls per sample:
1. **Detection pass**: "Which words in this text appear to have OCR errors? List them."
2. **Correction pass**: "Correct only these words: [error_word_1, error_word_2, ...]"

**Why it's better**: The detection pass scopes the correction to suspected errors, preventing the LLM from "correcting" words that are already correct.

**Prompt for detection pass (Arabic)**:
```
أنت خبير في أخطاء OCR العربي.
حدد الكلمات التي تبدو فيها أخطاء تعرف ضوئي في النص التالي.
أجب بقائمة من الكلمات المشكوك فيها فقط، كل كلمة في سطر.
إذا كان النص صحيحاً، أجب بـ "لا أخطاء".
```

**Prompt for correction pass (Arabic)**:
```
صحح هذه الكلمات فقط من النص:
الكلمات المشكوك فيها: {suspected_words}
النص الكامل: {ocr_text}
أعد النص كاملاً مع تصحيح الكلمات المحددة فقط.
```

**Trade-off**: 2× LLM calls per sample. Detection pass may miss errors or flag correct words (false positives).

---

### 3.4 Structured JSON Output

**Concept**: Instead of asking the LLM to return corrected full text, ask for a structured list of (original_span, corrected_span) changes.

**System prompt (Arabic)**:
```
أنت مصحح OCR عربي متخصص.
أعد تصحيحاتك بصيغة JSON فقط:
{"changes": [{"original": "الكلمة الخاطئة", "corrected": "الكلمة الصحيحة"}]}
إذا لم توجد أخطاء، أعد: {"changes": []}
```

**Post-processing**: Apply each change as a string replacement on the original OCR text.

**Advantages**:
- Transparent: you can see exactly what the LLM changed
- Conservative: only listed spans are modified (no hallucinated word insertions/deletions)
- Auditable: every change can be independently evaluated against Phase 1 confusion matrix

**Risks**:
- LLM may not produce valid JSON (use `json.loads()` with fallback to Phase 2)
- May miss complex errors (merged words, word boundary changes)

---

### 3.5 Ensemble Across Prompt Types

**Concept**: For each sample, run Phase 2, Phase 3, Phase 4A, and Phase 4B independently. Then take a word-level vote.

**Why it could help**: Different knowledge types (rules, examples, confusion matrix) address different error types. Their agreement signals high-confidence corrections; their disagreement signals ambiguity.

**Implementation**:
```python
# Collect outputs from 4 phases
outputs = {
    "zero_shot": p2_result,
    "ocr_aware": p3_result,
    "rules": p4a_result,
    "fewshot": p4b_result,
}
# Word-level majority vote
ensemble = word_majority_vote(list(outputs.values()))
```

**Cost**: 4× inference time. Can be done in post-processing by loading the 4 `corrections.jsonl` files locally without additional inference.

---

### 3.6 Edit-Distance Gating

**Concept**: After LLM correction, check if the output differs from OCR text by more than a threshold K (normalized edit distance). If so, revert to OCR.

**Rationale**: Legitimate OCR corrections should be *small edits* — a few character substitutions. If the LLM rewrites the entire sentence (hallucination), the edit distance will be large.

**Implementation**:
```python
import editdistance

def edit_distance_gate(ocr: str, llm_output: str, threshold: float = 0.3) -> str:
    """Revert to OCR if LLM changed more than threshold fraction of characters."""
    dist = editdistance.eval(ocr, llm_output)
    normalized = dist / max(len(ocr), 1)
    if normalized > threshold:
        return ocr  # Too many changes — revert
    return llm_output
```

**Threshold tuning**: Try 0.1 (very conservative), 0.2 (moderate), 0.3 (liberal).
- At 0.1: only accept corrections with ≤10% character change rate
- A threshold of 0.2 aligns with the average CER on PATS (if OCR error rate is 5%, even perfect correction would only change ~5%)

---

### 3.7 Better Arabic Embeddings for RAG

**Current**: `paraphrase-multilingual-MiniLM-L12-v2` — trained on 50+ languages, not specialized for Arabic.

**Better alternatives** (in order of preference):

| Model | Size | Arabic Quality | Notes |
|-------|------|----------------|-------|
| `intfloat/multilingual-e5-large` | 560M | High | Better multilingual quality, still accessible |
| `CAMeL-Lab/bert-base-arabic-camelbert-msa` | 110M | Very High | Arabic-specific, trained on MSA |
| `aubmindlab/bert-base-arabert` | 110M | Very High | AraBERT, strong Arabic understanding |
| `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | 270M | Medium-High | Better than MiniLM |

**Implementation**: Change in `configs/config.yaml`:
```yaml
rag:
  embedding_model: "intfloat/multilingual-e5-large"
```

**Note**: Arabic-specific models may not produce good sentence embeddings without fine-tuning. `multilingual-e5-large` is the safest upgrade with known strong multilingual sentence similarity performance.

---

### 3.8 Hybrid BM25 + Dense Retrieval

**Concept**: Combine sparse (BM25, keyword-matching) and dense (embedding-based) retrieval for RAG.

**Why it helps for OCR**: OCR errors are character-level — the input text may share exact substrings with corpus sentences (shared correct parts). BM25 captures these exact matches that dense embeddings may miss.

**Implementation** (using `rank_bm25` library):
```python
from rank_bm25 import BM25Okapi
import numpy as np

class HybridRetriever:
    def __init__(self, corpus: list[str], dense_retriever: RAGRetriever):
        self.bm25 = BM25Okapi([doc.split() for doc in corpus])
        self.dense = dense_retriever
        self.corpus = corpus

    def retrieve(self, query: str, k: int = 3, alpha: float = 0.5) -> list[str]:
        # BM25 scores (sparse)
        bm25_scores = self.bm25.get_scores(query.split())
        bm25_scores = bm25_scores / (bm25_scores.max() + 1e-9)  # normalize

        # Dense scores
        dense_results = self.dense.retrieve(query, k=len(self.corpus))
        dense_scores = np.zeros(len(self.corpus))
        for r in dense_results:
            dense_scores[r.rank - 1] = r.score

        # Hybrid: weighted combination
        hybrid_scores = alpha * dense_scores + (1 - alpha) * bm25_scores
        top_k_indices = hybrid_scores.argsort()[-k:][::-1]
        return [self.corpus[i] for i in top_k_indices]
```

**Alpha tuning**: Try alpha=0.3 (lean BM25), 0.5 (balanced), 0.7 (lean dense).

---

### 3.9 Fine-Tuning on OCR Correction Pairs (Optional, High-Impact)

**Concept**: Fine-tune a small Arabic language model on (ocr_text → gt_text) pairs extracted from the PATS training split.

**Why it could dominate**: All other phases use frozen Qwen3-4B with in-context learning. Fine-tuning directly adapts the model weights to the specific OCR error distribution of Qaari.

**Data**: From Phase 1 analysis, you have (ocr_sample, gt_sample) pairs for all 18 training datasets. This is clean, domain-specific training data.

**Options**:
1. **Seq2Seq fine-tuning on Qwen3-4B**: Using LoRA/QLoRA, fine-tune on PATS training pairs. Minimal GPU requirement (~8GB with 4-bit quantization).
2. **Fine-tune AraBART**: A seq2seq Arabic model pre-trained on Arabic; potentially better than Qwen3 for this structured correction task.
3. **Prefix-tuning**: Learn a soft prefix for the OCR correction task without changing base model weights.

**Caveats**:
- This shifts the work from "prompting" to "fine-tuning" — different research contribution
- Could be framed as an extension/future work if the thesis timeline doesn't allow
- If you do this, keep the 8-phase pipeline as the comparison baseline

---

## 4. Prompt Engineering Deep Dive

### 4.1 Zero-Shot Prompt (Phase 2) — Improvements

**Current prompt**:
```
أنت مصحح نصوص عربية متخصص. مهمتك تصحيح أخطاء التعرف الضوئي (OCR) في النص العربي.
أعد النص المصحح فقط بدون أي شرح أو تعليق إضافي.
```

**Identified weaknesses**:
1. Does not say "be conservative" — model may over-correct
2. Does not specify output should be the same length
3. "متخصص" (specialist) is vague — add more specific OCR framing

**Improved version**:
```
أنت نظام تصحيح أخطاء OCR متخصص في النصوص العربية.
مهمتك: صحح أخطاء التعرف الضوئي (حروف محرفة أو مبدلة بسبب جودة الصورة) وأبقِ بقية النص كما هو.

قواعد أساسية:
- صحح الأحرف المحرفة فقط، لا تعيد صياغة الجملة
- إذا لم تكن متأكداً من وجود خطأ، أبقِ الكلمة كما هي
- أعد النص بنفس عدد الكلمات تقريباً
- لا تضف معلومات جديدة ولا تحذف كلمات

أعد النص المصحح فقط.
```

**Key additions**:
- "أبقِ الكلمة كما هي" (keep as is if uncertain) — conservative bias
- "لا تعيد صياغة الجملة" (don't rephrase) — prevents hallucination
- "نفس عدد الكلمات" (same word count) — length anchoring

---

### 4.2 OCR-Aware Prompt (Phase 3) — Improvements

**Current format** (flat_arabic):
```
أخطاء Qaari الشائعة في التعرف على الحروف:
- يستبدل (ة) بـ (ه) في 25% من الحالات
- يستبدل (ي) بـ (ى) في 18% من الحالات
```

**Issues**:
1. Probability percentages are generic — they apply to the pooled corpus, not this specific sentence
2. The format reads like a grammar lesson, not a targeted correction guide
3. No instruction on priority (which error type matters most?)

**Improved format** — make it *actionable*:
```
تنبيه: يقوم نظام Qaari بهذه الاستبدالات الخاطئة بشكل متكرر، تحقق منها في النص:
1. (ة) ← يكتبها أحياناً (ه)  — تحقق من نهايات الكلمات المؤنثة
2. (ي) ← يكتبها أحياناً (ى)  — تحقق من أواخر الأفعال والأسماء
3. (ق) ← يكتبها أحياناً (ف)  — النقطتان فوق مقابل نقطة واحدة فوق
```

**Change**: Use "← يكتبها أحياناً" (writes it as) rather than "يستبدل" (replaces) — more natural Arabic

**Add**: Include a "scan instruction" — direct the model to look at specific parts of the word (word endings for ة/ه, dot positions for ف/ق pairs).

---

### 4.3 Rules Prompt (Phase 4A) — Improvements

**Current**: All 6 rule categories injected with equal weight.

**Improvement — Input-adaptive rule selection**:

Before generating the prompt, analyze the OCR text to determine which rule categories are most relevant. If the text contains many words ending in ه or ة, inject the taa_marbuta rule first. If the text has many ا sequences, emphasize hamza rules.

```python
def select_relevant_rules(ocr_text: str, all_rules: list[ArabicRule]) -> list[ArabicRule]:
    """Rank rules by relevance to the specific input text."""
    scored = []
    for rule in all_rules:
        score = 0
        # Count trigger characters in the text
        if rule.category == "taa_marbuta" and re.search(r'[هة]', ocr_text):
            score += ocr_text.count('ه') + ocr_text.count('ة')
        elif rule.category == "hamza" and re.search(r'[اأإآء]', ocr_text):
            score += sum(ocr_text.count(c) for c in 'اأإآء')
        # ... etc
        scored.append((score, rule))
    return [r for _, r in sorted(scored, reverse=True)]
```

This makes the rules prompt **sample-specific** rather than static.

---

### 4.4 Few-Shot Prompt (Phase 4B) — Improvements

**Current format** (inline_arabic):
```
أمثلة على تصحيح أخطاء الكتابة العربية:
- خطأ:  هاده الجامعه تتمتع بسمعة عالمية
  صحيح: هذه الجامعة تتمتع بسمعة عالمية
```

**Improvement 1 — Add explanation per example**:
```
- خطأ:  هاده الجامعه تتمتع بسمعة عالمية
  صحيح: هذه الجامعة تتمتع بسمعة عالمية
  (التصحيح: ه→ذ في "هاذه"، وه→ة في "الجامعة")
```

Explaining *what* changed helps the LLM understand the pattern, not just memorize the output.

**Improvement 2 — Add a "no change" example**:
```
- خطأ:  الطالب يدرس في المكتبة كل يوم
  صحيح: الطالب يدرس في المكتبة كل يوم
  (لا تغيير — النص صحيح كما هو)
```

This explicitly teaches the model that "not every text has errors" — crucial for preventing over-correction on clean fonts.

**Improvement 3 — Label error type**:
```
مثال (خطأ نقطة):
- خطأ:  ذهب الرجل إلى السوق لشراء الفاكهه
  صحيح: ذهب الرجل إلى السوق لشراء الفاكهة
```

Categorizing the error type in the few-shot example helps the model generalize the pattern.

---

### 4.5 RAG Prompt (Phase 5) — Improvements

**Current**:
```
فيما يلي نصوص عربية صحيحة مشابهة للنص المراد تصحيحه، استخدمها كمرجع
```

**Issues**:
1. The instruction "use as reference" is ambiguous — should the model copy phrases? paraphrase? just use as style guide?
2. Retrieved OpenITI sentences are often classical Arabic (old vocabulary) — may mislead the model on modern text

**Improved instruction**:
```
فيما يلي أمثلة على كتابة عربية صحيحة. استخدمها فقط للتحقق من صحة الأحرف والكلمات في النص أدناه، ولا تنقل منها عبارات أو أفكاراً:
```

**Key change**: "للتحقق من صحة الأحرف" (to verify character correctness) rather than "كمرجع" (as a reference) — this constrains the model to use retrieval for character-level verification, not content copying.

**Additional idea**: Add cosine similarity score in the retrieval context so the model knows how relevant each sentence is:
```
نصوص مرجعية صحيحة (مرتبة حسب التشابه):
1. [تشابه: 0.78] استقبل الملك الزوار في القصر بكرم.
2. [تشابه: 0.65] المكتبة تضم آلاف الكتب النادرة.
```

---

### 4.6 Combined Prompt (Phase 6) — Section Ordering

**Current order**: confusion → rules → examples → retrieval

**Research insight**: In few-shot learning literature, examples shown *immediately before* the task (closest to the input) have the most influence on model behavior. The current order puts retrieval last, which may be optimal. However, for this specific case:

**Alternative ordering to test**: examples → confusion → retrieval → rules
- Examples first: establish the correction pattern via demonstration
- Confusion: reinforce which specific characters to watch
- Retrieval: provide correct Arabic context
- Rules: serve as a final check

**Reason to try**: Rules are the most abstract/generic knowledge — they may be better as a "background constraint" (last) rather than "upfront instruction" (first).

---

### 4.7 Advanced Prompt Strategies

#### Output Prefix (Force Continuation)

End the user message with the beginning of the expected output:

```python
user_message = f"{ocr_text}\n\nالنص المصحح:"
```

This exploits the autoregressive nature of the model — it will continue from "النص المصحح:" rather than starting from scratch. Effectively forces the output format and removes the risk of the model prefacing its answer.

---

#### Conservative Instruction Addition

Add to every system prompt (regardless of phase):
```
تحذير: إذا كانت الكلمة قد تكون صحيحة أصلاً، لا تغيّرها. الأفضل أن تبقي الكلمة كما هي من أن تغيّرها خطأً.
```

Translation: "Warning: If a word might already be correct, do not change it. It is better to leave a word unchanged than to change it incorrectly."

This shifts the model's behavior toward precision over recall — fewer false positives (introduced errors) at the cost of some false negatives (missed corrections).

---

#### Enable Thinking Mode (Qwen3 Specific)

Qwen3 has a reasoning scratchpad (think tokens). Currently disabled (`enable_thinking=False`) to keep output clean.

**Experiment**: Enable it for a subset of samples and compare output quality:
```python
formatted = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,   # Enable reasoning
)
```

Then strip `<think>...</think>` from output in post-processing. The reasoning steps may help the model make more accurate corrections, especially for ambiguous characters.

**Cost**: ~2× output tokens. Worth testing on a sample.

---

## 5. How to Write the Results in the Paper

### 5.1 Paper Structure (Recommended)

For a Master's thesis / research paper, the natural section structure is:

```
Abstract
1. Introduction
2. Related Work
3. Problem Definition & Datasets
4. System Design (8 Phases)
5. Experimental Setup
6. Results & Analysis
   6.1 Baseline (Phase 1)
   6.2 Zero-Shot LLM (Phase 2)
   6.3 Individual Knowledge Types (Phases 3-5)
   6.4 Combinations & Ablation (Phase 6)
   6.5 Error Analysis
7. Discussion
8. Limitations & Future Work
9. Conclusion
```

---

### 5.2 Dataset Description Table

Always include this table in the paper — reviewers need to understand the data complexity:

| Dataset | Type | Font | Samples (Train) | Samples (Val) | Baseline CER (%) | Baseline WER (%) |
|---------|------|------|-----------------|---------------|-------------------|-------------------|
| PATS-Akhbar | Synthetic | Akhbar | 2,213 | 553 | 1.94 | 5.03 |
| PATS-Andalus | Synthetic | Andalus | 2,211 | 551 | 13.67 | 35.94 |
| PATS-Arial | Synthetic | Arial | 2,207 | 551 | 3.16 | 6.89 |
| PATS-Naskh | Synthetic | Naskh | 2,212 | 553 | 4.74 | 12.94 |
| PATS-Simplified | Synthetic | Simplified | 2,213 | 553 | 2.72 | 5.90 |
| PATS-Tahoma | Synthetic | Tahoma | 2,203 | 549 | 5.26 | 8.76 |
| PATS-Thuluth | Synthetic | Thuluth | 2,213 | 553 | 12.57 | 33.27 |
| PATS-Traditional | Synthetic | Traditional | 2,213 | 553 | 2.81 | 9.18 |
| KHATT | Handwritten | — | 1,393 | 230 | 63.0 | 91.7 |

> Note: Report train and validation separately. Report normal samples only (exclude runaway samples), and note the % of runaway samples separately.

---

### 5.3 Main Results Table

**Table format for Phase 2-5 comparison** (compare each phase to Phase 2 baseline):

| Phase | Knowledge Added | CER (PATS-mean) | WER (PATS-mean) | CER (KHATT) | ΔCER vs Ph2 | p-value | Cohen's d |
|-------|----------------|-----------------|-----------------|-------------|-------------|---------|-----------|
| 1 (OCR baseline) | — | X.XX | X.XX | X.XX | — | — | — |
| 2 (Zero-Shot) | None | X.XX | X.XX | X.XX | — | — | — |
| 3 (OCR-Aware) | Confusion Matrix | X.XX | X.XX | X.XX | ±X.XX | p=X.XX | d=X.XX |
| 4A (Rules) | Arabic Rules | X.XX | X.XX | X.XX | ±X.XX | p=X.XX | d=X.XX |
| 4B (Few-Shot) | QALB Examples | X.XX | X.XX | X.XX | ±X.XX | p=X.XX | d=X.XX |
| 4C (CAMeL) | Morphological Valid. | X.XX | X.XX | X.XX | ±X.XX | p=X.XX | d=X.XX |
| 5 (RAG) | OpenITI Corpus | X.XX | X.XX | X.XX | ±X.XX | p=X.XX | d=X.XX |

**Formatting guidelines**:
- Bold the best result in each column
- Use arrows (↑/↓) for ΔCER: ↓ means improvement (lower CER), ↑ means regression
- Report PATS and KHATT separately — never average them together
- p-values: use `< 0.001`, `< 0.01`, `< 0.05`, or `n.s.` (not significant)

---

### 5.4 Phase 6 Ablation Table

**Standard ablation format** — each row removes one component from the full system:

| System | Confusion | Rules | Few-Shot | RAG | CAMeL | CER (PATS) | WER (PATS) | ΔCER vs Full |
|--------|-----------|-------|----------|-----|-------|------------|------------|--------------|
| Phase 2 (baseline) | — | — | — | — | — | X.XX | X.XX | — |
| Full System | ✓ | ✓ | ✓ | ✓ | ✓ | X.XX | X.XX | — |
| −Confusion | — | ✓ | ✓ | ✓ | ✓ | X.XX | X.XX | ±X.XX |
| −Rules | ✓ | — | ✓ | ✓ | ✓ | X.XX | X.XX | ±X.XX |
| −Few-Shot | ✓ | ✓ | — | ✓ | ✓ | X.XX | X.XX | ±X.XX |
| −RAG | ✓ | ✓ | ✓ | — | ✓ | X.XX | X.XX | ±X.XX |
| −CAMeL | ✓ | ✓ | ✓ | ✓ | — | X.XX | X.XX | ±X.XX |

**Interpretation guidance for the paper**:
- A large positive ΔCER when removing component X means X is *important*
- A near-zero ΔCER when removing X means X is *redundant* (already captured by other components)
- A negative ΔCER when removing X means X was *hurting* performance (remove it from final system)

---

### 5.5 Per-Font Results (PATS Variance Analysis)

Include this table to show that font difficulty matters and that the LLM handles different fonts differently:

| Font | OCR CER | Ph2 CER | Best Phase | Best CER | Relative Improvement |
|------|---------|---------|-----------|----------|---------------------|
| Akhbar | 1.94% | X.XX% | PhX | X.XX% | X.X% |
| Andalus | 13.67% | X.XX% | PhX | X.XX% | X.X% |
| Arial | 3.16% | X.XX% | PhX | X.XX% | X.X% |
| Naskh | 4.74% | X.XX% | PhX | X.XX% | X.X% |
| Simplified | 2.72% | X.XX% | PhX | X.XX% | X.X% |
| Tahoma | 5.26% | X.XX% | PhX | X.XX% | X.X% |
| Thuluth | 12.57% | X.XX% | PhX | X.XX% | X.X% |
| Traditional | 2.81% | X.XX% | PhX | X.XX% | X.X% |

**Finding to highlight**: "LLM correction shows the largest *relative* improvement on high-error fonts (Andalus, Thuluth), suggesting it is most useful when OCR quality is low but not catastrophic."

---

### 5.6 Statistical Reporting Guidelines

**Always report these together**:
1. **Mean difference** (ΔCER / ΔWER): "Phase 3 reduced mean CER by 0.52 percentage points"
2. **p-value** with Bonferroni correction: "p = 0.023 (Bonferroni-corrected for 9 comparisons, α = 0.05/9 = 0.0056)"
3. **Cohen's d**: "with a medium effect size (d = 0.42)"
4. **95% CI** (bootstrap): "95% CI [−0.81, −0.23]"

**Example sentence**:
> Phase 3 (OCR-Aware Prompting) reduced mean CER by 0.52 pp on PATS datasets (d = 0.42, 95% CI [−0.81, −0.23], paired t-test: t(17) = 3.12, p = 0.006, Bonferroni-corrected p = 0.054), representing a statistically marginal improvement at the corrected threshold.

**What "significant" means**: Use the Bonferroni-corrected threshold. If you test 9 combos with α=0.05, the corrected threshold is α/9 = 0.0056. Be honest — many comparisons may not reach this threshold.

---

### 5.7 Error Analysis Tables

**Table: Error Category Distribution Before and After Correction**

| Error Category | Phase 1 (OCR) Count | Phase 2 (LLM) Fixed | Phase 2 Introduced | Net Change |
|----------------|--------------------|--------------------|-------------------|------------|
| Taa Marbuta (ة/ه) | X | X | X | ±X |
| Hamza | X | X | X | ±X |
| Alef Maksura (ى/ي) | X | X | X | ±X |
| Dot Confusion | X | X | X | ±X |
| Similar Shapes | X | X | X | ±X |
| Insertion | X | X | X | ±X |
| Deletion | X | X | X | ±X |
| Merged Words | X | X | X | ±X |
| Split Words | X | X | X | ±X |
| Other | X | X | X | ±X |

**Key finding to look for**: Which error types does the LLM fix well (high "Fixed" count) and which does it make worse (high "Introduced" count)?

---

### 5.8 Failure Case Analysis

Always include a qualitative failure case table. Reviewers appreciate concrete examples:

| Example | OCR Text | GT Text | LLM Output | Error Type |
|---------|----------|---------|-----------|------------|
| Over-correction | ذهب إلى المدرسه | ذهب إلى المدرسة | ذهب إلى المدرسة | Correct ✓ |
| Over-correction | هذا الرجل يكتبه | هذا الرجل يكتبه | هذا الرجل يكتبة | LLM changed correct ه → ة |
| Hallucination | المكتـ... ية | المكتبة العامة | المكتبة الجديدة | LLM invented "جديدة" |
| Missed error | يستبدل (ف) بـ (ق) | يستبدل (ف) بـ (ف) | يستبدل (ف) بـ (ق) | Missed dot correction |
| Merge not fixed | المدرسةوالجامعة | المدرسة والجامعة | المدرسةوالجامعة | Word merge not corrected |

Show 2–3 examples per failure type. Use actual samples from your data.

---

### 5.9 Limitations Section (Be Specific)

Write concrete, specific limitations rather than generic disclaimers:

1. **Model size**: Qwen3-4B is a small model by current standards. Larger models (7B, 13B, 70B) may perform better but require more compute. This is a practical constraint, not a methodological flaw.

2. **QALB domain mismatch**: QALB contains human grammar errors (casual Arabic typing mistakes). OCR errors are caused by visual character confusion. While many error types overlap (ة/ه, hamza), the distributions differ. A purpose-built OCR correction corpus would be more appropriate.

3. **KHATT handwriting**: With 60–66% CER, handwritten text is largely unrecoverable by prompt-based LLM correction. Future work should address handwriting separately (e.g., with OCR-specific post-processing or specialized handwriting recognition models).

4. **Static rule encoding**: The Arabic orthographic rules in Phase 4A are hardcoded and manually curated. They do not adapt to dataset-specific error patterns or learn from results.

5. **FAISS index on general corpus**: OpenITI is a classical Arabic corpus. Modern OCR text may not have high semantic similarity to classical texts, reducing RAG effectiveness. A modern Arabic news or web corpus might be more appropriate.

6. **Single evaluation model**: All phases use Qwen3-4B. Results may not generalize to other LLMs (GPT-4, Llama-3, AraGPT2).

---

### 5.10 Framing the Contribution

**Research narrative structure** for the Introduction/Conclusion:

1. **The Gap**: Closed-source VLMs (GPT-4V, Gemini, Claude) demonstrate strong Arabic OCR capability, but are expensive and inaccessible. Open-source OCR (Qaari) produces high error rates, especially on complex fonts and handwriting.

2. **The Approach**: Can LLM post-processing bridge this gap? We systematically test 8 knowledge injection strategies with Qwen3-4B.

3. **The Findings** (anticipated structure):
   - "Vanilla LLM correction reduces CER by X% on typewritten Arabic" → Phase 2 finding
   - "OCR-specific knowledge (confusion matrix) provides additional Y% improvement" → Phase 3 finding
   - "Rule injection and few-shot examples contribute comparably" → Phase 4 finding
   - "RAG from a historical corpus provides Z% additional improvement" → Phase 5 finding
   - "The optimal combination achieves W% CER, closing X% of the gap to closed-source VLMs" → Phase 6 finding
   - "Handwritten text (KHATT) remains challenging at 65% baseline CER" → Honest limitation

4. **The Contribution**:
   - Systematic evaluation of 5 knowledge types for Arabic OCR correction
   - Open-source replicable pipeline (Kaggle/Colab compatible)
   - Ablation study quantifying each component's contribution
   - Empirical evidence on LLM post-correction feasibility for Arabic OCR

---

### 5.11 Visualizations

**Include these figures in the paper**:

1. **Bar chart**: CER comparison across phases (Phase 1 → 2 → 3 → 4A → 4B → 4C → 5 → 6-best) — one bar per phase, one chart per dataset type (PATS vs KHATT)

2. **Heatmap**: Confusion matrix of Qaari OCR errors — rows = GT characters, columns = OCR characters, color intensity = frequency

3. **Ablation bar chart**: Full system vs each ablated version — horizontal bars show ΔCER when each component is removed

4. **Font-level scatter plot**: X-axis = OCR baseline CER, Y-axis = LLM improvement (best phase vs baseline), one point per font — shows whether improvement correlates with error rate

5. **Error type stacked bar**: For Phase 1 and best phase — stacked bars showing counts of each error category, before and after LLM correction

6. **System architecture diagram**: Simple block diagram of the 3-stage pipeline (export → Kaggle inference → analyze) and which knowledge sources flow into which phase

---

### 5.12 Abstract Template

The abstract should summarize all key numbers. Draft:

> Arabic Optical Character Recognition (OCR) remains challenging for open-source systems. Qaari, a leading open-source Arabic OCR engine, achieves Character Error Rates (CER) ranging from **1.94% to 13.67%** on typewritten text and **60–66%** on handwritten text. We investigate whether Large Language Models (LLMs) can serve as a post-processing step to correct Qaari's errors, testing **five knowledge injection strategies**: OCR error patterns, linguistic rules, few-shot examples, morphological validation, and retrieval-augmented generation (RAG). Using **Qwen3-4B** on 18 datasets comprising 47,859 samples from PATS-A01 and KHATT, we find that [Key Finding 1]. Among isolated knowledge types, [Key Finding 2]. Our full combined system achieves [Best CER] on typewritten Arabic, representing a [X%] relative improvement over the zero-shot LLM baseline. Ablation analysis reveals that [Key Ablation Finding]. We release our pipeline as fully reproducible open-source code.

---

## Quick Reference: Experiment Priority

If you need to prioritize follow-up experiments, here is a ranked list:

| Priority | Experiment | Expected Impact | Implementation Effort |
|----------|-----------|-----------------|----------------------|
| 1 | Synthetic OCR pairs from PATS train as few-shot (replaces QALB) | High | Medium |
| 2 | Conservative instruction addition to all prompts | High | Low |
| 3 | Edit-distance gating post-processing | High | Low |
| 4 | Apply CAMeL revert to all phases (not just 4C) | Medium-High | Low |
| 5 | Self-consistency majority voting (3 runs) | Medium | Medium |
| 6 | Two-pass detect-then-correct | Medium | Medium |
| 7 | Better Arabic embedding model for RAG | Medium | Low |
| 8 | Greedy decoding (temperature=0) | Low-Medium | Low |
| 9 | Enable Qwen3 thinking mode | Low-Medium | Low |
| 10 | Hybrid BM25+dense RAG retrieval | Low-Medium | High |

---

*Last updated: 2026-03-01*
*Based on: Full codebase analysis, Phases 1–6 implementation*
