# Master's Thesis — Writing Guidelines
## Arabic Post-OCR Correction Using Knowledge-Augmented LLMs
### Cairo University, Faculty of Engineering

---

## 0. Thesis Identity

| Field | Value |
|-------|-------|
| **Working Title** | "Knowledge-Augmented Large Language Models for Arabic Post-OCR Error Correction" |
| **Template** | Cairo University Faculty of Engineering — `cufethesis.cls` |
| **Main file** | `mthesis.tex` |
| **Target length** | 100–130 pages (main matter, excluding front/back) |
| **Chapter count** | 6 chapters (as per template) |
| **Language** | English (with Arabic abstract page per CUFE requirement) |

---

## 1. Thesis Narrative — The Full Arc

A master's thesis must tell a complete story from "here is the problem" through
"here is everything that was tried before us" all the way to "here is what we found
and what it means." The reader should need no prior knowledge of Arabic NLP, OCR,
or LLMs to follow the argument.

### The Central Question
**Can a lightweight open-source LLM correct Arabic OCR errors through prompt
engineering alone, and does adding more linguistic or retrieval-based knowledge
to the prompt improve the correction?**

### The Story in Six Acts (one per chapter)
1. **Why this matters and what we set out to do** (Introduction)
2. **What the world already knows — Arabic, OCR, LLMs, and correction** (Background)
3. **How we built the experiment — data, tools, pipeline** (System Design)
4. **What each phase does and why** (Experimental Methodology)
5. **What the numbers say** (Results and Analysis)
6. **What it all means, what we got wrong, what comes next** (Discussion and Conclusion)

---

## 2. Chapter-by-Chapter Plan

---

### CHAPTER 1 — Introduction
**File:** `MainMatter/Chapter1.tex`
**Target length:** 12–16 pages
**Purpose:** Motivate the problem, state the gap, list contributions, map the thesis.

#### 1.1  The Problem with Arabic OCR (~2 pages)
Start with the real-world stakes. Arabic text digitization is critical for
preserving historical documents, building search engines, and enabling machine
translation. Optical character recognition turns document images into text, but
Arabic poses unique challenges that English OCR does not face:
- Cursive connected script with no clear inter-character boundaries
- Short vowels (diacritics / tashkeel) are often optional in written Arabic yet
  critical for disambiguation
- Many letter pairs share the same shape and differ only by dots (ba, ta, tha, nun, ya)
- Right-to-left writing direction with ligatures
- Wide variation: classical vs. modern standard vs. dialectal
- Font variation (typewritten) and handwriting variation (handwritten)

Describe the consequence: OCR errors compound in downstream tasks. A misread
character changes word meaning; a sequence of errors degrades machine translation
quality, information retrieval recall, and text-to-speech output.

#### 1.2  The Open-Source vs. Closed-Source Gap (~1.5 pages)
Acknowledge that commercial/closed-source VLMs (GPT-4V, Gemini, Claude) handle
Arabic document recognition well because they were trained on massive multimodal
corpora. Open-source alternatives like Qaari are more accessible but fall behind.
State the gap quantitatively (qualitatively at this point — exact numbers come
in Chapter 5): the question is whether post-OCR correction with an LLM can close
this gap without changing the OCR engine.

#### 1.3  Post-OCR Correction as a Text-to-Text Task (~1 page)
Explain the idea of treating OCR output as noisy text and using a language model
to denoise it. This decouples the vision pipeline (OCR) from the language
pipeline (correction). It requires no retraining, works with any OCR engine,
and can be applied retroactively to existing digitized collections.

#### 1.4  Research Questions (~1 page)
State all research questions formally and number them. Reference them again in
Chapter 5 when answering each one.

**RQ1:** What are the characteristics of errors produced by Qaari on Arabic
typewritten text? What error types dominate, and how severe is the overall
degradation?

**RQ2:** Can a zero-shot instruction-tuned LLM correct Arabic OCR errors without
any task-specific guidance?

**RQ3:** Does injecting OCR-specific error knowledge (confusion matrix) into the
prompt improve correction beyond zero-shot?

**RQ4:** Does injecting explicit Arabic orthographic rules improve correction?

**RQ5:** Does providing error-correction examples (few-shot) from the QALB corpus
improve correction?

**RQ6:** Does morphological post-processing via CAMeL Tools improve the quality
of LLM-corrected text?

**RQ7:** Does self-reflective prompting — injecting the model's own diagnosed
failure patterns — improve correction?

**RQ8:** Does retrieval-augmented generation grounded in the OpenITI corpus
improve correction?

**RQ9 (overarching):** Which combination of knowledge augmentation strategies
produces the best correction, and what does each component individually contribute?

#### 1.5  Contributions (~1 page)
List concretely:
1. A systematic evaluation framework for Arabic post-OCR correction with six
   isolated knowledge-augmentation conditions.
2. A three-stage pipeline (export–infer–analyze) that separates local data
   handling from GPU inference.
3. A novel self-reflective prompting strategy (Phase 4D) that derives correction
   guidance from the model's own training-split error statistics.
4. An empirical finding that zero-shot correction outperforms all
   knowledge-augmented variants on high-quality typewritten Arabic OCR.
5. An error taxonomy for Qaari on the PATS-A01 Akhbar font with morphological
   classification using CAMeL Tools.
6. Open-source code and reproducible configuration for all phases.

#### 1.6  Thesis Organization (~0.5 pages)
One paragraph per chapter, describing what it covers. End with a sentence that
says the reader can skip to Chapter 5 for results or Chapter 6 for conclusions
if they are familiar with the background.

---

### CHAPTER 2 — Background and Literature Review
**File:** `MainMatter/Chapter2.tex`
**Target length:** 28–35 pages
**Purpose:** Build the reader from zero. Cover Arabic script, OCR fundamentals,
language models, prompt engineering, and all related work. A reader unfamiliar
with any of these topics should be able to follow the rest of the thesis.

#### 2.1  The Arabic Language and Script (~4 pages)

**2.1.1  Linguistic Overview**
Brief overview: Arabic is a Semitic language with a root-and-pattern morphology.
Most words derive from a three-consonant (trilateral) root. Prefixes and suffixes
encode case, number, gender, definiteness, verb tense, and more. Explain why
this makes Arabic morphologically richer than English and harder for text
processing tools.

**2.1.2  The Arabic Script**
- 28 letters; most have 4 positional forms (isolated, initial, medial, final)
- Letters are connected within words; spaces separate words
- Diacritics (harakat/tashkeel): fatha, kasra, damma, sukun, shadda, tanwin variants
  — placed above/below the consonant; optional in most modern text; critical for
  disambiguation (e.g., كَتَبَ kataba "he wrote" vs. كُتُب kutub "books")
- Hamza: appears alone or on carriers (alef, waw, ya seat); governed by rules
  that even native speakers sometimes apply inconsistently
- Taa marbuta (ة): feminine ending, pronounced /a/ in context or /h/ at pause;
  visually similar to ha (ه)
- Alef maksura (ى): looks like ya without dots; used at word end to represent /a/;
  confused with ya (ي) by both humans and OCR

**2.1.3  Arabic Text in Computing**
Unicode encoding; right-to-left rendering; issues with bidirectional text;
normalization (NFKC, NFKD); the hazards of mixing normalized and non-normalized
text when computing CER. Note that diacritics have their own Unicode code points
and can be stripped by removing characters in the range U+0610–U+061A and
U+064B–U+065F.

**2.1.4  Challenges for NLP**
- Tokenization: word boundaries are clear but morpheme boundaries are not
- Sparsity: many word forms for each lemma inflates vocabulary
- Dialectal variation: Egyptian, Gulf, Levantine dialects differ significantly
  from Modern Standard Arabic
- Code-switching: mixing Arabic and English is common in modern text

#### 2.2  Optical Character Recognition (~5 pages)

**2.2.1  What Is OCR?**
Define OCR as the process of converting images of text into machine-readable
character sequences. Explain the classical pipeline: preprocessing (binarization,
deskew, denoising) → layout analysis (page segmentation, line detection) →
character recognition → post-processing (language model correction).

**2.2.2  OCR Architectures**
Describe the evolution:
- Template matching (pre-2000): character images matched against stored templates
- Hidden Markov Models: model character sequences probabilistically; dominant
  through the 2000s
- Deep learning era: convolutional + recurrent (CNN-LSTM-CTC) became dominant
  around 2015; Tesseract v4 adopted LSTM
- Transformer-based OCR: TrOCR, Donut, and similar vision-language models
  approach the task as image-to-sequence generation; currently state-of-the-art

**2.2.3  Arabic-Specific OCR Challenges**
Connect the script properties from §2.1.2 to concrete OCR failure modes:
- Connected script means segmentation errors directly produce recognition errors
- Dot placement ambiguity: slight ink irregularity shifts ba to ta or nun
- Ligature handling: certain letter combinations merge into single glyphs
- Diacritic sensitivity: OCR engines often ignore diacritics entirely or add
  them incorrectly
- Baseline variability in handwritten text

**2.2.4  Qaari: The OCR Engine Used in This Study**
Describe Qaari specifically. Based on Tesseract with Arabic LSTM models. Note
its known error patterns. Describe the repetition artifact observed in Phase 1:
when Qaari encounters a certain layout pattern it can enter a repetitive loop,
reproducing a short text span hundreds of times. This is a known but infrequent
failure mode.

**2.2.5  Evaluation Metrics for OCR**
Define CER and WER formally with the edit-distance formula. Discuss the choice
of CER as primary metric (captures character-level precision better suited to
Arabic morphology). Note the diacritics stripping decision: evaluating with
diacritics penalizes models for adding/removing vowel marks that may be
stylistically optional; the standard practice is to strip diacritics before
evaluation. Both variants are tracked.

#### 2.3  Large Language Models (~6 pages)

**2.3.1  From N-grams to Transformers**
Brief history: n-gram language models, neural language models (Bengio et al.
2003), recurrent LMs, the transformer (Vaswani et al. 2017). Do not linger —
the reader needs enough context to understand the LLM papers cited later.

**2.3.2  The Transformer Architecture**
Cover: self-attention mechanism, multi-head attention, positional encoding,
feed-forward sublayers, layer normalization, encoder-decoder vs. decoder-only
architectures. Explain why attention allows the model to relate any token to
any other, making long-range dependencies tractable.

**2.3.3  Pre-Training and Instruction Tuning**
Explain the two-stage paradigm:
- Pre-training: predict next token over a massive text corpus; learns language
  statistics, world knowledge, syntactic and semantic structure
- Instruction tuning / RLHF: fine-tune on human-labeled instruction–response
  pairs to make the model follow natural language instructions

Explain why instruction-tuned models are the right choice for post-OCR correction:
they follow a correction instruction reliably without needing task-specific
fine-tuning.

**2.3.4  Qwen3-4B-Instruct-2507**
Describe the model used in this study:
- Qwen3 family from Alibaba DAMO Academy
- 4 billion parameters; instruction-tuned variant
- Strong multilingual capabilities, native Arabic support
- `enable_thinking` flag: when enabled, produces a chain-of-thought scratchpad
  before the answer; disabled in this study to reduce latency and token cost
- Available openly for research use; deployed on Kaggle T4 GPUs for inference

**2.3.5  Arabic LLMs and Arabic NLP with LLMs**
Review prior work on Arabic-focused LLMs: AraGPT2, AraT5, CAMeLBERT, AraBERT.
Discuss the shift from encoder-only BERT models toward decoder-only instruction-
tuned models. Note that Qwen3 and similar multilingual models now outperform
Arabic-specific models on many benchmarks due to scale.

#### 2.4  Prompt Engineering (~5 pages)

**2.4.1  Zero-Shot Prompting**
Explain: the model is given an instruction and an input without any examples.
Performance depends entirely on instruction clarity and the model's pre-training.
Discuss conservative vs. aggressive prompt design choices and why conservatism
(instruct the model to return text unchanged if correct) matters for correction
tasks where most tokens are already right.

**2.4.2  Few-Shot Prompting**
In-context learning: provide input–output demonstration pairs in the prompt
to show the model the task format. Discuss: how many examples to use; how to
select them (random vs. diverse vs. error-type-matched); the QALB corpus as a
source of Arabic error-correction pairs; the transfer gap between human typing
errors and OCR errors.

**2.4.3  Chain-of-Thought and Instruction Augmentation**
Explain chain-of-thought prompting (Wei et al. 2022) and why it is not used
in this study: correction is a one-pass transformation task, not a multi-step
reasoning task; CoT adds tokens without benefit and risks verbose output that
breaks the output format.

**2.4.4  Retrieval-Augmented Generation (RAG)**
Explain the RAG paradigm (Lewis et al. 2020): build an index over an external
corpus; at inference time retrieve the K most relevant passages; include them
in the prompt as grounding context. Cover: embedding models, vector indexes
(FAISS), similarity metrics, chunk size considerations. Describe the specific
configuration used in Phase 5: OpenITI corpus, MiniLM embeddings, FAISS flat
index, top-3 retrieval.

**2.4.5  Self-Reflective and Meta-Cognitive Prompting**
Discuss the idea of feeding a model information about its own failure patterns.
Connect to: Reflexion (Shinn et al. 2023), self-critique prompting, and verbal
reinforcement learning. Explain how Phase 4D operationalizes this: rather than
asking the model to critique its output at inference time, it is given pre-computed
statistics about systematic errors observed on training data. This is a lightweight
meta-learning approach.

#### 2.5  Knowledge Sources Used in This Study (~3 pages)

**2.5.1  Confusion Matrix as a Knowledge Source**
Explain what an OCR confusion matrix captures: for each true character, what
characters did the OCR engine produce? Describe how the top confusions from
Phase 1 are formatted as a prompt injection in Phase 3.

**2.5.2  Arabic Orthographic Rules**
Enumerate the specific rules used in Phase 4A: hamza al-qat' vs. hamza al-wasl,
taa marbuta vs. ha, alef maqsura vs. ya, tanwin rules, the definite article
with sun and moon letters. Provide a brief explanation of each rule with an
example Arabic word. These rules are grounded in classical Arabic grammar and
are taught in Arabic primary education.

**2.5.3  The QALB Corpus**
Describe QALB (Qatar Arabic Language Bank): collected for the Shared Task on
Automatic Arabic Text Correction. Contains human-written sentences paired with
corrections annotating grammatical, spelling, and punctuation errors. Explain
the filtering process for Phase 4B: selecting pairs whose error types overlap
with known OCR error types; filtering by sentence length (max 300 characters);
final count of OCR-relevant pairs available.

**2.5.4  The OpenITI Corpus**
Describe OpenITI: a large-scale digitized collection of Islamicate texts
spanning classical Arabic, Persian, and other languages of the Islamic scholarly
tradition. Contains hundreds of millions of words of Arabic text. Explain why
it is used for RAG: it is one of the largest freely available Arabic corpora,
which helps build a diverse retrieval index. Note the domain mismatch risk:
OpenITI texts are classical/historical; PATS-A01 contains modern newspaper text.

**2.5.5  CAMeL Tools**
Describe CAMeL Tools (NYU Abu Dhabi): an open-source Python toolkit for Arabic
NLP. Cover the components used:
- MorphologicalAnalyzer: given a word, returns all valid morphological analyses
  (lemma, root, pattern, POS, features) from the MSA database
- MorphologicalDisambiguator: context-aware selection of the best analysis
- Application in Phase 1 (error categorization) and Phase 4C (post-processing)
Describe the revert strategy: a corrected word that fails morphological analysis
while the original OCR word passes is reverted to the OCR original.

#### 2.6  Related Work on Post-OCR Correction (~4 pages)

**2.6.1  Classical Approaches**
Language model re-scoring, dictionary lookup, and hybrid methods. Cite key
early papers: Tong and Evans (1996), Kolak and Resnik (2005). Discuss their
limitations: no context beyond local n-grams, require large domain-specific
language models.

**2.6.2  Neural Sequence-to-Sequence Correction**
CNN-LSTM models for post-OCR correction (Dong and Smith 2018). Transformer-based
correction. Results for English: substantial improvement over baseline OCR,
especially on historical documents. Discuss why these methods need training data
(paired OCR output / ground truth), which is scarce for Arabic.

**2.6.3  LLM-Based Post-OCR Correction**
Kang et al. (2024): showed that ChatGPT-level LLMs are effective at English OCR
post-correction without any training data. Emphasize that this motivates using
LLMs for Arabic but that Arabic-specific evaluation is absent from their study.

**2.6.4  Arabic-Specific OCR Correction**
Survey the limited existing work: BERT-based correction for Arabic (AraBERT
fine-tuned on QALB-like data), rule-based Hunspell approaches. Identify the gap:
no study has compared multiple prompt engineering strategies for Arabic post-OCR
correction using a modern instruction-tuned LLM. This is the gap this thesis fills.

**2.6.5  Positioning of This Work**
A paragraph that explicitly states: this thesis is the first to (a) evaluate
six distinct knowledge augmentation strategies under controlled conditions for
Arabic OCR correction, (b) propose self-reflective prompting for OCR correction,
and (c) study the over-correction problem in the Arabic post-OCR setting.

---

### CHAPTER 3 — System Design and Datasets
**File:** `MainMatter/Chapter3.tex`
**Target length:** 18–22 pages
**Purpose:** Describe everything about the experimental infrastructure. A reader
should be able to reproduce the entire pipeline from this chapter.

#### 3.1  Overall Architecture (~2 pages)
Describe the three-stage pipeline:
1. **Export stage** (local): load dataset, build prompts, write inference_input.jsonl
2. **Inference stage** (Kaggle/GPU): load model, run prompts, write corrections.jsonl
3. **Analyze stage** (local): read corrections, compute CER/WER, produce reports

Explain why this separation was chosen: the research group does not have local
GPU access; this architecture lets inference run on any cloud GPU while data and
analysis remain local. Include a system architecture diagram (Fig. 3.1).

Describe the JSONL format: each line is a JSON object with fields `sample_id`,
`dataset`, `ocr_text`, `prompt`, `prompt_version`, and (after inference) `correction`.

#### 3.2  Datasets (~5 pages)

**3.2.1  PATS-A01**
- Full name and citation
- Eight Arabic typewritten fonts: Akhbar, Arabic Typesetting, Simplified Arabic,
  Traditional Arabic, DecoType Naskh, Lotus, Motken, Andalus (list all eight)
- Document structure: page images with line-level or page-level ground truth
- Ground truth encoding: CP-1256 (Windows Arabic codepage); converted to UTF-8
  for processing
- Special case: Traditional font GT file uses `TraditionalText.txt` instead of
  the A01-prefixed naming convention
- Split strategy: 80/20 train/validation per font, seed=42, using
  `scripts/generate_pats_splits.py`; split file at `data/ocr-raw-data/PATS_A01_Dataset/pats_splits.json`
- Total: 2,213 train / 553 validation per font; 2,766 total per font
- Dataset keys used in pipeline (18 total): `PATS-A01-{font}-train` and
  `PATS-A01-{font}-val` for all 8 fonts + KHATT-train + KHATT-validation
- **Experiments in this thesis**: Akhbar font, training split, 200 samples

**3.2.2  KHATT**
- Handwritten Arabic text dataset (Mahmoud et al. 2014)
- Real handwriting from multiple writers; natural variation in style and quality
- Train and validation splits
- Expected to be more challenging than PATS-A01 due to unconstrained handwriting
- **Experiments in this thesis**: not yet evaluated (future work)

**3.2.3  Dataset Statistics Table**
Include a table (Table 3.1) with: dataset name, type (typewritten/handwritten),
split, total samples, samples evaluated in this thesis, encoding, notes.

**3.2.4  OCR Results Format**
Describe Qaari output files: one `.txt` file per image with recognized text.
File structure under `data/ocr-results/qaari-results/`. Explain the DataLoader
(`src/data/data_loader.py`) and how it maps dataset keys to file paths.

#### 3.3  Preprocessing and Evaluation (~3 pages)

**3.3.1  Text Normalization**
Describe what is and is not normalized before computing metrics:
- Diacritics are stripped (Unicode range U+064B–U+065F and related)
- Tatweel (kashida, U+0640) is removed
- No letter normalization (alef variants, hamza forms are kept as-is for evaluation)
Explain the rationale: letter normalization would hide certain classes of errors
that the study aims to measure.

**3.3.2  CER and WER Computation**
Use the `jiwer` library (or custom edit-distance implementation). Describe:
- Token boundary: characters for CER; whitespace-delimited words for WER
- Normalization applied before metric computation
- Both raw (`corrected`) and diacritic-stripped (`corrected_no_diacritics`)
  variants are computed; the thesis reports the diacritic-stripped variant
  as primary
- Per-sample CER is computed first; aggregate statistics (mean, std, median,
  p95) are derived from the per-sample distribution

**3.3.3  Error Taxonomy Construction**
Describe the seven error categories and the detection heuristics for each:
- Dot confusion: substitution between a character and a known dot-group peer
- Hamza errors: substitution between alef variants (أ/إ/آ/ا) or hamza forms
- Taa marbuta: substitution between ة and ه
- Alef maksura: substitution between ى and ي
- Similar shapes: substitution between pairs sharing the base skeleton (ر/ز, د/ذ, etc.)
- Merged words: fewer words in OCR output than in ground truth
- Split words: more words in OCR output than in ground truth
- Insertion, deletion, other substitution: catch-all

#### 3.4  Knowledge Base Components (~4 pages)

**3.4.1  Confusion Matrix Loader**
Describe `src/data/knowledge_base.py::ConfusionMatrixLoader`: reads Phase 1
confusion matrix JSON, returns top-N confusion pairs formatted as an Arabic-
language list suitable for prompt injection.

**3.4.2  Rules Loader**
Describe `src/data/knowledge_base.py::RulesLoader`: loads Arabic spelling rules
from `data/rules/`. Explain the rules file format and the core rules used.
Show the full set of rules used in Phase 4A in a table (Table 3.2).

**3.4.3  QALB Loader and Filtering**
Describe `src/data/knowledge_base.py::QALBLoader`:
- Reads QALB-2014/2015 train/dev files
- Applies length filter (max 300 characters) — note why max_length had to be
  raised from the default 100: QALB sentences average 161–594 characters,
  so a limit of 100 filters everything
- Applies OCR-relevance filter: keeps only pairs where the error overlaps with
  known OCR error types (dot substitutions, similar-shape swaps)
- Final count: 528 OCR-relevant pairs available for selection
- Phase 4B selects 5 diverse pairs per inference call

**3.4.4  OpenITI Corpus and RAG Index**
Describe `src/data/knowledge_base.py::OpenITILoader`:
- CSV inventory of OpenITI files with local paths
- Path resolution bug fixed: `local_path` in CSV uses `../data/...` relative
  to openiti_root, which must be stripped before joining
- Sentence splitting: split on sentence-ending punctuation (period, question,
  exclamation) and Arabic-specific sentence boundaries (U+060C, U+061B, U+061F)
- Minimum sentence length filter: 20 characters
- Total indexed sentences: 200,000 from a 200k-sentence subset

Describe `src/core/rag_retriever.py`:
- Embedding model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Index type: FAISS flat L2
- Build time: saved to disk; loaded for inference
- Retrieval: top-3 sentences per query

**3.4.5  CAMeL Tools Integration**
Describe `src/linguistic/morphology.py::MorphAnalyzer`:
- Wraps CAMeL Tools MorphologicalAnalyzer for MSA
- LRU cache on `analyze()` to avoid redundant lookups
- Graceful degradation: if camel_tools is not installed, returns an empty
  analysis rather than crashing

Describe `src/linguistic/validator.py::WordValidator`:
- `validate_text(ocr_text, llm_text)`: word-by-word comparison
- `validate_correction()`: implements the revert strategy
- Returns validation statistics: total words, reverted count, revert rate

**3.4.6  LLM Insights Loader (Phase 4D)**
Describe `src/data/knowledge_base.py::LLMInsightsLoader`:
- Reads JSON files from `results/phase4d/insights/`
- Each file contains per-error-type fix_rate and introduction_rate
- Converts statistics to Arabic-language diagnostic sentences
- Aggregated by dataset type (PATS-A01, KHATT) for more robust estimates

#### 3.5  The LLM Correction Module (~2 pages)

**3.5.1  Architecture**
Describe the class hierarchy in `src/core/llm_corrector.py`:
- `BaseLLMCorrector`: abstract base class; defines `correct()` and `correct_batch()`
- `TransformersCorrector`: loads Qwen3 via HuggingFace Transformers; handles
  tokenization, generation, response extraction
- `MockCorrector`: returns the OCR text unchanged; used for local testing without GPU

**3.5.2  Generation Configuration**
Document the exact parameters used:
- `max_new_tokens`: 512
- `do_sample`: False (greedy decoding for reproducibility)
- `enable_thinking`: False (Qwen3-specific; suppresses scratchpad)
- Temperature: not applicable (greedy)

**3.5.3  Response Extraction**
Describe how the model output is parsed: the prompt instructs the model to
return only the corrected text; the response extractor strips the assistant
role marker and any preamble. Handling of model refusals and empty outputs.

#### 3.6  PromptBuilder (~2 pages)
Describe `src/core/prompt_builder.py` and all prompt variants:

| Method | Used In | Prompt Version |
|--------|---------|----------------|
| `build_zero_shot()` | Phase 2 | p2v2 |
| `build_ocr_aware()` | Phase 3 | p3v1 |
| `build_rule_augmented()` | Phase 4A | p4av1 |
| `build_few_shot()` | Phase 4B | p4bv1 |
| `build_self_reflective()` | Phase 4D | p4dv1 |
| `build_rag()` | Phase 5 | p5v1 |
| `build_combined()` | Phase 6 | p6v1 |

For each, include the full prompt template in a `\begin{verbatim}` block
(or as a figure) showing both system and user components. Explain the
conservative instruction common to all: "if the text is correct, return it
unchanged." Discuss the design decision to keep all prompts in Arabic to
avoid cross-lingual confusion.

---

### CHAPTER 4 — Experimental Methodology
**File:** `MainMatter/Chapter4.tex`
**Target length:** 22–28 pages
**Purpose:** Describe each phase as a complete, self-contained experiment with
hypothesis, method, and expected output.

#### 4.1  Experimental Design Principles (~1.5 pages)
Explain the overarching design:
- **Isolation principle**: each Phase 3–5 experiment changes exactly one variable
  relative to Phase 2. This allows attribution of observed changes to a single
  cause.
- **Phase 2 as hub**: all isolated experiments compare to Phase 2, not to Phase 1
  or to each other. Comparing to Phase 1 would confound OCR quality with LLM
  quality; the question is what the LLM adds given that it is already running.
- **Why not test all 32 combinations upfront**: at 5 knowledge sources, 2^5 = 32
  combinations; running all 32 on the full dataset is computationally expensive
  and most would not be interpretable. The hierarchical approach (isolated first,
  then best combinations in Phase 6) extracts more insight per compute budget.
- **Reproducibility**: each phase exports a JSONL file to a known path; the analyze
  stage checks for its presence and resumes rather than re-running. All random
  seeds are fixed.

#### 4.2  Phase 1 — Baseline and Error Taxonomy (~3 pages)

**Hypothesis:** Qaari produces systematic, categorizable errors on Arabic text.

**Method:**
1. Load all OCR predictions and ground-truth pairs from the Akhbar dataset
2. Compute per-sample CER and WER; aggregate statistics
3. Build character-level confusion matrix: count occurrences of (ground-truth
   char, OCR char) pairs across the corpus; normalize by ground-truth frequency
4. Classify each substitution error into one of the seven taxonomic categories
5. Apply CAMeL Tools morphological analysis to each OCR token; classify as
   non-word or valid-but-wrong

**Output files:**
- `results/phase1/{dataset}/baseline_metrics.json` — CER/WER statistics
- `results/phase1/{dataset}/confusion_matrix.json` — character-level confusion counts
- `results/phase1/{dataset}/error_taxonomy.json` — error counts by category
- `results/phase1/{dataset}/error_examples.json` — sample OCR/GT pairs per category
- `results/phase1/{dataset}/morphological_analysis.json` — word validity stats
- `results/phase1/{dataset}/report.md` — human-readable summary

**Research question answered:** RQ1

#### 4.3  Phase 2 — Zero-Shot LLM Correction (~3 pages)

**Hypothesis:** A zero-shot instruction-tuned LLM can improve Arabic OCR output
without task-specific guidance.

**Method:**
1. For each sample, construct a zero-shot prompt using `build_zero_shot()`
2. Write prompts to `inference_input.jsonl`
3. Run inference (Kaggle/GPU): for each sample, generate the corrected text
4. Write corrections to `corrections.jsonl`
5. Compute post-correction CER/WER; compare to Phase 1

**Prompt design rationale:** The system prompt establishes the model as an Arabic
text corrector. The conservative instruction "if the text looks correct, return
it unchanged" prevents the model from making gratuitous changes to accurate text.
This addresses a known failure mode of instruction-following models: they tend to
interpret every correction instruction as a mandate to rewrite, even when the
input is already correct.

**Output files:**
- `results/phase2/{dataset}/metrics.json` — post-correction CER/WER
- `results/phase2/{dataset}/comparison_vs_phase1.json` — delta vs. OCR baseline
- `results/phase2/{dataset}/error_changes.json` — which errors were fixed vs. introduced

**Role in the study:** Phase 2 output is the baseline for all subsequent phases.
Any phase that does not beat Phase 2 is a negative result.

**Research question answered:** RQ2

#### 4.4  Phase 3 — OCR-Aware Prompting (~2.5 pages)

**Hypothesis:** Telling the LLM which specific character confusions Qaari makes
will direct its attention to those pairs and improve correction.

**Method:**
1. Load Phase 1 confusion matrix
2. Extract the top-10 confusion pairs (true character → OCR character with highest
   count)
3. Format as an Arabic-language bullet list: "يخلط بين X و Y"
4. Inject this list into the system prompt using `build_ocr_aware()`
5. Run inference and analyze as Phase 2

**Variables:**
- Number of confusions to include: 10 (fixed based on prior pilot)
- Format: flat Arabic list (alternative: table with probabilities — not tested)

**Research question answered:** RQ3

#### 4.5  Phase 4A — Rule-Augmented Prompting (~2.5 pages)

**Hypothesis:** Explicit Arabic orthographic rules improve correction by giving
the LLM grounded guidance for rule-governed phenomena.

**Method:**
1. Load rules from `data/rules/`
2. Format as a numbered instruction list in Arabic
3. Inject into system prompt using `build_rule_augmented()`
4. Run inference and analyze

**Rules covered:** (list all rules from the data/rules/ directory)

**Research question answered:** RQ4

#### 4.6  Phase 4B — Few-Shot Examples (~3 pages)

**Hypothesis:** Demonstrating the correction task with examples from QALB will
help the model understand what kind of corrections are expected.

**Method:**
1. Load QALB corpus, apply OCR-relevance and length filters
2. Select 5 diverse pairs using diversity-maximizing selection (ensuring coverage
   of multiple error types rather than randomly drawn from the same type)
3. Format as Arabic input–output demonstrations in the prompt
4. Run inference and analyze

**Transfer gap discussion:** QALB examples come from human typing errors. OCR
errors and human typing errors are different distributions: humans tend to make
phonological errors (writing words as they sound), while OCR makes visual
confusion errors (confusing characters that look alike). The transfer gap between
these distributions is a source of variance.

**Research question answered:** RQ5

#### 4.7  Phase 4C — CAMeL Morphological Post-Processing (~2.5 pages)

**Hypothesis:** Morphological validation can catch LLM hallucinations — cases
where the LLM introduces morphologically invalid Arabic words — and revert them
to the (valid) OCR original.

**Method:**
1. Run Phase 2 zero-shot correction (same as Phase 2 outputs)
2. For each word in the LLM output:
   a. Check morphological validity with CAMeL Tools
   b. If the LLM word is invalid AND the corresponding OCR word is valid →
      revert to OCR word
3. Recompute CER/WER on the post-processed output

**Revert strategy rationale:** The revert strategy is deliberately conservative.
It cannot introduce new words; it can only prevent the LLM from breaking a word
that the OCR got right. This means its maximum downside is zero (it never adds
errors), but its upside is limited to cases where LLM hallucination is the
source of error.

**Note on CAMeL setup:** Requires `pip install camel-tools && camel_data -i
morphology-db-msa-r13`. The morphology database must be downloaded separately.

**Research question answered:** RQ6

#### 4.8  Phase 4D — Self-Reflective Prompting (~4 pages)

**Hypothesis:** Feeding the model statistical summaries of its own systematic
errors will cause it to apply those corrections more carefully.

**Method — Step 1 (analyze-train mode):**
1. Run Phase 2 on the training split (where ground truth is available)
2. For each training sample, compare LLM corrections to ground truth using
   `src/analysis/llm_error_analyzer.py::LLMErrorAnalyzer`
3. For each error type, compute:
   - `fix_rate`: fraction of baseline OCR errors of that type that the LLM corrected
   - `introduction_rate`: fraction of ground-truth tokens of that type that the LLM
     incorrectly modified (introduced a new error where none existed)
4. Aggregate across all training samples to get dataset-type-level statistics

**Method — Step 2 (export mode):**
5. Load the aggregated insights from `results/phase4d/insights/`
6. Convert to Arabic-language diagnostic sentences: for error types with low
   fix_rate, generate a warning; for types with high introduction_rate, generate
   a caution note
7. Inject into the system prompt using `build_self_reflective()`
8. Export validation-split inference input

**Method — Step 3 (analyze mode):**
9. Run inference on the validation split
10. Compute CER/WER; compare to Phase 2

**Insights derived from training data:**
- Insertion errors (OCR repetitions): 97.35% fix rate — the LLM is excellent at
  collapsing repetitions
- Deletion errors: 3.12% fix rate — the LLM rarely reconstructs deleted characters
- Other substitutions: 16.33% fix rate — the LLM handles explicit substitution
  errors poorly
- Error introduction rate: 25.06% overall — 1 in 4 tokens that were correct in
  the OCR is incorrectly changed by the LLM

Show the actual Arabic-language insight text generated from these statistics.

**Research question answered:** RQ7

#### 4.9  Phase 5 — Retrieval-Augmented Generation (~3 pages)

**Hypothesis:** Retrieved passages from a large Arabic corpus provide contextual
grounding that helps the LLM correct OCR text more accurately.

**Method — Build mode:**
1. Load OpenITI corpus files using `OpenITILoader`
2. Split into sentences at punctuation boundaries
3. Filter: minimum 20 characters; maximum 500 characters
4. Embed all sentences with `paraphrase-multilingual-MiniLM-L12-v2`
5. Build FAISS flat L2 index; save to disk

**Method — Export mode:**
6. For each OCR sample, embed the OCR text as a query
7. Retrieve top-3 similar sentences from the index
8. Format retrieved sentences as an Arabic context block in the prompt
9. Build prompt using `build_rag()`

**Method — Analyze mode:**
10. Compute CER/WER; compare to Phase 2
11. Compute retrieval quality metrics: average cosine similarity of retrieved
    passages to the query; percentage of retrievals above similarity threshold 0.5

**Domain mismatch consideration:** The OpenITI corpus contains classical and
historical Arabic texts. PATS-A01 Akhbar contains modern newspaper text. The
semantic and lexical overlap between these domains may be low, which could explain
why RAG underperforms.

**Research question answered:** RQ8

#### 4.10  Phase 6 — Combinations and Ablation (~3 pages)

**Purpose:** Test meaningful combinations of the best-performing knowledge
sources; then perform ablation to quantify each component's contribution to
the full system.

**Selection of combinations for testing:**
The 12 combinations tested include:
- Top pairwise combinations: confusion matrix + few-shot; rules + few-shot;
  confusion matrix + rules; few-shot + self-reflective
- Triple combination: top-performing pair + third knowledge source
- Full system: all applicable knowledge sources combined
- CAMeL post-processing applied to selected combinations (2 additional)
- Phase 4D combinations (3 additional: Phase 4D alone is already Phase 4D;
  here it is tested as an add-on to other combinations)

**Ablation design:**
Starting from the full system combination, ablate one component at a time:
- Full system CER = reference
- Remove confusion matrix → CER delta = confusion matrix contribution
- Remove rules → CER delta = rules contribution
- Remove few-shot → CER delta = few-shot contribution
- Remove self-reflective → CER delta = self-reflective contribution
- Remove RAG → CER delta = RAG contribution
- Remove CAMeL post-processing → CER delta = CAMeL contribution

**Research question answered:** RQ9

---

### CHAPTER 5 — Results and Analysis
**File:** `MainMatter/Chapter5.tex`
**Target length:** 22–28 pages
**Purpose:** Report all numerical results, answer each research question, and
provide thorough analysis of both expected and unexpected findings.

**Writing guideline:** Every result must be stated twice — once as a number in a
table and once as an interpretive sentence in the text. Tables are for comparison;
prose is for understanding. Never let a table speak for itself.

#### 5.1  Phase 1 Results — OCR Error Characterization (~5 pages)

**5.1.1  Aggregate Error Rates**
Report CER and WER for the 200-sample Akhbar-train evaluation:
- All samples (including runaway): CER 17.01%, WER 20.14%
- Normal samples only (excluding 1 runaway): CER 1.37%, WER 5.04%

Discuss the runaway artifact: Qaari occasionally enters an infinite repetition
loop on a document segment, generating thousands of copies of a short text span.
This affects 0.5% of samples (1/200) but dominates the aggregate CER because the
edit distance between a repeated phrase and the ground truth is extremely high.
This artifact is real-world behavior of the OCR engine and is included in the
evaluation to maintain ecological validity.

**5.1.2  Error Type Distribution**
Report the taxonomy breakdown:
- Insertion: 1,964 errors (95.2%) — dominated by the runaway sample
- Other substitution: 49 errors (2.38%)
- Deletion: 32 errors (1.55%)
- Dot confusion: 14 errors (0.68%)
- Alef maksura errors: 3 errors (0.15%)
- Similar shape: 1 error (0.05%)

Discuss: on normal samples, the error profile shifts. Most insertions disappear;
dot confusion and substitution errors become the dominant categories.

**5.1.3  Confusion Matrix Analysis**
Present the top-10 character confusions in a table (Table 5.1). Connect each
confusion to the script properties described in §2.1: ط/ظ confusion (similar
shape, differ only by a dot); ي/ب confusion (dot position); ى/ي confusion
(alef maqsura vs. ya); ف/ق confusion (one vs. two dots below).

**5.1.4  Morphological Validity Analysis**
Report the CAMeL Tools morphological analysis results. On normal samples, what
percentage of OCR tokens are morphologically valid (valid-but-wrong) vs.
morphologically invalid (non-words)? Interpret: if most errors are valid-but-wrong,
simple dictionary lookup will not suffice — context is required.

**Answers RQ1.**

#### 5.2  Phase 2 Results — Zero-Shot LLM Baseline (~4 pages)

**5.2.1  Aggregate Performance**
Report: CER 5.77%, WER 10.60%, CER std 0.12, CER median 3.28%.

**5.2.2  Comparison to Phase 1**
Compare to all-samples OCR (CER 17.01%): LLM achieves 66% relative CER reduction.
This is largely attributable to the repetition artifact correction.

Compare to normal-samples OCR (CER 1.37%): the LLM increases CER from 1.37% to
5.77% on normal samples. This is the over-correction problem stated numerically.
Explain: on text that Qaari rendered correctly, the LLM introduces errors at a
rate that exceeds the errors it fixes.

**5.2.3  Per-Sample Analysis**
Discuss the high standard deviation (0.12) relative to the mean (0.058). Show
the CER distribution (Fig. 5.1 — histogram of per-sample CER values). Identify
the bimodal pattern: most samples cluster near 0–0.05 CER; a tail of high-CER
samples (mostly the repetition artifact and over-correction cases) pulls the
mean upward.

**Answers RQ2.**

#### 5.3  Phases 3–5 Results — Knowledge Augmentation Evaluation (~10 pages)

Present Table 5.2 (the main comparison table from the paper, reproduced in
full here). Then dedicate one subsection to each phase.

**5.3.1  Phase 3 — Confusion Matrix Injection**
CER: 19.62%, WER: 24.35% — 240% worse than Phase 2 in relative CER.

Analyze why. The confusion list tells the model to pay attention to dot-group
confusions. On a dataset where dot confusions account for only 0.68% of all
errors, this creates a large signal-to-noise problem. The model interprets the
confusion list as evidence that the text is unreliable in all dot-group positions
and substitutes characters it would otherwise leave alone.

Note the very high CER standard deviation (2.19): the confusion injection does
not uniformly degrade all samples; it catastrophically rewrites a small fraction
while leaving most samples similar to Phase 2. This suggests the model latches
onto the confusion information only when it encounters ambiguous passages.

**Answers RQ3.**

**5.3.2  Phase 4A — Rule-Augmented Prompting**
CER: 11.03%, WER: 17.70% — 91% worse than Phase 2 in relative CER.

The rules cover phenomena (hamza al-qat', taa marbuta, alef maqsura) that
appear throughout Arabic text. Unlike the confusion matrix (which targets a
narrow error type), the rules are broadly applicable. The model applies them
even to tokens that were correct, over-normalizing forms that are acceptable
in context. For example, the rule about hamza al-wasl may cause the model to
change hamza forms on words that already use the correct form.

**Answers RQ4.**

**5.3.3  Phase 4B — Few-Shot Examples**
CER: 10.09%, WER: 15.64% — 75% worse than Phase 2. Best among augmented phases.

The few-shot examples constrain the model's behavior more precisely than rules
or confusion lists because they show specific input–output transformations
rather than abstract principles. The model has a clearer template for what
"correction" looks like. The degradation relative to Phase 2 is smaller because
the examples also implicitly convey conservatism: if the example shows minor
changes, the model is less likely to rewrite aggressively.

Discuss the QALB transfer gap: human typing errors (adding hamza where it does
not belong, using ha for taa marbuta) overlap with some OCR errors but the
distributions differ. A few-shot set drawn from genuine OCR error-correction
pairs might perform better.

**Answers RQ5.**

**5.3.4  Phase 4C — CAMeL Morphological Post-Processing**
CER: 5.77%, WER: 10.60% — identical to Phase 2.

The revert strategy triggered zero times on the 200-sample evaluation. No word
in the Phase 2 output was simultaneously morphologically invalid and sourced
from a morphologically valid OCR token. Two explanations:
1. Phase 2's LLM errors are primarily valid-but-wrong substitutions, not
   morphological invalids. The corrected words look like real Arabic words but
   are contextually wrong. CAMeL cannot catch these.
2. The OCR errors in the Akhbar dataset at 1.37% CER are sparse; most tokens
   are identical in the OCR output and LLM correction, leaving few candidates
   for the revert condition.

Discuss the implication for Phase 6: adding CAMeL post-processing to a combined
system will likely also have zero effect unless the combined system introduces
more non-word outputs.

**Answers RQ6.**

**5.3.5  Phase 4D — Self-Reflective Prompting**
CER: 10.48%, WER: 16.12% — 82% worse than Phase 2.

The training-split analysis revealed accurate statistics: the model fixes
97.35% of insertion errors but introduces new deletion errors at a significant
rate (25.06% overall error introduction rate). However, translating these
statistics into improved behavior via a natural-language instruction does not
work. The model receives the diagnostic notes but its inference behavior does
not change in the direction predicted.

Possible explanation: the diagnostic information is at odds with the conservative
zero-shot instruction. The system prompt says "be careful, you make deletions and
over-correct"; the model's instruction-following training interprets this as
additional license to intervene, rather than as a constraint to hold back.

Note the high error introduction rate (25.06%) discovered during analysis:
this means that for every 4 correct OCR tokens, the LLM changes one of them
incorrectly. This is a systematic over-intervention problem that diagnostics
alone cannot solve.

**Answers RQ7.**

**5.3.6  Phase 5 — RAG with OpenITI**
CER: 22.10%, WER: 26.93% — 283% worse than Phase 2. Worst result overall.

Retrieved passages from OpenITI are drawn from classical Arabic texts. The
Akhbar newspaper sentences use modern Arabic vocabulary and short sentence
structures. The retrieved passages have low cosine similarity to the queries
(report the average similarity score here — retrieve from retrieval_analysis.json).
The model uses these dissimilar passages as templates, importing vocabulary and
constructions from the retrieved text that do not belong in the input sentence.

The high CER standard deviation (1.99) confirms inconsistent behavior: some
samples are heavily affected by retrieval (when a retrieved passage happens to
use similar vocabulary, the model gets confused about what to correct) while
others are barely affected.

**Answers RQ8.**

#### 5.4  Phase 6 Results — Combinations and Ablation (~4 pages)
[To be written after Phase 6 is executed. Describe structure here:]

Report:
- Best pairwise combination vs. Phase 2 and best isolated phase
- Full system vs. best combination
- Ablation table: full system with each component removed one at a time
- Which component contributes most? Which are redundant?

**Answers RQ9.**

#### 5.5  Summary and Ranking (~1 page)
Reproduce the final ranking table. State which research questions received
positive answers (RQ1, RQ2 partially), which received negative answers
(RQ3–RQ8 for the isolated conditions), and what the combined system shows (RQ9).

---

### CHAPTER 6 — Discussion and Conclusion
**File:** `MainMatter/Chapter6.tex`
**Target length:** 14–18 pages
**Purpose:** Interpret the findings, situate them in the broader literature,
acknowledge limitations honestly, and chart future directions.

#### 6.1  Revisiting the Research Questions (~3 pages)
Go through RQ1–RQ9 systematically. For each:
- State the question
- State the finding (one sentence)
- Explain what it means for the broader literature
- Note any caveats

#### 6.2  The Over-Correction Problem (~3 pages)
This is the central unexpected finding. Develop the argument fully:

**What over-correction is:** The LLM changes tokens that the OCR rendered
correctly. On high-quality typewritten input, where the baseline CER is 1.37%,
any unsolicited change is likely to introduce an error. The model's 25.06%
error introduction rate means it corrupts 1 in 4 previously correct tokens.

**Why it happens:** Instruction-tuned LLMs are trained to fulfill requests.
When the request is "correct this text," the model interprets any ambiguity
as an invitation to intervene. The conservative instruction ("return unchanged
if correct") partially mitigates this but cannot fully counteract the trained
correction reflex.

**Why augmentation makes it worse:** Every additional knowledge element in the
prompt — confusion lists, rules, examples, diagnostics, retrieved passages —
signals to the model that the input is problematic and that active correction
is expected. Each element raises the model's intervention threshold, causing
more unsolicited rewrites.

**Connection to literature:** Similar effects have been observed in fact-checking
(models given more context sometimes introduce more hallucinations) and in
chain-of-thought prompting (models that reason step-by-step sometimes talk
themselves into wrong answers they would have got right directly).

#### 6.3  When Would Augmentation Help? (~2 pages)
Discuss the conditions under which the findings would likely reverse:

1. **Higher baseline CER:** If Qaari's output were 20–30% CER on all samples
   (e.g., on a difficult font or handwritten input), there would be more errors
   to fix and less risk of over-correcting correct tokens. Knowledge augmentation
   might then produce net benefit.

2. **Better retrieval quality:** If the retrieved passages were domain-matched
   (newspaper corpus for newspaper text), the RAG condition might behave
   differently. OpenITI–PATS-A01 is a worst-case domain mismatch.

3. **Few-shot with OCR-native examples:** Replacing QALB pairs with pairs
   generated from genuine OCR runs (run Qaari on clean text, pair output with
   original) would eliminate the transfer gap.

4. **Fine-tuned models:** An LLM fine-tuned specifically to be conservative
   — trained on examples where the correct action is to return the text unchanged
   — would likely benefit more from augmentation because it would not over-activate
   on the injected context.

#### 6.4  Implications for System Design (~1.5 pages)
Practical recommendations derived from the findings:
- For high-quality typewritten Arabic OCR: use zero-shot correction, do not augment
- Apply LLM correction selectively: only to samples exceeding a CER threshold
  (e.g., > 10%), where the presence of errors is more certain
- The LLM is a reliable safety net for catastrophic OCR failures (the repetition
  artifact); routing known-problematic samples to LLM correction is worthwhile
- CAMeL post-processing adds no cost and no harm; it can be included as a
  defensive layer without risk of degradation
- RAG with a domain-mismatched corpus is actively harmful; do not deploy it
  unless the retrieval corpus matches the document domain closely

#### 6.5  Limitations (~2 pages)
Be specific and honest:
1. **Single dataset**: all quantitative results are from PATS-A01 Akhbar-train,
   200 samples. Results may not generalize to other fonts, other OCR engines,
   or handwritten text.
2. **Single OCR engine**: Qaari-specific error patterns (especially the repetition
   artifact) shaped the results. A different OCR engine with different error
   characteristics might yield different Phase 3 behavior.
3. **Single LLM**: Qwen3-4B results may not generalize to larger models or to
   Arabic-specific models.
4. **No statistical significance testing on isolated phases**: Phase 6 includes
   statistical analysis; individual phase comparisons do not include paired
   t-tests due to the single-dataset scope.
5. **Phase 6 not yet completed**: The combination and ablation study is designed
   but not yet executed. The most important research question (RQ9) is not yet
   answered.
6. **QALB transfer gap**: Phase 4B uses human typing error examples; the transfer
   to OCR errors is imperfect and untested.
7. **Classical corpus for RAG**: OpenITI domain mismatch makes Phase 5 a
   worst-case RAG evaluation; results may not generalize to a well-matched corpus.

#### 6.6  Future Work (~1.5 pages)
1. Evaluate all phases on KHATT (handwritten) and all 8 PATS-A01 fonts
2. Generate OCR-native few-shot pairs for Phase 4B
3. Build a domain-matched retrieval corpus (modern Arabic newspaper corpus)
4. Fine-tune Qwen3-4B on an OCR correction task with explicit "do not change"
   training examples
5. Test with larger models (Qwen3-14B, GPT-4) to assess whether scale reduces
   over-correction
6. Study the over-correction problem as a standalone research question:
   can a calibrated refusal mechanism be trained into an LLM?
7. Extend to dialectal Arabic OCR
8. Online correction: combine OCR confidence scores with LLM correction decisions

#### 6.7  Conclusion (~1 page)
Restate the research question. Summarize the main finding: zero-shot correction
is the best LLM strategy for high-quality typewritten Arabic OCR; knowledge
augmentation causes over-correction. State the practical implication. End with
a sentence on the broader significance: negative results in prompt engineering
research are valuable because they identify failure modes that successful
deployments must guard against.

---

## 3. Front Matter

### Abstract (`FrontMatter/Abstract.tex`)
~300 words. Cover:
- Problem: Arabic OCR errors
- Gap: no systematic study of prompt augmentation for Arabic post-OCR correction
- Method: six isolated conditions + combination study; PATS-A01; Qwen3-4B
- Key finding: zero-shot CER 5.77%; all augmentations worse; RAG worst (22.10%);
  few-shot best among augmented (10.09%); CAMeL post-processing neutral
- Implication: over-correction is the dominant failure mode; simpler is better
  for high-quality OCR

### Acknowledgements (`FrontMatter/Acknowledgements.tex`)
Thank supervisor, committee, university. Mention any compute resources used
(Kaggle GPU quota, university HPC if applicable).

### List of Publications (`FrontMatter/Publications.tex`)
Include the IEEE conference paper if submitted/accepted.

### List of Symbols (`FrontMatter/Symbols.tex`)
| Symbol | Definition |
|--------|------------|
| CER | Character Error Rate |
| WER | Word Error Rate |
| LLM | Large Language Model |
| OCR | Optical Character Recognition |
| RAG | Retrieval-Augmented Generation |
| MSA | Modern Standard Arabic |
| RLHF | Reinforcement Learning from Human Feedback |
| FAISS | Facebook AI Similarity Search |
| PATS | Printed Arabic Text Set |
| KHATT | Khatt Arabic Text (handwritten dataset) |
| QALB | Qatar Arabic Language Bank |
| OpenITI | Open Islamicate Texts Initiative |

---

## 4. Back Matter — Appendices

### Appendix A — Full Prompt Templates
Include the complete text of every prompt used in Phases 2–6, in Arabic,
with English translations. This lets a reader reproduce the exact prompts
without referring to the source code.

### Appendix B — Phase 4D Insights
Include the full Arabic-language diagnostic text generated from training-split
analysis for PATS-A01. This documents exactly what was injected into the
Phase 4D system prompt.

### Appendix C — Configuration Files
Include the relevant sections of `configs/config.yaml` showing:
- All dataset keys evaluated
- Model and inference settings
- Evaluation settings (strip_diacritics, report_both)
- Phase-specific settings (QALB max_length, RAG top_k, etc.)

### Appendix D — PATS-A01 Error Examples
A table of ~20 representative OCR error examples from Phase 1: OCR text, ground
truth, error type, and (where applicable) LLM correction from Phase 2. Provides
qualitative texture to the quantitative results.

---

## 5. Figures Plan

| Figure | Content | Chapter | File path |
|--------|---------|---------|-----------|
| Fig. 1.1 | Problem overview diagram: OCR pipeline + LLM correction stage | Ch. 1 | Figures/Ch1/ |
| Fig. 2.1 | Arabic letter positional forms (isolated/initial/medial/final) | Ch. 2 | Figures/Ch2/ |
| Fig. 2.2 | Transformer architecture (self-attention) | Ch. 2 | Figures/Ch2/ |
| Fig. 2.3 | RAG pipeline diagram | Ch. 2 | Figures/Ch2/ |
| Fig. 3.1 | Three-stage pipeline architecture | Ch. 3 | Figures/Ch3/ |
| Fig. 3.2 | Dataset hierarchy diagram | Ch. 3 | Figures/Ch3/ |
| Fig. 3.3 | PromptBuilder class diagram | Ch. 3 | Figures/Ch3/ |
| Fig. 4.1 | Experimental phase flow diagram (fork from Phase 2) | Ch. 4 | Figures/Ch4/ |
| Fig. 4.2 | Phase 4D self-reflective loop diagram | Ch. 4 | Figures/Ch4/ |
| Fig. 5.1 | Histogram of per-sample CER values (Phase 1 vs Phase 2) | Ch. 5 | Figures/Ch5/ |
| Fig. 5.2 | Confusion matrix heatmap (top 15 confusions) | Ch. 5 | Figures/Ch5/ |
| Fig. 5.3 | Bar chart: CER all phases side by side | Ch. 5 | Figures/Ch5/ |
| Fig. 5.4 | Scatter: Phase 2 CER vs Phase 4B CER per sample | Ch. 5 | Figures/Ch5/ |
| Fig. 5.5 | Ablation waterfall chart (Phase 6, when available) | Ch. 5 | Figures/Ch5/ |

---

## 6. Tables Plan

| Table | Content | Chapter |
|-------|---------|---------|
| Table 3.1 | Dataset statistics | Ch. 3 |
| Table 3.2 | Arabic spelling rules used in Phase 4A | Ch. 3 |
| Table 3.3 | Prompt version registry | Ch. 3 |
| Table 4.1 | Phase summary: name, hypothesis, variable, RQ | Ch. 4 |
| Table 5.1 | Top-10 Qaari character confusions | Ch. 5 |
| Table 5.2 | Error taxonomy counts and percentages | Ch. 5 |
| Table 5.3 | Main comparison: all phases, CER, WER, delta vs Ph2 | Ch. 5 |
| Table 5.4 | Phase 4D: error-type fix rates and introduction rates | Ch. 5 |
| Table 5.5 | Phase 6: combination results (when available) | Ch. 5 |
| Table 5.6 | Phase 6: ablation table (when available) | Ch. 5 |
| Table 6.1 | Summary: RQ → finding → status | Ch. 6 |

---

## 7. Numbers to Fill In (same as paper, but more detail needed)

Before writing Chapters 4–5, extract all of the following from `results_/`:

### Phase 1 (`results_/phase1/PATS-A01-Akhbar-train/`)
- [x] CER all samples: 17.01%, WER: 20.14%
- [x] CER normal samples: 1.37%, WER: 5.04%
- [x] Error counts by type (see taxonomy summary)
- [x] Top confusion pairs (from confusion_matrix.json)
- [ ] Morphological analysis stats (% non-word vs valid-but-wrong)
- [ ] Example OCR/GT pairs per error category (for Appendix D)

### Phase 2 (`results_/phase2/PATS-A01-Akhbar-train/`)
- [x] CER: 5.77%, WER: 10.60%
- [x] CER std: 0.12, median: 3.28%, p95: 19.57%
- [ ] Per-sample CER distribution data (for histogram Fig. 5.1)
- [ ] Error changes JSON: how many errors fixed vs. introduced

### Phases 3–5 (all confirmed from comparison JSON files)
- [x] Ph3: CER 19.62%, WER 24.35%, delta CER +240%
- [x] Ph4A: CER 11.03%, WER 17.70%, delta CER +91%
- [x] Ph4B: CER 10.09%, WER 15.64%, delta CER +75%
- [x] Ph4C: CER 5.77%, WER 10.60%, delta 0%
- [x] Ph4D: CER 10.48%, WER 16.12%, delta +82%
- [x] Ph5: CER 22.10%, WER 26.93%, delta +283%
- [ ] Ph4C revert count: 0 (confirmed from validation_stats.json)
- [ ] Ph5 retrieval similarity scores (from retrieval_analysis.json)
- [ ] Ph4D insights: full Arabic text (from insights/*.json)

### Phase 6 — Not yet available
- [ ] All combination results
- [ ] Ablation results

---

## 8. `mthesis.tex` Chapter Mapping

Update `mthesis.tex` as follows:

```latex
\Chapter{Introduction}{MainMatter/Chapter1}\label{ch_intro}
\Chapter{Background and Literature Review}{MainMatter/Chapter2}\label{ch_background}
\Chapter{System Design and Datasets}{MainMatter/Chapter3}\label{ch_system}
\Chapter{Experimental Methodology}{MainMatter/Chapter4}\label{ch_methodology}
\Chapter{Results and Analysis}{MainMatter/Chapter5}\label{ch_results}
\Chapter{Discussion and Conclusion}{MainMatter/Chapter6}\label{ch_conclusion}
```

Update thesis title in `mthesis.tex`:
```latex
\ThesisTitle{Knowledge-Augmented Large Language Models for Arabic Post-OCR Error Correction}
```

Update keywords:
```latex
\ThesisKeywords{Arabic OCR; Post-OCR Correction; Large Language Models; Prompt
Engineering; Retrieval-Augmented Generation; Qwen3; PATS-A01; Error Taxonomy}
```

---

## 9. Writing Workflow (Recommended Order)

1. **Update `mthesis.tex`** — titles, keywords, personal info placeholders
2. **Chapter 3** first — the system description is factual and writes quickly;
   it also forces you to read the code and understand it deeply before writing
   the methodology
3. **Chapter 2** — background can be written while Chapter 3 is settling;
   do not over-write; 3 pages per subsection is the max
4. **Chapter 4** — one section per phase; connect each section to the
   corresponding `results_/` file
5. **Chapter 5** — write the tables first (they are already filled above),
   then write the prose around them; never write prose first and tables second
6. **Chapter 6** — write after Chapter 5 is stable; the discussion must
   reference specific numbers from Chapter 5
7. **Chapter 1** — write last except for the abstract; now that you know
   all findings, the introduction is easier to frame
8. **Abstract** — absolutely last; it is a 300-word compression of the whole thesis
9. **Front matter** (symbols, publications) — any time; these are mechanical

---

## 10. Style Notes Specific to This Thesis

- Use `\setstretch{1.5}` inside each chapter file for line spacing
  (CUFE norm is 1.5 for main text)
- Every table uses `\renewcommand{\arraystretch}{1.2}` for readability
- All chapter labels follow the template convention: `\label{ch_XXXX}`
- Figure labels: `\label{fig_XXXX}`; Table labels: `\label{table_XXXX}`
- Cite with `\cite{key}` inline; bibliography uses `biblatex` with IEEE style
- Add references to `thesis.bib` (already set up in `mthesis.tex`)
- Arabic text in figures: save as PDF/PNG with Arabic rendered externally
  (use Python + matplotlib with arabic_reshaper for charts with Arabic labels)
- The `cufethesis.cls` requires author info fields to be filled in `mthesis.tex`:
  `\EngineerName`, `\BirthDate`, `\Nationality`, `\EMail`, `\Phone`, `\Address`
