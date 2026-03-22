# IEEE Conference Paper — Writing Guidelines
## Arabic Post-OCR Correction Using LLMs

---

## 1. Paper Identity

| Field | Value |
|-------|-------|
| **Working Title** | "Enhancing Arabic OCR Post-Correction via Knowledge-Augmented Large Language Models" |
| **Venue** | IEEE conference (e.g., ICDAR, ACL-related, EMNLP, or Arabic NLP workshop) |
| **Format** | IEEE double-column, `IEEEtran` class, `[conference]` option |
| **Target Length** | 8 pages + references (standard IEEE conference) |
| **Template** | `conference_101719.tex` (already in this folder) |

---

## 2. Core Research Narrative

### The Story Arc (in one paragraph)
Arabic OCR systems like Qaari produce systematic errors — character confusions, hamza mistakes, taa marbuta swaps — that degrade downstream NLP tasks. Closed-source VLMs handle this well, but open-source alternatives lag behind. We ask: **can a lightweight open-source LLM (Qwen3-4B) close this gap through prompt engineering alone?** We test six knowledge augmentation strategies — confusion-matrix injection, spelling rules, few-shot examples, morphological post-processing, self-reflective prompting, and RAG — then combine the best in an ablation study. Results on two datasets (PATS-A01 typewritten + KHATT handwritten) show what works, what does not, and why.

### Key Claims to Support with Numbers
1. Qaari produces significant CER/WER on both datasets (Phase 1 baseline establishes this)
2. Zero-shot LLM already improves over raw OCR (Phase 2 vs Phase 1)
3. Each knowledge type contributes differently — some help, some hurt, some are dataset-specific
4. The best combination from Phase 6 outperforms all isolated phases
5. Performance varies between typewritten (PATS-A01) and handwritten (KHATT) — this is a finding, not a bug

---

## 3. Paper Structure & Section Plan

### 3.1 Abstract (~150 words)
- Problem: Arabic OCR errors in open-source systems
- Gap: No systematic study of LLM prompt strategies for Arabic post-OCR correction
- Method: 8-phase experiment on two datasets with six augmentation strategies
- Key result: best CER/WER numbers vs baseline
- Implication: prompt engineering alone can substantially bridge the gap

### 3.2 Introduction (~0.75 col)
Cover in order:
1. Arabic OCR importance + challenges (diacritics, cursive, dot ambiguity)
2. Open-source vs closed-source VLM gap
3. Post-OCR correction as a text-to-text task for LLMs
4. Our research question: which knowledge augmentations help most?
5. Contributions (bulleted list — 3–4 items)
6. Paper outline sentence

**Contributions to state:**
- First systematic comparison of six LLM knowledge augmentation strategies for Arabic post-OCR correction
- Novel self-reflective prompting approach using the model's own error statistics
- Evaluation on two contrasting datasets: synthetic typewritten (PATS-A01) and real handwritten (KHATT)
- Ablation study revealing contribution of each component in the combined system

### 3.3 Related Work (~0.5 col)
Organize into subsections:
1. **Arabic OCR systems** — Qaari, KRAKEN, commercial alternatives; their known failure modes
2. **Post-OCR correction** — Traditional methods (language models, rule-based); neural approaches
3. **LLMs for text correction** — Prompt engineering for correction tasks; Arabic NLP with LLMs
4. **RAG for NLP** — Brief: OpenITI as knowledge source; retrieval-augmented correction
5. **Gap** — No prior work systematically compares prompt augmentation strategies for Arabic post-OCR correction with this breadth

### 3.4 Datasets (~0.4 col)
Two subsections:

**PATS-A01:**
- Typewritten Arabic, 8 font families (Akhbar, Traditional, ...)
- Synthetic/controlled
- Split: 80/20 train/validation per font (2213/553 pages, seed=42)
- 18 dataset keys total across fonts × splits

**KHATT:**
- Handwritten Arabic, real-world variation
- Train + validation splits
- More challenging; higher expected CER

**Preprocessing note:** Diacritics stripped before CER/WER evaluation (configurable; both variants reported in results)

### 3.5 Experimental Setup (~0.5 col)
- **OCR Engine**: Qaari (open-source Arabic OCR)
- **LLM**: Qwen3-4B-Instruct-2507 (hosted on Kaggle/Colab; no local GPU required)
- **Metrics**: CER (Character Error Rate), WER (Word Error Rate) — primary metric is CER
- **Evaluation**: Per-dataset aggregated; `thinking` mode disabled (`enable_thinking=False`)
- **Statistical testing**: Paired t-test + Cohen's d; Bonferroni correction for multiple comparisons
- **Inference**: Three-stage pipeline — export → infer (Kaggle) → analyze (local)

### 3.6 Methodology — The 8-Phase Experiment (~1.5 col)
This is the core technical section. Structure as:

#### Phase 1: Baseline & Error Taxonomy
- What Qaari errors look like; confusion matrix; error categories (dot confusion, hamza, taa marbuta, alef maksura, similar shapes, merges, splits)
- Morphological analysis via CAMeL Tools: non-word vs valid-but-wrong errors
- **Output**: establishes problem severity; feeds confusion matrix into Phase 3

#### Phase 2: Zero-Shot LLM (Baseline for All Comparisons)
- Simple Arabic correction prompt
- Conservative v2 prompt: only fix clear errors, return unchanged if correct
- **Role**: all subsequent phases compare to this, not to Phase 1
- Show prompt template (Arabic, abbreviated)

#### Phases 3–5 & 4A–4D: Isolated Knowledge Augmentations
Present as a table + brief per-phase description:

| Phase | Name | Knowledge Injected | Method |
|-------|------|--------------------|--------|
| 3 | OCR-Aware | Confusion matrix (top confusions) | Prompt injection |
| 4A | Rule-Augmented | Arabic orthographic rules (hamza, taa marbuta, etc.) | Prompt injection |
| 4B | Few-Shot | QALB OCR-filtered error-correction pairs | Prompt examples |
| 4C | CAMeL Validation | Morphological post-processing | Post-LLM revert strategy |
| 4D | Self-Reflective | LLM's own error statistics (train split analysis) | Prompt injection |
| 5 | RAG | OpenITI corpus retrieval (FAISS + sentence embeddings) | Retrieved context |

**Key design principle to state explicitly**: Each phase is an isolated experiment — only one variable changes vs Phase 2. This controls for interaction effects, which are studied separately in Phase 6.

**Phase 4D detail** (novel contribution — emphasize):
1. Run Phase 2 on training splits
2. Analyze LLM errors with `LLMErrorAnalyzer` → per-ErrorType fix_rate / introduction_rate
3. Aggregate by dataset type (PATS-A01, KHATT)
4. Feed weaknesses + over-correction patterns back as Arabic-language insights in system prompt

**Phase 4C detail** (hybrid neural-symbolic):
- Revert strategy: if LLM output word fails CAMeL morphological analysis but OCR word passes → revert to OCR
- Conservative by design: never introduces new words, only prevents hallucination

#### Phase 6: Combinations & Ablation
- Hierarchical: pairs → triples (if warranted) → full system
- 12 inference combinations tested (including 3 Phase 4D combos + 2 CAMeL combos)
- Ablation: start from full system, remove one component at a time
- **Outputs**: synergy detection, diminishing returns analysis, final ranking

### 3.7 Results (~1.5 col)
Structure:

#### 3.7.1 Phase 1 — OCR Baseline Error Analysis
- Table: CER/WER per dataset (PATS-A01 avg, KHATT train, KHATT val)
- Error type breakdown (% by category)
- Morphological analysis: % non-word errors vs valid-but-wrong

#### 3.7.2 Phase 2 — Zero-Shot LLM Performance
- Table: CER/WER Phase 1 vs Phase 2 per dataset
- Delta: absolute and relative improvement
- Observation: LLM already helps without any guidance

#### 3.7.3 Isolated Augmentation Results (Phases 3–5)
- Main comparison table: all phases vs Phase 2 (CER / WER / delta)
- Separate columns for PATS-A01 (avg) and KHATT (avg)
- Discussion per phase: what worked, what did not, why (connect to error taxonomy)

**Suggested table format:**
| Phase | PATS-A01 CER | Δ vs P2 | KHATT CER | Δ vs P2 |
|-------|-------------|---------|-----------|---------|
| P2 (baseline) | — | — | — | — |
| P3 (confusion matrix) | | | | |
| P4A (rules) | | | | |
| P4B (few-shot) | | | | |
| P4C (CAMeL) | | | | |
| P4D (self-reflect) | | | | |
| P5 (RAG) | | | | |

#### 3.7.4 Phase 6 — Combination & Ablation Results
- Best combination table
- Ablation table: full system → remove each component → delta
- Key finding: which component contributes most? Which are redundant?

#### 3.7.5 Statistical Significance
- Paired t-test results (p-values)
- Cohen's d effect sizes
- Note which improvements are significant after Bonferroni correction

### 3.8 Discussion (~0.75 col)
Address in order:
1. **Why does PATS-A01 behave differently from KHATT?** (typewritten vs handwritten error profiles)
2. **Which knowledge type is most transferable?** (rules generalize; few-shot may overfit to QALB error distribution)
3. **Self-reflective prompting** — novel contribution: works because it targets the model's own systematic biases
4. **RAG trade-offs** — retrieval quality matters; OpenITI domain may not match test documents
5. **CAMeL as safeguard** — does not always improve CER but prevents catastrophic hallucinations
6. **Limitations**:
   - Single LLM tested (Qwen3-4B-Instruct-2507)
   - One OCR engine (Qaari)
   - Inference cost scales linearly (not deployable at scale without optimization)
   - QALB examples are human typing errors, not true OCR errors (transfer gap)

### 3.9 Conclusion (~0.3 col)
- Restate research question
- Best-performing configuration and numbers
- Answer: yes/no/partially to "can LLMs bridge the gap?"
- One-sentence future work: fine-tuning, multi-engine, real-time correction

---

## 4. Figures & Tables Plan

| Item | Content | Section |
|------|---------|---------|
| **Fig 1** | System architecture / pipeline diagram (export → infer → analyze) | §3.5 or §3.6 |
| **Fig 2** | Confusion matrix heatmap (top character confusions from Phase 1) | §3.7.1 |
| **Fig 3** | CER comparison bar chart — all phases on PATS-A01 and KHATT | §3.7.3 |
| **Fig 4** | Ablation waterfall chart — contribution per component | §3.7.4 |
| **Table I** | Dataset statistics (size, split, characteristics) | §3.4 |
| **Table II** | Phase 2 vs Phase 1 across datasets | §3.7.2 |
| **Table III** | All isolated phases vs Phase 2 (main comparison) | §3.7.3 |
| **Table IV** | Phase 6 ablation results | §3.7.4 |

**Figure guidance:**
- All figures must be in PDF or high-res PNG (≥300 DPI)
- Use `\includegraphics[width=\columnwidth]{fig.png}`
- Caption below figure, title above table
- Arabic text in figures: use `arabtex` or include as rasterized images

---

## 5. Numbers to Fill In (from results)

Before writing any section, extract these numbers from the results files:

### From Phase 1 (`results_/phase1/`)
- [ ] Baseline CER per dataset (PATS-A01 per font + KHATT train/val)
- [ ] Average CER: PATS-A01 (across fonts), KHATT
- [ ] Top 5 character confusions from confusion matrix
- [ ] Error type distribution (% per category)
- [ ] % morphologically invalid words (non-word errors)

### From Phase 2 (`results_/phase2/`)
- [ ] Post-correction CER/WER per dataset
- [ ] Average improvement: PATS-A01, KHATT
- [ ] Absolute delta CER (Phase 1 → Phase 2)

### From Phases 3–5 (`results_/phase3/`, `results_/phase4a/`, etc.)
- [ ] CER/WER and delta vs Phase 2 for each phase, per dataset
- [ ] Whether improvement is statistically significant (p-value, Cohen's d)

### From Phase 6 (`results_/phase6/` or `results/phase6/`)
- [ ] Best combination CER vs all prior phases
- [ ] Ablation delta per component
- [ ] Final ranking of all approaches

**Note**: Both `corrected` and `corrected_no_diacritics` CER/WER variants are in metrics.json files. **Use `corrected_no_diacritics` as the primary reported metric** (consistent with evaluation config `strip_diacritics: true`).

---

## 6. Writing Style & IEEE Conventions

### Language
- Past tense for experiments: "We evaluated...", "Results showed..."
- Present tense for claims: "Table III shows...", "Phase 4D outperforms..."
- Avoid "we believe"; state findings directly
- No contractions

### IEEE-Specific Rules
- Figures: `\begin{figure}[t]` (top of column preferred)
- Tables: use `\begin{table}` with `\caption{}` above the table
- Equations: use `\begin{equation}` environment; number all equations
- Citations: `\cite{key}` inline; use `IEEEtran.bst` bibliography style
- Section numbering: Roman numerals (I, II, III...) — `IEEEtran` does this automatically
- No widow/orphan lines (LaTeX handles most; check at final stage)
- Author affiliations: use `\IEEEauthorblockN` + `\IEEEauthorblockA`

### CER/WER Formula (for the paper)
$$\text{CER} = \frac{S + D + I}{N}$$
where S = substitutions, D = deletions, I = insertions, N = reference character count.

### Arabic Text in LaTeX
- Use `\usepackage{arabtex}` or include Arabic text as images to avoid encoding issues
- Alternative: use `polyglossia` + `fontspec` with XeLaTeX — more reliable for Arabic
- If switching to XeLaTeX, change `pdflatex` to `xelatex` in compilation

---

## 7. Bibliography Notes

Key references to find/add:

| Topic | What to cite |
|-------|-------------|
| Qaari OCR | Qaari paper / repo / GitHub |
| PATS-A01 dataset | Original PATS-A01 dataset paper |
| KHATT dataset | KHATT paper (Mahmoud et al.) |
| QALB corpus | Mohit et al. QALB dataset paper |
| OpenITI | OpenITI project / Romanov et al. |
| CAMeL Tools | Obeid et al. CAMeL Tools paper |
| Qwen3 model | Qwen technical report |
| Arabic post-OCR correction (prior) | Relevant ACL/ICDAR papers |
| CER/WER metrics | Standard reference (Klakow & Peters 2002 or similar) |
| RAG | Lewis et al. 2020 RAG paper |
| Prompt engineering | Wei et al. CoT; Brown et al. GPT-3 for few-shot |

---

## 8. Writing Workflow

1. **Read results first** — extract all numbers into a scratch table (do not write without numbers)
2. **Write in section order**: Abstract last; start with §3.4 (Datasets) and §3.5 (Setup) — these are factual and easy
3. **Tables before prose** — fill the comparison tables first, then write the narrative around them
4. **Figures** — generate after results are final; placeholder `\includegraphics` during drafting
5. **Related Work** — write after methodology is stable (so you know what to contrast with)
6. **Abstract** — write last, after all sections are complete
7. **Proofread pass** — check all numbers match results files; check all citations exist in `.bib`

---

## 9. Open Questions to Resolve Before Writing

- [ ] **Author list & affiliations** — who are the co-authors? (supervisor, university dept.)
- [ ] **Target venue** — which specific IEEE conference? (affects page limit, deadline, focus)
- [ ] **Funding** — any grant to acknowledge in `\thanks{}`?
- [ ] **Phase 6 results available?** — if Phase 6 is not run, paper can still cover Phases 1–5 (mention Phase 6 as future work)
- [ ] **Prompt versions** — confirm which prompt version (v1 aggressive vs v2 conservative) was used for final reported results; be consistent across all phases
- [ ] **Statistical tests** — confirm scipy available in analysis; Bonferroni correction applied to all 6 isolated comparisons

---

## 10. Quick Reference: Key File Locations

| Content | Path |
|---------|------|
| Phase results | `results_/phase{N}/` |
| Architecture detail | `docs/Architecture.md` |
| Phase design docs | `docs/Phase{N}_Design.md` |
| Configs | `configs/config.yaml` |
| LaTeX template | `publication/ieee-conference/IEEE_Conference_Paper/conference_101719.tex` |
| Figures output | `publication/ieee-conference/IEEE_Conference_Paper/` |
| This file | `publication/ieee-conference/IEEE_Conference_Paper/PAPER_GUIDELINES.md` |
