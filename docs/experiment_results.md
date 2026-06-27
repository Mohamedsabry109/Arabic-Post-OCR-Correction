# Experiment Results — Arabic Post-OCR Correction

**Metric convention throughout**: normal samples only (OCR/GT length ratio ≤ 5.0), diacritics stripped before CER/WER.  
**Runaway corrector**: applied as post-processing where noted; ratio threshold = 3.0.  
**Model**: Qwen3-4B-Instruct-2507 (primary); additional models in Experiment 3.

---

## Experiment 1 — Phased Knowledge Augmentation (PATS-A01 + KHATT, 8 Phases)

First systematic experiment. One model (Qwen3-4B), two datasets (PATS-A01 typewritten + KHATT handwritten), 8 experimental phases each isolating one variable.

### Phase 1 — OCR Baseline (Qaari)

| Font | CER | WER | N (normal) | Runaway % |
|------|-----|-----|-----------|-----------|
| Akhbar | 2.12% | 4.84% | 437 | 4.8% |
| Simplified | 2.94% | 5.81% | 432 | 5.7% |
| Traditional | 3.63% | 9.35% | 435 | 5.0% |
| Arial | 3.65% | 6.84% | 436 | 4.6% |
| Naskh | 4.10% | 9.56% | 443 | 3.5% |
| Tahoma | 5.82% | 9.07% | 407 | 10.3% |
| Thuluth | 10.21% | 31.11% | 441 | 3.9% |
| Andalus | 12.12% | 33.67% | 427 | 6.2% |
| **PATS-A01 avg** | **5.57%** | **13.78%** | 3458 total | ~5.5% avg |
| **KHATT** | **34.24%** | **75.60%** | 186 | 17.7% |

Source: `results_first_experiment_pats_khatt_only_8_phases_complete/phase1/`

### Phases 2–9 — LLM Correction Strategies

| Phase | Strategy | PATS CER | PATS WER | KHATT CER | KHATT WER | Δ PATS CER | Δ KHATT CER |
|-------|----------|----------|----------|-----------|-----------|------------|-------------|
| P1 | OCR Baseline | 5.57% | 13.78% | 34.24% | 75.60% | — | — |
| P2 | Zero-Shot | 5.84% | 14.42% | 33.25% | 73.94% | +0.27% | −0.99% |
| P3 | OCR-Aware (confusion matrix) | 5.74% | 14.16% | 33.26% | 74.12% | +0.17% | −0.98% |
| P4 | Self-Reflective (training artifacts) | 5.59% | 13.66% | 33.24% | 72.67% | +0.02% | −1.00% |
| P5 | CAMeL Validation† | 21.89% | 90.74% | 41.62% | 94.10% | +16.32% | +7.38% |
| P6 | Combination (conf_self) | 5.76% | 13.81% | 33.27% | 72.98% | +0.19% | −0.97% |
| P8 | RAG BM25 (char n-gram) | **5.45%** | 12.04% | 33.83% | 73.76% | **−0.12%** | −0.41% |
| P9 | Error-Signature RAG | 5.47% | 13.63% | **33.13%** | 73.06% | −0.10% | **−1.11%** |

†Phase 5 had implementation bugs (wrong MorphAnalyzer init + position-keyed revert). Results are not reliable; excluded from conclusions.

Source: `results_first_experiment_pats_khatt_only_8_phases_complete/phase{2,3,4,5,6,8,9}/`

### Per-Font Phase 2–9 CER

| Phase | Akhbar | Andalus | Arial | Naskh | Simplified | Tahoma | Thuluth | Traditional | **Avg** |
|-------|--------|---------|-------|-------|-----------|--------|---------|------------|---------|
| OCR | 2.12% | 12.12% | 3.65% | 4.10% | 2.94% | 5.82% | 10.21% | 3.63% | **5.57%** |
| P2 | 2.71% | 12.27% | 4.50% | 4.27% | 2.93% | 6.13% | 10.92% | 2.96% | 5.84% |
| P3 | 2.57% | 11.75% | 4.48% | 4.23% | 2.97% | 6.12% | 10.94% | 2.90% | 5.74% |
| P4 | 2.59% | 11.51% | 4.08% | 4.22% | 2.86% | 5.91% | 10.67% | 2.87% | 5.59% |
| P6 | 2.46% | 13.22% | 4.09% | 4.13% | 2.72% | 6.15% | 10.56% | 2.75% | 5.76% |
| P8 | 3.01% | 10.41% | 3.83% | 4.36% | 3.01% | 6.21% | 9.68% | 3.07% | **5.45%** |
| P9 | 2.27% | 11.98% | 3.76% | 3.94% | 2.56% | 5.76% | 10.61% | 2.88% | 5.47% |

### Key Findings — Experiment 1

1. LLM correction slightly **hurts** PATS on average (all phases: +0.02% to +0.27% CER). PATS is already so accurate that false positives dominate.
2. Only P8 (RAG) and P9 (Error-Sig RAG) achieve below-baseline PATS (−0.12%, −0.10%).
3. LLM correction modestly **helps** KHATT: ~1 pp CER reduction across P2–P6/P9. Not enough to be clinically significant on its own.
4. P9 best on KHATT (33.13%); P8 best on PATS (5.45%).
5. **Over-correction threshold**: fonts with CER < ~3% are consistently harmed; fonts with CER > ~6% consistently improve.

---

## Experiment 2 — Prompt Design Study (8 Trials, 13 Datasets)

Systematic prompt engineering study. One model (Qwen3-4B), 13 validation datasets (PATS-A01 8 fonts + KHATT + KHATT-Paragraph + Yarmouk + Muharaf + Historical), 8 prompt configurations.

### Trial Definitions

| # | ID | Prompt Style | Retrieval |
|---|-----|-------------|-----------|
| T1 | base_zs | Base (aggressive, "correct this text") | None |
| T2 | base_rag | Base | BM25 RAG (top-5 examples) |
| T3 | cons_zs | Conservative ("return as-is if correct, only fix clear errors") | None |
| T4 | cons_rag | Conservative | BM25 RAG |
| T5 | hp_zs | Error-pattern augmented (confusion matrix hints) | None |
| T6 | hp_rag | Error-pattern augmented | BM25 RAG |
| T7 | hp_cons_zs | Error-pattern + Conservative | None |
| T8 | hp_cons_rag | Error-pattern + Conservative | BM25 RAG |

### PATS-A01 + KHATT Results

| Trial | PATS CER | PATS WER | KHATT CER | KHATT WER |
|-------|----------|----------|-----------|-----------|
| OCR baseline | 5.57% | 13.78% | 34.24% | 75.60% |
| T1 base_zs | 5.84% | 14.43% | 33.25% | 73.93% |
| T2 base_rag | 5.44% | 12.03% | 33.82% | 73.76% |
| **T3 cons_zs** | **5.27%** | **13.35%** | 32.93% | 74.15% |
| T4 cons_rag | 27.02% | 28.10% | 32.69% | 73.29% |
| T5 hp_zs | 5.85% | 13.95% | 32.90% | 72.78% |
| T6 hp_rag | 19.64% | 24.52% | **32.64%** | **73.40%** |
| T7 hp_cons_zs | ⚠ 190.60% | ⚠ 198.12% | ⚠ 51.64% | ⚠ 90.58% |
| T8 hp_cons_rag | 46.56% | 48.79% | **32.54%** | 73.67% |

⚠ T7: catastrophic failure — prompt-template tags leaked into LLM output.

### Per-Font PATS CER (selected trials)

| Trial | Akhbar | Andalus | Arial | Naskh | Simplified | Tahoma | Thuluth | Traditional | **Avg** |
|-------|--------|---------|-------|-------|-----------|--------|---------|------------|---------|
| OCR | 2.12% | 12.12% | 3.65% | 4.10% | 2.94% | 5.82% | 10.21% | 3.63% | **5.57%** |
| T1 | 2.71% | 12.26% | 4.50% | 4.26% | 2.94% | 6.14% | 10.93% | 2.97% | 5.84% |
| T2 | 2.97% | 10.42% | 3.83% | 4.36% | 3.01% | 6.20% | 9.68% | 3.07% | 5.44% |
| **T3** | **2.32%** | 11.68% | **3.74%** | **3.69%** | **2.43%** | **5.54%** | 10.23% | **2.58%** | **5.27%** |
| T5 | 2.83% | 11.87% | 4.24% | 4.21% | 3.01% | 6.12% | 11.26% | 3.25% | 5.85% |

Source: `results/experiment2/trial{1..8}_*/`

### Key Findings — Experiment 2

1. **T3 (conservative zero-shot)** is best for PATS: 5.27% avg CER (−5.4% relative vs OCR baseline).
2. **T2 (base_rag)** is best overall: balances PATS (5.44%) and KHATT (33.82%).
3. Error-pattern prompts (T5–T8) are risky: T4/T6/T8 cause catastrophic per-font failures; T7 completely fails.
4. **Decision for Experiment 3**: use T2 (base_rag) as the single best overall prompt.
5. New full-page datasets (Yarmouk, Muharaf, Historical) show OCR CER >100% — dominated by Qaari runaway bug on these domains; not correctable with current approach.

---

## Experiment 3 — Final Experiment (9 Runs, 3 Categories)

Best prompt (T2 = base_rag), multiple models, multiple OCR sources, multiple domain benchmarks.

All CER values: normal samples only, diacritics stripped.  
CER = raw corrected output. CER† = after runaway correction (threshold 3.0×GT length).

### Category A — Model Comparison

**Setting**: validation set, Qaari OCR, T2 prompt  
**Datasets**: PATS-A01 (8 fonts) + KHATT + Full-page (KHATT-Para + Yarmouk + Muharaf + Historical)

| Model | Params | PATS CER | PATS CER† | KHATT CER | KHATT CER† | Full-page CER† |
|-------|--------|----------|-----------|-----------|------------|----------------|
| OCR Baseline | — | 5.57% | — | 34.24% | — | 86.50% |
| Qwen3-4B | 4B | 5.44% | **5.44%** | **33.82%** | **33.82%** | **68.18%** |
| Qwen3-14B | 14B | 15.30% | 5.60% | 175.69% | 35.55% | 88.51% |
| Gemma-3-4B | 4B | 7.47% | 7.47% | 36.14% | 36.14% | 71.12% |
| Gemma-3-12B | 12B | 5.33% | **5.33%** | 35.98% | 35.98% | 70.01% |

#### Per-font PATS CER† (runaway-corrected)

| Font | OCR | Qwen3-4B | Qwen3-14B | Gemma-3-4B | Gemma-3-12B |
|------|-----|----------|-----------|------------|-------------|
| Akhbar | 2.12% | 2.97% | 3.27% | 4.80% | 3.05% |
| Andalus | 12.12% | 10.42% | 9.80% | 13.26% | 10.00% |
| Arial | 3.65% | 3.83% | 4.50% | 5.90% | 3.67% |
| Naskh | 4.10% | 4.36% | 4.75% | 5.94% | 4.41% |
| Simplified | 2.94% | 3.01% | 3.39% | 4.93% | 3.26% |
| Tahoma | 5.82% | 6.20% | 6.10% | 7.64% | 6.03% |
| Thuluth | 10.21% | 9.68% | 9.06% | 11.81% | 8.86% |
| Traditional | 3.63% | 3.07% | 3.96% | 5.48% | 3.35% |
| **Avg** | **5.57%** | **5.44%** | **5.60%** | **7.47%** | **5.33%** |

#### Key findings — 3A

- **Runaway corrector is critical for Qwen3-14B**: without it, 175.69% KHATT (3A raw); with it, 35.55% — still worse than Qwen3-4B (33.82%).
- **Gemma-3-12B best on PATS** (5.33%); **Qwen3-4B best on KHATT + Full-page**.
- Larger model ≠ always better: Qwen3-14B underperforms Qwen3-4B after runaway correction.
- Gemma-3-4B consistently underperforms Gemma-3-12B (both PATS and KHATT).
- Full-page datasets show significant gains with runaway correction: OCR 86.50% → Qwen3-4B 68.18% (−21% abs).

---

### Category B — OCR Source Quality Impact

**Setting**: Qwen3-4B, validation set, T2 prompt  
**Scope**: Gemma-3 VLM OCR is **excluded for PATS and KHATT** — line-strip images (10–16:1 aspect ratio) cause the Gemma preprocessor to squash images to near-square, producing hallucinations and repetition loops. Results are invalid and not reported. Gemma results are reported only for full-page datasets (~0.6:1 aspect ratio) where the comparison is valid.

#### PATS-A01 — Qaari OCR only (Gemma excluded)

| Font | Qaari OCR | Qaari Corr† |
|------|-----------|-------------|
| Akhbar | 2.12% | 2.97% |
| Andalus | 12.12% | 10.42% |
| Arial | 3.65% | 3.83% |
| Naskh | 4.10% | 4.36% |
| Simplified | 2.94% | 3.01% |
| Tahoma | 5.82% | 6.20% |
| Thuluth | 10.21% | 9.68% |
| Traditional | 3.63% | 3.07% |
| **PATS Avg** | **5.57%** | **5.44%** |

#### KHATT — Qaari OCR only (Gemma excluded)

| | Qaari OCR | Qaari Corr† |
|-|-----------|-------------|
| KHATT | 34.24% | 33.82% |

#### Full-page datasets — Qaari vs Gemma (valid comparison)

| Dataset | Qaari OCR | Qaari Corr† | Gemma OCR | Gemma Corr† |
|---------|-----------|-------------|-----------|-------------|
| KHATT-Para-val | 61.68% | 45.06% | 36.14% | 35.82% |
| Yarmouk-testing | 49.85% | 43.35% | 95.76% | 74.37% |
| Muharaf-val | 129.39% | 89.78% | 141.15% | 72.85% |
| Historical | 105.10% | 94.53% | 76.26% | 74.09% |
| **Full-page Avg** | **86.50%** | **68.18%** | **87.33%** | **64.28%** |

#### Key findings — 3B

- **Gemma excluded for PATS/KHATT** — line strips with 10–16:1 aspect ratio exceed Gemma-3 VLM preprocessor design scope. Numbers would be 170%+ CER (invalid), not a fair comparison.
- **Full-page comparison is valid**: Gemma OCR 87.33% ≈ Qaari 86.50%; after LLM correction, Gemma 64.28% vs Qaari 68.18% — Gemma-sourced corrected output is marginally better.
- **Implication**: OCR source quality does not matter for full-page documents when LLM correction is applied. For line-strip images, Qaari is the only reliable open-source OCR option.
- VLM-based OCR (Gemma-3) needs aspect-ratio-aware image preprocessing to handle line strips.

---

### Category C — Generalization to New Domains

**Setting**: Qwen3-4B, T2 prompt, RDI-Test Benchmark + Kitab Benchmark

#### RDI-Test-Lines Benchmark

> **Note on BM dataset structure**: The RDI-Test benchmark has two task types:  
> - **Line Recognition** (this experiment): line-strip images with text GT → CER/WER computable. ✅  
> - **Line Segmentation**: full-page images with polygon-coordinate GT (no text transcription) → cannot compute CER/WER. ❌ Excluded.  
> **All BM results below are Line Recognition only.**

**Interesting finding**: Gemma OCR outperforms Qaari on RDI-Test-Lines — likely because Qaari was fine-tuned on different image types and struggles with diverse manuscript/handwritten document styles, while Gemma's general VLM capabilities give it more robustness.

| Subset | OCR Source | N (normal) | OCR CER | Corr CER† | Δ CER |
|--------|-----------|-----------|---------|-----------|-------|
| LR-Handwritten | Qaari | 385 | 98.82% | 91.55% | −7.3% |
| LR-Handwritten | Gemma | 762 | **62.53%** | **57.71%** | −4.8% |
| LR-Manuscripts | Qaari | 1009 | 81.94% | 79.00% | −2.9% |
| LR-Manuscripts | Gemma | 1175 | **78.10%** | **70.36%** | −7.7% |
| LR-Typewritten | Qaari | 671 | 105.19% | 84.79% | −20.4% |
| LR-Typewritten | Gemma | 1314 | **64.86%** | **55.31%** | −9.5% |
| **RDI-Test-Lines Overall** | **Qaari** | **2065** | **95.31%** | **85.11%** | **−10.7%** |
| **RDI-Test-Lines Overall** | **Gemma** | **3251** | **68.49%** | **61.13%** | **−10.7%** |

#### Kitab Benchmark

| Subset | OCR Source | N (normal) | OCR CER | Corr CER† | Δ CER |
|--------|-----------|-----------|---------|-----------|-------|
| kitab-adab | Qaari | 153 | 44.08% | 47.32% | +3.2% |
| kitab-adab | Gemma | 186 | 57.09% | 60.46% | +3.4% |
| kitab-arabicocr | Qaari | 50 | 1.80% | 7.21% | +5.4% |
| kitab-arabicocr | Gemma | 50 | 27.43% | 21.75% | −5.7% |
| kitab-evarest | Qaari | 580 | 32.92% | 60.67% | +27.8% |
| kitab-evarest | Gemma | 649 | 37.27% | 62.14% | +24.9% |
| kitab-hindawi | Qaari | 199 | 31.46% | 24.31% | **−7.2%** |
| kitab-hindawi | Gemma | 196 | 28.97% | 26.26% | −2.7% |
| kitab-historyar | Qaari | 78 | 63.48% | 51.69% | −11.8% |
| kitab-historyar | Gemma | 152 | 79.01% | 61.73% | −17.3% |
| kitab-isippt | Qaari | 225 | 26.52% | 29.34% | +2.8% |
| kitab-isippt | Gemma | 374 | 87.48% | 68.75% | −18.7% |
| kitab-khatt | Qaari | 148 | 39.71% | 40.68% | +1.0% |
| kitab-khatt | Gemma | 177 | 82.21% | 74.43% | −7.8% |
| kitab-khattparagraph | Qaari | 27 | 205.27% | 90.60% | **−114.7%** |
| kitab-khattparagraph | Gemma | 92 | 318.33% | 131.65% | −186.7% |
| kitab-muharaf | Qaari | 93 | 76.04% | 58.72% | −17.3% |
| kitab-muharaf | Gemma | 171 | 88.47% | 71.03% | −17.4% |
| kitab-onlinekhatt | Qaari | 175 | 36.78% | 37.26% | +0.5% |
| kitab-onlinekhatt | Gemma | 196 | 54.35% | 54.01% | −0.3% |
| kitab-patsocr | Qaari | 466 | 4.79% | 5.98% | +1.2% |
| kitab-patsocr | Gemma | 372 | 87.98% | 69.95% | −18.0% |
| kitab-synthesizear | Qaari | 374 | 21.17% | 22.42% | +1.3% |
| kitab-synthesizear | Gemma | 498 | 35.52% | 37.07% | +1.5% |
| **Kitab Overall** | **Qaari** | **—** | **49.28%** | **40.73%** | **−17.3%** |
| **Kitab Overall** | **Gemma** | **—** | **79.45%** | **60.07%** | **−24.4%** |

#### Key findings — 3C

- LLM correction **generalizes** to unseen benchmarks: RDI-Test-Lines −10.7%, Kitab −17 to −24% relative.
- **BM surprising finding**: Gemma OCR (68.49%) is significantly better than Qaari (95.31%) on RDI-Test-Lines — Qaari struggles with diverse document styles where Gemma's general VLM capabilities are more robust.
- Kitab datasets with very high OCR CER (khattparagraph: 205%/318%) benefit most from runaway correction.
- Low-CER subsets (kitab-arabicocr: 1.80%, kitab-patsocr: 4.79%) are harmed — over-correction threshold confirmed in new domains.
- Gemma OCR on Kitab is generally worse than Qaari but LLM corrects more aggressively (−24% vs −17%).

---

## Summary — Best Configurations Per Task

| Task | Best Config | CER | vs OCR Baseline |
|------|------------|-----|----------------|
| PATS typewritten correction | Gemma-3-12B, T2, Qaari OCR | 5.33% | −4.3% relative |
| KHATT handwritten correction | Qwen3-4B, T2, Qaari OCR | 33.82% | −1.2% relative |
| Full-page multi-domain | Qwen3-4B, T2, Qaari OCR | 68.18% | −21.2% absolute |
| RDI-Test Benchmark | Qwen3-4B, T2, Qaari OCR | 85.11% | −10.7% absolute |
| Kitab Benchmark | Qwen3-4B, T2, Qaari OCR | 40.73% | −17.3% relative |
| Prompt-only (no RAG) | T3 cons_zs | 5.27% PATS | −5.4% relative |

## Data Sources

| Experiment | Directory |
|-----------|-----------|
| Exp1 Phase 1 baseline | `results_first_experiment_pats_khatt_only_8_phases_complete/phase1/` |
| Exp1 Phases 2–9 | `results_first_experiment_pats_khatt_only_8_phases_complete/phase{N}/` |
| Exp2 (8 trials) | `results/experiment2/trial{N}_*/` |
| Exp3 corrections | `results/experiment3/corrections/` |
| Exp3 analysis script | `scripts/analyze_experiment3_full.py` |
