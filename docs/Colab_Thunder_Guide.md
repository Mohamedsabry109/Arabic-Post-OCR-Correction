# Testing Thunder Scripts on Google Colab

## GPU Compatibility

| Script | Free T4 (15 GB) | Pro T4 (15 GB) | Pro+ A100 (40 GB) |
|--------|:-:|:-:|:-:|
| `qaari_infer_seq.py` | ✅ | ✅ | ✅ |
| `qwen_infer_seq.py` | ✅ | ✅ | ✅ |
| `qaari_infer.py --backend transformers` | ✅ | ✅ | ✅ |
| `qwen_infer.py --backend transformers` | ✅ | ✅ | ✅ |
| `qaari_infer.py` (vLLM, default) | ⚠️ tight | ✅ reduced flags | ✅ |
| `qwen_infer.py` (vLLM, default) | ⚠️ tight | ✅ reduced flags | ✅ |

For free T4, use `--backend transformers`. vLLM works on Pro T4 with a reduced
context window but offers no real throughput benefit at small test sizes anyway.

---

## Prerequisites — What to Upload to Google Drive First

### For Qwen scripts (testing LLM correction)

Generate the inference input locally first:
```bash
python pipelines/run_phase2.py --mode export
# or phase3, phase4, etc.
```

Then upload to Drive:
```
MyDrive/arabic-ocr/
└── phase2/
    └── inference_input.jsonl
```

### For Qaari scripts (testing OCR)

Upload a small subset of images from `data/images/`:
```
MyDrive/arabic-ocr/
└── images/
    └── pats-a01-data/
        └── A01-Akhbar/
            ├── Akhbar_1.png
            └── ...
```

Upload via: Google Drive → New → Folder upload (keep the directory structure intact).

---

## Step-by-Step Colab Setup

### Step 0 — Select GPU runtime

> Runtime → Change runtime type → GPU
> - Free tier: T4 (15 GB)
> - Pro+: A100 (40 GB) — required for vLLM default settings

### Cell 1 — Mount Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Cell 2 — Clone repo

```python
REPO_URL = "https://github.com/YOUR_USERNAME/Arabic-Post-OCR-Correction.git"
# Private repo: "https://YOUR_TOKEN@github.com/USERNAME/Arabic-Post-OCR-Correction.git"

!git clone {REPO_URL} /content/project
%cd /content/project
```

### Cell 3 — Install dependencies

```python
# Core — always needed
!pip install -q \
    torch torchvision \
    transformers accelerate huggingface_hub tokenizers \
    qwen-vl-utils \
    Pillow tqdm pyyaml editdistance

# vLLM — only if testing the vLLM path (skip on free T4)
# !pip install -q vllm

# Flash Attention 2 — optional, speeds up Qaari on A100
# !pip install -q flash-attn --no-build-isolation
```

---

## Running Each Script

Set the Drive path once, then use it in every command:

```python
DRIVE = "/content/drive/MyDrive/arabic-ocr"
```

---

### Qwen sequential (simplest — best first test)

```python
!python thunder/qwen_infer_seq.py \
    --input  {DRIVE}/phase2/inference_input.jsonl \
    --output {DRIVE}/phase2/corrections_seq.jsonl \
    --limit  20
```

The prompt preview and per-sample tqdm are printed before inference starts.
Output lands directly in Drive — survives Colab disconnects.
Re-run the same command to resume: already-done `sample_id`s are skipped.

---

### Qwen batched transformers (faster than sequential, no vLLM required)

```python
!python thunder/qwen_infer.py \
    --backend    transformers \
    --batch-size 8 \
    --input  {DRIVE}/phase2/inference_input.jsonl \
    --output {DRIVE}/phase2/corrections.jsonl \
    --limit  20
```

---

### Qwen vLLM (A100 recommended)

```python
# A100 (Pro+) — default settings
!python thunder/qwen_infer.py \
    --input  {DRIVE}/phase2/inference_input.jsonl \
    --output {DRIVE}/phase2/corrections.jsonl

# Pro T4 — reduce context window to fit in 15 GB
!python thunder/qwen_infer.py \
    --gpu-memory-util 0.85 \
    --max-model-len   8192 \
    --input  {DRIVE}/phase2/inference_input.jsonl \
    --output {DRIVE}/phase2/corrections.jsonl
```

---

### Qaari sequential

```python
!python thunder/qaari_infer_seq.py \
    --image-root  {DRIVE}/images \
    --output-root {DRIVE}/ocr-results/qaari-seq-results \
    --limit 50
```

---

### Qaari batched transformers

```python
!python thunder/qaari_infer.py \
    --backend    transformers \
    --batch-size 4 \
    --image-root  {DRIVE}/images \
    --output-root {DRIVE}/ocr-results/qaari-results \
    --limit 50
```

---

### Qaari vLLM (A100 recommended)

```python
# A100 (Pro+)
!python thunder/qaari_infer.py \
    --image-root  {DRIVE}/images \
    --output-root {DRIVE}/ocr-results/qaari-results

# Pro T4 — shrink context window
!python thunder/qaari_infer.py \
    --gpu-memory-util 0.80 \
    --image-root  {DRIVE}/images \
    --output-root {DRIVE}/ocr-results/qaari-results
```

---

## Comparing Outputs Locally

Download the output files from Drive, then run:

```bash
# Qwen: diff corrected_text between sequential and batched/vLLM runs
python - <<'EOF'
import json

def load(path):
    return {r["sample_id"]: r["corrected_text"]
            for l in open(path, encoding="utf-8")
            if (r := json.loads(l))}

a = load("results/phase2/corrections.jsonl")       # vLLM / batched
b = load("results/phase2/corrections_seq.jsonl")   # sequential

diff = [(sid, a[sid], b[sid]) for sid in a if sid in b and a[sid] != b[sid]]
print(f"{len(diff)} / {len(a)} records differ")
for sid, va, vb in diff[:5]:
    print(f"\n{sid}")
    print(f"  batch/vLLM : {va[:100]}")
    print(f"  sequential : {vb[:100]}")
EOF
```

```bash
# Qaari: diff .txt files between sequential and batched runs
diff -r data/ocr-results/qaari-results/ data/ocr-results/qaari-seq-results/
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Colab disconnects mid-run | Output is in Drive + per-record flush. Re-run the same command — already-done IDs/files are skipped automatically. |
| `CUDA out of memory` on T4 | Use `--batch-size 4` (batched path) or switch to the `_seq` script. |
| `vllm` not installed | Add `!pip install -q vllm` in the install cell and re-run. |
| Colab runtime expires (12 h free limit) | Re-run all setup cells (mount, clone, install) then re-run the inference cell. It resumes from the last written record. |
| Drive path not found | Verify the folder structure in Drive exactly matches the path in the command. Check with `!ls {DRIVE}`. |
| Model download slow on first run | Pre-download during setup: `!python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-4B-Instruct-2507')"` |
| Wrong `prompt_type` in preview | The preview reflects whatever is in `inference_input.jsonl`. Re-export from the correct phase pipeline if it looks wrong. |

---

## Recommended Test Sequence

1. **Start with `qwen_infer_seq.py --limit 20`** on free T4.
   - Validates the full pipeline end-to-end in ~5 minutes.
   - Output goes to Drive; prompt preview confirms the right prompt is used.

2. **Run `qwen_infer.py --backend transformers --limit 20`** on the same input.
   - Compare with seq output to verify batch path produces identical corrections.

3. **If on A100**, run `qwen_infer.py` (vLLM default) and compare again.
   - Minor float non-determinism is expected; corrections should be functionally identical.

4. **For Qaari**, repeat the same sequence with a 50-image subset from one font folder.
