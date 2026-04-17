# Thunder Compute -- Sequential Run Quickstart (venv, no flash-attn)

Minimal steps to run the **sequential** inference scripts (`qwen_infer_seq.py`
and `qaari_infer_seq.py`) on a Thunder instance, using a Python venv and
skipping flash-attn entirely.

---

## 1. Export inference inputs locally

```bash
python pipelines/run_phase2.py --mode export
# + any other phases you want: phase3, phase4, phase6, phase7
```

Commit & push so the instance can `git pull`:

```bash
git add -A && git commit -m "export inputs" && git push
```

---

## 2. Create the Thunder instance

```bash
pip install tnr --upgrade
tnr login
tnr create --gpu a100xl --disk-size-gb 150
tnr status --wait
tnr connect 0
```

---

## 3. Clone the repo

```bash
cd /root
git clone https://github.com/YOUR_USERNAME/Arabic-Post-OCR-Correction.git
cd Arabic-Post-OCR-Correction
```

---

## 4. Create and activate a venv

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

> The venv inherits CUDA from the host. Do NOT reinstall CUDA.

---

## 5. Install minimum dependencies (no flash-attn, no vLLM, no CAMeL)

Only the packages strictly needed to run the `_seq` scripts on Thunder.
Analysis deps (camel-tools, scipy, editdistance) are installed locally, not
here.

```bash
# Torch (CUDA 12.1 wheels for A100)
pip install "torch>=2.2.0" "torchvision" \
    --index-url https://download.pytorch.org/whl/cu121

# HuggingFace stack + Qaari QLoRA deps.
# Version-pinned together because Qaari is a QLoRA adapter — mismatched
# versions of peft / transformers / bitsandbytes trigger:
#   AttributeError: 'Parameter' object has no attribute 'compress_statistics'
# Skip peft/bitsandbytes/qwen-vl-utils if you're only running qwen_infer_seq.py.
pip install -q --upgrade \
    huggingface_hub \
    "transformers>=4.44.0,<5.0.0" \
    "accelerate>=0.30.0" \
    "peft>=0.11.0" \
    qwen_vl_utils \
    bitsandbytes

# Misc utils
pip install Pillow tqdm pyyaml
```

That's it. No vLLM, no flash-attn, no camel-tools, no scipy.

---

## 6. Pre-download the models

Saves billed GPU time during inference.

```bash
python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-4B-Instruct-2507",
                  ignore_patterns=["*.msgpack","*.h5","flax_model*","tf_model*"])
snapshot_download("NAMAA-Space/Qari-OCR-0.1-VL-2B-Instruct",
                  ignore_patterns=["*.msgpack","*.h5","flax_model*","tf_model*"])
print("Models cached.")
EOF
```

---

## 7. Verify GPU + imports

```bash
nvidia-smi                                           # should show A100 80 GB
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "from src.core.prompt_builder import PromptBuilder; print('imports OK')"
```

---

## 8. Start tmux (so inference survives disconnects)

```bash
tmux new -s run
```

Inside tmux, activate the venv again (new shell):

```bash
cd /root/Arabic-Post-OCR-Correction
source .venv/bin/activate
```

---

## 9. Transfer inference inputs

Inputs are already in the repo if you committed them -- just `git pull`.
Otherwise, from your **local** machine:

```bash
scp -r results/ root@<INSTANCE_IP>:/root/Arabic-Post-OCR-Correction/results/
```

---

## 10. Run sequential inference

The `_seq` scripts use plain HF transformers, one sample at a time -- no
flash-attn, no vLLM needed.

### Smoke test (20 samples)

```bash
python thunder/qwen_infer_seq.py \
    --input  results/phase2/inference_input.jsonl \
    --output results/phase2/corrections.jsonl \
    --limit 20
```

### Full Phase 2 run

```bash
python thunder/qwen_infer_seq.py \
    --input  results/phase2/inference_input.jsonl \
    --output results/phase2/corrections.jsonl
```

### Other phases (same pattern)

```bash
python thunder/qwen_infer_seq.py \
    --input results/phase3/inference_input.jsonl \
    --output results/phase3/corrections.jsonl

python thunder/qwen_infer_seq.py \
    --input results/phase4/inference_input.jsonl \
    --output results/phase4/corrections.jsonl

for combo in conf_only self_only conf_self; do
    python thunder/qwen_infer_seq.py \
        --input  results/phase6/$combo/inference_input.jsonl \
        --output results/phase6/$combo/corrections.jsonl
done

python thunder/qwen_infer_seq.py \
    --input results/phase7/inference_input.jsonl \
    --output results/phase7/corrections.jsonl
```

### Qaari OCR (if re-running OCR on images)

```bash
python thunder/qaari_infer_seq.py \
    --image-root  data/images \
    --output-root data/ocr-results/qaari-results \
    --no-flash-attn
```

> `--no-flash-attn` tells the Qaari script to load the model without
> requesting Flash Attention 2.

---

## 11. Resume after disconnect

Detach tmux: `Ctrl+B`, then `D`.
Reconnect later:

```bash
tnr connect 0
tmux attach -t run
```

If the instance crashed, just re-run the same command -- already-processed
sample IDs are skipped automatically.

---

## 12. Transfer results back

From your **local** terminal:

```bash
scp -r root@<INSTANCE_IP>:/root/Arabic-Post-OCR-Correction/results/ ./results/
```

---

## 13. Clean up

```bash
# From local -- snapshot first if you want to restore later
tnr snapshot create 0
tnr delete 0
```

---

## 14. Analyze locally

```bash
python pipelines/run_phase2.py --mode analyze
python pipelines/run_phase3.py --mode analyze
python pipelines/run_phase4.py --mode analyze
python pipelines/run_phase5.py --mode validate
python pipelines/run_phase6.py --mode analyze --combo all
python pipelines/run_phase6.py --mode validate --combo best_camel
python pipelines/run_phase6.py --mode summarize --combo all
python pipelines/run_phase7.py --mode analyze
```
