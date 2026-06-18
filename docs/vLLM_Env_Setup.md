# vLLM Environment Setup (Verified)

**Hardware**: A100 GPU  
**Driver**: 580.159.04 (CUDA 13.0)  
**Python**: 3.12 (system default on Ubuntu)

---

## Install Steps

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools
```

### 1. PyTorch (cu124 — required for CUDA 13.0 + torch 2.6)

PyTorch dropped cu121 support after 2.5.1. Use the cu124 index; CUDA 13.0 driver
is backward compatible with cu124 wheels.

```bash
pip install "torch==2.6.0" "torchvision==0.21.0" \
    --index-url https://download.pytorch.org/whl/cu124
```

Verify:
```bash
python -c "import torch; assert torch.cuda.is_available(); print('OK:', torch.version.cuda, torch.cuda.get_device_name(0))"
```

### 2. vLLM

```bash
pip install "vllm==0.8.5"
```

> vLLM 0.8.5 requires exactly `torch==2.6.0` and `torchvision==0.21.0`.
> torch 2.7 breaks the compiled C extensions (`_C.abi3.so`).

### 3. HuggingFace Stack

```bash
pip install \
    "huggingface_hub>=0.23.0" \
    "transformers>=4.44.0,<5.0.0" \
    "accelerate>=1.0.0" \
    "peft>=0.11.0" \
    "bitsandbytes>=0.45.0" \
    qwen_vl_utils
```

### 4. Project Dependencies

```bash
pip install Pillow tqdm pyyaml editdistance scipy
```

---

## Running infer_vllm.py

### Option A — enforce-eager (works immediately, slightly slower)

Triton JIT compilation fails because gcc can't find `libcuda.so` on some cloud VMs.
`--enforce-eager` disables CUDA graphs and skips Triton entirely.

```bash
python scripts/infer_vllm.py \
    --input  results/phase2/inference_input.jsonl \
    --output results/phase2/corrections.jsonl \
    --enforce-eager \
    --chunk-size 500
```

### Option B — full speed (fix libcuda stub first)

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:$LD_LIBRARY_PATH

python scripts/infer_vllm.py \
    --input  results/phase2/inference_input.jsonl \
    --output results/phase2/corrections.jsonl \
    --chunk-size 500
```

If `libcuda.so` stub is missing:
```bash
sudo ln -s /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/local/cuda/lib64/stubs/libcuda.so
```

---

## Key Notes

| Topic | Detail |
|-------|--------|
| `--chunk-size` | Controls resume granularity (flush to disk every N samples). Use 500–1000. vLLM manages memory internally regardless of chunk size — no OOM risk from large chunks. |
| Throughput | vLLM throughput scales with batch size. Small chunks (n=10) underutilize the A100. Larger chunks give significantly better samples/sec. |
| Resume | Script reads completed `sample_id`s from output JSONL on restart and skips them automatically. |
| Qwen3 thinking | `enable_thinking=False` is passed via `apply_chat_template` to suppress `<think>` scratchpad. |
