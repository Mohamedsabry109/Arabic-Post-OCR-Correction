#!/usr/bin/env bash
# =============================================================================
# Thunder Compute (A100 80 GB) — One-time setup script
# Run ONCE on the instance before starting inference.
# Safe to re-run: pip and snapshot_download are idempotent.
# =============================================================================
set -euo pipefail

echo "============================================================"
echo "  Thunder Compute Setup — Arabic OCR Correction"
echo "============================================================"

# --------------------------------------------------------------------------
# 1. System packages
# --------------------------------------------------------------------------
echo ""
echo "[1/6] Installing system packages..."
apt-get update -qq && apt-get install -y -qq git wget curl build-essential

# --------------------------------------------------------------------------
# 2. Python dependencies
# --------------------------------------------------------------------------
echo ""
echo "[2/6] Installing Python packages..."

pip install --upgrade pip --quiet

# --- Core ML stack (CUDA 12.1 wheel for A100 sm_80) ---
pip install --quiet \
    "torch>=2.2.0" \
    "torchvision" \
    --index-url https://download.pytorch.org/whl/cu121

# --- vLLM (primary high-throughput engine — must come after torch) ---
pip install --quiet "vllm>=0.6.3"

# --- HuggingFace stack ---
pip install --quiet \
    "transformers>=4.45.0" \
    "accelerate>=0.30.0" \
    "huggingface_hub>=0.23.0" \
    "tokenizers>=0.19.0" \
    "datasets>=2.18.0"

# --- Qwen2VL image utilities (required for Qaari batching) ---
pip install --quiet "qwen-vl-utils>=0.0.8"

# --- Flash Attention 2 (significant speedup for long VLM sequences on A100) ---
pip install --quiet flash-attn --no-build-isolation || \
    echo "  [warn] flash-attn unavailable — continuing without it (slower for long sequences)"

# --- CAMeL Tools (Arabic NLP — Phases 1, 5, 6) ---
pip install --quiet "camel-tools>=1.5.0" || \
    echo "  [warn] camel-tools install failed — Phases 1/5/6 morphological validation will be skipped"

# --- DSPy (Phase 7 automated prompt optimization) ---
pip install --quiet "dspy-ai>=2.4.0" || \
    echo "  [warn] dspy-ai install failed — Phase 7 optimization will be unavailable"

# --- Scientific computing (metrics, statistics) ---
pip install --quiet \
    "scipy>=1.11.0" \
    "numpy>=1.26.0"

# --- Misc utilities ---
pip install --quiet \
    "Pillow>=10.0.0" \
    "tqdm>=4.66.0" \
    "pyyaml>=6.0" \
    "editdistance" \
    "regex"

echo "  [ok] All packages installed."

# --------------------------------------------------------------------------
# 3. CAMeL Tools data download (morphological analyser DB for Phases 1/5/6)
# --------------------------------------------------------------------------
echo ""
echo "[3/6] Downloading CAMeL Tools morphology database..."
python - <<'PYEOF'
try:
    import camel_tools
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "-m", "camel_tools.cli.camel_data", "-i", "morphology-db-msa-r13"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print("  [ok] CAMeL morphology DB installed.")
    else:
        print(f"  [warn] camel_data failed: {result.stderr.strip()}")
except ImportError:
    print("  [skip] camel-tools not installed — skipping morphology DB download.")
PYEOF

# --------------------------------------------------------------------------
# 4. Pre-download models to HuggingFace cache
# --------------------------------------------------------------------------
# Doing this during setup avoids wasting billed inference time on downloads.

echo ""
echo "[4/6] Pre-downloading Qaari OCR model (~4 GB)..."
python - <<'PYEOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="NAMAA-Space/Qari-OCR-0.1-VL-2B-Instruct",
    ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
)
print("  [ok] Qaari model cached.")
PYEOF

echo ""
echo "[5/6] Pre-downloading Qwen3-4B correction model (~8 GB)..."
python - <<'PYEOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Qwen/Qwen3-4B-Instruct-2507",
    ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
)
print("  [ok] Qwen3-4B model cached.")
PYEOF

# --------------------------------------------------------------------------
# 5. Clone / update project repo
# --------------------------------------------------------------------------
echo ""
echo "[6/6] Cloning project repo..."
if [ -d "/workspace/Arabic-Post-OCR-Correction/.git" ]; then
    echo "  [skip] Repo already exists at /workspace/Arabic-Post-OCR-Correction"
    echo "         Run 'git pull' inside it to update."
else
    git clone https://github.com/YOUR_USERNAME/Arabic-Post-OCR-Correction.git \
        /workspace/Arabic-Post-OCR-Correction 2>/dev/null || \
        echo "  [warn] git clone failed — set your repo URL above or clone manually."
fi

# --------------------------------------------------------------------------
# 6. Verify GPU
# --------------------------------------------------------------------------
echo ""
python - <<'PYEOF'
import torch, sys

if not torch.cuda.is_available():
    print("  [ERROR] No GPU detected! Check Thunder instance setup.")
    sys.exit(1)

dev = torch.cuda.get_device_properties(0)
vram_gb = dev.total_memory / 1024**3

print(f"  [ok] GPU  : {dev.name}")
print(f"  [ok] VRAM : {vram_gb:.1f} GB")
print(f"  [ok] CUDA : {torch.version.cuda}")

if vram_gb < 40:
    print(f"  [warn] Less than 40 GB VRAM — consider enabling 4-bit quantization "
          f"(config: model.quantize_4bit: true)")
elif vram_gb >= 79:
    print(f"  [ok] A100 80 GB confirmed — optimal for full FP16 inference.")
PYEOF

echo ""
echo "============================================================"
echo "  Setup complete!"
echo ""
echo "  Quick-start commands:"
echo ""
echo "  # Step 1 — Re-run Qaari OCR on all dataset images:"
echo "  python thunder/qaari_infer.py \\"
echo "      --image-root ./data/images \\"
echo "      --output-root ./data/ocr-results/qaari-results"
echo ""
echo "  # Step 1 (sequential reference, for comparison):"
echo "  python thunder/qaari_infer_seq.py \\"
echo "      --image-root ./data/images \\"
echo "      --output-root ./data/ocr-results/qaari-seq-results"
echo ""
echo "  # Step 2 — Export inference input for LLM correction:"
echo "  python pipelines/run_phase2.py --mode export"
echo ""
echo "  # Step 3 — High-throughput Qwen3 correction with vLLM:"
echo "  python thunder/qwen_infer.py \\"
echo "      --input  results/phase2/inference_input.jsonl \\"
echo "      --output results/phase2/corrections.jsonl"
echo ""
echo "  # Step 3 (sequential reference, for comparison):"
echo "  python thunder/qwen_infer_seq.py \\"
echo "      --input  results/phase2/inference_input.jsonl \\"
echo "      --output results/phase2/corrections_seq.jsonl"
echo ""
echo "  # Step 4 (local) — Analyse results:"
echo "  python pipelines/run_phase2.py --mode analyze"
echo "============================================================"
