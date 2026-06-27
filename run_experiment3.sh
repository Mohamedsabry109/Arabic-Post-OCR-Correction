#!/bin/bash

# Activate the virtual environment
source .venv/bin/activate

# Stub libcuda so Triton/gemma transformers builds find -lcuda
export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib/stubs:$LD_LIBRARY_PATH

# Define the retry function
run_with_retry() {
    echo "--- Starting: $@ ---"
    until python "$@"; do
        echo "Script crashed with exit code $?. Retrying in 2 seconds..."
        sleep 2
    done
    echo "Success! Moving to the next step."
    echo "-----------------------------------"
}

# -- VAL / QAARI ----------------------------------------

# Run 01: qwen3-4b | val_qaari | t2
# run_with_retry scripts/infer_vllm.py \
#     --input '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/inputs/val_qaari_t2.jsonl' \
#     --output '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/corrections/qwen3-4b_val_qaari_t2.jsonl' \
#     --system-prompt '/home/ubuntu/Arabic-Post-OCR-Correction/configs/crafted_system_prompt.txt' \
#     --experiment-model qwen3-4b \
#     --chunk-size 200 \
#     #--enforce-eager

# Run 02: qwen3-14b | val_qaari | t2
run_with_retry scripts/infer_vllm.py \
    --input '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/inputs/val_qaari_t2.jsonl' \
    --output '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/corrections/qwen3-14b_val_qaari_t2.jsonl' \
    --system-prompt '/home/ubuntu/Arabic-Post-OCR-Correction/configs/crafted_system_prompt.txt' \
    --experiment-model qwen3-14b \
    --chunk-size 200 \
    #--enforce-eager

# Run 03: gemma-3-4b | val_qaari | t2
run_with_retry scripts/infer.py \
    --input '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/inputs/val_qaari_t2.jsonl' \
    --output '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/corrections/gemma-3-4b_val_qaari_t2.jsonl' \
    --system-prompt '/home/ubuntu/Arabic-Post-OCR-Correction/configs/crafted_system_prompt.txt' \
    --experiment-model gemma-3-4b \
    --batch-size 8

# Run 04: gemma-3-12b | val_qaari | t2
run_with_retry scripts/infer.py \
    --input '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/inputs/val_qaari_t2.jsonl' \
    --output '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/corrections/gemma-3-12b_val_qaari_t2.jsonl' \
    --system-prompt '/home/ubuntu/Arabic-Post-OCR-Correction/configs/crafted_system_prompt.txt' \
    --experiment-model gemma-3-12b \
    --batch-size 4

# Run 05: qwen3-4b | val_qaari | t3
# run_with_retry scripts/infer_vllm.py \
#     --input '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/inputs/val_qaari_t3.jsonl' \
#     --output '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/corrections/qwen3-4b_val_qaari_t3.jsonl' \
#     --system-prompt '/home/ubuntu/Arabic-Post-OCR-Correction/configs/crafted_system_prompt_v2.txt' \
#     --experiment-model qwen3-4b \
#     --chunk-size 200 \
#     #--enforce-eager

# Run 06: qwen3-14b | val_qaari | t3
run_with_retry scripts/infer_vllm.py \
    --input '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/inputs/val_qaari_t3.jsonl' \
    --output '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/corrections/qwen3-14b_val_qaari_t3.jsonl' \
    --system-prompt '/home/ubuntu/Arabic-Post-OCR-Correction/configs/crafted_system_prompt_v2.txt' \
    --experiment-model qwen3-14b \
    --chunk-size 200 \
    #--enforce-eager

# Run 07: gemma-3-4b | val_qaari | t3
run_with_retry scripts/infer.py \
    --input '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/inputs/val_qaari_t3.jsonl' \
    --output '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/corrections/gemma-3-4b_val_qaari_t3.jsonl' \
    --system-prompt '/home/ubuntu/Arabic-Post-OCR-Correction/configs/crafted_system_prompt_v2.txt' \
    --experiment-model gemma-3-4b \
    --batch-size 8

# Run 08: gemma-3-12b | val_qaari | t3
run_with_retry scripts/infer.py \
    --input '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/inputs/val_qaari_t3.jsonl' \
    --output '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/corrections/gemma-3-12b_val_qaari_t3.jsonl' \
    --system-prompt '/home/ubuntu/Arabic-Post-OCR-Correction/configs/crafted_system_prompt_v2.txt' \
    --experiment-model gemma-3-12b \
    --batch-size 4

# -- VAL / GEMMA ----------------------------------------

# Run 09: qwen3-4b | val_gemma | t2
run_with_retry scripts/infer_vllm.py \
    --input '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/inputs/val_gemma_t2.jsonl' \
    --output '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/corrections/qwen3-4b_val_gemma_t2.jsonl' \
    --system-prompt '/home/ubuntu/Arabic-Post-OCR-Correction/configs/crafted_system_prompt.txt' \
    --experiment-model qwen3-4b \
    --chunk-size 200 \
    #--enforce-eager

# Run 10: qwen3-14b | val_gemma | t2
run_with_retry scripts/infer_vllm.py \
    --input '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/inputs/val_gemma_t2.jsonl' \
    --output '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/corrections/qwen3-14b_val_gemma_t2.jsonl' \
    --system-prompt '/home/ubuntu/Arabic-Post-OCR-Correction/configs/crafted_system_prompt.txt' \
    --experiment-model qwen3-14b \
    --chunk-size 200 \
    #--enforce-eager

# Run 11: gemma-3-4b | val_gemma | t2
run_with_retry scripts/infer.py \
    --input '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/inputs/val_gemma_t2.jsonl' \
    --output '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/corrections/gemma-3-4b_val_gemma_t2.jsonl' \
    --system-prompt '/home/ubuntu/Arabic-Post-OCR-Correction/configs/crafted_system_prompt.txt' \
    --experiment-model gemma-3-4b \
    --batch-size 8

# Run 12: gemma-3-12b | val_gemma | t2
run_with_retry scripts/infer.py \
    --input '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/inputs/val_gemma_t2.jsonl' \
    --output '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/corrections/gemma-3-12b_val_gemma_t2.jsonl' \
    --system-prompt '/home/ubuntu/Arabic-Post-OCR-Correction/configs/crafted_system_prompt.txt' \
    --experiment-model gemma-3-12b \
    --batch-size 4

# Run 13: qwen3-4b | val_gemma | t3
run_with_retry scripts/infer_vllm.py \
    --input '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/inputs/val_gemma_t3.jsonl' \
    --output '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/corrections/qwen3-4b_val_gemma_t3.jsonl' \
    --system-prompt '/home/ubuntu/Arabic-Post-OCR-Correction/configs/crafted_system_prompt_v2.txt' \
    --experiment-model qwen3-4b \
    --chunk-size 200 \
    #--enforce-eager

# Run 14: qwen3-14b | val_gemma | t3
run_with_retry scripts/infer_vllm.py \
    --input '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/inputs/val_gemma_t3.jsonl' \
    --output '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/corrections/qwen3-14b_val_gemma_t3.jsonl' \
    --system-prompt '/home/ubuntu/Arabic-Post-OCR-Correction/configs/crafted_system_prompt_v2.txt' \
    --experiment-model qwen3-14b \
    --chunk-size 200 \
    #--enforce-eager

# Run 15: gemma-3-4b | val_gemma | t3
run_with_retry scripts/infer.py \
    --input '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/inputs/val_gemma_t3.jsonl' \
    --output '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/corrections/gemma-3-4b_val_gemma_t3.jsonl' \
    --system-prompt '/home/ubuntu/Arabic-Post-OCR-Correction/configs/crafted_system_prompt_v2.txt' \
    --experiment-model gemma-3-4b \
    --batch-size 8

# Run 16: gemma-3-12b | val_gemma | t3
run_with_retry scripts/infer.py \
    --input '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/inputs/val_gemma_t3.jsonl' \
    --output '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/corrections/gemma-3-12b_val_gemma_t3.jsonl' \
    --system-prompt '/home/ubuntu/Arabic-Post-OCR-Correction/configs/crafted_system_prompt_v2.txt' \
    --experiment-model gemma-3-12b \
    --batch-size 4

# -- BM / QAARI ----------------------------------------

# Run 17: qwen3-4b | bm_qaari | t2
run_with_retry scripts/infer_vllm.py \
    --input '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/inputs/bm_qaari_t2.jsonl' \
    --output '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/corrections/qwen3-4b_bm_qaari_t2.jsonl' \
    --system-prompt '/home/ubuntu/Arabic-Post-OCR-Correction/configs/crafted_system_prompt.txt' \
    --experiment-model qwen3-4b \
    --chunk-size 200 \
    #--enforce-eager

# Run 18: qwen3-14b | bm_qaari | t2
run_with_retry scripts/infer_vllm.py \
    --input '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/inputs/bm_qaari_t2.jsonl' \
    --output '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/corrections/qwen3-14b_bm_qaari_t2.jsonl' \
    --system-prompt '/home/ubuntu/Arabic-Post-OCR-Correction/configs/crafted_system_prompt.txt' \
    --experiment-model qwen3-14b \
    --chunk-size 200 \
    #--enforce-eager

# Run 19: gemma-3-4b | bm_qaari | t2
run_with_retry scripts/infer.py \
    --input '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/inputs/bm_qaari_t2.jsonl' \
    --output '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/corrections/gemma-3-4b_bm_qaari_t2.jsonl' \
    --system-prompt '/home/ubuntu/Arabic-Post-OCR-Correction/configs/crafted_system_prompt.txt' \
    --experiment-model gemma-3-4b \
    --batch-size 8

# Run 20: gemma-3-12b | bm_qaari | t2
run_with_retry scripts/infer.py \
    --input '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/inputs/bm_qaari_t2.jsonl' \
    --output '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/corrections/gemma-3-12b_bm_qaari_t2.jsonl' \
    --system-prompt '/home/ubuntu/Arabic-Post-OCR-Correction/configs/crafted_system_prompt.txt' \
    --experiment-model gemma-3-12b \
    --batch-size 4

# Run 21: qwen3-4b | bm_qaari | t3
run_with_retry scripts/infer_vllm.py \
    --input '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/inputs/bm_qaari_t3.jsonl' \
    --output '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/corrections/qwen3-4b_bm_qaari_t3.jsonl' \
    --system-prompt '/home/ubuntu/Arabic-Post-OCR-Correction/configs/crafted_system_prompt_v2.txt' \
    --experiment-model qwen3-4b \
    --chunk-size 200 \
    #--enforce-eager

# Run 22: qwen3-14b | bm_qaari | t3
run_with_retry scripts/infer_vllm.py \
    --input '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/inputs/bm_qaari_t3.jsonl' \
    --output '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/corrections/qwen3-14b_bm_qaari_t3.jsonl' \
    --system-prompt '/home/ubuntu/Arabic-Post-OCR-Correction/configs/crafted_system_prompt_v2.txt' \
    --experiment-model qwen3-14b \
    --chunk-size 200 \
    #--enforce-eager

# Run 23: gemma-3-4b | bm_qaari | t3
run_with_retry scripts/infer.py \
    --input '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/inputs/bm_qaari_t3.jsonl' \
    --output '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/corrections/gemma-3-4b_bm_qaari_t3.jsonl' \
    --system-prompt '/home/ubuntu/Arabic-Post-OCR-Correction/configs/crafted_system_prompt_v2.txt' \
    --experiment-model gemma-3-4b \
    --batch-size 8

# Run 24: gemma-3-12b | bm_qaari | t3
run_with_retry scripts/infer.py \
    --input '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/inputs/bm_qaari_t3.jsonl' \
    --output '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/corrections/gemma-3-12b_bm_qaari_t3.jsonl' \
    --system-prompt '/home/ubuntu/Arabic-Post-OCR-Correction/configs/crafted_system_prompt_v2.txt' \
    --experiment-model gemma-3-12b \
    --batch-size 4

# -- BM / GEMMA ----------------------------------------

# Run 25: qwen3-4b | bm_gemma | t2
run_with_retry scripts/infer_vllm.py \
    --input '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/inputs/bm_gemma_t2.jsonl' \
    --output '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/corrections/qwen3-4b_bm_gemma_t2.jsonl' \
    --system-prompt '/home/ubuntu/Arabic-Post-OCR-Correction/configs/crafted_system_prompt.txt' \
    --experiment-model qwen3-4b \
    --chunk-size 200 \
    #--enforce-eager

# Run 26: qwen3-14b | bm_gemma | t2
run_with_retry scripts/infer_vllm.py \
    --input '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/inputs/bm_gemma_t2.jsonl' \
    --output '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/corrections/qwen3-14b_bm_gemma_t2.jsonl' \
    --system-prompt '/home/ubuntu/Arabic-Post-OCR-Correction/configs/crafted_system_prompt.txt' \
    --experiment-model qwen3-14b \
    --chunk-size 200 \
    #--enforce-eager

# Run 27: gemma-3-4b | bm_gemma | t2
run_with_retry scripts/infer.py \
    --input '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/inputs/bm_gemma_t2.jsonl' \
    --output '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/corrections/gemma-3-4b_bm_gemma_t2.jsonl' \
    --system-prompt '/home/ubuntu/Arabic-Post-OCR-Correction/configs/crafted_system_prompt.txt' \
    --experiment-model gemma-3-4b \
    --batch-size 8

# Run 28: gemma-3-12b | bm_gemma | t2
run_with_retry scripts/infer.py \
    --input '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/inputs/bm_gemma_t2.jsonl' \
    --output '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/corrections/gemma-3-12b_bm_gemma_t2.jsonl' \
    --system-prompt '/home/ubuntu/Arabic-Post-OCR-Correction/configs/crafted_system_prompt.txt' \
    --experiment-model gemma-3-12b \
    --batch-size 4

# Run 29: qwen3-4b | bm_gemma | t3
run_with_retry scripts/infer_vllm.py \
    --input '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/inputs/bm_gemma_t3.jsonl' \
    --output '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/corrections/qwen3-4b_bm_gemma_t3.jsonl' \
    --system-prompt '/home/ubuntu/Arabic-Post-OCR-Correction/configs/crafted_system_prompt_v2.txt' \
    --experiment-model qwen3-4b \
    --chunk-size 200 \
    #--enforce-eager

# Run 30: qwen3-14b | bm_gemma | t3
run_with_retry scripts/infer_vllm.py \
    --input '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/inputs/bm_gemma_t3.jsonl' \
    --output '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/corrections/qwen3-14b_bm_gemma_t3.jsonl' \
    --system-prompt '/home/ubuntu/Arabic-Post-OCR-Correction/configs/crafted_system_prompt_v2.txt' \
    --experiment-model qwen3-14b \
    --chunk-size 200 \
    #--enforce-eager

# Run 31: gemma-3-4b | bm_gemma | t3
run_with_retry scripts/infer.py \
    --input '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/inputs/bm_gemma_t3.jsonl' \
    --output '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/corrections/gemma-3-4b_bm_gemma_t3.jsonl' \
    --system-prompt '/home/ubuntu/Arabic-Post-OCR-Correction/configs/crafted_system_prompt_v2.txt' \
    --experiment-model gemma-3-4b \
    --batch-size 8

# Run 32: gemma-3-12b | bm_gemma | t3
run_with_retry scripts/infer.py \
    --input '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/inputs/bm_gemma_t3.jsonl' \
    --output '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/corrections/gemma-3-12b_bm_gemma_t3.jsonl' \
    --system-prompt '/home/ubuntu/Arabic-Post-OCR-Correction/configs/crafted_system_prompt_v2.txt' \
    --experiment-model gemma-3-12b \
    --batch-size 4

# -- KITAB / QAARI ----------------------------------------

# Run 33: qwen3-4b | kitab_qaari | t2
# run_with_retry scripts/infer_vllm.py \
#     --input '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/inputs/kitab_qaari_t2.jsonl' \
#     --output '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/corrections/qwen3-4b_kitab_qaari_t2.jsonl' \
#     --system-prompt '/home/ubuntu/Arabic-Post-OCR-Correction/configs/crafted_system_prompt.txt' \
#     --experiment-model qwen3-4b \
#     --chunk-size 200 \
#     #--enforce-eager

# Run 34: qwen3-14b | kitab_qaari | t2
# run_with_retry scripts/infer_vllm.py \
#     --input '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/inputs/kitab_qaari_t2.jsonl' \
#     --output '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/corrections/qwen3-14b_kitab_qaari_t2.jsonl' \
#     --system-prompt '/home/ubuntu/Arabic-Post-OCR-Correction/configs/crafted_system_prompt.txt' \
#     --experiment-model qwen3-14b \
#     --chunk-size 200 \
#     #--enforce-eager

# Run 35: gemma-3-4b | kitab_qaari | t2
# run_with_retry scripts/infer.py \
#     --input '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/inputs/kitab_qaari_t2.jsonl' \
#     --output '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/corrections/gemma-3-4b_kitab_qaari_t2.jsonl' \
#     --system-prompt '/home/ubuntu/Arabic-Post-OCR-Correction/configs/crafted_system_prompt.txt' \
#     --experiment-model gemma-3-4b \
#     --batch-size 8

# Run 36: gemma-3-12b | kitab_qaari | t2
# run_with_retry scripts/infer.py \
#     --input '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/inputs/kitab_qaari_t2.jsonl' \
#     --output '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/corrections/gemma-3-12b_kitab_qaari_t2.jsonl' \
#     --system-prompt '/home/ubuntu/Arabic-Post-OCR-Correction/configs/crafted_system_prompt.txt' \
#     --experiment-model gemma-3-12b \
#     --batch-size 4

# Run 37: qwen3-4b | kitab_qaari | t3
# run_with_retry scripts/infer_vllm.py \
#     --input '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/inputs/kitab_qaari_t3.jsonl' \
#     --output '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/corrections/qwen3-4b_kitab_qaari_t3.jsonl' \
#     --system-prompt '/home/ubuntu/Arabic-Post-OCR-Correction/configs/crafted_system_prompt_v2.txt' \
#     --experiment-model qwen3-4b \
#     --chunk-size 200 \
#     #--enforce-eager

# Run 38: qwen3-14b | kitab_qaari | t3
# run_with_retry scripts/infer_vllm.py \
#     --input '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/inputs/kitab_qaari_t3.jsonl' \
#     --output '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/corrections/qwen3-14b_kitab_qaari_t3.jsonl' \
#     --system-prompt '/home/ubuntu/Arabic-Post-OCR-Correction/configs/crafted_system_prompt_v2.txt' \
#     --experiment-model qwen3-14b \
#     --chunk-size 200 \
#     #--enforce-eager

# Run 39: gemma-3-4b | kitab_qaari | t3
# run_with_retry scripts/infer.py \
#     --input '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/inputs/kitab_qaari_t3.jsonl' \
#     --output '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/corrections/gemma-3-4b_kitab_qaari_t3.jsonl' \
#     --system-prompt '/home/ubuntu/Arabic-Post-OCR-Correction/configs/crafted_system_prompt_v2.txt' \
#     --experiment-model gemma-3-4b \
#     --batch-size 8

# Run 40: gemma-3-12b | kitab_qaari | t3
# run_with_retry scripts/infer.py \
#     --input '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/inputs/kitab_qaari_t3.jsonl' \
#     --output '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/corrections/gemma-3-12b_kitab_qaari_t3.jsonl' \
#     --system-prompt '/home/ubuntu/Arabic-Post-OCR-Correction/configs/crafted_system_prompt_v2.txt' \
#     --experiment-model gemma-3-12b \
#     --batch-size 4

# -- KITAB / GEMMA ----------------------------------------

# Run 41: qwen3-4b | kitab_gemma | t2
# run_with_retry scripts/infer_vllm.py \
#     --input '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/inputs/kitab_gemma_t2.jsonl' \
#     --output '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/corrections/qwen3-4b_kitab_gemma_t2.jsonl' \
#     --system-prompt '/home/ubuntu/Arabic-Post-OCR-Correction/configs/crafted_system_prompt.txt' \
#     --experiment-model qwen3-4b \
#     --chunk-size 200 \
#     #--enforce-eager

# Run 42: qwen3-14b | kitab_gemma | t2
# run_with_retry scripts/infer_vllm.py \
#     --input '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/inputs/kitab_gemma_t2.jsonl' \
#     --output '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/corrections/qwen3-14b_kitab_gemma_t2.jsonl' \
#     --system-prompt '/home/ubuntu/Arabic-Post-OCR-Correction/configs/crafted_system_prompt.txt' \
#     --experiment-model qwen3-14b \
#     --chunk-size 200 \
#     #--enforce-eager

# Run 43: gemma-3-4b | kitab_gemma | t2
# run_with_retry scripts/infer.py \
#     --input '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/inputs/kitab_gemma_t2.jsonl' \
#     --output '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/corrections/gemma-3-4b_kitab_gemma_t2.jsonl' \
#     --system-prompt '/home/ubuntu/Arabic-Post-OCR-Correction/configs/crafted_system_prompt.txt' \
#     --experiment-model gemma-3-4b \
#     --batch-size 8

# Run 44: gemma-3-12b | kitab_gemma | t2
# run_with_retry scripts/infer.py \
#     --input '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/inputs/kitab_gemma_t2.jsonl' \
#     --output '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/corrections/gemma-3-12b_kitab_gemma_t2.jsonl' \
#     --system-prompt '/home/ubuntu/Arabic-Post-OCR-Correction/configs/crafted_system_prompt.txt' \
#     --experiment-model gemma-3-12b \
#     --batch-size 4

# Run 45: qwen3-4b | kitab_gemma | t3
# run_with_retry scripts/infer_vllm.py \
#     --input '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/inputs/kitab_gemma_t3.jsonl' \
#     --output '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/corrections/qwen3-4b_kitab_gemma_t3.jsonl' \
#     --system-prompt '/home/ubuntu/Arabic-Post-OCR-Correction/configs/crafted_system_prompt_v2.txt' \
#     --experiment-model qwen3-4b \
#     --chunk-size 200 \
#     #--enforce-eager

# Run 46: qwen3-14b | kitab_gemma | t3
# run_with_retry scripts/infer_vllm.py \
#     --input '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/inputs/kitab_gemma_t3.jsonl' \
#     --output '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/corrections/qwen3-14b_kitab_gemma_t3.jsonl' \
#     --system-prompt '/home/ubuntu/Arabic-Post-OCR-Correction/configs/crafted_system_prompt_v2.txt' \
#     --experiment-model qwen3-14b \
#     --chunk-size 200 \
#     #--enforce-eager

# Run 47: gemma-3-4b | kitab_gemma | t3
# run_with_retry scripts/infer.py \
#     --input '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/inputs/kitab_gemma_t3.jsonl' \
#     --output '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/corrections/gemma-3-4b_kitab_gemma_t3.jsonl' \
#     --system-prompt '/home/ubuntu/Arabic-Post-OCR-Correction/configs/crafted_system_prompt_v2.txt' \
#     --experiment-model gemma-3-4b \
#     --batch-size 8

# Run 48: gemma-3-12b | kitab_gemma | t3
# run_with_retry scripts/infer.py \
#     --input '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/inputs/kitab_gemma_t3.jsonl' \
#     --output '/home/ubuntu/Arabic-Post-OCR-Correction/results/experiment3/corrections/gemma-3-12b_kitab_gemma_t3.jsonl' \
#     --system-prompt '/home/ubuntu/Arabic-Post-OCR-Correction/configs/crafted_system_prompt_v2.txt' \
#     --experiment-model gemma-3-12b \
#     --batch-size 4

echo "All runs finished successfully!"
