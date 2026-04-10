# Kitab-Bench OCR: Step-by-Step Guide

Get Qaari OCR output for all 3,760 kitab-bench images using the existing Kaggle notebook.

## Overview

| Item | Value |
|------|-------|
| Images | 3,760 across 13 sub-datasets |
| Total size | ~372 MB (images only) |
| Notebook | `notebooks/qaari-infer-kaggle.ipynb` |
| Estimated time | ~1-2 hours on T4x2 |
| Output | `.txt` files mirroring the image structure |

## Step 1: Upload kitab-bench images to Kaggle as a Dataset

The notebook reads images from `/kaggle/input/`. You need to upload `data/kitab-bench/` as a Kaggle dataset containing **only the images** (GT files are not needed on Kaggle).

### Option A: Kaggle CLI (recommended)

```bash
# 1. Install Kaggle CLI if not already
pip install kaggle

# 2. Make sure ~/.kaggle/kaggle.json has your API key
#    (Download from kaggle.com -> Settings -> API -> Create New Token)

# 3. Create a metadata file for the new dataset
mkdir -p /tmp/kitab-bench-upload
```

Create `/tmp/kitab-bench-upload/dataset-metadata.json`:
```json
{
  "title": "kitab-bench-images",
  "id": "YOUR_KAGGLE_USERNAME/kitab-bench-images",
  "licenses": [{"name": "CC0-1.0"}]
}
```

```bash
# 4. Copy ONLY the images (no GT, no metadata) into upload folder
#    From project root:
cd data/kitab-bench
for ds in */; do
    mkdir -p /tmp/kitab-bench-upload/"$ds"images/
    cp "$ds"images/*.png /tmp/kitab-bench-upload/"$ds"images/
done

# 5. Upload
kaggle datasets create -p /tmp/kitab-bench-upload --dir-mode zip
```

### Option B: Kaggle Web UI

1. Go to https://www.kaggle.com/datasets → **New Dataset**
2. Name it `kitab-bench-images`
3. Upload the **images** folders only. Structure should be:
   ```
   kitab-bench-images/
   ├── adab/images/0.png, 1.png, ...
   ├── arabicocr/images/0.png, 1.png, ...
   ├── evarest/images/0.png, 1.png, ...
   ├── hindawi/images/0.png, 1.png, ...
   ├── historicalbooks/images/0.png, 1.png, ...
   ├── historyar/images/0.png, 1.png, ...
   ├── isippt/images/0.png, 1.png, ...
   ├── khatt/images/0.png, 1.png, ...
   ├── khattparagraph/images/0.png, 1.png, ...
   ├── muharaf/images/0.png, 1.png, ...
   ├── onlinekhatt/images/0.png, 1.png, ...
   ├── patsocr/images/0.png, 1.png, ...
   └── synthesizear/images/0.png, 1.png, ...
   ```
4. **Tip**: If 372 MB is slow to upload via browser, zip the folder first and upload the zip — Kaggle auto-extracts it.

### Option C: Zip and upload (easiest for large data)

```bash
# From project root
cd data
zip -r kitab-bench-images.zip kitab-bench/*/images/
# Upload the zip via Kaggle web UI as a new dataset named "kitab-bench-images"
# Kaggle will auto-extract it
```

> **Note**: After upload, verify the dataset on Kaggle. The images should be accessible at paths like `/kaggle/input/kitab-bench-images/kitab-bench/adab/images/0.png` (if you zipped from `data/`) or `/kaggle/input/kitab-bench-images/adab/images/0.png` (if you uploaded folders directly). The exact prefix depends on how you uploaded — check the "Data" tab on your Kaggle dataset page to confirm the structure.

## Step 2: Add the dataset to your Kaggle notebook

1. Open `notebooks/qaari-infer-kaggle.ipynb` on Kaggle (or create a new notebook)
2. Click **Add Data** (right sidebar) → search for your `kitab-bench-images` dataset → **Add**
3. Verify it appears under `/kaggle/input/kitab-bench-images/`

You can verify the path in a notebook cell:
```python
!ls /kaggle/input/kitab-bench-images/
```

## Step 3: Update DATASET_DIRS in the notebook

In cell-2 (the big code cell), find the `DATASET_DIRS` list and change it to point to kitab-bench:

```python
# Configuration
DATASET_DIRS = [
    "/kaggle/input/kitab-bench-images",       # if uploaded folders directly
    # OR
    # "/kaggle/input/kitab-bench-images/kitab-bench",  # if zipped from data/
]
```

**Important**: The pipeline scans recursively for image files (`.png`, `.jpg`, etc.) under `DATASET_DIRS`, so just point it to the root — it will find all 3,760 images across all 13 sub-datasets automatically.

### Output path mapping

The notebook maps input → output like this:
```
Input:  /kaggle/input/kitab-bench-images/adab/images/0.png
Output: /kaggle/working/results/kitab-bench-images/adab/images/0.txt
```

The path structure under `results/` mirrors whatever is under `/kaggle/input/`.

## Step 4: Run the notebook

1. Set GPU: **Settings** → **Accelerator** → **GPU T4 x2**
2. Set internet: **On** (needed for HF sync)
3. Run all cells

The pipeline will:
- List remote files already synced (skip those)
- Process all pending images
- Sync results to HF repo `Mohamed109/ocr-results` every 50 images
- Retry any OOM failures automatically

### Expected runtime

| Sub-dataset | Images | Est. time |
|-------------|--------|-----------|
| adab | 200 | ~3 min |
| arabicocr | 50 | ~1 min |
| evarest | 800 | ~12 min |
| hindawi | 200 | ~3 min |
| historicalbooks | 10 | <1 min |
| historyar | 200 | ~3 min |
| isippt | 500 | ~8 min |
| khatt | 200 | ~3 min |
| khattparagraph | 200 | ~3 min |
| muharaf | 200 | ~3 min |
| onlinekhatt | 200 | ~3 min |
| patsocr | 500 | ~8 min |
| synthesizear | 500 | ~8 min |
| **Total** | **3,760** | **~60 min** |

*Times are rough estimates assuming ~1 img/sec on T4x2 with 2 workers.*

## Step 5: Download the OCR results

After the notebook finishes, the results are synced to HF repo `Mohamed109/ocr-results`. Download them locally.

### Option A: From HF Hub

```bash
# From project root — download only the kitab-bench results
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'Mohamed109/ocr-results',
    repo_type='dataset',
    local_dir='./data/ocr-results/qaari-results/kitab-bench-dl',
    allow_patterns='results/kitab-bench-images/**/*.txt',
)
"
```

### Option B: From Kaggle output

If HF sync didn't work, download from the notebook's output:
1. On the notebook page, go to **Output** tab
2. Download the `results/` folder
3. Place it locally

### Option C: Direct from notebook (add a cell)

Add a final cell to the notebook to zip and download:
```python
!cd /kaggle/working && zip -r results.zip results/
# Then download results.zip from the Output tab
```

## Step 6: Place results in the correct local directory

The OCR results need to be under `data/ocr-results/qaari-results/` with a structure matching the images:

```
data/ocr-results/qaari-results/kitab-bench/
├── adab/images/0.txt
├── adab/images/1.txt
├── ...
├── arabicocr/images/0.txt
├── ...
└── synthesizear/images/499.txt
```

Depending on how you downloaded, you may need to rename/move:

```bash
# If downloaded from HF (files are under results/kitab-bench-images/...):
mv data/ocr-results/qaari-results/kitab-bench-dl/results/kitab-bench-images \
   data/ocr-results/qaari-results/kitab-bench

# Verify
ls data/ocr-results/qaari-results/kitab-bench/
# Should show: adab  arabicocr  evarest  hindawi  ...
```

## Step 7: Verify

```bash
# Count OCR result files — should be ~3,760
find data/ocr-results/qaari-results/kitab-bench -name "*.txt" | wc -l

# Spot-check a result
cat data/ocr-results/qaari-results/kitab-bench/adab/images/0.txt
```

## Troubleshooting

### Kaggle session times out
The notebook has built-in resume. Just re-run — it skips already-processed images (checks both local output and HF remote).

### Rate limit (429) on HF sync
The pipeline has exponential backoff built in. If it persists, increase `SYNC_INTERVAL` from 50 to 100.

### OOM errors
The pipeline auto-retries with reduced image quality. If a specific image keeps failing, it gets logged to `failed_images.json` in the notebook output.

### Wrong path structure after download
The key is that the path under `qaari-results/` matches the input structure. If you get an extra nesting level like `kitab-bench-images/kitab-bench/adab/...`, just move the inner folder up.
