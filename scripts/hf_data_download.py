from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Mohamed109/ocr-results",
    repo_type="dataset",
    local_dir="./ocr-results",
    max_workers=2,          # reduce parallel calls
    resume_download=True
)