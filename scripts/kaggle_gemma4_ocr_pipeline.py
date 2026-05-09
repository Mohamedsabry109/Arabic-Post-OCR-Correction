import os
import time
import torch
import multiprocessing
import gc
import json
import fcntl
import tempfile
import shutil
from pathlib import Path
from typing import List, Tuple, Optional, Set
from PIL import Image
from huggingface_hub import HfApi

try:
    from kaggle_secrets import UserSecretsClient
    HF_TOKEN = UserSecretsClient().get_secret("HF_WRITE")
    print("HF token loaded")
except:
    HF_TOKEN = None
    print("HF token not found - sync disabled")

# Configuration
DATASET_DIRS = [
    "/kaggle/input/datasets/mohamedsabry109/khatt-paragraph/proc_images",
]
HF_REPO_ID = "Mohamed109/ocr-results"
HF_REPO_FOLDER = "new_results"  # <--- SET YOUR NEW HF FOLDER NAME HERE
MODEL_PATH = "google/gemma-3-4b-it"  # <--- SET YOUR GEMMA 4 MODEL PATH HERE
BASE_OUTPUT_DIR = "/kaggle/working/results"
FAILED_IMAGES_LOG = "/kaggle/working/failed_images.json"
SYNC_STATE_FILE = "/kaggle/working/sync_state.json"
SYNC_INTERVAL = 20
USE_BOTH_GPUS = False
INSTANCES_PER_GPU = 1
MAX_NEW_TOKENS = 2000
LOG_INTERVAL = 10

# Performance tuning
AGGRESSIVE_CLEANUP = False
CLEANUP_INTERVAL = 50
MAX_RETRIES_PER_WORKER = 2
ENABLE_CROSS_WORKER_RETRY = True

# Rate limit / retry config
MAX_API_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 10
MAX_BACKOFF_SECONDS = 300

OCR_PROMPT = (
    "Below is the image of one page of a document. "
    "Just return the plain text representation of this document as if you were reading it naturally. "
    "Do not hallucinate."
)


def retry_with_backoff(func, *args, max_retries=MAX_API_RETRIES, initial_backoff=INITIAL_BACKOFF_SECONDS, **kwargs):
    """Generic retry wrapper with exponential backoff for HF API calls."""
    last_exception = None
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            error_str = str(e).lower()
            is_rate_limit = "429" in str(e) or "rate limit" in error_str
            is_server_error = any(code in str(e) for code in ["500", "502", "503", "504"])

            if not (is_rate_limit or is_server_error):
                raise

            backoff = min(initial_backoff * (2 ** attempt), MAX_BACKOFF_SECONDS)
            if is_rate_limit:
                import re
                wait_match = re.search(r'waiting\s+(\d+\.?\d*)\s*s', error_str)
                if wait_match:
                    suggested = float(wait_match.group(1))
                    backoff = max(backoff, suggested + 5)

            print(f"  API error (attempt {attempt + 1}/{max_retries}): {type(e).__name__}")
            print(f"  Retrying in {backoff:.0f}s...", flush=True)
            time.sleep(backoff)

    raise last_exception


class SyncStateTracker:
    """Thread-safe tracker with file locking for tracking synced files."""
    def __init__(self, state_file: str):
        self.state_file = state_file
        self.lock_file = state_file + ".lock"

    def _load_state(self) -> Set[str]:
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    return set(data.get("synced_files", []))
            except:
                pass
        return set()

    def _save_state(self, synced_files: Set[str]):
        temp_file = self.state_file + ".tmp"
        with open(temp_file, 'w') as f:
            json.dump({"synced_files": list(synced_files)}, f)
        os.replace(temp_file, self.state_file)

    def mark_synced(self, file_paths: List[str]):
        if not os.path.exists(self.lock_file):
            open(self.lock_file, 'w').close()

        with open(self.lock_file, 'r') as lock_fd:
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)
            try:
                current_state = self._load_state()
                current_state.update(file_paths)
                self._save_state(current_state)
            finally:
                fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)

    def get_synced_files(self) -> Set[str]:
        return self._load_state()

    def get_unsynced_files(self, base_dir: str) -> List[str]:
        synced_files = self._load_state()
        all_files = []
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file.endswith('.txt'):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, base_dir)
                    if rel_path not in synced_files:
                        all_files.append(full_path)
        return all_files


class Gemma4OCR:
    """OCR engine using Gemma 4 (or Gemma 3 multimodal) via AutoModelForImageTextToText."""

    def __init__(self, model_name: str, max_tokens: int, device: str, use_flash_attn: bool = False):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        import torch

        self.device = device

        attn_impl = "flash_attention_2" if use_flash_attn else "eager"
        print(f"Loading Gemma4 with attn_implementation={attn_impl} on {device}...")

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
            device_map=device,
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.max_tokens = max_tokens
        self.model.eval()

    def __call__(self, _: str, image: Image.Image, worker_id: int, reduced_quality: bool = False) -> str:
        import torch

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": OCR_PROMPT},
                ],
            }
        ]

        # apply_chat_template with return_dict=True produces input_ids + pixel_values
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)

        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=False,
            )

        # Decode only the newly generated tokens
        output_ids = generated_ids[0][input_len:]
        output_text = self.processor.decode(output_ids, skip_special_tokens=True)

        del inputs, generated_ids, output_ids

        return output_text


class FailedImagesTracker:
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.local_failures = []

    def add_failed(self, img_path: str, out_path: str, worker_id: int, error: str):
        self.local_failures.append({
            "img_path": img_path,
            "out_path": out_path,
            "worker_id": worker_id,
            "error": str(error)[:100]
        })

    def flush_to_disk(self):
        if not self.local_failures:
            return

        existing = {}
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as f:
                    existing = json.load(f)
            except:
                pass

        for failure in self.local_failures:
            key = failure["img_path"]
            if key not in existing:
                existing[key] = {
                    "img_path": failure["img_path"],
                    "out_path": failure["out_path"],
                    "attempts": []
                }
            existing[key]["attempts"].append({
                "worker_id": failure["worker_id"],
                "error": failure["error"]
            })

        with open(self.log_file, 'w') as f:
            json.dump(existing, f, indent=2)

        self.local_failures.clear()

    def load_failures(self) -> dict:
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def get_retry_candidates(self, max_attempts: int = 4) -> List[Tuple[str, str]]:
        failures = self.load_failures()
        retry_list = []
        for key, data in failures.items():
            if len(data["attempts"]) < max_attempts and not os.path.exists(data["out_path"]):
                retry_list.append((data["img_path"], data["out_path"]))
        return retry_list

    def get_final_failures(self) -> List[dict]:
        failures = self.load_failures()
        final_failures = []
        for key, data in failures.items():
            if not os.path.exists(data["out_path"]) and len(data["attempts"]) >= 4:
                final_failures.append(data)
        return final_failures


class OCRWorker:
    def __init__(self, worker_id: int, gpu_id: int, should_sync: bool = False):
        self.worker_id = worker_id
        self.gpu_id = gpu_id
        self.device = f"cuda:{gpu_id}"
        self.should_sync = should_sync and HF_TOKEN
        self.api = HfApi(token=HF_TOKEN) if self.should_sync else None
        self.sync_tracker = SyncStateTracker(SYNC_STATE_FILE) if self.should_sync else None
        self.ocr_engine = None
        self.failed_tracker = FailedImagesTracker(FAILED_IMAGES_LOG)
        self.images_since_cleanup = 0
        self.newly_created_files = []

    def load_model(self) -> bool:
        print(f"[Worker {self.worker_id} | GPU {self.gpu_id}] Loading model...", flush=True)
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.ocr_engine = Gemma4OCR(
                model_name=MODEL_PATH,
                max_tokens=MAX_NEW_TOKENS,
                device=self.device,
                use_flash_attn=False
            )

            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(self.gpu_id) / 1024**3
                print(f"[Worker {self.worker_id} | GPU {self.gpu_id}] Memory: {allocated:.2f}GB allocated", flush=True)

            print(f"[Worker {self.worker_id} | GPU {self.gpu_id}] Ready!", flush=True)
            return True
        except Exception as e:
            print(f"[Worker {self.worker_id} | GPU {self.gpu_id}] FAILED: {e}", flush=True)
            return False

    def cleanup_memory(self, force: bool = False):
        if force or AGGRESSIVE_CLEANUP:
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()

    def periodic_cleanup(self):
        self.images_since_cleanup += 1
        if self.images_since_cleanup >= CLEANUP_INTERVAL:
            self.cleanup_memory(force=True)
            self.images_since_cleanup = 0

    def process_image(self, img_path: str, out_path: str, retry_count: int = 0) -> bool:
        try:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            image = Image.open(img_path).convert("RGB")
            reduced_quality = retry_count > 0
            output = self.ocr_engine("", image, self.worker_id, reduced_quality=reduced_quality)

            with open(out_path, "w", encoding="utf-8") as f:
                f.write(output)

            self.newly_created_files.append(out_path)
            self.periodic_cleanup()
            return True

        except torch.cuda.OutOfMemoryError:
            error_msg = f"OOM (attempt {retry_count + 1})"
            print(f"[Worker {self.worker_id} | GPU {self.gpu_id}] {error_msg}: {os.path.basename(img_path)}", flush=True)

            self.failed_tracker.add_failed(img_path, out_path, self.worker_id, error_msg)
            self.cleanup_memory(force=True)

            if retry_count < MAX_RETRIES_PER_WORKER:
                time.sleep(2)
                return self.process_image(img_path, out_path, retry_count + 1)
            return False

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)[:80]}"
            print(f"[Worker {self.worker_id} | GPU {self.gpu_id}] Error on {os.path.basename(img_path)}: {error_msg}", flush=True)
            self.failed_tracker.add_failed(img_path, out_path, self.worker_id, error_msg)
            self.cleanup_memory(force=True)
            return False

    def sync_incremental(self, processed_count: int) -> bool:
        """Batch upload new files in a single commit with retry logic."""
        if not self.should_sync or not self.newly_created_files:
            return True

        num_new = len(self.newly_created_files)
        print(f"\n[Worker {self.worker_id} | GPU {self.gpu_id}] [SYNC] Uploading {num_new} files...", flush=True)

        temp_dir = tempfile.mkdtemp()

        try:
            for file_path in self.newly_created_files:
                rel_path = os.path.relpath(file_path, BASE_OUTPUT_DIR)
                temp_file_path = os.path.join(temp_dir, rel_path)
                os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)
                shutil.copy2(file_path, temp_file_path)

            def _do_upload():
                self.api.upload_folder(
                    folder_path=temp_dir,
                    repo_id=HF_REPO_ID,
                    repo_type="dataset",
                    path_in_repo=HF_REPO_FOLDER,
                    commit_message=f"Worker {self.worker_id}: Add {num_new} files to {HF_REPO_FOLDER} (checkpoint at {processed_count})"
                )

            retry_with_backoff(_do_upload)

            rel_paths = [os.path.relpath(f, BASE_OUTPUT_DIR) for f in self.newly_created_files]
            self.sync_tracker.mark_synced(rel_paths)
            self.newly_created_files.clear()

            print(f"[Worker {self.worker_id} | GPU {self.gpu_id}] [SYNC] {num_new} files uploaded", flush=True)
            return True

        except Exception as e:
            print(f"[Worker {self.worker_id} | GPU {self.gpu_id}] [SYNC] Failed after retries: {e}", flush=True)
            return False
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def run(self, tasks: List[Tuple[str, str]]):
        if not self.load_model():
            return

        total = len(tasks)
        processed = skipped = failed = new_since_sync = 0
        start_time = time.time()

        for idx, (img, out) in enumerate(tasks, 1):
            if os.path.exists(out):
                skipped += 1
                continue

            if self.process_image(img, out):
                processed += 1
                new_since_sync += 1
            else:
                failed += 1

            if self.should_sync and new_since_sync >= SYNC_INTERVAL:
                self.sync_incremental(processed)
                new_since_sync = 0

            if idx % LOG_INTERVAL == 0:
                elapsed = time.time() - start_time
                speed = processed / elapsed if elapsed > 0 else 0
                eta = (total - idx) / speed if speed > 0 else 0

                mem_info = ""
                if torch.cuda.is_available() and idx % (LOG_INTERVAL * 5) == 0:
                    allocated = torch.cuda.memory_allocated(self.gpu_id) / 1024**3
                    mem_info = f" | Mem:{allocated:.2f}GB"

                print(
                    f"[Worker {self.worker_id} | GPU {self.gpu_id}] {idx}/{total} | "
                    f"Done:{processed} Skip:{skipped} Fail:{failed} | "
                    f"{speed:.2f} img/s | ETA:{eta/60:.1f}m{mem_info}",
                    flush=True
                )

        self.failed_tracker.flush_to_disk()

        if self.should_sync and self.newly_created_files:
            self.sync_incremental(processed)

        print(f"\n[Worker {self.worker_id} | GPU {self.gpu_id}] DONE! Processed:{processed} Skipped:{skipped} Failed:{failed}")


def worker_process(worker_id: int, gpu_id: int, tasks: List[Tuple[str, str]], should_sync: bool):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["XLA_FLAGS"] = "--xla_gpu_force_compilation_parallelism=1"

    if worker_id > 0:
        time.sleep(worker_id * 5)

    worker = OCRWorker(worker_id, gpu_id, should_sync)
    worker.run(tasks)


def fetch_existing_remote_files() -> Set[str]:
    """List files in the HF repo (single API call, no downloads)."""
    if not HF_TOKEN:
        return set()

    print(f"Listing existing files in {HF_REPO_ID}...", flush=True)

    try:
        api = HfApi(token=HF_TOKEN)

        def _list_files():
            return api.list_repo_files(repo_id=HF_REPO_ID, repo_type="dataset")

        all_files = retry_with_backoff(_list_files)

        prefix = f"{HF_REPO_FOLDER}/"
        remote_results = {
            f[len(prefix):]
            for f in all_files
            if f.startswith(prefix) and f.endswith(".txt")
        }

        print(f"Found {len(remote_results)} existing result files in '{HF_REPO_FOLDER}'", flush=True)
        return remote_results

    except Exception as e:
        print(f"Could not list remote files: {e}", flush=True)
        return set()


def scan_datasets() -> List[Tuple[str, str]]:
    print("\n" + "="*70)
    print("SCANNING")
    print("="*70)
    tasks = []
    extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')
    for ds_dir in DATASET_DIRS:
        if not os.path.exists(ds_dir):
            print(f"Not found: {ds_dir}")
            continue
        print(f"Dir: {ds_dir}")
        count = 0
        for root, _, files in os.walk(ds_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in extensions):
                    inp = os.path.join(root, file)
                    rel = os.path.relpath(inp, "/kaggle/input")
                    out = os.path.join(BASE_OUTPUT_DIR, os.path.splitext(rel)[0] + ".txt")
                    tasks.append((inp, out))
                    count += 1
        print(f"   {count} images")
    print(f"\nTotal: {len(tasks)}")
    return tasks


def retry_failed_images():
    if not ENABLE_CROSS_WORKER_RETRY:
        return

    tracker = FailedImagesTracker(FAILED_IMAGES_LOG)
    retry_candidates = tracker.get_retry_candidates(max_attempts=4)

    if not retry_candidates:
        print("\nNo images need retry")
        return

    print(f"\n" + "="*70)
    print(f"RETRY PHASE - {len(retry_candidates)} images")
    print("="*70)

    num_gpus = 2 if USE_BOTH_GPUS else 1
    chunk_size = len(retry_candidates) // num_gpus

    processes = []
    for gpu_id in range(num_gpus):
        start_idx = gpu_id * chunk_size
        end_idx = start_idx + chunk_size if gpu_id < num_gpus - 1 else len(retry_candidates)
        gpu_tasks = retry_candidates[start_idx:end_idx]

        if gpu_tasks:
            print(f"GPU {gpu_id}: {len(gpu_tasks)} retry tasks")
            p = multiprocessing.Process(
                target=worker_process,
                args=(100 + gpu_id, gpu_id, gpu_tasks, True)
            )
            processes.append(p)

    for p in processes:
        p.start()
    for p in processes:
        p.join()

    print("="*70)


def print_final_report():
    tracker = FailedImagesTracker(FAILED_IMAGES_LOG)
    final_failures = tracker.get_final_failures()

    if final_failures:
        print(f"\n" + "="*70)
        print(f"FINAL REPORT - {len(final_failures)} images could not be processed")
        print("="*70)
        for failure in final_failures[:10]:
            print(f"  - {os.path.basename(failure['img_path'])} ({len(failure['attempts'])} attempts)")
        if len(final_failures) > 10:
            print(f"  ... and {len(final_failures) - 10} more")
        print(f"\nFull log: {FAILED_IMAGES_LOG}")
        print("="*70)
    else:
        print(f"\nAll images processed successfully!")


def main():
    print("\n" + "="*70)
    print("KAGGLE OCR PIPELINE - GEMMA 4")
    print("="*70)
    print(f"Model: {MODEL_PATH}")
    print(f"Output: {BASE_OUTPUT_DIR}")
    print(f"Remote Folder: {HF_REPO_FOLDER}")
    print(f"Sync: {'ON (incremental)' if HF_TOKEN else 'OFF'}")
    print(f"GPUs: {'Both' if USE_BOTH_GPUS else 'Single'}")
    print(f"Instances per GPU: {INSTANCES_PER_GPU}")
    print("="*70)

    print("Pre-fetching model weights to cache...")
    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor
        import gc

        AutoProcessor.from_pretrained(MODEL_PATH)

        temp_model = AutoModelForImageTextToText.from_pretrained(
            MODEL_PATH,
            device_map="cpu",
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
        )

        del temp_model
        gc.collect()
        print("All weights cached locally!")
    except Exception as e:
        print(f"Pre-fetch failed: {e}")
    print("="*70)

    remote_files = fetch_existing_remote_files()

    if remote_files:
        sync_tracker = SyncStateTracker(SYNC_STATE_FILE)
        sync_tracker.mark_synced(list(remote_files))

    all_tasks = scan_datasets()

    if not all_tasks:
        print("\nNo images found!")
        return

    print("\nFiltering...")
    pending = []
    done_local = 0
    done_remote = 0
    for img_path, out_path in all_tasks:
        if os.path.exists(out_path):
            done_local += 1
            continue
        rel_path = os.path.relpath(out_path, BASE_OUTPUT_DIR)
        if rel_path in remote_files:
            done_remote += 1
            continue
        pending.append((img_path, out_path))

    print(f"   Done locally: {done_local}")
    print(f"   Done remotely: {done_remote}")
    print(f"   Remaining: {len(pending)}")

    if not pending:
        print("\nAll done!")
        print_final_report()
        return

    num_gpus = 2 if USE_BOTH_GPUS else 1
    num_workers = num_gpus * INSTANCES_PER_GPU

    chunk_size = len(pending) // num_workers
    remainder = len(pending) % num_workers

    worker_tasks = []
    start_idx = 0
    for i in range(num_workers):
        size = chunk_size + (1 if i < remainder else 0)
        end_idx = start_idx + size
        worker_tasks.append(pending[start_idx:end_idx])
        start_idx = end_idx

    print("\n" + "="*70)
    print("PHASE 1: INITIAL PROCESSING")
    print("="*70)

    worker_id = 0
    for gpu_id in range(num_gpus):
        for instance in range(INSTANCES_PER_GPU):
            print(f"Worker {worker_id} (GPU {gpu_id}): {len(worker_tasks[worker_id])} tasks")
            worker_id += 1
    print("="*70)

    processes = []
    worker_id = 0

    for gpu_id in range(num_gpus):
        for instance in range(INSTANCES_PER_GPU):
            p = multiprocessing.Process(
                target=worker_process,
                args=(worker_id, gpu_id, worker_tasks[worker_id], True)
            )
            processes.append(p)
            worker_id += 1

    for p in processes:
        p.start()
    for p in processes:
        p.join()

    if ENABLE_CROSS_WORKER_RETRY:
        retry_failed_images()

    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print_final_report()


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
