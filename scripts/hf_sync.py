#!/usr/bin/env python3
"""Sync inference outputs to / from a HuggingFace dataset repository.

Supports three modes:
  push  — upload local files to HF (new files + updated files only)
  pull  — download HF files into the local tree (merge: remote wins on conflict)
  sync  — pull first (pick up remote progress), then push (publish local progress)

Designed to run on Thunder, Colab, Kaggle, or locally.

Usage
-----
    # Push all of results/ to HF
    python scripts/hf_sync.py push

    # Pull results/ from HF
    python scripts/hf_sync.py pull

    # Sync (pull then push) — safest option when multiple machines write in parallel
    python scripts/hf_sync.py sync

    # Push only specific subdirectories or files
    python scripts/hf_sync.py push --paths results/phase2 results/phase3

    # Also sync Qaari OCR txt results
    python scripts/hf_sync.py push --paths results data/ocr-results/qaari-results

    # Dry-run: show what would be uploaded without actually uploading
    python scripts/hf_sync.py push --dry-run

    # Override repo / token
    python scripts/hf_sync.py push --repo YOUR_HF_USERNAME/your-repo --token hf_xxx

Authentication
--------------
Set HF_TOKEN in the environment (preferred) or pass --token.
On Colab: store as a Colab secret named HF_TOKEN (Secrets panel).
On Thunder / Kaggle: set via env or pass on the CLI.

Repo layout
-----------
The HF repo mirrors the local directory structure verbatim.
Local  : results/phase2/corrections.jsonl
HF     : results/phase2/corrections.jsonl

File filters (defaults)
-----------------------
Included : *.jsonl  *.json  *.txt  *.md  *.log
Excluded : __pycache__  *.py  *.pyc  .git  *.png  *.jpg  *.jpeg
           (raw images are never pushed — they're too large)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_REPO      = "Mohamed109/ocr-results"
DEFAULT_REPO_TYPE = "dataset"

# Patterns passed to upload_folder (HF glob syntax)
_UPLOAD_ALLOW = [
    "**/*.jsonl",
    "**/*.json",
    "**/*.txt",
    "**/*.md",
    "**/*.log",
]

# Patterns to always exclude
_UPLOAD_IGNORE = [
    "**/__pycache__/**",
    "**/*.py",
    "**/*.pyc",
    "**/.git/**",
    "**/*.png",
    "**/*.jpg",
    "**/*.jpeg",
    "**/*.tif",
    "**/*.tiff",
    "**/*.bmp",
    "**/*.gif",
]

# Default local paths to sync (relative to project root)
_DEFAULT_PATHS = ["results"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve(p: str | Path) -> Path:
    """Return an absolute path, resolving relative paths from project root."""
    p = Path(p)
    return p if p.is_absolute() else _PROJECT_ROOT / p


def _get_token(cli_token: Optional[str]) -> Optional[str]:
    token = cli_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        logger.warning(
            "No HF token found. Set HF_TOKEN env var or pass --token. "
            "Private repos will fail; public repos may work without a token."
        )
    return token


def _require_huggingface_hub() -> None:
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        logger.error("huggingface_hub is not installed. Run: pip install huggingface_hub")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Push
# ---------------------------------------------------------------------------


def push(
    paths: list[str],
    repo: str,
    token: Optional[str],
    dry_run: bool,
    commit_message: Optional[str],
) -> None:
    """Upload each path (file or directory) to the HF repo."""
    from huggingface_hub import HfApi

    api = HfApi()

    # Ensure the repo exists (create if missing)
    try:
        api.repo_info(repo_id=repo, repo_type=DEFAULT_REPO_TYPE, token=token)
    except Exception:
        if not dry_run:
            logger.info("Repo %s not found — creating it...", repo)
            api.create_repo(
                repo_id=repo,
                repo_type=DEFAULT_REPO_TYPE,
                private=True,
                token=token,
                exist_ok=True,
            )
        else:
            logger.info("[dry-run] Would create repo %s", repo)

    msg = commit_message or "sync: upload inference outputs"

    for raw_path in paths:
        local = _resolve(raw_path)

        if not local.exists():
            logger.warning("Path does not exist, skipping: %s", local)
            continue

        if local.is_file():
            # Single-file upload — path_in_repo mirrors results/... structure
            rel = _file_path_in_repo(local)
            logger.info("push file: %s -> %s/%s", local.name, repo, rel)
            if not dry_run:
                try:
                    api.upload_file(
                        path_or_fileobj=str(local),
                        path_in_repo=rel,
                        repo_id=repo,
                        repo_type=DEFAULT_REPO_TYPE,
                        token=token,
                        commit_message=msg,
                    )
                    logger.info("  [ok] uploaded %s", rel)
                except Exception as exc:
                    logger.error("  [fail] %s — %s", rel, exc)
        else:
            # Directory upload — preserves sub-tree structure
            folder_in_repo = _folder_path_in_repo(local)
            logger.info(
                "push folder: %s -> %s/%s  (allow=%s)",
                local, repo, folder_in_repo or "(root)", _UPLOAD_ALLOW,
            )
            if not dry_run:
                try:
                    url = api.upload_folder(
                        folder_path=str(local),
                        path_in_repo=folder_in_repo,
                        repo_id=repo,
                        repo_type=DEFAULT_REPO_TYPE,
                        token=token,
                        allow_patterns=_UPLOAD_ALLOW,
                        ignore_patterns=_UPLOAD_IGNORE,
                        commit_message=msg,
                    )
                    logger.info("  [ok] folder uploaded -> %s", url)
                except Exception as exc:
                    logger.error("  [fail] folder %s — %s", local, exc)
            else:
                # Show what would be uploaded
                files = sorted(local.rglob("*"))
                matched = _filter_files(files, local)
                logger.info("  [dry-run] %d files would be uploaded:", len(matched))
                for f in matched[:20]:
                    logger.info("    %s", f.relative_to(local))
                if len(matched) > 20:
                    logger.info("    ... and %d more", len(matched) - 20)


# ---------------------------------------------------------------------------
# Pull
# ---------------------------------------------------------------------------


def pull(
    paths: list[str],
    repo: str,
    token: Optional[str],
    dry_run: bool,
) -> None:
    """Download files from the HF repo into the local tree.

    Files are placed at their original local path (mirroring repo layout).
    If a file already exists locally and differs from HF, the remote version wins.
    """
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()

    # List all files in the repo
    try:
        all_remote = list(api.list_repo_files(
            repo_id=repo, repo_type=DEFAULT_REPO_TYPE, token=token,
        ))
    except Exception as exc:
        logger.error("Cannot list repo %s: %s", repo, exc)
        sys.exit(1)

    logger.info("Remote repo has %d files.", len(all_remote))

    # Filter to only the requested paths (prefix match)
    resolved_prefixes = [
        _folder_path_in_repo(_resolve(p)) for p in paths
    ]
    # An empty prefix means "everything under project root" — keep all
    if resolved_prefixes and any(p == "" for p in resolved_prefixes):
        resolved_prefixes = [""]

    def _matches(repo_path: str) -> bool:
        if not resolved_prefixes:
            return True
        for prefix in resolved_prefixes:
            if not prefix or repo_path.startswith(prefix):
                return True
        return False

    to_download = [p for p in all_remote if _matches(p)]
    logger.info("%d files match the requested paths.", len(to_download))

    if dry_run:
        logger.info("[dry-run] Would download:")
        for p in to_download[:20]:
            logger.info("  %s", p)
        if len(to_download) > 20:
            logger.info("  ... and %d more", len(to_download) - 20)
        return

    n_ok = n_skip = n_fail = 0
    for repo_path in to_download:
        local_dest = _project_path_from_repo_path(repo_path)
        try:
            tmp = hf_hub_download(
                repo_id=repo,
                filename=repo_path,
                repo_type=DEFAULT_REPO_TYPE,
                token=token,
            )
            local_dest.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(tmp, local_dest)
            n_ok += 1
            logger.info("  pulled: %s", repo_path)
        except Exception as exc:
            n_fail += 1
            logger.warning("  [fail] %s — %s", repo_path, exc)

    logger.info(
        "Pull complete: %d pulled, %d skipped, %d failed.",
        n_ok, n_skip, n_fail,
    )


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _file_path_in_repo(local: Path) -> str:
    """Map an absolute local path to its path-in-repo string.

    We find the nearest ancestor that is 'results' or 'data' and use that
    subtree. If neither is found we use the filename only.
    """
    parts = local.resolve().parts
    for anchor in ("results", "data"):
        for i, part in enumerate(parts):
            if part == anchor:
                return "/".join(parts[i:])
    return local.name


def _folder_path_in_repo(local: Path) -> str:
    """Map a local folder to its path-in-repo prefix (may be empty string)."""
    parts = local.resolve().parts
    for anchor in ("results", "data"):
        for i, part in enumerate(parts):
            if part == anchor:
                return "/".join(parts[i:])
    # The path is the project root itself (or somewhere above anchor folders)
    return ""


def _project_path_from_repo_path(repo_path: str) -> Path:
    """Convert a repo-relative path back to an absolute local path."""
    return _PROJECT_ROOT / repo_path


def _filter_files(files: list[Path], base: Path) -> list[Path]:
    """Apply allow / ignore patterns (simple suffix-based approximation for dry-run)."""
    allowed_suffixes = {".jsonl", ".json", ".txt", ".md", ".log"}
    ignored_names    = {"__pycache__", ".git"}
    ignored_suffixes = {".py", ".pyc", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"}

    result = []
    for f in files:
        if not f.is_file():
            continue
        if any(part in ignored_names for part in f.parts):
            continue
        if f.suffix.lower() in ignored_suffixes:
            continue
        if f.suffix.lower() in allowed_suffixes:
            result.append(f)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sync inference outputs to/from a HuggingFace dataset repo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "mode",
        choices=["push", "pull", "sync"],
        help=(
            "push: upload local -> HF  |  "
            "pull: download HF -> local  |  "
            "sync: pull then push"
        ),
    )
    p.add_argument(
        "--paths", nargs="+", default=_DEFAULT_PATHS, metavar="PATH",
        help=(
            "Local paths (files or directories) to sync. "
            f"Default: {_DEFAULT_PATHS}. "
            "Example: --paths results/phase2 results/phase3 data/ocr-results/qaari-results"
        ),
    )
    p.add_argument(
        "--repo", default=DEFAULT_REPO,
        help=f"HuggingFace dataset repo id (default: {DEFAULT_REPO}).",
    )
    p.add_argument(
        "--token", default=None,
        help="HF access token. Falls back to HF_TOKEN env var.",
    )
    p.add_argument(
        "--message", default=None, metavar="MSG",
        help="Commit message for push (default: 'sync: upload inference outputs').",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be done without making any changes.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _require_huggingface_hub()

    token = _get_token(args.token)

    sep = "=" * 60
    logger.info(sep)
    logger.info("HF Sync  mode=%s  repo=%s", args.mode, args.repo)
    logger.info("Paths   : %s", args.paths)
    if args.dry_run:
        logger.info("DRY RUN — no files will be read or written")
    logger.info(sep)

    if args.mode in ("pull", "sync"):
        logger.info("--- PULL ---")
        pull(args.paths, args.repo, token, dry_run=args.dry_run)

    if args.mode in ("push", "sync"):
        logger.info("--- PUSH ---")
        push(args.paths, args.repo, token, dry_run=args.dry_run, commit_message=args.message)

    logger.info(sep)
    logger.info("Done.")
    logger.info(sep)


if __name__ == "__main__":
    main()
