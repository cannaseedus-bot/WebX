#!/usr/bin/env python3
"""
fetch_oss_weights.py — Download GPT OSS weights from HuggingFace for micronaut models.

Usage:
    python fetch_oss_weights.py --repo microsoft/Phi-3-mini-4k-instruct --file model.gguf --out models/oss-cache/
    python fetch_oss_weights.py --repo Qwen/Qwen2.5-7B-Instruct-GGUF --file qwen2.5-7b-instruct-q4_k_m.gguf --out models/oss-cache/
    python fetch_oss_weights.py --list-catalog
    python fetch_oss_weights.py --info microsoft/Phi-3-mini-4k-instruct
"""

import argparse
import os
import sys
import json
import hashlib
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

# ─── OSS Weight Catalog ───────────────────────────────────────────────────────
# Tested HuggingFace repos with license info and recommended GGUF files.
# All listed models are MIT or Apache 2.0 unless noted.

OSS_CATALOG = {
    "microsoft/Phi-3-mini-4k-instruct": {
        "description": "Phi-3 Mini 4K — 3.8B, high quality for size",
        "license": "MIT",
        "recommended_file": "Phi-3-mini-4k-instruct-q4.gguf",
        "vocab_size": 32064,
        "num_layers": 32,
        "hidden_size": 3072,
        "quantization": "Q4_K_M",
        "context_length": 4096,
        "fold": "COMPUTE",
        "notes": "Great for code/math tasks. Fast inference.",
    },
    "microsoft/Phi-3-medium-4k-instruct": {
        "description": "Phi-3 Medium 4K — 14B, strong reasoning",
        "license": "MIT",
        "recommended_file": "Phi-3-medium-4k-instruct-q4.gguf",
        "vocab_size": 32064,
        "num_layers": 40,
        "hidden_size": 5120,
        "quantization": "Q4_K_M",
        "context_length": 4096,
        "fold": "COMPUTE",
        "notes": "Needs ~8GB VRAM. Use for planning/reasoning micronauts.",
    },
    "microsoft/phi-2": {
        "description": "Phi-2 — 2.7B, small but capable",
        "license": "MIT",
        "recommended_file": "phi-2.Q4_K_M.gguf",
        "vocab_size": 51200,
        "num_layers": 32,
        "hidden_size": 2560,
        "quantization": "Q4_K_M",
        "context_length": 2048,
        "fold": "COMPUTE",
        "notes": "Very fast. Good for quick classification/extraction tasks.",
    },
    "Qwen/Qwen2.5-7B-Instruct-GGUF": {
        "description": "Qwen 2.5 7B Instruct — strong multilingual",
        "license": "Apache 2.0",
        "recommended_file": "qwen2.5-7b-instruct-q4_k_m.gguf",
        "vocab_size": 152064,
        "num_layers": 28,
        "hidden_size": 3584,
        "quantization": "Q4_K_M",
        "context_length": 32768,
        "fold": "COMPUTE",
        "notes": "Good for code + multilingual. 32K context.",
    },
    "Qwen/Qwen2.5-3B-Instruct-GGUF": {
        "description": "Qwen 2.5 3B Instruct — small and fast",
        "license": "Apache 2.0",
        "recommended_file": "qwen2.5-3b-instruct-q4_k_m.gguf",
        "vocab_size": 152064,
        "num_layers": 36,
        "hidden_size": 2048,
        "quantization": "Q4_K_M",
        "context_length": 32768,
        "fold": "COMPUTE",
        "notes": "Very fast. Good for edge/low-memory micronauts.",
    },
    "mistralai/Mistral-7B-Instruct-v0.3": {
        "description": "Mistral 7B v0.3 — excellent instruction follower",
        "license": "Apache 2.0",
        "recommended_file": "Mistral-7B-Instruct-v0.3.Q4_K_M.gguf",
        "vocab_size": 32768,
        "num_layers": 32,
        "hidden_size": 4096,
        "quantization": "Q4_K_M",
        "context_length": 32768,
        "fold": "COMPUTE",
        "notes": "Sliding window attention. Strong at following structured instructions.",
    },
    "NousResearch/Hermes-3-Llama-3.1-8B-GGUF": {
        "description": "Hermes 3 Llama 3.1 8B — tool use, agentic",
        "license": "Llama 3.1 Community License",
        "recommended_file": "Hermes-3-Llama-3.1-8B.Q4_K_M.gguf",
        "vocab_size": 128256,
        "num_layers": 32,
        "hidden_size": 4096,
        "quantization": "Q4_K_M",
        "context_length": 131072,
        "fold": "PATTERN",
        "notes": "Best for agentic/tool-use micronauts. 128K context. Check Llama license.",
    },
    "google/gemma-2-9b-it-GGUF": {
        "description": "Gemma 2 9B IT — strong reasoning",
        "license": "Gemma Terms of Use",
        "recommended_file": "gemma-2-9b-it-Q4_K_M.gguf",
        "vocab_size": 256000,
        "num_layers": 42,
        "hidden_size": 3584,
        "quantization": "Q4_K_M",
        "context_length": 8192,
        "fold": "COMPUTE",
        "notes": "Check Gemma license terms. Strong math/reasoning.",
    },
    "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF": {
        "description": "Llama 3.1 8B Instruct (bartowski quants)",
        "license": "Llama 3.1 Community License",
        "recommended_file": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "vocab_size": 128256,
        "num_layers": 32,
        "hidden_size": 4096,
        "quantization": "Q4_K_M",
        "context_length": 131072,
        "fold": "COMPUTE",
        "notes": "Check Llama license. High-quality quants from bartowski.",
    },
    "lmstudio-community/DeepSeek-R1-Distill-Qwen-7B-GGUF": {
        "description": "DeepSeek R1 Distill 7B — strong reasoning/CoT",
        "license": "MIT",
        "recommended_file": "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf",
        "vocab_size": 152064,
        "num_layers": 28,
        "hidden_size": 3584,
        "quantization": "Q4_K_M",
        "context_length": 131072,
        "fold": "PATTERN",
        "notes": "Excellent for reasoning/chain-of-thought micronauts. MIT license.",
    },
}


def list_catalog():
    """Print the OSS weight catalog in a readable format."""
    print("\n=== OSS Weight Catalog ===\n")
    for repo, info in OSS_CATALOG.items():
        license_flag = "⚠" if info["license"] not in ("MIT", "Apache 2.0") else "✓"
        print(f"{license_flag} {repo}")
        print(f"   {info['description']}")
        print(f"   License: {info['license']}  |  File: {info['recommended_file']}")
        print(f"   Fold: {info['fold']}  |  Layers: {info['num_layers']}  |  Hidden: {info['hidden_size']}")
        print(f"   Notes: {info['notes']}")
        print()


def get_repo_info(repo: str) -> dict:
    """Return catalog entry for a repo."""
    if repo in OSS_CATALOG:
        info = OSS_CATALOG[repo].copy()
        info["repo"] = repo
        return info
    return {"repo": repo, "notes": "Not in catalog — check license manually."}


def hf_url(repo: str, filename: str) -> str:
    """Build the HuggingFace direct download URL."""
    return f"https://huggingface.co/{repo}/resolve/main/{filename}"


def sha256_file(path: Path) -> str:
    """Compute SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest: Path, show_progress: bool = True) -> Path:
    """Download a file with progress display."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")

    req = urllib.request.Request(url, headers={"User-Agent": "micronaut-model-factory/1.0"})

    try:
        with urllib.request.urlopen(req) as response:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 65536

            with open(tmp, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if show_progress and total:
                        pct = downloaded / total * 100
                        mb = downloaded / 1_048_576
                        total_mb = total / 1_048_576
                        print(f"\r  {pct:.1f}%  {mb:.1f} / {total_mb:.1f} MB", end="", flush=True)

        if show_progress:
            print()

        tmp.rename(dest)
        return dest

    except urllib.error.HTTPError as e:
        if tmp.exists():
            tmp.unlink()
        if e.code == 404:
            raise FileNotFoundError(
                f"File not found: {url}\n"
                f"Check that the file exists in the repo. Use --info <repo> to see recommended file."
            )
        raise RuntimeError(f"HTTP {e.code}: {e.reason}")
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def fetch_weights(
    repo: str,
    filename: Optional[str],
    out_dir: str,
    skip_existing: bool = True,
    verify: bool = False,
) -> dict:
    """
    Download OSS weights from HuggingFace.

    Returns dict with: path, hash, repo, filename, size_bytes
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Resolve filename
    if not filename:
        if repo in OSS_CATALOG:
            filename = OSS_CATALOG[repo]["recommended_file"]
            print(f"  Using recommended file: {filename}")
        else:
            raise ValueError(
                f"No filename specified and '{repo}' is not in the catalog.\n"
                f"Run --list-catalog to see available repos, or specify --file explicitly."
            )

    dest = out_path / filename
    url = hf_url(repo, filename)

    # Check for existing file
    if dest.exists() and skip_existing:
        size = dest.stat().st_size
        print(f"  Already exists: {dest}  ({size / 1_048_576:.1f} MB)")
        file_hash = sha256_file(dest)
        return {"path": str(dest), "hash": f"sha256:{file_hash}", "repo": repo, "filename": filename, "size_bytes": size}

    # Download
    print(f"  Downloading: {url}")
    print(f"  → {dest}")
    dest = download_file(url, dest)

    # Hash
    print("  Computing SHA-256...", end="", flush=True)
    file_hash = sha256_file(dest)
    print(f" {file_hash[:16]}...")

    size = dest.stat().st_size
    result = {
        "path": str(dest),
        "hash": f"sha256:{file_hash}",
        "repo": repo,
        "filename": filename,
        "size_bytes": size,
    }

    # Write sidecar metadata
    meta_path = dest.with_suffix(".meta.json")
    catalog_info = OSS_CATALOG.get(repo, {})
    meta = {**result, "catalog": catalog_info}
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata: {meta_path}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Fetch GPT OSS weights from HuggingFace for micronaut models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--repo", help="HuggingFace repo (e.g. microsoft/Phi-3-mini-4k-instruct)")
    parser.add_argument("--file", help="Filename to download (uses recommended if omitted)")
    parser.add_argument("--out", default="models/oss-cache/", help="Output directory (default: models/oss-cache/)")
    parser.add_argument("--list-catalog", action="store_true", help="List all tested OSS models")
    parser.add_argument("--info", metavar="REPO", help="Show catalog info for a specific repo")
    parser.add_argument("--no-skip", action="store_true", help="Re-download even if file exists")
    parser.add_argument("--json", action="store_true", help="Output result as JSON")

    args = parser.parse_args()

    if args.list_catalog:
        list_catalog()
        return 0

    if args.info:
        info = get_repo_info(args.info)
        print(json.dumps(info, indent=2))
        return 0

    if not args.repo:
        parser.error("--repo is required (or use --list-catalog / --info)")

    try:
        result = fetch_weights(
            repo=args.repo,
            filename=args.file,
            out_dir=args.out,
            skip_existing=not args.no_skip,
        )

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"\n✓ Ready: {result['path']}")
            print(f"  Hash:  {result['hash']}")
            print(f"  Size:  {result['size_bytes'] / 1_048_576:.1f} MB")
            print()
            # Show scaffold hint
            name = Path(result["filename"]).stem.replace("-", "_").replace(".", "_")
            print(f"  Scaffold a micronaut with these weights:")
            print(f"  python scaffold_micronaut.py new <ID> <Name> --backend local_gguf --weights {result['path']} --fold COMPUTE")

    except (FileNotFoundError, RuntimeError, ValueError) as e:
        print(f"\n[FAIL] {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
