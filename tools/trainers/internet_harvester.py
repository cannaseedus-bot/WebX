#!/usr/bin/env python3
"""internet_harvester.py — ASX Prime OS internet data harvesting training pipeline.

Fetches JSONL batches written by data-harvester.mjs and converts them into
incremental training records for MM-CODER (GPT-2 coder fine-tune).

Usage:
    python internet_harvester.py --batch-dir data/harvested [--model-out E:\\models\\GPT2\\med-GPT]

Output (stdout, last line): JSON summary for learning-engine.mjs to parse.

muPY style — no framework, minimal imports.
"""

import sys
import json
import pathlib
import urllib.request
import urllib.error
import time
import argparse

# ── Config ────────────────────────────────────────────────────────────────────

CODER_SOURCES = [
    # GitHub trending — coder training gold
    "https://api.github.com/search/repositories?q=language:python+stars:>100&sort=stars&per_page=5",
    # StackOverflow recent python questions
    "https://api.stackexchange.com/2.3/questions?order=desc&sort=activity&tagged=python&site=stackoverflow&pagesize=5",
    # arXiv CS.LG latest
    "https://export.arxiv.org/api/query?search_query=cat:cs.LG&max_results=3&sortBy=submittedDate",
]

HEADERS = {
    "User-Agent": "KUHUL-MM-CODER-Harvester/1.0 (training-data-collection; respectful)",
    "Accept": "application/json",
}

RATE_DELAY = 1.2   # seconds between requests


# ── Fetch ─────────────────────────────────────────────────────────────────────

def fetch(url, timeout=10):
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}") from e
    except Exception as e:
        raise RuntimeError(str(e)) from e


# ── Extraction ────────────────────────────────────────────────────────────────

def extract_coder_pairs(source_id, raw):
    """Convert raw API response to (prompt, completion) training pairs."""
    pairs = []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # arXiv returns XML — just store raw chunk as a completion target
        return [{"prompt": f"Explain research from {source_id}:", "completion": raw[:1024]}]

    if source_id == "github":
        for item in data.get("items", []):
            name = item.get("full_name", "")
            desc = item.get("description") or ""
            lang = item.get("language") or "unknown"
            pairs.append({
                "prompt":     f"What does the GitHub repo '{name}' do?",
                "completion": f"{desc} (Language: {lang})",
                "source":     source_id,
                "fold":       "⟁COMPUTE_FOLD⟁",
            })

    elif source_id == "stackoverflow":
        for item in data.get("items", []):
            title = item.get("title", "")
            body  = (item.get("body") or "")[:512]
            if title:
                pairs.append({
                    "prompt":     title,
                    "completion": body,
                    "source":     source_id,
                    "fold":       "⟁COMPUTE_FOLD⟁",
                })

    else:
        pairs.append({"prompt": f"Data from {source_id}:", "completion": raw[:1024], "source": source_id})

    return pairs


# ── Batch ingest ──────────────────────────────────────────────────────────────

def load_batch_dir(batch_dir):
    """Read all JSONL files written by data-harvester.mjs."""
    records = []
    p = pathlib.Path(batch_dir)
    for f in sorted(p.glob("batch_*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def harvest_live():
    """Fallback: fetch directly if no batch files exist."""
    pairs = []
    for url in CODER_SOURCES:
        src_id = url.split("/")[2].split(".")[1]  # rough domain slug
        try:
            raw = fetch(url)
            pairs.extend(extract_coder_pairs(src_id, raw))
            print(f"[HARVESTER] Sek {src_id} → {len(raw)}b", flush=True)
        except RuntimeError as e:
            print(f"[HARVESTER] Xul {src_id}: {e}", file=sys.stderr, flush=True)
        time.sleep(RATE_DELAY)
    return pairs


# ── Training write-out ────────────────────────────────────────────────────────

def write_training_jsonl(pairs, out_path):
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    return out_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-dir", default=None)
    ap.add_argument("--model-out", default=r"E:\models\GPT2\med-GPT")
    ap.add_argument("--out-jsonl", default=None)
    args = ap.parse_args()

    # 1 — ingest from harvester batch files or fetch live
    if args.batch_dir:
        records = load_batch_dir(args.batch_dir)
        pairs = []
        for rec in records:
            pairs.extend(extract_coder_pairs(rec.get("source_id", "unknown"), rec.get("raw", "")))
    else:
        pairs = harvest_live()

    if not pairs:
        print(json.dumps({"error": "no_data", "pairs": 0}))
        return

    # 2 — write training JSONL
    ts = int(time.time())
    out = pathlib.Path(args.out_jsonl) if args.out_jsonl else \
          pathlib.Path(args.model_out) / "training" / f"harvest_{ts}.jsonl"
    written = write_training_jsonl(pairs, out)

    # 3 — summary for learning-engine.mjs (must be last stdout line, valid JSON)
    summary = {
        "pairs":      len(pairs),
        "training_file": str(written),
        "model_path": args.model_out,
        "fold":       "⟁COMPUTE_FOLD⟁",
        "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
