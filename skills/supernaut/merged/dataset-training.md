---
name: dataset-training
description: Manage dataset discovery, manifest generation, and preparation for LoRA/LoRA-like training runs using the project corpus.
---

# Dataset Training Skill

Use this skill whenever you need to build, inspect, or refine a dataset for training a LoRA-style or prompt-engine model.

## When to Use

- You are gathering tokens/examples from the repository `data/` directory (chat logs, prompts, diagrams, RLHF traces).
- You want a manifest that reports file sizes, line counts, sample rows, and rough token estimates before training.
- You need to compare dataset candidates, compute train/validation splits, or generate feeds for `pipx`/LoRA runs.

## What This Skill Adds

1. **Dataset manifest generation** via `skills/dataset-training/scripts/generate_dataset_manifest.js`. It scans `data/*.jsonl`, counts lines, captures sample records, and optionally writes a manifest to `artifacts/dataset-manifest.json`.
2. **Structured naming guidance** so you can describe a set of datasets as training, evaluation, or RLHF sources before handing them to a trainer.
3. **Preparation checklist** covering metadata, splits, filtering downstream, and bucket paths for LoRA/QLoRA training.

## Workflow

1. Run `DATA_DIR=C:\\public_html\\data node skills/dataset-training/scripts/generate_dataset_manifest.js --out artifacts/dataset-manifest.json` to capture the dataset landscape. The script honors `DATA_DIR` when your corpus sits outside the repo.
2. Use the generated manifest to pick the model (fast vs deep) and assign each file to train/validation/test splits.
3. Apply filters (e.g., `--pattern rhy`) to limit the dataset to the code/abstract sections you want to reinforce.
4. Feed the manifest into your training flow (`scripts/forge-model-binary.js`, `scripts/gsnr-train.py`, etc.) so the trainer can reference cleaned, documented sources.

## Tips

- Keep `data/*.jsonl` organized; prefer descriptive filenames (e.g., `chat-dolphin.jsonl`, `prompts.xml`).
- If you need more than line counts, add derived fields to the manifest using `scripts/gsnr-training` helpers in `src/gsnr_training`.
- For very large datasets, add a `--sample` flag to `generate_dataset_manifest.js` (see the script comments) to avoid scanning every row in every pass.
