#!/usr/bin/env python3
"""
Phase 3: SMGM-16 Training Pipeline
Trains the Mixture-of-Experts model on combined datasets.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, IterableDataset
import sys
import time

# Import model from smgm16.py
sys.path.insert(0, str(Path(__file__).parent.parent))
from smgm16 import SMGM16, training_step

try:
    from tokenizers import Tokenizer
except Exception:
    Tokenizer = None


class TrainingDataset(Dataset):
    """PyTorch Dataset wrapper."""
    
    def __init__(self, jsonl_path: Path, tokenizer=None, max_len: int = 512):
        self.data = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        self.data.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        record = self.data[idx]
        prompt_text, response_text = self._extract_chat(record)
        input_ids, labels = self._encode_chat_example(prompt_text, response_text)

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'output_ids': torch.tensor(labels, dtype=torch.long),
            'type': record.get('type', ''),
            'domain': record.get('domain', '')
        }

    def _extract_chat(self, record):
        messages = record.get("messages")
        if isinstance(messages, list) and messages:
            prompt_parts = []
            response_parts = []
            assistant_seen = False
            for item in messages:
                if not isinstance(item, dict):
                    continue
                role = str(item.get("role") or "").strip().lower()
                content = str(item.get("content") or "").strip()
                if not content:
                    continue
                if role == "assistant":
                    assistant_seen = True
                    response_parts.append(content)
                elif role in {"system", "user"}:
                    prompt_parts.append(f"<|{role}|>\n{content}")
            if assistant_seen and response_parts:
                return "\n".join(prompt_parts) + "\n<|assistant|>\n", "\n\n".join(response_parts)

        prompt_text = str(record.get("input") or record.get("question") or record.get("prompt") or "").strip()
        response_text = str(record.get("output") or record.get("response") or record.get("chosen") or "").strip()
        if prompt_text and not prompt_text.startswith("<|"):
            prompt_text = f"<|user|>\n{prompt_text}\n<|assistant|>\n"
        return prompt_text, response_text

    def _encode_chat_example(self, prompt_text: str, response_text: str):
        eos_id = 1

        if self.tokenizer is not None:
            try:
                prompt_ids = list(self.tokenizer.encode(prompt_text).ids)
                response_ids = list(self.tokenizer.encode(response_text).ids)
                if hasattr(self.tokenizer, "token_to_id"):
                    eos_candidate = self.tokenizer.token_to_id("<|eos|>")
                    if eos_candidate is not None:
                        eos_id = int(eos_candidate)
            except Exception:
                prompt_ids = [ord(c) % 256 for c in prompt_text]
                response_ids = [ord(c) % 256 for c in response_text]
        else:
            prompt_ids = [ord(c) % 256 for c in prompt_text]
            response_ids = [ord(c) % 256 for c in response_text]

        seq = (prompt_ids + [eos_id] + response_ids + [eos_id])[: self.max_len + 1]
        if len(seq) < 2:
            seq = [eos_id, eos_id]

        input_ids = seq[:-1]
        labels = seq[1:]

        prompt_boundary = min(len(prompt_ids), len(labels))
        for i in range(prompt_boundary):
            labels[i] = -100

        if len(input_ids) > self.max_len:
            input_ids = input_ids[: self.max_len]
            labels = labels[: self.max_len]

        pad_len = self.max_len - len(input_ids)
        if pad_len > 0:
            input_ids += [0] * pad_len
            labels += [-100] * pad_len

        return input_ids, labels


class StreamingTrainingDataset(IterableDataset):
    """Stream JSONL rows one at a time to keep the working set bounded."""

    def __init__(self, jsonl_path: Path, tokenizer=None, max_len: int = 512):
        self.jsonl_path = Path(jsonl_path)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __iter__(self):
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                prompt_text, response_text = TrainingDataset._extract_chat(self, record)
                input_ids, labels = TrainingDataset._encode_chat_example(self, prompt_text, response_text)

                yield {
                    'input_ids': torch.tensor(input_ids, dtype=torch.long),
                    'output_ids': torch.tensor(labels, dtype=torch.long),
                    'type': record.get('type', ''),
                    'domain': record.get('domain', '')
                }


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_tokenizer(tokenizer_path: str | None):
    if not tokenizer_path:
        return None
    path = Path(tokenizer_path)
    if not path.exists() or Tokenizer is None:
        return None
    return Tokenizer.from_file(str(path))


def apply_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Apply CLI overrides to the loaded configuration."""
    if args.device:
        config["device"] = args.device
    if args.train_jsonl:
        config.setdefault("datasets", {})["train"] = args.train_jsonl
    if args.val_jsonl:
        config.setdefault("datasets", {})["val"] = args.val_jsonl
    if args.test_jsonl:
        config.setdefault("datasets", {})["test"] = args.test_jsonl
    if args.max_epochs is not None:
        config["max_epochs"] = args.max_epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.max_seq_length is not None:
        config["max_seq_length"] = args.max_seq_length
    if args.max_train_batches is not None:
        config["max_train_batches"] = args.max_train_batches
    if args.max_eval_batches is not None:
        config["max_eval_batches"] = args.max_eval_batches
    if args.checkpoint_dir is not None:
        config["checkpoint_dir"] = args.checkpoint_dir
    return config


def build_dataset(jsonl_path: Path, max_len: int, streaming: bool, tokenizer=None):
    if streaming:
        return StreamingTrainingDataset(jsonl_path, tokenizer=tokenizer, max_len=max_len)
    return TrainingDataset(jsonl_path, tokenizer=tokenizer, max_len=max_len)


def dataset_count(dataset) -> str:
    try:
        return f"{len(dataset)}"
    except Exception:
        return "streaming"


def masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """Cross-entropy that returns a zero scalar when a batch has no valid targets."""
    flat_targets = targets.reshape(-1)
    valid_mask = flat_targets != ignore_index
    if not torch.any(valid_mask):
        return torch.zeros((), device=logits.device, dtype=logits.dtype)
    flat_logits = logits.reshape(-1, logits.size(-1))
    return nn.functional.cross_entropy(
        flat_logits[valid_mask],
        flat_targets[valid_mask],
    )


def train_epoch(model, dataloader, optimizer, config, device, epoch):
    """Train one epoch."""
    model.train()
    total_loss = 0
    lambdas = config.get('loss_weights', {
        'task': 1.0,
        'balance': 0.1,
        'stage_balance': 0.05,
        'stage_entropy': 0.02
    })
    
    for batch_idx, batch in enumerate(dataloader):
        # Predict assistant tokens from the staged conversation tokens.
        x = batch['input_ids'].long().to(device)
        y = batch['output_ids'].long().to(device)
        
        # Forward pass with token inputs.
        loss = training_step(model, (x, y), optimizer, lambdas)
        total_loss += loss
        
        if (batch_idx + 1) % config['log_interval'] == 0:
            avg_loss = total_loss / (batch_idx + 1)
            print(f"  [Epoch {epoch}, Batch {batch_idx}] Loss: {avg_loss:.4f}")
        
        if (batch_idx + 1) >= int(config.get("max_train_batches", 100)):
            break

    avg_epoch_loss = total_loss / min(batch_idx + 1, int(config.get("max_train_batches", 100)))
    return avg_epoch_loss


def eval_epoch(model, dataloader, config, device, epoch):
    """Evaluate one epoch."""
    model.eval()
    total_loss = 0
    lambdas = config.get('loss_weights', {})
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            x = batch['input_ids'].long().to(device)
            y = batch['output_ids'].long().to(device)
            
            # Forward pass
            out, all_gates, all_stage_probs = model(x)
            loss = masked_cross_entropy(out, y, ignore_index=-100)
            total_loss += loss.item()
            
            if (batch_idx + 1) >= int(config.get("max_eval_batches", 50)):
                break

    avg_loss = total_loss / min(batch_idx + 1, int(config.get("max_eval_batches", 50)))
    return avg_loss


def save_checkpoint(model, checkpoint_dir: Path, name: str) -> Path:
    """Save a raw SMGM16 state dict and return the written path."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / name
    torch.save(model.state_dict(), checkpoint_path)
    return checkpoint_path


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description="SMGM-16 training pipeline.")
    parser.add_argument("--config", type=str, default=str(Path(__file__).parent / "config.yaml"))
    parser.add_argument("--train-jsonl", type=str, default=None)
    parser.add_argument("--val-jsonl", type=str, default=None)
    parser.add_argument("--test-jsonl", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--tokenizer-path", type=str, default=None)
    parser.add_argument("--base-checkpoint", type=str, default=None)
    parser.add_argument("--lora-rank", type=int, default=None)
    parser.add_argument("--lora-alpha", type=float, default=None)
    parser.add_argument("--lora-dropout", type=float, default=None)
    parser.add_argument("--train-layer-norms", action="store_true")
    parser.add_argument("--train-pi-time", action="store_true")
    args = parser.parse_args()

    print("="*70)
    print("PHASE 3: SMGM-16 Training Pipeline")
    print("="*70)
    
    # Load configuration
    config_path = Path(args.config)
    config = load_config(config_path)
    config = apply_overrides(config, args)
    print(f"\nLoaded config from {config_path.name}")
    
    # Setup device
    device = torch.device(config.get('device', 'cpu'))
    print(f"Using device: {device}")

    tokenizer_path = args.tokenizer_path or config.get("tokenizer_path")
    tokenizer = load_tokenizer(tokenizer_path)
    if tokenizer is not None:
        print(f"Using tokenizer: {tokenizer_path}")
    else:
        print("Tokenizer unavailable; falling back to byte-level char encoding.")
    
    # Create data loaders
    print("\nLoading datasets...")
    datasets_config = config['datasets']
    streaming_cfg = config.get("streaming", {})
    streaming_enabled = bool(streaming_cfg.get("enabled", False))
    train_dataset = build_dataset(
        Path(datasets_config['train']),
        max_len=config['max_seq_length'],
        streaming=streaming_enabled,
        tokenizer=tokenizer,
    )
    val_dataset = build_dataset(
        Path(datasets_config['val']),
        max_len=config['max_seq_length'],
        streaming=streaming_enabled,
        tokenizer=tokenizer,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=not streaming_enabled,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0
    )
    print(f"  Train samples: {dataset_count(train_dataset)}")
    print(f"  Val samples: {dataset_count(val_dataset)}")
    
    # Initialize model
    print("\nInitializing SMGM-16 model...")
    lora_rank = int(args.lora_rank if args.lora_rank is not None else config.get("lora_rank", 0))
    model = SMGM16(
        d_model=config['d_model'],
        layers=config['num_layers'],
        stage_dims=tuple(config['stage_dims']),
        k=config['k'],
        patch_dim=config['patch_dim'],
    ).to(device)
    lora_supported = hasattr(model, "enable_lora_training")
    if lora_rank > 0 and not lora_supported:
        print("  LoRA requested but local SMGM16 has no LoRA adapter hooks; using full trainable model.")

    base_checkpoint = args.base_checkpoint or config.get("base_checkpoint")
    if base_checkpoint:
        cp_path = Path(base_checkpoint)
        if cp_path.exists():
            state = torch.load(cp_path, map_location=device)
            try:
                model.load_state_dict(state, strict=(lora_rank == 0 and lora_supported))
            except RuntimeError as exc:
                print(f"  Strict checkpoint load failed; retrying non-strict load: {exc}")
                model.load_state_dict(state, strict=False)
            print(f"  Base checkpoint loaded from {cp_path.name}")
        else:
            print(f"  Base checkpoint not found: {cp_path}")

    if lora_rank > 0 and lora_supported:
        model.enable_lora_training(
            train_layer_norms=bool(args.train_layer_norms or config.get("train_layer_norms", False)),
            train_pi_time=bool(args.train_pi_time or config.get("train_pi_time", False)),
        )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Setup optimizer
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Training loop
    print("\nStarting training...")
    start_time = time.time()
    checkpoint_path = None
    checkpoint_dir = Path(config['checkpoint_dir'])
    
    for epoch in range(config['max_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['max_epochs']}")
        
        # Training
        train_loss = train_epoch(model, train_loader, optimizer, config, device, epoch + 1)
        print(f"  Train loss: {train_loss:.4f}")
        
        # Validation
        val_loss = eval_epoch(model, val_loader, config, device, epoch + 1)
        print(f"  Val loss:   {val_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 2 == 0:
            checkpoint_path = save_checkpoint(model, checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt")
            print(f"  Checkpoint saved to {checkpoint_path.name}")
    
    elapsed_time = time.time() - start_time
    print("\n" + "="*70)
    print(f"Training complete in {elapsed_time:.1f}s")
    final_epoch_path = save_checkpoint(model, checkpoint_dir, f"checkpoint_epoch_{config['max_epochs']}.pt")
    final_path = save_checkpoint(model, checkpoint_dir, "checkpoint_final.pt")
    if checkpoint_path is not None and checkpoint_path == final_epoch_path:
        print(f"Final epoch checkpoint refreshed: {final_epoch_path.name}")
    else:
        print(f"Final epoch checkpoint saved to {final_epoch_path.name}")
    print(f"Final checkpoint saved to {final_path.name}")
    print("="*70)


if __name__ == "__main__":
    main()
