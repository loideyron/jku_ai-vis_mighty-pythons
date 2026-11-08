"""Fine-tune on train; select by validation macro-F1. Never evaluate test here."""
import argparse
import csv
import hashlib
import importlib.metadata
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from common import ROOT, TextDataset, classification_metrics, load_data, read_config, require_cuda, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Alternative JSON configuration")
    parser.add_argument("--run-name", help="Override output name; existing runs are never overwritten")
    args = parser.parse_args()
    config = read_config(args.config)
    if args.run_name:
        if Path(args.run_name).name != args.run_name or args.run_name in {".", ".."}:
            raise ValueError("Invalid run name")
        config["run_name"] = args.run_name
    model_dir = ROOT / "models" / config["run_name"]
    results_dir = ROOT / "results" / config["run_name"]
    if model_dir.exists() or results_dir.exists():
        raise FileExistsError("Run already exists. Use --run-name with a new name; do not overwrite a shared model.")
    parts = load_data(config)

    import numpy as np
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments, set_seed

    bf16 = require_cuda()
    set_seed(config["seed"])
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], revision=config["model_revision"])
    selected = {}
    for split in ("train", "validation"):
        rows = parts[split]
        selected[split] = TextDataset(rows, tokenizer, config["max_length"])
        print(f"{split}: {len(selected[split])} rows; {selected[split].truncated} truncated at {config['max_length']} tokens")
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"], revision=config["model_revision"], num_labels=2,
        id2label={0: "unambiguous", 1: "ambiguous"}, label2id={"unambiguous": 0, "ambiguous": 1},
        use_safetensors=True)
    # Full fine-tuning: backbone and new classification head both remain trainable.
    def metrics(prediction):
        logits = prediction.predictions
        scores = np.exp(logits - logits.max(axis=1, keepdims=True))
        scores /= scores.sum(axis=1, keepdims=True)
        result = classification_metrics(prediction.label_ids, (scores[:, 1] >= config["threshold"]).astype(int))
        result.pop("confusion_matrix")
        return result

    training_args = TrainingArguments(
        output_dir=str(model_dir / "checkpoints"),
        num_train_epochs=config["epochs"],
        learning_rate=config["learning_rate"], weight_decay=config["weight_decay"],
        warmup_ratio=config["warmup_ratio"],
        per_device_train_batch_size=config["train_batch_size"],
        per_device_eval_batch_size=config["eval_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        gradient_checkpointing=config["gradient_checkpointing"],
        bf16=bf16, fp16=not bf16, tf32=False, optim="adamw_torch",
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1, save_only_model=True,
        load_best_model_at_end=True, metric_for_best_model="macro_f1", greater_is_better=True,
        seed=config["seed"], data_seed=config["seed"], logging_steps=10,
        dataloader_num_workers=0, report_to="none", save_safetensors=True)
    trainer = Trainer(model=model, args=training_args, train_dataset=selected["train"],
                      eval_dataset=selected["validation"], processing_class=tokenizer,
                      data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8), compute_metrics=metrics)
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        git_dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        git_commit, git_dirty = None, None
    metadata = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "data_sha256": {key: hashlib.sha256((ROOT / config[key]).read_bytes()).hexdigest()
                        for key in ("data_file", "split_file")},
        "split_summary": {name: {"rows": len(rows),
                                 "sources": len({r["source_file"] for r in rows}),
                                 "label_0": sum(r["label"] == 0 for r in rows),
                                 "label_1": sum(r["label"] == 1 for r in rows)}
                          for name, rows in parts.items()},
        "git_commit": git_commit, "git_dirty": git_dirty,
        "python_version": platform.python_version(),
        "packages": {name: importlib.metadata.version(name)
                     for name in ("torch", "transformers", "accelerate", "numpy", "scikit-learn")},
        "gpu": torch.cuda.get_device_name(0),
        "cuda_runtime": torch.version.cuda,
        "precision": "bf16" if bf16 else "fp16",
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "truncated": {name: dataset.truncated for name, dataset in selected.items()},
        "selection_metric": "validation macro-F1",
    }
    metadata_path = results_dir / "run_metadata.json"
    write_json(metadata_path, metadata)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        trainer.train()
        validation = trainer.evaluate()
        trainer.save_model(str(model_dir))
        tokenizer.save_pretrained(model_dir)
        write_json(model_dir / "training_config.json", config)
    except (Exception, KeyboardInterrupt) as error:
        metadata.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                        error=f"{type(error).__name__}: {error}",
                        finished_utc=datetime.now(timezone.utc).isoformat(),
                        elapsed_seconds=time.perf_counter() - started,
                        peak_allocated_vram_gib=torch.cuda.max_memory_allocated() / 2**30)
        write_json(metadata_path, metadata)
        raise
    write_json(results_dir / "training_summary.json", {
        "validation_metrics": validation,
        "best_validation_macro_f1": trainer.state.best_metric,
        "truncated": {s: d.truncated for s, d in selected.items()}})
    history = trainer.state.log_history
    fields = sorted({key for entry in history for key in entry})
    with (results_dir / "training_history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)
    metadata.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat(),
                    elapsed_seconds=time.perf_counter() - started,
                    best_checkpoint=str(Path(trainer.state.best_model_checkpoint).relative_to(ROOT)),
                    best_validation_macro_f1=trainer.state.best_metric,
                    validation_metrics=validation,
                    peak_allocated_vram_gib=torch.cuda.max_memory_allocated() / 2**30)
    write_json(metadata_path, metadata)
    print(f"Saved selected model: {model_dir}\nNo test predictions have been generated.")


if __name__ == "__main__":
    main()
