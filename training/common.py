"""Shared data integrity checks; this module also works without ML packages."""
import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def read_config(path=None):
    path = Path(path) if path else ROOT / "training/config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    name = config["run_name"]
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("run_name must be a single directory name")
    if not 0 < config["threshold"] < 1:
        raise ValueError("threshold must be between zero and one")
    return config


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_data(config):
    paths = [ROOT / config[k] for k in ("data_file", "split_file")]
    tables = []
    for path in paths:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        table = {row["instance_id"]: row for row in rows}
        if len(table) != len(rows):
            raise ValueError(f"Duplicate instance IDs in {path}")
        tables.append(table)
    data, manifest = tables
    if data.keys() != manifest.keys():
        raise ValueError("Data and split manifest must contain exactly the same IDs")
    partitions = {key: [] for key in ("train", "validation", "test")}
    source_splits, text_splits = {}, {}
    for key, row in data.items():
        assignment = manifest[key]
        split = assignment["split"]
        if split not in partitions or row["label"] not in {"0", "1"}:
            raise ValueError(f"Invalid label or split: {key}")
        if any(row[k] != assignment[k] for k in ("label", "source_file")):
            raise ValueError(f"Inconsistent manifest metadata: {key}")
        text = " ".join(row["requirement_text"].casefold().split())
        if not text or not row["source_file"].strip():
            raise ValueError(f"Missing text or source: {key}")
        for mapping, value in ((source_splits, row["source_file"]), (text_splits, text)):
            if value in mapping and mapping[value] != split:
                raise ValueError(f"Source or duplicate-text leakage: {key}")
            mapping[value] = split
        partitions[split].append({**row, "label": int(row["label"]), "split": split})
    if any(not rows or {r["label"] for r in rows} != {0, 1} for rows in partitions.values()):
        raise ValueError("Each partition must be nonempty and contain both labels")
    return partitions


class TextDataset:
    def __init__(self, rows, tokenizer, max_length):
        self.items = []
        self.truncated = 0
        for row in rows:
            encoded = tokenizer(row["requirement_text"], truncation=False, verbose=False)
            self.truncated += len(encoded["input_ids"]) > max_length
            encoded = tokenizer(row["requirement_text"], truncation=True, max_length=max_length)
            encoded["labels"] = row["label"]
            self.items.append(encoded)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def classification_metrics(labels, predictions):
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
    return {"accuracy": float(accuracy_score(labels, predictions)),
            "f1": float(f1_score(labels, predictions, zero_division=0)),
            "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
            "precision": float(precision_score(labels, predictions, zero_division=0)),
            "recall": float(recall_score(labels, predictions, zero_division=0)),
            "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist()}


def require_cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run training/setup.ps1 and check your NVIDIA driver.")
    # Execute a kernel, rather than relying only on driver detection.
    (torch.ones((32, 32), device="cuda") @ torch.ones((32, 32), device="cuda")).sum().item()
    return torch.cuda.is_bf16_supported()
