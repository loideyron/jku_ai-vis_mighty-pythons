"""Generate held-out predictions using a saved, validation-selected checkpoint."""
import argparse
import csv
import json
from pathlib import Path

from common import ROOT, TextDataset, classification_metrics, load_data, require_cuda, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="models/roberta-pure-seed42")
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--cpu", action="store_true", help="Allow prediction on computers without CUDA")
    args = parser.parse_args()
    model_dir = Path(args.model_dir)
    if not model_dir.is_absolute():
        model_dir = ROOT / model_dir
    config = json.loads((model_dir / "training_config.json").read_text(encoding="utf-8"))
    rows = load_data(config)[args.split]
    out = ROOT / "results" / config["run_name"]
    prediction_path = out / f"{args.split}_predictions.csv"
    if prediction_path.exists():
        raise FileExistsError(f"Preserving existing predictions: {prediction_path}")
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
    if not args.cpu:
        require_cuda()
    device = torch.device("cpu" if args.cpu else "cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True).to(device)
    model.eval()
    dataset = TextDataset(rows, tokenizer, config["max_length"])
    loader = DataLoader(dataset, batch_size=config["eval_batch_size"], shuffle=False,
                        collate_fn=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8))
    scores = []
    with torch.inference_mode():
        for batch in loader:
            batch.pop("labels")
            logits = model(**{k: v.to(device) for k, v in batch.items()}).logits
            scores.extend(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().tolist())
    predictions = [int(p >= config["threshold"]) for p in scores]
    out.mkdir(parents=True, exist_ok=True)
    fields = ["instance_id", "source_file", "requirement_text", "true_label", "predicted_label", "probability_ambiguous", "reasons", "split", "was_truncated"]
    with prediction_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row, score, prediction in zip(rows, scores, predictions):
            length = len(tokenizer(row["requirement_text"], truncation=False, verbose=False)["input_ids"])
            writer.writerow({"instance_id": row["instance_id"], "source_file": row["source_file"],
                             "requirement_text": row["requirement_text"], "true_label": row["label"],
                             "predicted_label": prediction, "probability_ambiguous": score,
                             "reasons": row["reasons"], "split": args.split, "was_truncated": length > config["max_length"]})
    metrics = classification_metrics([r["label"] for r in rows], predictions)
    metrics.update(rows=len(rows), truncated=dataset.truncated, threshold=config["threshold"],
                   confusion_matrix_convention="rows=true, columns=predicted; labels=[0,1]")
    write_json(out / f"{args.split}_metrics.json", metrics)
    print(f"Saved {len(rows)} predictions: {prediction_path}")


if __name__ == "__main__":
    main()
