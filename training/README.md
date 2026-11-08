# Fine-tune RoBERTa on PURE

Run these PowerShell commands from the team repository root.

## 1. Set up the environment (once per computer)

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/) and an NVIDIA GPU with a compatible driver.

```powershell
powershell -ExecutionPolicy Bypass -File training/setup.ps1
```

This installs Python 3.12 and dependencies in `.venv/`, including PyTorch with
CUDA 12.8 support. No separate CUDA Toolkit installation is required. The script
checks the GPU. Skip this step if the project environment is already installed.

## 2. Train

```powershell
.venv/Scripts/python.exe training/train.py
```

Settings are in `training/config.json`. Defaults: full fine-tuning of RoBERTa-base
and its classification head, 3 epochs, batch size 8, accumulation 2, max length 256.
The first run downloads the pretrained model. The best validation macro-F1 selects
the final checkpoint; test data is not used for training or model selection.

The selected model and tokenizer are saved in `models/roberta-pure-seed42/`.
Training results are in `results/roberta-pure-seed42/`.

Existing runs are protected. To start another run, use `--run-name roberta-v2`.

## 3. Generate test predictions

```powershell
.venv/Scripts/python.exe training/predict.py
```

Outputs: `results/roberta-pure-seed42/test_predictions.csv` and `test_metrics.json`.
For a different run, add `--model-dir models/roberta-v2`. Add `--cpu` to predict
without a GPU. Existing prediction files are not overwritten.
