# jku_ai-vis_mighty-pythons
JKU WS2026 AI and Visualization Course Assignment : Group Mighty Pythons

# Members
- K12147981
- K12148218
- K12111267
- K12434880

# Task 
Ambiguity Detection in Natural-Language Software Requirements

# Model 
RoBERTa @ https://huggingface.co/docs/transformers/model_doc/roberta

## Shared fine-tuning pipeline

See [training/README.md](training/README.md) for environment setup,
fine-tuning, and held-out predictions. The binary classifier trains
both RoBERTa-base and its classification head. Data and the source-disjoint PURE
seed-42 split are included under `data/`; annotation reasons are analysis metadata.
Large model files remain local under `models/` and are excluded from Git.

# DataSet 
- PURE (Dataset of Public Requirements Documents) - Reference Dataset, without Annotation:
  - https://zenodo.org/records/1414117
- Annotated PURE Dataset:
  - https://github.com/profzivkovic/software-requirements-ambiguity-detection-replication


