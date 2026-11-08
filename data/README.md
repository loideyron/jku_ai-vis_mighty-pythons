# PURE data

Unchanged copies from [the annotated dataset repository](https://github.com/profzivkovic/software-requirements-ambiguity-detection-replication),
commit `b7e416532ba1cb7d92f1d413d8f5377e41a5a574`:

- `pure_clean.csv`: upstream `data/processed/pure_clean.csv`.
- `PURE_seed42.csv`: upstream `data/splits/intra/PURE_seed42.csv`.

Join by `instance_id`. The split contains **4,578 training, 648 validation and
1,308 test requirements**, with different source documents in each partition.
The loader checks IDs, labels and source/text overlap automatically.

Model input: `requirement_text`. Target: `label` (0 = unambiguous, 1 = ambiguous
under the dataset's definition). `reasons` is annotation metadata for analysis;
do not feed it to the classifier. Annotation quality has not been independently verified.

Original corpus: Ferrari, Spagnolo and Gnesi (2017), [PURE](https://zenodo.org/records/1414117).
The upstream repository's license is preserved in `UPSTREAM_LICENSE`;
original source documents retain their applicable rights.
