# Dataset Adapters

This package owns train/eval views over canonical artifacts emitted by `src.data`.

- `dataset.py`: legacy jagged torch dataset kept for current experiments.
- `hstu_dataset.py`: dense padded HSTU/SASRec-style adapter.
- `classical_dataset.py`: flat train/eval interactions for classical baselines.
- `splits.py`: sequence-level split policies over already preprocessed sequences.

TODO:

- Add sparse matrix builders on top of `classical_dataset.py` for Polara/EASE/ALS.
- Keep model-specific tensor contracts here instead of adding them to `src.data`.
