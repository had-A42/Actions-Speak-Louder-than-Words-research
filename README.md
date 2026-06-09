# Actions Speak Louder than Words Research

This repository contains experiments with neural recommender models for sequential user actions. The main source code is in `src/`, and all experiments are run from Jupyter notebooks.

## Project Structure

- `src/data/` - dataset loading, temporal train/test split, user and item reindexing.
- `src/evaluation/` - recommendation metrics and normalized entropy for action prediction tasks.
- `src/HSTU/` - HSTU and typed HSTU implementations, dataloaders, and training/evaluation pipelines.
- `src/DLRM/` - ranking models for DLRM experiments.
- `src/HSTU/notebooks/` - main HSTU experiments.
- `src/examples/dlrm_models.ipynb` - example notebook for running DLRM models.
- `tmp/` - local dataset files.
- `hf_cache/` - Hugging Face dataset cache.

## Installation

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install jupyter ipykernel
python -m ipykernel install --user --name actions-speak --display-name "Actions Speak"
```

If you use CUDA, install the `torch` version that matches your CUDA setup before running the notebooks.

## Data

Datasets can be downloaded from the shared Google Drive folder:

[https://drive.google.com/drive/folders/18xyk7ul7MaSrUXxFpGO6-jWIho6wk0D_](https://drive.google.com/drive/folders/18xyk7ul7MaSrUXxFpGO6-jWIho6wk0D_)

Each notebook has a `config` dictionary in the first cells. Be careful to set the correct dataset path in `interactions_path` before running an experiment; several notebooks use relative paths such as `../../../tmp/...`, which only work when the dataset files are placed exactly where the notebook expects them.

Dataset mapping by experiment:

- `src/HSTU/notebooks/ml-20m-hstu.ipynb` - MovieLens 20M, expected path in config: `../../../tmp/ml-20m.zip`.
- `src/HSTU/notebooks/amazon-reviews-hstu.ipynb` - Amazon Books ratings, expected path in config: `../../../tmp/ratings_Books.csv.gz`.
- `src/HSTU/notebooks/yambda-retrieval-hstu.ipynb` - Yambda retrieval subset, expected path in config: `../../../tmp/yambda-10m.parquet`.
- `src/HSTU/notebooks/yambda-retrieval-hstu-stochastic-length.ipynb` - Yambda retrieval subset, expected path in config currently points to a Kaggle input path; update it to the local `yambda-10m.parquet` path, for example `../../../tmp/yambda-10m.parquet`.
- `src/HSTU/notebooks/hstu_test_typed.ipynb` - Yambda lag features with action labels. The notebook uses `load_yambda_lag(interactions_path=None, ...)`, so it downloads `listens.parquet` and feature mappings through `huggingface_hub` unless you set explicit local paths in the config.
- `src/examples/dlrm_models.ipynb` - Yambda lag features with dense/sparse item features. The notebook downloads `listens.parquet` through `huggingface_hub`; use the local dataset path instead if you want to run fully from downloaded files.

## Running Experiments

All experiments are run in notebooks. The most important setup step is checking the notebook `config` before execution: dataset paths, `max_seq_len`, `test_quantile`, model hyperparameters, and device settings are defined directly in notebook cells.

Main notebooks:

- `src/HSTU/notebooks/ml-20m-hstu.ipynb` - HSTU on MovieLens 20M.
- `src/HSTU/notebooks/amazon-reviews-hstu.ipynb` - HSTU on Amazon Books.
- `src/HSTU/notebooks/yambda-retrieval-hstu.ipynb` - HSTU on Yambda retrieval.
- `src/HSTU/notebooks/yambda-retrieval-hstu-stochastic-length.ipynb` - HSTU on Yambda with stochastic length sampling.
- `src/HSTU/notebooks/hstu_test_typed.ipynb` - typed HSTU on Yambda lag features with action label prediction.
- `src/examples/dlrm_models.ipynb` - DLRM/ranker models on Yambda lag features.
- `hstu.ipynb` - early exploratory notebook with an HSTU prototype.

## Notebook Config Checklist

Before running an experiment, check the notebook config carefully:

- `interactions_path` points to the correct local dataset file.
- `max_seq_len` matches the experiment setup.
- `test_quantile` matches the intended temporal split.
- `HSTUExperimentConfig` or `TypedHSTUExperimentConfig` contains the intended model hyperparameters.
- `device` is set correctly for the available hardware.

Models use `cuda` automatically when `torch.cuda.is_available()` returns `True`; otherwise, they run on CPU.

## Metrics

Standard HSTU experiments compute retrieval metrics from `src/evaluation/metrics.py` over top-k recommendations. Typed HSTU computes normalized entropy (`ne_e_task`, `ne_c_task`) for action labels.
