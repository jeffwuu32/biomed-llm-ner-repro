# biomed-llm-ner-repro

## About

Reproduction repository for an LLM-based NER pipeline in biomedical fields. It is generally outperformed by conventional BERT token classifiers. However, we still decided to publish this repo for expository purposes.


## Overview

The `.ipynb` notebook loads a published `google/gemma-4-E2B-it` LoRA adapter from the Hugging
Face Hub and reproduces published results (see "Adapter Releases"). Currently covers the following datasets:

- [BC5CDR](https://pubmed.ncbi.nlm.nih.gov/27161011/): Chemical or Disease
- [BioRED](https://academic.oup.com/bib/article/23/5/bbac282/6645993): Gene, Chemical, Disease, Variant, Species, or CellLine


## Requirements

To run the notebook, you will need:

- An internet connection (no auth required)
- ~10 GB System RAM
- ~12 GB GPU RAM


## Setup

Only relevant if running locally -- Colab takes care of everything here.

1. Install NVIDIA drivers and a CUDA build of PyTorch.
2. Python 3.12+ is a safe baseline.
3. Run `python -m venv .venv` to create a virtual environment, then activate it:
    - Windows: `.venv\Scripts\activate`
    - Linux: `source .venv/bin/activate`
4. Install Jupyter -- `pip install jupyterlab && jupyter lab`, or VS Code's notebook extension -- and open `eval.ipynb` there.


## How to Run

1. In `eval.ipynb`, configure "Free variables". Leave "Control variables" alone if you wish to reproduce published results.
2. Run everything. The notebook downloads the raw corpus itself and builds everything else in-session.
3. Results are saved to `eval_results_<dataset>_<timestamp>.json`.


## Adapter Releases

See an adapter's model card for its reported result metrics.

| dataset | `EVAL_ADAPTER_ID` | `HUB_BRANCH` | as of |
|---|---|---|---|
| bc5cdr | [`jeffwuu32/bc5cdr-ner-gemma-4-E2B-it`](https://huggingface.co/jeffwuu32/bc5cdr-ner-gemma-4-E2B-it) | `v1.0.0` | 2026-08-29 |
| biored | [`jeffwuu32/biored-ner-gemma-4-E2B-it`](https://huggingface.co/jeffwuu32/biored-ner-gemma-4-E2B-it) | `v1.0.0` | 2026-08-29 |


## Pinning

Everything below is fixed so an upstream change can't silently change what gets reproduced. Most package versions are deliberately *not* pinned to be compatible with Colab.

| what | pinned to |
|---|---|
| adapter | see "Adapter Releases" above (`HUB_BRANCH`) |
| base model (`google/gemma-4-E2B-it`) | commit `3e22461f65e8` |
| BC5CDR corpus | commit `dd0b3165a943`, SHA256-verified on download |
| BioRED corpus | a static, dated NCBI FTP file, SHA256-verified on download |


## What is Reproduced

The reported numbers are generation-based P/R/F1 on the dataset's official **test** split, on two scales: category+text (exact match) and positional (char-level).

- macro = unweighted mean across categories
- micro = pooled across all spans

Ground-truth annotations that could not be structurally represented are penalized as misses, not silently dropped from the metrics.


## Statistics

Measured on Colab under default settings (with `EVAL_GEN_BATCH_SIZE = 16`) unless specified otherwise. Numbers below are estimates only.

### Inference time

| dataset | T4 | L4 |
|---|---|---|
| bc5cdr | 120 min | 45 min |
| biored | 30 min  | 10 min |


## Scope

The repo is intended for result reproduction, so the following are omitted:

- alternative approaches (e.g. BERT)
- LLM zero-shot baselines
- training pipelines
- hyperparameter sweeps
