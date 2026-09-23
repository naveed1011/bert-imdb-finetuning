# Fine-Tuning BERT for Sentiment Analysis-IMDB
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/🤗-Transformers-yellow)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

My project where I fine-tune BERT on the IMDB movie review dataset for sentiment classification, and run some experiments to understand how BERT actually learns task-specific information.

This repository fine-tunes [`bert-base-uncased`](https://huggingface.co/google-bert/bert-base-uncased) (~110M parameters) on the [IMDB dataset](https://huggingface.co/datasets/imdb) (50,000 reviews, binary sentiment). I also include a **layer-freezing experiment** to see where task knowledge lives inside a fine-tuned BERT — and how much accuracy you get for each fraction of trainable parameters.

I went through three versions while building this, and **[`docs/LEARNING_LOG.md`](docs/LEARNING_LOG.md)** documents what went wrong in each (including a test-set data leakage bug in v1), how I fixed it, and what I learned.

---

## Results

Run completed on **Kaggle Tesla T4 GPU** (PyTorch 2.10, transformers 5.0). Full fine-tuning hit **93.72% test accuracy** on the sealed IMDB test set (25,000 reviews), right in line with the literature expectation of 92-94%.

The layer-freezing ablation shows a clear accuracy-vs-compute trade-off: full fine-tuning gives the ceiling, training only the top-4 layers comes within 0.4 percentage points using just 26% of parameters, while a linear probe (head only, 0.001% of params) caps out around 63% — showing that sentiment isn't fully linearly separable in the raw pre-trained representation.

| Configuration | Trainable params | Test accuracy | Test F1 | Train time | Peak VRAM |
|---|---|---|---|---|---|
| Full fine-tuning (12 layers) | 100% | **93.72%** | 0.9372 | 107.9 min | 4.57 GB |
| Top-4 layers + head | 26.44% | **93.33%** | 0.9333 | 64.6 min | 4.07 GB |
| Head only (linear probe) | 0.001% | **63.28%** | 0.6139 | 44.1 min | 3.20 GB |

### Per-class breakdown (full fine-tuning)
| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| NEGATIVE | 0.9371 | 0.9374 | 0.9373 | 12,500 |
| POSITIVE | 0.9374 | 0.9370 | 0.9372 | 12,500 |
| **Accuracy** | | | **0.9372** | **25,000** |

All figures and raw data are in the [`results/`](results/) folder. Key numbers are machine-readable in [`results/summary.json`](results/summary.json).

---

## Key Design Choices

| Decision | Why I did it this way |
|---|---|
| **Sealed test set, one-shot evaluation** | My first version used the test set for model selection — classic data leakage. Now I carve validation out of the training split (90/10) and only touch the test set once, at the very end. |
| **Dynamic padding** (`DataCollatorWithPadding`) | I used to pad every review to 512 tokens, but IMDB reviews average ~230 words. Per-batch padding cuts wasted computation roughly in half — this is why the full study fits in a Kaggle session. |
| **Layer-freezing ablation** | Full fine-tune (100%) vs top-4 layers (~29%) vs head-only linear probe (~0.15%), all under the same protocol (same seed, data, learning rate, early stopping). I report accuracy along with compute time and VRAM usage. |
| **Reproducibility artifacts** | Fixed seeds, `environment.json` with library versions, `summary.json` with all key numbers, saved figures, CSVs, and a dump of misclassified examples for qualitative analysis. |
| **`SMOKE_TEST=1` mode** | The same notebook runs on CPU in minutes over tiny data subsets to verify every code path before spending GPU time. |
| **Kaggle-native paths + offline fallback** | Results go to `/kaggle/working` (the only persistent, downloadable directory on Kaggle). Data auto-detects from: `DATA_CSV` env → Hugging Face Hub → attached Kaggle CSV. Model resolution follows the same pattern. |

---

## Repository Structure

bert-imdb-finetuning/
├── README.md
├── LICENSE                         # MIT
├── requirements.txt                # For local runs — Kaggle has everything pre-installed
├── notebooks/
│   └── bert_imdb_finetuning.ipynb  # Main notebook — Run All on Kaggle GPU
├── scripts/
│   ├── notebook_source_part*.txt   # Readable, diffable notebook source files
│   ├── build_notebook.py          # Builds and validates the .ipynb from source files
│   └── test_data_loading.py        # Unit test for the data pipeline (runs on CPU)
├── docs/
│   └── LEARNING_LOG.md             # v1 → v2 → v3: mistakes, fixes, and what I learned
└── results/                        # Results from the full Kaggle GPU run
    ├── summary.json                # All key numbers, machine-readable
    ├── test_classification_report.txt
    ├── experiment_comparison.csv
    ├── misclassified_examples.csv
    ├── eda_overview.png
    ├── token_length_distribution.png
    ├── training_curves.png
    ├── confusion_matrix_full_ft.png
    ├── freezing_experiment_comparison.png
    └── all_experiments_curves.png
