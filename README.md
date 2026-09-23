# Fine-Tuning BERT for Sentiment Analysis — IMDB
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/🤗-Transformers-yellow)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

My project where I fine-tune BERT on the IMDB movie review dataset for sentiment classification, and run some experiments to understand how BERT actually learns task-specific information.

This repository fine-tunes [`bert-base-uncased`](https://huggingface.co/google-bert/bert-base-uncased) (~110M parameters) on the [IMDB dataset](https://huggingface.co/datasets/imdb) (50,000 reviews, binary sentiment). I also include a **layer-freezing experiment** to see where task knowledge lives inside a fine-tuned BERT — and how much accuracy you get for each fraction of trainable parameters.

I went through three versions while building this, and **[`docs/LEARNING_LOG.md`](docs/LEARNING_LOG.md)** documents what went wrong in each (including a test-set data leakage bug in v1), how I fixed it, and what I learned.

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

## Results

> Numbers below are placeholders until the full Kaggle GPU run completes. They'll be replaced with the actual output from `results/summary.json`. The quick smoke test validates the pipeline, not the actual model performance.
>
> Literature reference: BERT-base on IMDB typically gets about **92-94%** test accuracy; random guessing would be 50%.

| Configuration | Trainable params | Test accuracy | Test F1 | Train time | Peak VRAM |
|---|---|---|---|---|---|
| Full fine-tuning (12 layers) | 100% | *pending Kaggle run* | — | — | — |
| Top-4 layers + head | ~29% | *pending* | — | — | — |
| Head only (linear probe) | ~0.15% | *pending* | — | — | — |

---

## Repository Structure

```
bert-imdb-finetuning/
├── README.md
├── PUSH_TO_GITHUB.md                # Step-by-step commands for publishing
├── LICENSE                          # MIT
├── requirements.txt                 # For local runs — Kaggle has everything pre-installed
├── notebooks/
│   └── bert_imdb_finetuning.ipynb   # Main notebook — Run All on Kaggle GPU
├── scripts/
│   ├── notebook_source_part*.txt    # Readable, diffable notebook source files
│   ├── build_notebook.py            # Builds and validates the .ipynb from source files
│   └── test_data_loading.py         # Unit test for the data pipeline (runs on CPU)
├── docs/
│   ├── LEARNING_LOG.md              # v1 → v2 → v3: mistakes, fixes, and what I learned
│   └── KAGGLE_SETUP.md              # Step-by-step Kaggle run guide (online & offline)
└── results/                         # Curated artifacts to commit after the real run
    └── .gitkeep
```

*Why build the notebook from text files?* `.ipynb` files are JSON and hard to read in diffs. The actual source lives in plain text; `build_notebook.py` assembles and validates the notebook. Edit the source, rebuild, commit both.

---

## Running the Project

### On Kaggle (full experiment, ~2-3.5 hours)

1. New Notebook → **Accelerator: GPU T4 x2** → **Internet: On**
2. Upload `notebooks/bert_imdb_finetuning.ipynb` → **Run All**
3. Download artifacts from the Output panel (`saved_model/`, `results/`, figures)

Full details, including **offline** mode (attach IMDB CSV + BERT weights as Kaggle Datasets), are in [`docs/KAGGLE_SETUP.md`](docs/KAGGLE_SETUP.md).

### Locally (quick test — no GPU needed, ~5-10 minutes)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
SMOKE_TEST=1 python -m nbconvert --to notebook --execute \
    notebooks/bert_imdb_finetuning.ipynb --output smoke_executed.ipynb
```

This runs every code path (data loading, tokenization, training, early stopping, save/reload, test evaluation, both experiments, all figures and artifacts) on tiny data subsets.

### Unit test for the data pipeline

```bash
python scripts/test_data_loading.py     # No GPU, no downloads needed
```

---

## Project Evolution

| Version | Environment | Key change |
|---|---|---|
| v1 | Colab | First working fine-tune — **but model selection used the official test split (data leakage)** |
| v2 | Colab | Fixed the protocol with a sealed test set, added early stopping, save/reload, and freezing experiments |
| **v3 (this repo)** | **Kaggle GPU** | Dynamic padding (~2× faster), environment-aware paths, offline data/model fallbacks, smoke test mode, reproducibility artifacts, version fingerprinting, honest limitations section |

The full story is in [`docs/LEARNING_LOG.md`](docs/LEARNING_LOG.md).

---

## Validation Status

| Check | Status |
|---|---|
| Notebook structure (`nbformat.validate`) | Passed (31 cells) |
| Data pipeline unit test — CSV branch, stratification, smoke subsetting | Passed |
| Full end-to-end smoke execution (all code cells) | Passed on CPU |
| Hugging Face official-split branch | Exercised during smoke run |
| Kaggle-CSV offline branch | Exercised by unit test |
| transformers 4.41–5.x compatibility | Built-in — `make_training_args()` filters unsupported args |
| Full bert-base-uncased GPU run | Pending — run on Kaggle, then commit `results/` |
| Real result numbers in README table | Pending full run |

---

## What I Learned

Working through this project gave me hands-on experience with:

- **Data leakage & proper evaluation protocols** — including the subtle kind where model selection uses test data
- **Transfer learning** — pre-trained trunk + fresh head; why the learning rate needs to be small during fine-tuning; what freezing layers actually measures
- **Parameter-efficient fine-tuning** — how head-only training relates to modern methods like LoRA and adapters
- **Tokenization mechanics** — WordPiece, `<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>` pooling, attention masks, truncation trade-offs
- **Training engineering** — gradient accumulation, fp16 on T4 tensor cores, learning rate warmup, early stopping, dynamic vs static padding
- **Reproducibility** — fixed seeds everywhere, version fingerprinting, machine-readable result summaries
- **Library version compatibility** — APIs drift between major releases; small compatibility shims beat version pinning for portability

---

## References

- Devlin et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL.
- Maas et al. (2011). *Learning Word Vectors for Sentiment Analysis.* ACL. (IMDB dataset)
- Howard & Ruder (2018). *Universal Language Model Fine-tuning for Text Classification.* ACL.
- Rogers et al. (2020). *A Primer in BERTology.* TACL.
- Sun et al. (2019). *How to Fine-Tune BERT for Text Classification?*
- Hu et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR.

---

## Author

**Naveed Ahmad**

If you found this project useful, please consider giving it a star on GitHub!

---

## License

MIT — see [LICENSE](LICENSE).
