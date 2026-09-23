# Learning Log — v1 → v2 → v3

This document tracks how the project evolved: what broke, why it mattered, and what I learned along the way. I'm keeping this because the actual learning happened in the process — catching the leakage bug, figuring out how much compute I was wasting, designing the experiments — not just in the final working code.

---

## v1 — "It runs!" (Colab)

**What it did:** Loaded IMDB from Hugging Face, fine-tuned `bert-base-uncased`, reported test accuracy.

**The bug I didn't know I had: test-set leakage.**

My first version used the official IMDB `test` split as the evaluation set during training — both early stopping and best-checkpoint selection looked at it. The final "test accuracy" was therefore measured on data that had already influenced model selection hundreds of times. The number was real, but it measured "how well does this pipeline do when tuned on these 25k reviews," not "how well does it generalize to truly unseen data."

**Lesson:** The test set is something you only get to use once. Any decision that consults test data — even indirectly, even "just to pick a checkpoint" — turns it into another validation set and biases the final number upward. This is probably the most common evaluation mistake in ML, and I made it firsthand.

---

## v2 — Fixing the protocol (Colab)

**Changes I made:**
- Carved a 2,500-sample validation set out of the official *train* split (90/10, seeded); the official test split became **sealed** until a single final evaluation pass
- Added `EarlyStoppingCallback` (patience 2 on validation accuracy) + `load_best_model_at_end`, so the saved checkpoint is the validation-best one, chosen without touching test data
- Added save/reload round-trip to prove the artifact works from disk, not just from RAM
- Added learning curves, confusion matrix, and misclassification inspection
- Added the **layer-freezing experiments**: full fine-tune vs top-4 layers vs head-only

**What v2 still got wrong (found during v3 review):**
1. **Wasted compute:** `padding="max_length"` made every review a 512-token tensor. IMDB reviews average ~230 words, so most of the attention computation was spent on padding tokens.
2. **A silently empty plot:** The "train loss per epoch" curve tried to read `train_loss` from evaluation log entries, where that key doesn't exist. A guard condition hid the failure instead of showing it.
3. **Fragile labels:** The sanity check compared `pred["label"] == "LABEL_1"` — a string assumption that breaks the moment label mapping changes.
4. **No environment fingerprint:** Results without recorded library versions aren't reproducible.
5. **Hardcoded paths:** Artifacts landed in whatever the current working directory happened to be; nothing was organized for download.
6. **No cost measurement:** The freezing comparison reported accuracy but not training time or VRAM usage, which is half of the trade-off being studied.

**Lesson:** A correct protocol is necessary but not sufficient. Silent failures (empty plots, string-matched labels) and unmeasured costs quietly degrade the whole experiment.

---

## v3 — Reproducible, portable, measured (this repo, Kaggle GPU)

**Engineering changes:**
- **Dynamic padding** via `DataCollatorWithPadding`: Sequences stored at natural length, padded per batch to the batch maximum. About half as many padded positions on average → the whole study (full fine-tune + 2 experiments + evaluations) fits comfortably in a Kaggle session.
- **`SMOKE_TEST=1` mode:** The identical notebook validates every code path on CPU in minutes over tiny subsets. Also handles a subtle detail — the raw IMDB split is label-ordered, so naive subsetting would produce single-class data. Always smoke test before spending GPU hours.
- **Environment-aware paths:** Artifacts always go to `/kaggle/working` on Kaggle (the only persistent, downloadable directory) and `./outputs` locally.
- **Data-source fallback chain:** `DATA_CSV` env → Hugging Face Hub official split → attached Kaggle CSV with a stratified 45/5/50 rebuild. Model resolution follows the same pattern, so the notebook also runs with Kaggle **Internet OFF**.
- **Reproducibility artifacts:** `environment.json` (python/torch/transformers/datasets/GPU info), `summary.json` (all key numbers), `experiment_comparison.csv`, `test_classification_report.txt`, `misclassified_examples.csv`, all figures as PNGs.
- **Fixed v2 bugs:** Epoch train-loss now properly aggregated from step logs; human-readable labels via `id2label`/`label2id`; explicit `seed` + `data_seed`; per-experiment wall-clock and peak-VRAM measurement.
- **Version-drift resilience:** Running the notebook against bleeding-edge libraries surfaced four breaking API changes:
  1. `load_dataset("imdb")` — bare aliases rejected in newer versions → now tries canonical `stanfordnlp/imdb` first, legacy alias as fallback
  2. `TrainingArguments(warmup_ratio=...)` — removed in transformers 5.x → warmup computed as explicit step count
  3. `TrainingArguments(logging_dir=...)` — removed in 5.x → a `make_training_args()` factory filters unsupported kwargs against the installed signature
  4. `TrainerState.best_epoch` — removed in 5.x → `best_epoch_of()` recovers it from the eval log
- **Notebook built from diffable plain-text sources** (`scripts/`) with structural validation — `.ipynb` JSON is hard to review; the source of truth should be readable.

**Scientific framing added:**
- The ablation now tests an explicit hypothesis (task knowledge concentrates in upper layers) under identical protocol, and reports **accuracy per unit of compute**
- Confidence statistics on correct vs misclassified test examples (errors are usually less confident — the model often "knows" it's unsure)
- An honest **limitations** section: single seed (no error bars), 512-token truncation drops review endings, softmax confidence ≠ calibration, out-of-domain transfer untested

---

## Concepts I Now Understand From Experience

- **Data leakage & proper evaluation** — including the subtle kind where model selection uses test data
- **Transfer learning** — pre-trained trunk + fresh head; why learning rate must be small (2e-5) during fine-tuning; what freezing layers actually measures
- **Parameter-efficient fine-tuning lineage** — head-only training is the ancestor of adapters/LoRA: freeze the trunk, train a small new parameter set
- **Tokenization mechanics** — WordPiece, `<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>` pooling for classification, attention masks, truncation trade-offs
- **Training engineering** — gradient accumulation for effective batch 32 on 16 GB VRAM, fp16 on T4 tensor cores, warmup + linear decay, early stopping, checkpoint rotation, dynamic vs static padding
- **Reproducibility practice** — seeds everywhere, version fingerprinting, machine-readable result summaries, one-command smoke tests
- **Defensive version compatibility** — library APIs drift; executing against the newest stack (not just the one you wrote on) is the only way to find out

---

## What I'd Do Next

1. Multi-seed runs → mean ± std, and proper statistical tests before claiming one configuration beats another
2. Add a **LoRA** arm to the ablation to connect this study with modern LLM fine-tuning
3. Compare against DistilBERT (faster) and DeBERTa-v3 (better) at equal compute budget



### Quick test first (recommended)

Before spending GPU hours, verify the pipeline end-to-end in minutes:
Add a first cell with `%env SMOKE_TEST=1`, then Run All. Every code path executes on tiny data subsets; the numbers are meaningless by design. Remove the env var (or restart the session) for the real run.
