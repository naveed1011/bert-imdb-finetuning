#!/usr/bin/env python3
"""
test_data_loading.py — CPU-only integration test of the notebook's data pipeline.

Instead of re-implementing (and drifting from) the notebook code, this test
EXTRACTS the actual cells from scripts/notebook_source_part*.txt and executes
them in a controlled environment against a synthetic IMDB-format CSV.

Validated code paths:
  1. config cell            — runs headless, creates artifact dirs
  2. data loading cell      — DATA_CSV branch (offline/Kaggle-CSV mode)
  3. split summary cell     — SMOKE subsetting (shuffle-before-select!)
  4. label-distribution and leakage-protocol invariants

Usage:  python scripts/test_data_loading.py
Exit code 0 = all assertions passed.
"""
import os
import re
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PARTS = [os.path.join(HERE, "notebook_source_part1.txt")]

MARKER = re.compile(r"^<<<(MD|CODE)>>>$", re.MULTILINE)


def code_cells(text):
    out, matches = [], list(MARKER.finditer(text))
    for i, m in enumerate(matches):
        if m.group(1) != "CODE":
            continue
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out.append(text[start:end].strip("\n"))
    return out


def make_synthetic_csv(path, n=200):
    """IMDB-CSV format: review,sentiment (balanced, deterministic)."""
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["review", "sentiment"])
        for i in range(n):
            sent = "positive" if i % 2 == 0 else "negative"
            w.writerow([f"synthetic review number {i} with some words about a movie", sent])


def main():
    src = open(PARTS[0], encoding="utf-8").read()
    cells = code_cells(src)

    cfg_cell = next(c for c in cells if "0. CONFIGURATION" in c)
    data_cell = next(c for c in cells if "1a. DATA LOADING" in c)
    split_cell = next(c for c in cells if "1b. SPLIT SUMMARY" in c)

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "IMDB Dataset.csv")
        make_synthetic_csv(csv_path, n=1000)   # big enough that SMOKE subsets (64/32/32) are fully populated

        os.environ["SMOKE_TEST"] = "0"
        os.environ["DATA_CSV"] = csv_path

        ns = {"__name__": "__main__"}
        # keep artifact dirs inside the temp dir
        os.chdir(tmp)

        print("── executing notebook config cell ──")
        exec(compile(cfg_cell, "<config cell>", "exec"), ns)

        print("\n── executing notebook data-loading cell (DATA_CSV branch) ──")
        exec(compile(data_cell, "<data cell>", "exec"), ns)

        train, val, test = ns["actual_train"], ns["actual_val"], ns["actual_test"]
        assert len(train) == 450 and len(val) == 50 and len(test) == 500, \
            f"unexpected 45/5/50 split sizes: {len(train)}/{len(val)}/{len(test)}"
        for split, name in [(train, "train"), (val, "val"), (test, "test")]:
            labels = split["label"]
            assert set(labels) == {0, 1}, f"{name}: labels not stratified -> {set(labels)}"
            assert abs(sum(labels) / len(labels) - 0.5) < 0.05, f"{name}: not balanced"
            assert all(isinstance(t, str) and len(t) > 0 for t in split["text"][:5])
        assert "CSV via DATA_CSV" in ns["data_source"]
        print("OK: 1000-row CSV -> stratified 450/50/500 split, balanced labels, text intact")

        print("\n── executing split-summary cell in SMOKE mode ──")
        os.environ["SMOKE_TEST"] = "1"
        ns2 = {"__name__": "__main__"}
        exec(compile(cfg_cell, "<config cell>", "exec"), ns2)
        exec(compile(data_cell, "<data cell>", "exec"), ns2)
        exec(compile(split_cell, "<split cell>", "exec"), ns2)
        assert ns2["SMOKE"] is True
        st, sv, ste = ns2["actual_train"], ns2["actual_val"], ns2["actual_test"]
        assert len(st) == 64 and len(sv) == 32 and len(ste) == 32, \
            f"smoke subsets wrong: {len(st)}/{len(sv)}/{len(ste)}"
        # shuffle-before-select: first 64 of a shuffled train must contain BOTH classes
        assert set(st["label"]) == {0, 1}, "smoke train subset is single-class (shuffle missing?)"
        print("OK: SMOKE subsets = 64/32/32, both classes present (shuffle-before-select works)")

    print("\n✅ ALL DATA-PIPELINE TESTS PASSED")


if __name__ == "__main__":
    sys.exit(main())
