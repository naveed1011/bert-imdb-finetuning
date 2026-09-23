#!/usr/bin/env python3
"""
build_notebook.py — assemble notebooks/bert_imdb_finetuning.ipynb from the
sentinel-delimited source parts (scripts/notebook_source_part*.txt).

Why build programmatically instead of hand-editing .ipynb JSON?
  * the .txt sources are readable/diffable in code review (no escaped JSON blobs)
  * the output is guaranteed to be structurally valid nbformat
  * cell content can be linted/executed independently (see docs/LEARNING_LOG.md)

Usage:  python scripts/build_notebook.py
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PARTS = [
    os.path.join(HERE, "notebook_source_part1.txt"),
    os.path.join(HERE, "notebook_source_part2.txt"),
]
OUT = os.path.join(ROOT, "notebooks", "bert_imdb_finetuning.ipynb")

MARKER = re.compile(r"^<<<(MD|CODE)>>>$", re.MULTILINE)


def parse_cells(text):
    cells = []
    matches = list(MARKER.finditer(text))
    for i, m in enumerate(matches):
        kind = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip("\n")
        cells.append((kind, body))
    return cells


def main():
    all_cells = []
    for part in PARTS:
        with open(part, "r", encoding="utf-8") as f:
            all_cells.extend(parse_cells(f.read()))

    nb_cells = []
    for kind, body in all_cells:
        if kind == "MD":
            nb_cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": body.splitlines(keepends=True),
            })
        else:
            nb_cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": body.splitlines(keepends=True),
            })

    notebook = {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
        f.write("\n")

    n_md = sum(1 for c in nb_cells if c["cell_type"] == "markdown")
    n_code = len(nb_cells) - n_md
    print(f"Wrote {OUT}")
    print(f"  {len(nb_cells)} cells ({n_md} markdown, {n_code} code)")

    # Validate with nbformat if available
    try:
        import nbformat
        with open(OUT, "r", encoding="utf-8") as f:
            nbformat.validate(nbformat.read(f, as_version=4))
        print("  nbformat validation: PASSED")
    except ImportError:
        print("  nbformat not installed — skipped structural validation")
    except Exception as e:
        print(f"  nbformat validation: FAILED -> {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
