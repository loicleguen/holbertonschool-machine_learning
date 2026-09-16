#!/usr/bin/env python3
"""Drop-in replacement for `tfds.load('ted_hrlr_translate/pt_to_en', ...)`.

The original TFDS builder downloads its data from
http://www.phontron.com/data/qi18naacl-dataset.tar.gz, which no longer
serves the archive. This helper loads the same dataset from a manually
extracted copy on disk and returns it as a tf.data.Dataset of (pt, en)
tf.string pairs, identical in shape to what tfds.load used to return.

Expected layout (any of these is fine):
    $TED_HRLR_DIR/datasets/pt_to_en/{pt,en}.{train,dev,test}
    ~/.cache/ted_hrlr/datasets/pt_to_en/{pt,en}.{train,dev,test}
    ./datasets/pt_to_en/{pt,en}.{train,dev,test}

Usage in your training code:
    from setup import load_pt2en
    pt2en_train = load_pt2en('train')        # or 'validation', 'test'
"""

import os
import sys
from pathlib import Path

import tensorflow as tf

SOURCE_LANG = "pt"
TARGET_LANG = "en"
PAIR_DIR = "pt_to_en"

_SPLIT_SUFFIX = {"train": "train", "validation": "dev", "test": "test"}


def _candidate_roots():
    roots = []
    env = os.environ.get("TED_HRLR_DIR")
    if env:
        roots.append(Path(env))
    roots.append(Path.home() / ".cache" / "ted_hrlr")
    roots.append(Path.cwd())
    return roots


def resolve_data_dir() -> Path:
    """Find the extracted datasets/pt_to_en directory, or exit with guidance."""
    for root in _candidate_roots():
        data_dir = root / "datasets" / PAIR_DIR
        if data_dir.is_dir() and any(data_dir.glob(f"{SOURCE_LANG}.*")):
            return data_dir
    looked = "\n  ".join(str(r / "datasets" / PAIR_DIR) for r in _candidate_roots())
    sys.exit(
        "Extracted data not found. Looked in:\n  " + looked + "\n\n"
        "Extract the archive first (fast, shows progress):\n"
        "  mkdir -p ~/.cache/ted_hrlr\n"
        "  tar -xzvf ted_hrlr_pt_to_en.tar.gz -C ~/.cache/ted_hrlr\n\n"
        "Or point the script at your own location:\n"
        "  TED_HRLR_DIR=/path/to/extracted python setup.py"
    )


def _read_pairs(data_dir: Path, split: str):
    """Yield (pt, en) pairs, mirroring the official TFDS builder's parse logic."""
    suffix = _SPLIT_SUFFIX[split]
    src_file = data_dir / f"{SOURCE_LANG}.{suffix}"
    tgt_file = data_dir / f"{TARGET_LANG}.{suffix}"
    for f in (src_file, tgt_file):
        if not f.is_file():
            sys.exit(f"Missing data file: {f}")

    src = src_file.read_text(encoding="utf-8").split("\n")
    tgt = tgt_file.read_text(encoding="utf-8").split("\n")
    if len(src) != len(tgt):
        sys.exit(
            f"Line count mismatch for {split}: "
            f"{len(src)} ({src_file.name}) vs {len(tgt)} ({tgt_file.name})"
        )
    for pt, en in zip(src, tgt):
        if pt and en:  # drop pairs where either side is empty
            yield pt, en


def load_pt2en(split: str = "train") -> tf.data.Dataset:
    """Return a tf.data.Dataset of (pt, en) tf.string scalars for the split.

    Drop-in replacement for:
        tfds.load('ted_hrlr_translate/pt_to_en', split=split, as_supervised=True)
    """
    if split not in _SPLIT_SUFFIX:
        raise ValueError(f"split must be one of {list(_SPLIT_SUFFIX)}, got {split!r}")
    data_dir = resolve_data_dir()

    def gen():
        yield from _read_pairs(data_dir, split)

    return tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.string),  # pt
            tf.TensorSpec(shape=(), dtype=tf.string),  # en
        ),
    )


def main() -> None:
    data_dir = resolve_data_dir()
    print(f"Using data dir: {data_dir}\n")

    for split, suffix in _SPLIT_SUFFIX.items():
        src = (data_dir / f"{SOURCE_LANG}.{suffix}").read_text(encoding="utf-8").split("\n")
        tgt = (data_dir / f"{TARGET_LANG}.{suffix}").read_text(encoding="utf-8").split("\n")
        n = sum(1 for a, b in zip(src, tgt) if a and b)
        print(f"{split:>10}: {n} pairs")

    print("\nFirst training example:")
    for pt, en in load_pt2en("train").take(1):
        print("  PT:", pt.numpy().decode("utf-8"))
        print("  EN:", en.numpy().decode("utf-8"))
    print("\nReady. Import load_pt2en() in your training code.")


if __name__ == "__main__":
    main()
