# FER2013 HOG Emotion Recognition

This project builds an end-to-end facial emotion recognition pipeline on the original `fer2013.csv` dataset using a hand-written Histogram of Oriented Gradients (HOG) feature extractor.

The main goal is not to chase state-of-the-art accuracy. The goal is to show a clear progression from:

- a correct but slow baseline
- to NumPy-friendly improvements
- to Numba JIT acceleration
- to optional parallel feature extraction across images

The project is organized to be beginner-friendly and easy to run locally.

## What It Does

- Loads FER2013 directly from the original CSV format
- Parses each image from the pixel string into a `(48, 48)` NumPy array
- Preserves the official split using the `Usage` column:
  - `Training`
  - `PublicTest`
  - `PrivateTest`
- Implements HOG from scratch with:
  - unsigned orientations in `[0, 180)`
  - cell size `8x8`
  - `9` orientation bins
  - block size `2x2` cells
  - block stride `1` cell
- Trains a simple multiclass classifier with scikit-learn
- Benchmarks CSV parsing, HOG extraction, and classifier train/inference
- Compares baseline, NumPy, Numba, and optional parallel HOG variants
- Checks optimized HOG outputs against the baseline on a few sample images

## Project Layout

```text
fer_hog_project/
  README.md
  requirements.txt
  run_pipeline.py
  fer_hog/
    __init__.py
    benchmark.py
    classifier.py
    data.py
    hog_baseline.py
    hog_common.py
    hog_optimized.py
    pipeline.py
```

## Recommended Python Environment

You said `dedalus22` is available locally and already contains the needed packages. This project is written to run there directly.

Example:

```bash
~/miniforge3/envs/dedalus22/bin/python run_pipeline.py --help
```

## Install Requirements

If you ever want a fresh environment, install:

```bash
pip install -r requirements.txt
```

## Quick Start

From inside `fer_hog_project`:

```bash
~/miniforge3/envs/dedalus22/bin/python run_pipeline.py \
  --csv ../fer2013.csv \
  --implementation numpy_numba \
  --classifier linear_svc \
  --benchmark-images 200 \
  --correctness-images 3
```

For a smaller smoke test:

```bash
~/miniforge3/envs/dedalus22/bin/python run_pipeline.py \
  --csv ../fer2013.csv \
  --implementation baseline \
  --limit-train 300 \
  --limit-public 100 \
  --limit-private 100 \
  --benchmark-images 20 \
  --correctness-images 2
```

## Main Options

- `--implementation`
  - `baseline`
  - `numpy`
  - `numpy_numba`
  - `numpy_parallel`
  - `numpy_numba_parallel`
- `--classifier`
  - `linear_svc`
  - `logreg`
- `--parser`
  - `baseline`
  - `fast`
- `--limit-train`, `--limit-public`, `--limit-private`
  - useful for fast testing
- `--n-jobs`
  - worker count for parallel extraction
- `--skip-correctness`
  - skip the baseline-vs-optimized consistency check

## Notes

- The baseline HOG is intentionally loop-heavy so that optimization targets are easy to see.
- The optimized HOG keeps the same feature definition and should produce outputs very close to the baseline.
- FER2013 is large enough that the pure Python baseline can be slow on the full dataset. That is expected.
- Preferred implementation naming is compositional:
  - `numpy_numba` means NumPy preprocessing plus Numba JIT
  - `numpy_parallel` means NumPy plus multiprocessing across images
  - `numpy_numba_parallel` combines NumPy, Numba JIT, and parallelism
