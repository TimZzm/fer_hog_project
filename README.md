# FER2013 HOG Emotion Recognition

This project implements an end-to-end FER2013 facial emotion recognition pipeline based on a hand-written Histogram of Oriented Gradients (HOG) descriptor.

The purpose of the project is not to maximize classification accuracy. The purpose is to implement HOG correctly and compare how much the same HOG representation can be accelerated with NumPy, Numba, and parallel processing.

## What HOG Is

Histogram of Oriented Gradients (HOG) is a classical image descriptor that represents local shape information using edge directions.

In this project, the HOG procedure is:

1. Compute horizontal and vertical image gradients.
2. For each pixel, compute gradient magnitude and gradient orientation.
3. Divide the image into `8 x 8` cells.
4. For each cell, build a histogram with `9` orientation bins over `0` to `180` degrees.
5. Use the gradient magnitude as the vote weight for the corresponding orientation bin.
6. Group neighboring cells into overlapping `2 x 2` blocks.
7. Normalize each block to reduce sensitivity to brightness and contrast changes.
8. Flatten all normalized blocks into one HOG feature vector.

For a 48 x 48 FER2013 image with this configuration:

- cell size = 8 x 8
- number of cells per dimension = 48 / 8 = 6
- block size = 2 x 2
- block stride = 1 cell
- orientation bins = 9
the final HOG feature length is:

(6 - 2 + 1) x (6 - 2 + 1) x (2 x 2 x 9) = 5 x 5 x 36 = 900

The classifier is only the final stage of the pipeline. It is included mainly to verify that the optimized implementations preserve the practical effect of the HOG features rather than breaking them.

## What The Project Does

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
- Compares baseline, NumPy, Numba, and parallel HOG variants
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

## Python Environment

Use any Python environment that already contains the required packages. Replace the placeholder below with the Python executable from your own environment.

Example:

```bash
<PATH_TO_YOUR_ENV_PYTHON> run_pipeline.py --help
```

## Install Requirements

If you want to prepare a fresh environment:

```bash
pip install -r requirements.txt
```

## Implementations

### `baseline`

This is the simple reference implementation. It uses explicit Python loops and is intentionally slow, so it is useful as the correctness baseline.

```bash
<PATH_TO_YOUR_ENV_PYTHON> run_pipeline.py \
  --csv ../fer2013.csv \
  --implementation baseline \
  --classifier logreg \
  --log-file logreg_baseline.log
```

### `numpy`

This implementation reduces Python-level overhead by replacing parts of the baseline logic with NumPy-based array operations.

```bash
<PATH_TO_YOUR_ENV_PYTHON> run_pipeline.py \
  --csv ../fer2013.csv \
  --implementation numpy \
  --classifier logreg \
  --log-file logreg_numpy.log
```

### `numpy_numba`

This implementation keeps the NumPy-based workflow and uses Numba JIT to accelerate the loop-heavy HOG stages.

```bash
<PATH_TO_YOUR_ENV_PYTHON> run_pipeline.py \
  --csv ../fer2013.csv \
  --implementation numpy_numba \
  --classifier logreg \
  --log-file logreg_numpy_numba.log
```

### `numpy_parallel`

This implementation uses the NumPy HOG path and applies multiprocessing across images.

```bash
<PATH_TO_YOUR_ENV_PYTHON> run_pipeline.py \
  --csv ../fer2013.csv \
  --implementation numpy_parallel \
  --classifier logreg \
  --log-file logreg_numpy_parallel.log
```

### `numpy_numba_parallel`

This implementation combines the NumPy+Numba HOG path with multiprocessing across images.

```bash
<PATH_TO_YOUR_ENV_PYTHON> run_pipeline.py \
  --csv ../fer2013.csv \
  --implementation numpy_numba_parallel \
  --classifier logreg \
  --log-file logreg_numpy_numba_parallel.log
```

## Optional Smaller Smoke Test

If you want to test the pipeline quickly on a subset:

```bash
<PATH_TO_YOUR_ENV_PYTHON> run_pipeline.py \
  --csv ../fer2013.csv \
  --implementation baseline \
  --limit-train 300 \
  --limit-public 100 \
  --limit-private 100 \
  --benchmark-images 20 \
  --correctness-images 2 \
  --log-file baseline_small.log
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
- `--log-file`
  - save terminal output to a log file
- `--progress-every`
  - control how often progress is written during long runs

## Output And Logs

Each run prints:

- dataset summary
- correctness checks
- timing benchmark table
- train/test evaluation results

The same information is also written to the file given by `--log-file`.

Important timing rows in the logs:

- `feature_train:*`
- `feature_public:*`
- `feature_private:*`

These are the full HOG feature-extraction times on the three FER2013 splits.

## Notes

- The baseline HOG is intentionally loop-heavy so that optimization targets are easy to see.
- The optimized HOG keeps the same feature definition and should produce outputs very close to the baseline.
- FER2013 is large enough that the pure Python baseline can be slow on the full dataset. That is expected.
- Preferred implementation naming is compositional:
  - `numpy_numba` means NumPy preprocessing plus Numba JIT
  - `numpy_parallel` means NumPy plus multiprocessing across images
  - `numpy_numba_parallel` combines NumPy, Numba JIT, and parallelism
