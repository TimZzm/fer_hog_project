from __future__ import annotations

import argparse
from pathlib import Path
from time import strftime

import numpy as np

from .benchmark import BenchmarkResult, format_benchmark_table, timed_call
from .classifier import evaluate_classifier, train_classifier
from .data import DatasetSplits, load_fer2013
from .hog_baseline import extract_hog_features_baseline, hog_features_baseline
from .hog_common import HOGConfig, expected_feature_length
from .hog_optimized import (
    NUMBA_AVAILABLE,
    extract_hog_features_numba,
    extract_hog_features_numpy,
    extract_hog_features_parallel,
    hog_features_numba,
    hog_features_numpy,
)

def _extract_features(
    images: np.ndarray,
    implementation: str,
    config: HOGConfig,
    n_jobs: int | None = None,
    progress_callback=None,
    progress_every: int = 200,
    stage_name: str = "features",
) -> np.ndarray:
    if implementation == "baseline":
        return extract_hog_features_baseline(
            images,
            config,
            progress_callback=progress_callback,
            progress_every=progress_every,
            stage_name=stage_name,
        )
    if implementation == "numpy":
        return extract_hog_features_numpy(
            images,
            config,
            progress_callback=progress_callback,
            progress_every=progress_every,
            stage_name=stage_name,
        )
    if implementation == "numpy_numba":
        return extract_hog_features_numba(
            images,
            config,
            progress_callback=progress_callback,
            progress_every=progress_every,
            stage_name=stage_name,
        )
    if implementation == "numpy_parallel":
        return extract_hog_features_parallel(
            images,
            "numpy",
            config,
            n_jobs=n_jobs,
            progress_callback=progress_callback,
            progress_every=progress_every,
            stage_name=stage_name,
        )
    if implementation == "numpy_numba_parallel":
        return extract_hog_features_parallel(
            images,
            "numba",
            config,
            n_jobs=n_jobs,
            progress_callback=progress_callback,
            progress_every=progress_every,
            stage_name=stage_name,
        )
    raise ValueError(f"Unknown HOG implementation '{implementation}'.")


def _single_image_hog(image: np.ndarray, implementation: str, config: HOGConfig) -> np.ndarray:
    if implementation == "baseline":
        return hog_features_baseline(image, config)
    if implementation == "numpy":
        return hog_features_numpy(image, config)
    if implementation == "numpy_numba":
        return hog_features_numba(image, config)
    raise ValueError(f"Unsupported single-image implementation '{implementation}'.")


def run_correctness_checks(
    images: np.ndarray,
    config: HOGConfig,
    implementations: list[str],
) -> list[str]:
    messages: list[str] = []
    for implementation in implementations:
        max_abs_diff = 0.0
        max_rel_diff = 0.0
        for image in images:
            baseline = hog_features_baseline(image, config)
            other = _single_image_hog(image, implementation, config)
            abs_diff = float(np.max(np.abs(baseline - other)))
            denom = float(np.max(np.abs(baseline))) + 1e-8
            rel_diff = abs_diff / denom
            max_abs_diff = max(max_abs_diff, abs_diff)
            max_rel_diff = max(max_rel_diff, rel_diff)
        messages.append(
            f"Correctness check for {implementation}: "
            f"max_abs_diff={max_abs_diff:.6e}, max_rel_diff={max_rel_diff:.6e}"
        )
    return messages


def benchmark_hog_implementations(
    images: np.ndarray,
    config: HOGConfig,
    n_jobs: int | None = None,
) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []
    candidate_implementations = ["baseline", "numpy"]
    if NUMBA_AVAILABLE:
        candidate_implementations.append("numpy_numba")
    if images.shape[0] > 1:
        candidate_implementations.extend(["numpy_parallel"])
        if NUMBA_AVAILABLE:
            candidate_implementations.extend(["numpy_numba_parallel"])

    for implementation in candidate_implementations:
        try:
            _, benchmark = timed_call(
                _extract_features,
                images,
                implementation,
                config,
                n_jobs,
                _stage_name=f"hog:{implementation}",
                _items=images.shape[0],
            )
        except Exception as exc:
            print(f"Skipping benchmark for {implementation}: {exc}")
            continue
        results.append(benchmark)
    return results


def _print_dataset_summary(splits: DatasetSplits, config: HOGConfig) -> None:
    feature_length = expected_feature_length((48, 48), config)
    print("Dataset summary")
    print(f"  Training images:   {splits.training.size}")
    print(f"  PublicTest images: {splits.public_test.size}")
    print(f"  PrivateTest images:{splits.private_test.size}")
    print(f"  HOG feature length:{feature_length}")


def make_logger(log_file: Path | None):
    handle = None
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handle = log_file.open("a", encoding="utf-8")

    def log(message: str = "") -> None:
        timestamped = f"[{strftime('%Y-%m-%d %H:%M:%S')}] {message}" if message else ""
        print(message)
        if handle is not None:
            handle.write(timestamped + "\n")
            handle.flush()

    return log, handle


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FER2013 HOG emotion recognition pipeline")
    parser.add_argument("--csv", type=Path, required=True, help="Path to fer2013.csv")
    parser.add_argument(
        "--implementation",
        choices=[
            "baseline",
            "numpy",
            "numpy_numba",
            "numpy_parallel",
            "numpy_numba_parallel",
        ],
        default="numpy",
        help=(
            "HOG implementation to use. Choose baseline, numpy, numpy_numba, "
            "numpy_parallel, or numpy_numba_parallel."
        ),
    )
    parser.add_argument("--classifier", choices=["linear_svc", "logreg"], default="linear_svc")
    parser.add_argument("--parser", choices=["baseline", "fast"], default="fast")
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-public", type=int, default=None)
    parser.add_argument("--limit-private", type=int, default=None)
    parser.add_argument("--benchmark-images", type=int, default=100)
    parser.add_argument("--correctness-images", type=int, default=3)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--log-file", type=Path, default=Path("run.log"))
    parser.add_argument("--progress-every", type=int, default=500)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = HOGConfig()
    benchmarks: list[BenchmarkResult] = []
    log, log_handle = make_logger(args.log_file)

    try:
        log(f"Logging to {args.log_file}")
        log("Starting FER2013 pipeline")
        log("Loading FER2013 data")
        try:
            splits, load_benchmark = timed_call(
                load_fer2013,
                args.csv,
                args.parser,
                args.limit_train,
                args.limit_public,
                args.limit_private,
                progress_callback=log,
                progress_every=args.progress_every,
                _stage_name=f"csv_load:{args.parser}",
                _items=None,
            )
        except Exception as exc:
            log(f"Failed to load FER2013: {exc}")
            return 1

        benchmarks.append(load_benchmark)
        _print_dataset_summary(splits, config)
        log("Dataset summary printed")

        if not args.skip_correctness:
            log("Running correctness checks")
            sample_count = min(args.correctness_images, splits.training.size)
            if sample_count > 0:
                sample_images = splits.training.images[:sample_count]
                implementations = ["numpy"]
                if NUMBA_AVAILABLE:
                    implementations.append("numpy_numba")
                for message in run_correctness_checks(sample_images, config, implementations):
                    log(message)

        benchmark_count = min(args.benchmark_images, splits.training.size)
        if benchmark_count > 0:
            log(f"Benchmarking HOG implementations on {benchmark_count} training images")
            benchmark_images = splits.training.images[:benchmark_count]
            benchmarks.extend(benchmark_hog_implementations(benchmark_images, config, n_jobs=args.n_jobs))

        try:
            log(f"Extracting training features with {args.implementation}")
            train_features, hog_train_benchmark = timed_call(
                _extract_features,
                splits.training.images,
                args.implementation,
                config,
                args.n_jobs,
                progress_callback=log,
                progress_every=args.progress_every,
                stage_name="train",
                _stage_name=f"feature_train:{args.implementation}",
                _items=splits.training.size,
            )
            log(f"Extracting public test features with {args.implementation}")
            public_features, hog_public_benchmark = timed_call(
                _extract_features,
                splits.public_test.images,
                args.implementation,
                config,
                args.n_jobs,
                progress_callback=log,
                progress_every=args.progress_every,
                stage_name="public",
                _stage_name=f"feature_public:{args.implementation}",
                _items=splits.public_test.size,
            )
            log(f"Extracting private test features with {args.implementation}")
            private_features, hog_private_benchmark = timed_call(
                _extract_features,
                splits.private_test.images,
                args.implementation,
                config,
                args.n_jobs,
                progress_callback=log,
                progress_every=args.progress_every,
                stage_name="private",
                _stage_name=f"feature_private:{args.implementation}",
                _items=splits.private_test.size,
            )
        except Exception as exc:
            log(f"Feature extraction failed: {exc}")
            return 1

        benchmarks.extend([hog_train_benchmark, hog_public_benchmark, hog_private_benchmark])

        try:
            log(f"Training classifier: {args.classifier}")
            model, fit_benchmark = timed_call(
                train_classifier,
                train_features,
                splits.training.labels,
                args.classifier,
                _stage_name=f"classifier_fit:{args.classifier}",
                _items=splits.training.size,
            )
            log("Evaluating on PublicTest")
            public_eval, public_eval_benchmark = timed_call(
                evaluate_classifier,
                model,
                public_features,
                splits.public_test.labels,
                _stage_name="classifier_eval:public",
                _items=splits.public_test.size,
            )
            log("Evaluating on PrivateTest")
            private_eval, private_eval_benchmark = timed_call(
                evaluate_classifier,
                model,
                private_features,
                splits.private_test.labels,
                _stage_name="classifier_eval:private",
                _items=splits.private_test.size,
            )
            combined_features = np.concatenate([public_features, private_features], axis=0)
            combined_labels = splits.combined_test_labels
            log("Evaluating on combined test set")
            combined_eval, combined_eval_benchmark = timed_call(
                evaluate_classifier,
                model,
                combined_features,
                combined_labels,
                _stage_name="classifier_eval:combined",
                _items=combined_labels.shape[0],
            )
        except Exception as exc:
            log(f"Classifier stage failed: {exc}")
            return 1

        benchmarks.extend(
            [
                fit_benchmark,
                public_eval_benchmark,
                private_eval_benchmark,
                combined_eval_benchmark,
            ]
        )

        print()
        print("Evaluation")
        print(f"  PublicTest accuracy:  {public_eval.accuracy:.4f}")
        print(f"  PrivateTest accuracy: {private_eval.accuracy:.4f}")
        print(f"  Combined accuracy:    {combined_eval.accuracy:.4f}")
        print("  Combined confusion matrix:")
        print(combined_eval.confusion_matrix)
        log(f"PublicTest accuracy: {public_eval.accuracy:.4f}")
        log(f"PrivateTest accuracy: {private_eval.accuracy:.4f}")
        log(f"Combined accuracy: {combined_eval.accuracy:.4f}")

        print()
        print("Benchmarks")
        benchmark_text = format_benchmark_table(benchmarks)
        print(benchmark_text)
        log("Benchmarks")
        for line in benchmark_text.splitlines():
            log(line)
        log("Pipeline complete")
        return 0
    finally:
        if log_handle is not None:
            log_handle.close()
