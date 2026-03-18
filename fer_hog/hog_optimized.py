from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np

from .hog_common import HOGConfig, expected_feature_length, validate_hog_config

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ModuleNotFoundError:
    njit = None
    NUMBA_AVAILABLE = False


def compute_gradients_numpy(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    image = image.astype(np.float32, copy=False)
    gx = np.empty_like(image, dtype=np.float32)
    gy = np.empty_like(image, dtype=np.float32)

    gx[:, 1:-1] = image[:, 2:] - image[:, :-2]
    gx[:, 0] = image[:, 1] - image[:, 0]
    gx[:, -1] = image[:, -1] - image[:, -2]

    gy[1:-1, :] = image[2:, :] - image[:-2, :]
    gy[0, :] = image[1, :] - image[0, :]
    gy[-1, :] = image[-1, :] - image[-2, :]
    return gx, gy


def _cell_histograms_numpy(
    magnitude: np.ndarray,
    orientation: np.ndarray,
    config: HOGConfig,
) -> np.ndarray:
    cell_size = config.cell_size
    num_bins = config.num_bins
    bin_width = 180.0 / num_bins
    cells_y = magnitude.shape[0] // cell_size
    cells_x = magnitude.shape[1] // cell_size
    cell_histograms = np.zeros((cells_y, cells_x, num_bins), dtype=np.float32)

    for cell_row in range(cells_y):
        for cell_col in range(cells_x):
            row_start = cell_row * cell_size
            col_start = cell_col * cell_size
            for inner_row in range(cell_size):
                for inner_col in range(cell_size):
                    row = row_start + inner_row
                    col = col_start + inner_col
                    angle = orientation[row, col]
                    bin_index = int(angle / bin_width)
                    if bin_index >= num_bins:
                        bin_index = num_bins - 1
                    cell_histograms[cell_row, cell_col, bin_index] += magnitude[row, col]
    return cell_histograms


def hog_features_numpy(image: np.ndarray, config: HOGConfig | None = None) -> np.ndarray:
    if config is None:
        config = HOGConfig()
    validate_hog_config(config, image_shape=image.shape)

    gx, gy = compute_gradients_numpy(image)
    magnitude = np.hypot(gx, gy).astype(np.float32, copy=False)
    orientation = np.degrees(np.arctan2(gy, gx))
    orientation = np.where(orientation < 0.0, orientation + 180.0, orientation)
    orientation = np.where(orientation >= 180.0, 0.0, orientation)
    orientation = orientation.astype(np.float32, copy=False)

    cell_histograms = _cell_histograms_numpy(magnitude, orientation, config)
    cells_y, cells_x, _ = cell_histograms.shape
    blocks_y = (cells_y - config.block_size) // config.block_stride + 1
    blocks_x = (cells_x - config.block_size) // config.block_stride + 1
    block_length = config.block_size * config.block_size * config.num_bins
    features = np.empty(blocks_y * blocks_x * block_length, dtype=np.float32)

    write_index = 0
    for block_row in range(blocks_y):
        for block_col in range(blocks_x):
            block = cell_histograms[
                block_row : block_row + config.block_size,
                block_col : block_col + config.block_size,
                :,
            ].reshape(-1)
            norm = np.sqrt(np.sum(block * block) + config.epsilon * config.epsilon)
            features[write_index : write_index + block_length] = block / norm
            write_index += block_length
    return features


def extract_hog_features_numpy(
    images: np.ndarray,
    config: HOGConfig | None = None,
    progress_callback=None,
    progress_every: int = 200,
    stage_name: str = "numpy",
) -> np.ndarray:
    if config is None:
        config = HOGConfig()
    feature_length = expected_feature_length(images.shape[1:], config)
    features = np.empty((images.shape[0], feature_length), dtype=np.float32)
    for idx in range(images.shape[0]):
        features[idx] = hog_features_numpy(images[idx], config)
        if progress_callback is not None and (
            (idx + 1) % progress_every == 0 or idx + 1 == images.shape[0]
        ):
            progress_callback(f"HOG {stage_name}: processed {idx + 1}/{images.shape[0]} images")
    return features


if NUMBA_AVAILABLE:

    @njit(cache=True)
    def _hog_numba_histogram_impl(
        magnitude: np.ndarray,
        orientation: np.ndarray,
        cell_size: int,
        num_bins: int,
        block_size: int,
        block_stride: int,
        epsilon: float,
    ) -> np.ndarray:
        height, width = magnitude.shape
        cells_y = height // cell_size
        cells_x = width // cell_size
        bin_width = 180.0 / num_bins
        cell_histograms = np.zeros((cells_y, cells_x, num_bins), dtype=np.float32)

        for cell_row in range(cells_y):
            for cell_col in range(cells_x):
                row_start = cell_row * cell_size
                col_start = cell_col * cell_size
                for inner_row in range(cell_size):
                    for inner_col in range(cell_size):
                        row = row_start + inner_row
                        col = col_start + inner_col
                        angle = orientation[row, col]
                        bin_index = int(angle / bin_width)
                        if bin_index >= num_bins:
                            bin_index = num_bins - 1
                        cell_histograms[cell_row, cell_col, bin_index] += magnitude[row, col]

        blocks_y = (cells_y - block_size) // block_stride + 1
        blocks_x = (cells_x - block_size) // block_stride + 1
        block_length = block_size * block_size * num_bins
        feature_length = blocks_y * blocks_x * block_length
        features = np.empty(feature_length, dtype=np.float32)
        write_index = 0

        for block_row in range(blocks_y):
            for block_col in range(blocks_x):
                norm_accumulator = 0.0
                block_index = 0
                temp = np.empty(block_length, dtype=np.float32)
                for cell_offset_row in range(block_size):
                    for cell_offset_col in range(block_size):
                        hist = cell_histograms[
                            block_row + cell_offset_row,
                            block_col + cell_offset_col,
                        ]
                        for bin_index in range(num_bins):
                            value = hist[bin_index]
                            temp[block_index] = value
                            norm_accumulator += value * value
                            block_index += 1

                norm = np.sqrt(norm_accumulator + epsilon * epsilon)
                for idx in range(block_length):
                    features[write_index + idx] = temp[idx] / norm
                write_index += block_length

        return features


def hog_features_numba(image: np.ndarray, config: HOGConfig | None = None) -> np.ndarray:
    if not NUMBA_AVAILABLE:
        raise ModuleNotFoundError("numba is not installed in the current Python environment.")
    if config is None:
        config = HOGConfig()
    validate_hog_config(config, image_shape=image.shape)
    gx, gy = compute_gradients_numpy(image.astype(np.float32, copy=False))
    magnitude = np.hypot(gx, gy).astype(np.float32, copy=False)
    orientation = np.degrees(np.arctan2(gy, gx))
    orientation = np.where(orientation < 0.0, orientation + 180.0, orientation)
    orientation = np.where(orientation >= 180.0, 0.0, orientation).astype(np.float32, copy=False)
    return _hog_numba_histogram_impl(
        magnitude,
        orientation,
        config.cell_size,
        config.num_bins,
        config.block_size,
        config.block_stride,
        config.epsilon,
    )


def extract_hog_features_numba(
    images: np.ndarray,
    config: HOGConfig | None = None,
    progress_callback=None,
    progress_every: int = 200,
    stage_name: str = "numba",
) -> np.ndarray:
    if config is None:
        config = HOGConfig()
    feature_length = expected_feature_length(images.shape[1:], config)
    features = np.empty((images.shape[0], feature_length), dtype=np.float32)
    for idx in range(images.shape[0]):
        features[idx] = hog_features_numba(images[idx], config)
        if progress_callback is not None and (
            (idx + 1) % progress_every == 0 or idx + 1 == images.shape[0]
        ):
            progress_callback(f"HOG {stage_name}: processed {idx + 1}/{images.shape[0]} images")
    return features


def _parallel_worker(image: np.ndarray, implementation: str, config: HOGConfig) -> np.ndarray:
    if implementation == "baseline":
        from .hog_baseline import hog_features_baseline

        return hog_features_baseline(image, config)
    if implementation == "numpy":
        return hog_features_numpy(image, config)
    if implementation == "numba":
        return hog_features_numba(image, config)
    raise ValueError(f"Unsupported implementation for parallel extraction: {implementation}")


def extract_hog_features_parallel(
    images: np.ndarray,
    implementation: str,
    config: HOGConfig | None = None,
    n_jobs: int | None = None,
    progress_callback=None,
    progress_every: int = 200,
    stage_name: str = "parallel",
) -> np.ndarray:
    if config is None:
        config = HOGConfig()
    feature_length = expected_feature_length(images.shape[1:], config)
    features = np.empty((images.shape[0], feature_length), dtype=np.float32)
    worker = partial(_parallel_worker, implementation=implementation, config=config)
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        for idx, result in enumerate(executor.map(worker, images)):
            features[idx] = result
            if progress_callback is not None and (
                (idx + 1) % progress_every == 0 or idx + 1 == images.shape[0]
            ):
                progress_callback(f"HOG {stage_name}: processed {idx + 1}/{images.shape[0]} images")
    return features
