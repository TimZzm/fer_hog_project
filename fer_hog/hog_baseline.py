from __future__ import annotations

import numpy as np

from .hog_common import HOGConfig, expected_feature_length, validate_hog_config


def compute_gradients_baseline(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    image = image.astype(np.float32, copy=False)
    height, width = image.shape
    gx = np.zeros((height, width), dtype=np.float32)
    gy = np.zeros((height, width), dtype=np.float32)

    for row in range(height):
        for col in range(width):
            if col == 0:
                gx[row, col] = image[row, col + 1] - image[row, col]
            elif col == width - 1:
                gx[row, col] = image[row, col] - image[row, col - 1]
            else:
                gx[row, col] = image[row, col + 1] - image[row, col - 1]

            if row == 0:
                gy[row, col] = image[row + 1, col] - image[row, col]
            elif row == height - 1:
                gy[row, col] = image[row, col] - image[row - 1, col]
            else:
                gy[row, col] = image[row + 1, col] - image[row - 1, col]

    return gx, gy


def hog_features_baseline(image: np.ndarray, config: HOGConfig | None = None) -> np.ndarray:
    if config is None:
        config = HOGConfig()
    validate_hog_config(config, image_shape=image.shape)

    image = image.astype(np.float32, copy=False)
    gx, gy = compute_gradients_baseline(image)

    magnitude = np.zeros_like(image, dtype=np.float32)
    orientation = np.zeros_like(image, dtype=np.float32)
    height, width = image.shape

    for row in range(height):
        for col in range(width):
            mag = np.sqrt(gx[row, col] * gx[row, col] + gy[row, col] * gy[row, col])
            angle = np.degrees(np.arctan2(gy[row, col], gx[row, col]))
            if angle < 0.0:
                angle += 180.0
            if angle >= 180.0:
                angle = 0.0
            magnitude[row, col] = mag
            orientation[row, col] = angle

    cell_size = config.cell_size
    num_bins = config.num_bins
    bin_width = 180.0 / num_bins
    cells_y = height // cell_size
    cells_x = width // cell_size
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

    feature_length = expected_feature_length(image.shape, config)
    features = np.empty(feature_length, dtype=np.float32)
    write_index = 0
    blocks_y = (cells_y - config.block_size) // config.block_stride + 1
    blocks_x = (cells_x - config.block_size) // config.block_stride + 1

    for block_row in range(blocks_y):
        for block_col in range(blocks_x):
            block_vector = np.empty(
                config.block_size * config.block_size * num_bins,
                dtype=np.float32,
            )
            block_index = 0
            for cell_offset_row in range(config.block_size):
                for cell_offset_col in range(config.block_size):
                    hist = cell_histograms[
                        block_row + cell_offset_row,
                        block_col + cell_offset_col,
                    ]
                    for bin_index in range(num_bins):
                        block_vector[block_index] = hist[bin_index]
                        block_index += 1
            norm = np.sqrt(np.sum(block_vector * block_vector) + config.epsilon * config.epsilon)
            block_vector = block_vector / norm
            features[write_index : write_index + block_vector.size] = block_vector
            write_index += block_vector.size

    return features


def extract_hog_features_baseline(
    images: np.ndarray,
    config: HOGConfig | None = None,
    progress_callback=None,
    progress_every: int = 200,
    stage_name: str = "baseline",
) -> np.ndarray:
    if config is None:
        config = HOGConfig()
    num_images = images.shape[0]
    feature_length = expected_feature_length(images.shape[1:], config)
    features = np.empty((num_images, feature_length), dtype=np.float32)
    for idx in range(num_images):
        features[idx] = hog_features_baseline(images[idx], config)
        if progress_callback is not None and (
            (idx + 1) % progress_every == 0 or idx + 1 == num_images
        ):
            progress_callback(f"HOG {stage_name}: processed {idx + 1}/{num_images} images")
    return features
