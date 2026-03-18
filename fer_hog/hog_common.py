from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HOGConfig:
    cell_size: int = 8
    num_bins: int = 9
    block_size: int = 2
    block_stride: int = 1
    epsilon: float = 1e-5


def validate_hog_config(config: HOGConfig, image_shape: tuple[int, int] = (48, 48)) -> None:
    height, width = image_shape
    if height % config.cell_size != 0 or width % config.cell_size != 0:
        raise ValueError("Image shape must be divisible by the HOG cell size.")
    if config.block_size <= 0:
        raise ValueError("block_size must be positive.")
    if config.block_stride <= 0:
        raise ValueError("block_stride must be positive.")
    cells_y = height // config.cell_size
    cells_x = width // config.cell_size
    if config.block_size > cells_y or config.block_size > cells_x:
        raise ValueError("block_size is larger than the number of cells in the image.")


def expected_feature_length(
    image_shape: tuple[int, int] = (48, 48),
    config: HOGConfig | None = None,
) -> int:
    if config is None:
        config = HOGConfig()
    validate_hog_config(config, image_shape=image_shape)
    height, width = image_shape
    cells_y = height // config.cell_size
    cells_x = width // config.cell_size
    blocks_y = (cells_y - config.block_size) // config.block_stride + 1
    blocks_x = (cells_x - config.block_size) // config.block_stride + 1
    block_histograms = config.block_size * config.block_size * config.num_bins
    return blocks_y * blocks_x * block_histograms

