from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


@dataclass
class SplitData:
    images: np.ndarray
    labels: np.ndarray
    usage: str

    @property
    def size(self) -> int:
        return int(self.labels.shape[0])


@dataclass
class DatasetSplits:
    training: SplitData
    public_test: SplitData
    private_test: SplitData

    @property
    def combined_test_images(self) -> np.ndarray:
        return np.concatenate([self.public_test.images, self.private_test.images], axis=0)

    @property
    def combined_test_labels(self) -> np.ndarray:
        return np.concatenate([self.public_test.labels, self.private_test.labels], axis=0)


def _import_pandas():
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pandas is required for loading FER2013. Install it or use the recommended environment."
        ) from exc
    return pd


def parse_pixels_baseline(pixel_string: str) -> np.ndarray:
    parts = pixel_string.strip().split()
    if len(parts) != 48 * 48:
        raise ValueError(f"Expected 2304 pixels, found {len(parts)}.")
    image = np.empty((48, 48), dtype=np.float32)
    index = 0
    for row in range(48):
        for col in range(48):
            image[row, col] = float(int(parts[index]))
            index += 1
    return image


def parse_pixels_fast(pixel_string: str) -> np.ndarray:
    pixels = np.fromstring(pixel_string, sep=" ", dtype=np.float32)
    if pixels.size != 48 * 48:
        raise ValueError(f"Expected 2304 pixels, found {pixels.size}.")
    return pixels.reshape(48, 48)


def _load_one_split(
    frame,
    usage_name: str,
    parser: Callable[[str], np.ndarray],
    limit: int | None = None,
    progress_callback: Callable[[str], None] | None = None,
    progress_every: int = 500,
) -> SplitData:
    subset = frame.loc[frame["Usage"] == usage_name, ["emotion", "pixels"]]
    if limit is not None:
        subset = subset.iloc[:limit]
    num_rows = len(subset)
    images = np.empty((num_rows, 48, 48), dtype=np.float32)
    labels = np.empty(num_rows, dtype=np.int64)

    for out_idx, row in enumerate(subset.itertuples(index=False)):
        try:
            images[out_idx] = parser(row.pixels)
            labels[out_idx] = int(row.emotion)
        except Exception as exc:
            raise ValueError(
                f"Failed to parse row {out_idx} from split '{usage_name}': {exc}"
            ) from exc
        if progress_callback is not None and (
            (out_idx + 1) % progress_every == 0 or out_idx + 1 == num_rows
        ):
            progress_callback(f"Parsed {out_idx + 1}/{num_rows} rows from {usage_name}")

    return SplitData(images=images, labels=labels, usage=usage_name)


def load_fer2013(
    csv_path: str | Path,
    parser_name: str = "fast",
    limit_train: int | None = None,
    limit_public: int | None = None,
    limit_private: int | None = None,
    progress_callback: Callable[[str], None] | None = None,
    progress_every: int = 500,
) -> DatasetSplits:
    pd = _import_pandas()
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"FER2013 CSV not found: {csv_path}")

    parser_map: dict[str, Callable[[str], np.ndarray]] = {
        "baseline": parse_pixels_baseline,
        "fast": parse_pixels_fast,
    }
    if parser_name not in parser_map:
        raise ValueError(f"Unknown parser '{parser_name}'. Choose from {sorted(parser_map)}.")
    parser = parser_map[parser_name]

    if progress_callback is not None:
        progress_callback(f"Reading FER2013 CSV from {csv_path}")
    frame = pd.read_csv(csv_path)
    required_columns = {"emotion", "pixels", "Usage"}
    missing = required_columns.difference(frame.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    return DatasetSplits(
        training=_load_one_split(
            frame,
            "Training",
            parser,
            limit_train,
            progress_callback=progress_callback,
            progress_every=progress_every,
        ),
        public_test=_load_one_split(
            frame,
            "PublicTest",
            parser,
            limit_public,
            progress_callback=progress_callback,
            progress_every=progress_every,
        ),
        private_test=_load_one_split(
            frame,
            "PrivateTest",
            parser,
            limit_private,
            progress_callback=progress_callback,
            progress_every=progress_every,
        ),
    )
