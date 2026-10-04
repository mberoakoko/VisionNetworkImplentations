import typing

import grain
import numpy as np


def grain_pipeline(
    sensors: np.ndarray | typing.Sequence,
    mean: np.ndarray,
    std: np.ndarray,
    batch_size: int,
    seed: int,
):
    """Deterministic Grain map -> standardize -> batch -> prefetch iterator."""

    def standardize(row: np.ndarray) -> np.ndarray:
        return ((row - mean) / std).astype(np.float32)

    dataset = (
        grain.MapDataset.source(sensors)
        .shuffle(seed=seed)
        .map(standardize)
        .batch(batch_size, drop_remainder=True)
        .to_iter_dataset(grain.ReadOptions(num_threads=2, prefetch_buffer_size=8))
    )
    return dataset
