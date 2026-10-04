from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

from anomaly_detection_generative_models.config.anomaly_detection_config import (
    OBS_DIM,
)


def _verify_successful_write(path: Path) -> None:
    """Validates that the Parquet file was actually written and contains data."""
    if not path.exists():
        raise OSError(f"Failed to write Parquet file at {path}")
    if path.stat().st_size == 0:
        raise OSError(f"Written Parquet file at {path} is empty (0 bytes).")

    logger.info("Write Successful. ")

def write_parquet(sensors: np.ndarray, degraded: np.ndarray, path: Path) -> None:
    
    if not isinstance(sensors, np.ndarray) or not isinstance(degraded, np.ndarray):
        raise TypeError("Inputs 'sensors' and 'degraded' must be numpy ndarrays.")


    num_rows = sensors.shape[0]
    if num_rows == 0:
        raise ValueError("Cannot write an empty array to Parquet.")

    logger.info(f"Writing parquet file to dist {path=}")
    
    table = pa.table(
        {
            **{f"s{i:02d}": sensors[:, i] for i in range(sensors.shape[1])},
            "degraded": degraded.astype(np.int8),
        }
    )
    pq.write_table(table, path, compression="snappy")

    _verify_successful_write(path)

def load_parquet(path: Path) -> tuple[np.ndarray, np.ndarray]:
    table = pq.read_table(path)
    degraded = table.column("degraded").to_numpy()
    cols = [table.column(f"s{i:02d}").to_numpy() for i in range(OBS_DIM)]
    return np.stack(cols, axis=1).astype(np.float32), degraded


def healthy_split(sensors: np.ndarray, degraded: np.ndarray, train_frac: float = 0.8):
    """Train and calibrate are both healthy. Worn rows are evaluation only."""
    cut = int(np.argmax(degraded))
    healthy = sensors[:cut]
    n_train = int(train_frac * len(healthy))
    return {
        "train": healthy[:n_train],
        "calib": healthy[n_train:],
        "worn": sensors[cut:],
        "cut": cut,
        "labels_after_train": degraded[n_train:],
        "rows_after_train": sensors[n_train:],
    }


def standardize_fit(train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return train.mean(axis=0), train.std(axis=0) + 1e-6


def apply_standardize(rows: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((rows - mean) / std).astype(np.float32)
