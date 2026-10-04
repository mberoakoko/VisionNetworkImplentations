
import itertools

import numpy as np
from loguru import logger

from anomaly_detection_generative_models.config.anomaly_detection_config import (
    DATA_PATH,
)
from anomaly_detection_generative_models.models.ppca import PPCADetector, inspect_effective_k
from anomaly_detection_generative_models.simulation.simulate import simulate_machine
from anomaly_detection_generative_models.utils.io import (
    apply_standardize,
    healthy_split,
    load_parquet,
    standardize_fit,
    write_parquet,
)
from anomaly_detection_generative_models.utils.pipeline import grain_pipeline
from anomaly_detection_generative_models.utils.anomaly_detection_plotting_utils import plot_ard_column_norms

def _simulate_and_dump_to_disk() -> tuple:
    n_steps = 20_000
    logger.info(f"simulating {n_steps} steps ")
    sim = simulate_machine(n_steps)
    write_parquet(sim["sensors"], sim["degraded"], DATA_PATH)
    sensors, degraded = load_parquet(DATA_PATH)
    print("artifact", DATA_PATH, "bytes", DATA_PATH.stat().st_size, flush=True)
    return sensors, degraded

def main():
    if DATA_PATH.exists(): 
        sensors , degraded = _simulate_and_dump_to_disk()
        split = healthy_split(sensors, degraded)
        mean, std = standardize_fit(split["train"])
        train = apply_standardize(split["train"], mean, std)
        calib = apply_standardize(split["calib"], mean, std)
        eval_rows = apply_standardize(split["rows_after_train"], mean, std)
        eval_labels = split["labels_after_train"]

        detector = PPCADetector(latent_dim=10, steps=2000, batch_size=256)
        pipeline = grain_pipeline(train, mean, std, batch_size=256, seed=7)
        
        batches = np.array(list(itertools.islice(pipeline, 32)))
        logger.debug(batches[0])
        detector.fit(batches)

        logger.debug(f"{calib.shape=}")
        scores = detector.score(calib)
        logger.debug(f"{scores.shape}")
        plot_ard_column_norms(detector)





if __name__ == "__main__":
   main()
