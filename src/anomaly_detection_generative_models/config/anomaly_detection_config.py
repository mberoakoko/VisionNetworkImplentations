from pathlib import Path

ROOT = Path(__file__).parent.parent / "artifacts/ppca_pdm"
ROOT.mkdir(parents=True, exist_ok=True)

DATA_PATH = ROOT / "machine_sensors.parquet"

FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)


OBS_DIM = 30
LATENT_DIM = 3
DT = 0.02
HEALTHY_MU = 1.4
DEGRADED_MU = 4.2
HEALTHY_NOISE = 0.04
DEGRADED_NOISE = 0.16
