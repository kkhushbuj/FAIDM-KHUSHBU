from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Paths:
    project_root: Path = Path(__file__).resolve().parents[1]
    data_raw: Path = project_root / "data" / "raw"
    outputs: Path = project_root / "outputs"
    figures: Path = outputs / "figures"
    reports: Path = outputs / "reports"
    models: Path = outputs / "models"

CSV_NAME = "CDC Diabetes Dataset.csv"
TARGET_COL = "Diabetes_012"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Quick-run controls (safe + explainable)
SILHOUETTE_SAMPLE_N = 10000
PCA_SAMPLE_N = 20000
K_MIN = 2
K_MAX = 6
