from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OOF_DIR = PROJECT_ROOT / "oof"
SUBMISSION_DIR = PROJECT_ROOT / "submissions"

COMPETITION_SLUG = "playground-series-s6e11"
ID_COL = "id"
TARGET_COL = "is_fraud"

SEED = 42
N_FOLDS = 5


def ensure_dirs() -> None:
    for d in (DATA_RAW_DIR, DATA_PROCESSED_DIR, OOF_DIR, SUBMISSION_DIR):
        d.mkdir(parents=True, exist_ok=True)
