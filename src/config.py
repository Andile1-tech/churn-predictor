"""Central configuration: file paths, constants, and the data source URL."""
from pathlib import Path

# --- Paths ---------------------------------------------------------------
# __file__ is this file (src/config.py). .parent = src/. .parent.parent = project root.
ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = ROOT_DIR / "models"

RAW_DATA_PATH = RAW_DATA_DIR / "telco_churn.csv"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "churn_clean.csv"
MODEL_PATH = MODEL_DIR / "churn_model.joblib"

# --- Data source ---------------------------------------------------------
DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)

# --- Modeling constants --------------------------------------------------
TARGET_COLUMN = "Churn"
RANDOM_STATE = 42

# --- Side effect: make sure the folders exist ----------------------------
for _dir in (RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR):
    _dir.mkdir(parents=True, exist_ok=True)