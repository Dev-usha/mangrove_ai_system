import os

# =========================
# PROJECT PATHS
# =========================

PROJECT_DIR = os.getcwd()

DATA_DIR = os.path.join(PROJECT_DIR, "data")
DATASET_ROOT = os.path.join(DATA_DIR, "datamangrove")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

FEATURE_CSV = os.path.join(PROCESSED_DATA_DIR, "mangrove_features.csv")

MODEL_DIR = os.path.join(PROJECT_DIR, "saved_models")

OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")
PREDICTION_DIR = os.path.join(OUTPUT_DIR, "predictions")

# =========================
# FEATURE SETTINGS
# =========================

SAMPLES_PER_IMAGE = 300
PATCH_SIZE = 7
RANDOM_STATE = 42

# =========================
# CREATE REQUIRED FOLDERS
# =========================

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DATASET_ROOT, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(PREDICTION_DIR, exist_ok=True)