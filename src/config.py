from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODELS_DIR / "xgboost_fraud_model.joblib"
META_PATH = MODELS_DIR / "model_metadata.json"

THRESHOLD_DEFAULT = 0.5
THRESHOLD_BUSINESS = 0.25

COLUMNAS_A_DESCARTAR = [
    "trans_date_trans_time",
    "trans_num",
    "dob",
    "unix_time",
    "cc_num",
    "first",
    "last",
    "street",
]

COLUMNAS_NECESARIAS = ["lat", "long", "merch_lat", "merch_long", "trans_date_trans_time"]

COLUMNA_FECHA = "trans_date_trans_time"
COLUMNA_OBJETIVO = "is_fraud"

SMOTE_SAMPLING_STRATEGY = 0.1
SMOTE_K_NEIGHBORS = 3
SMOTE_RANDOM_STATE = 42

XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 5,
    "learning_rate": 0.03,
    "random_state": 42,
    "eval_metric": "aucpr",
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "max_delta_step": 1,
}

SPLIT_TRAIN_FRAC = 0.8
