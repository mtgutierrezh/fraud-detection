from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, f1_score, recall_score
from xgboost import XGBClassifier

from logger_config import setup_logging

logger = setup_logging(__name__)

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODELS_DIR / "xgboost_fraud_model.joblib"

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


def convertir_categoricas(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype("category")
    return df


def cargar_datos() -> pd.DataFrame:
    input_path = PROCESSED_DIR / "04_produccion.parquet"
    if not input_path.exists():
        raise FileNotFoundError(f"No se encontro {input_path}")

    logger.info("Cargando datos de produccion desde %s", input_path.name)
    df = pd.read_parquet(input_path)
    logger.info("Dataset cargado: %d filas, %d columnas", len(df), len(df.columns))

    columna_fecha = "trans_date_trans_time"
    if columna_fecha in df.columns and not pd.api.types.is_datetime64_any_dtype(df[columna_fecha]):
        df[columna_fecha] = pd.to_datetime(df[columna_fecha])

    return df


def split_temporal(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    columna_fecha = "trans_date_trans_time"
    if columna_fecha not in df.columns:
        raise KeyError(f"Columna '{columna_fecha}' requerida para split cronologico")

    df = df.sort_values(columna_fecha)
    corte = int(len(df) * 0.8)
    df_train = df.iloc[:corte]
    df_test = df.iloc[corte:]

    logger.info("Split temporal: Train %d filas (hasta %s), Test %d filas (desde %s)",
                len(df_train), df_train[columna_fecha].max(),
                len(df_test), df_test[columna_fecha].min())

    columnas_descartar = [c for c in COLUMNAS_A_DESCARTAR if c in df.columns]
    X_train = df_train.drop(columns=["is_fraud"] + columnas_descartar)
    y_train = df_train["is_fraud"]
    X_test = df_test.drop(columns=["is_fraud"] + columnas_descartar)
    y_test = df_test["is_fraud"]

    return X_train, X_test, y_train, y_test


def entrenar_modelo(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    ratio = (y_train == 0).sum() / (y_train == 1).sum()
    logger.info("Ratio de desbalance (no-fraude/fraude): %.2f", ratio)

    modelo = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=ratio,
        random_state=42,
        eval_metric="logloss",
        enable_categorical=True,
    )

    logger.info("Iniciando entrenamiento XGBoost (scale_pos_weight=%.2f)", ratio)
    modelo.fit(X_train, y_train)
    logger.info("Entrenamiento completado")

    return modelo


def evaluar_modelo(modelo: XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = modelo.predict(X_test)

    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    logger.info("=== Metricas del Modelo ===")
    logger.info("Recall:   %.4f", recall)
    logger.info("F1-Score: %.4f", f1)
    logger.info("Classification Report:\n%s", classification_report(y_test, y_pred, digits=4))

    return {"recall": recall, "f1_score": f1}


def guardar_modelo(modelo: XGBClassifier) -> Path:
    joblib.dump(modelo, MODEL_PATH)
    tamano_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
    logger.info("Modelo guardado en %s (%.2f MB)", MODEL_PATH, tamano_mb)
    return MODEL_PATH


def main() -> None:
    logger.info("=== INICIO PIPELINE: 05_model_training ===")
    try:
        df = cargar_datos()
        X_train, X_test, y_train, y_test = split_temporal(df)
        X_train = convertir_categoricas(X_train)
        X_test = convertir_categoricas(X_test)
        modelo = entrenar_modelo(X_train, y_train)
        metricas = evaluar_modelo(modelo, X_test, y_test)
        guardar_modelo(modelo)
        logger.info("Metricas finales - Recall: %.4f | F1-Score: %.4f",
                    metricas["recall"], metricas["f1_score"])
    except FileNotFoundError:
        logger.error("Archivo de datos no encontrado. Abortando entrenamiento.")
        raise
    except Exception:
        logger.exception("Error critico en entrenamiento. Abortando pipeline.")
        raise
    logger.info("=== FIN PIPELINE: 05_model_training ===")


if __name__ == "__main__":
    main()
