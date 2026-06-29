import json
import sqlite3
from datetime import datetime

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from xgboost import XGBClassifier

from config import (
    COLUMNA_FECHA,
    COLUMNA_OBJETIVO,
    COLUMNAS_A_DESCARTAR,
    DATA_DIR,
    META_PATH,
    MODEL_PATH,
    MODELS_DIR,
    PROCESSED_DIR,
    REPORTS_DIR,
    SMOTE_K_NEIGHBORS,
    SMOTE_RANDOM_STATE,
    SMOTE_SAMPLING_STRATEGY,
    SPLIT_TRAIN_FRAC,
    THRESHOLD_BUSINESS,
    THRESHOLD_DEFAULT,
    XGB_PARAMS,
)
from logger_config import setup_logging

logger = setup_logging(__name__)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def feature_engineering(df: pd.DataFrame, cat_means: dict) -> pd.DataFrame:
    df = df.copy()
    nuevas = {}

    if all(c in df.columns for c in ["lat", "long", "merch_lat", "merch_long"]):
        df["dist_km"] = df.apply(
            lambda r: haversine_km(r["lat"], r["long"], r["merch_lat"], r["merch_long"]), axis=1
        )
        nuevas["dist_km"] = "Distancia cliente-comercio (Haversine km)"

    if COLUMNA_FECHA in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df[COLUMNA_FECHA]):
            df[COLUMNA_FECHA] = pd.to_datetime(df[COLUMNA_FECHA])
        df["hora"] = df[COLUMNA_FECHA].dt.hour
        df["dia_semana"] = df[COLUMNA_FECHA].dt.dayofweek
        nuevas["hora"] = "Hora del dia (0-23)"
        nuevas["dia_semana"] = "Dia de la semana (0=Lunes)"

    if "category" in df.columns and "amt" in df.columns:
        df["amt_vs_cat_mean"] = df.apply(
            lambda r: r["amt"] - cat_means.get(r["category"], r["amt"]), axis=1
        )
        nuevas["amt_vs_cat_mean"] = "Diferencia vs monto promedio de la categoria"

    logger.info("Feature engineering: %d nuevas variables generadas", len(nuevas))
    for col, desc in nuevas.items():
        logger.info("  - %s: %s", col, desc)

    return df


def compute_cat_means(df_train: pd.DataFrame) -> dict:
    cat_means = df_train.groupby("category")["amt"].mean().to_dict()
    logger.info("cat_means calculadas sobre TRAIN: %d categorias", len(cat_means))
    return cat_means


def cargar_datos() -> pd.DataFrame:
    input_path = DATA_DIR / "processed" / "04_produccion.parquet"
    if not input_path.exists():
        raise FileNotFoundError(f"No se encontro {input_path}")
    logger.info("Cargando datos de produccion desde %s", input_path.name)
    df = pd.read_parquet(input_path)
    logger.info("Dataset cargado: %d filas, %d columnas", len(df), len(df.columns))
    return df


def split_temporal(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if COLUMNA_FECHA not in df.columns:
        raise KeyError(f"Columna '{COLUMNA_FECHA}' requerida para split cronologico")
    df = df.sort_values(COLUMNA_FECHA)
    corte = int(len(df) * SPLIT_TRAIN_FRAC)
    df_train = df.iloc[:corte]
    df_test = df.iloc[corte:]
    logger.info(
        "Split temporal: Train %d filas (hasta %s), Test %d filas (desde %s)",
        len(df_train), df_train[COLUMNA_FECHA].max(),
        len(df_test), df_test[COLUMNA_FECHA].min(),
    )
    return df_train, df_test


def preparar_features(
    df: pd.DataFrame, cat_means: dict | None = None, *, entrenamiento: bool = False
) -> tuple[pd.DataFrame, pd.Series | None, dict | None]:
    if entrenamiento:
        cat_means = compute_cat_means(df)
        df_feat = feature_engineering(df, cat_means)
    else:
        if cat_means is None:
            raise ValueError("cat_means es obligatorio para datos que no son de entrenamiento")
        df_feat = feature_engineering(df, cat_means)

    columnas_descartar = [c for c in COLUMNAS_A_DESCARTAR if c in df.columns]
    y = df_feat[COLUMNA_OBJETIVO] if COLUMNA_OBJETIVO in df_feat.columns else None
    X = df_feat.drop(columns=[COLUMNA_OBJETIVO] + columnas_descartar, errors="ignore")
    return X, y, cat_means


def codificar_categoricas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_categorical_dtype(df[col]):
            df[col] = df[col].cat.codes
        elif df[col].dtype == "object":
            df[col] = df[col].astype("category").cat.codes
    return df


def balancear_smote(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    conteo = y_train.value_counts()
    logger.info(
        "Antes de SMOTE - clase 0: %d, clase 1: %d (ratio: %.1f:1)",
        conteo.get(0, 0), conteo.get(1, 0), conteo.get(0, 0) / max(conteo.get(1, 0), 1),
    )

    smote = SMOTE(
        sampling_strategy=SMOTE_SAMPLING_STRATEGY,
        k_neighbors=SMOTE_K_NEIGHBORS,
        random_state=SMOTE_RANDOM_STATE,
    )
    X_res, y_res = smote.fit_resample(X_train, y_train)

    conteo_res = y_res.value_counts()
    logger.info(
        "Post SMOTE - clase 0: %d, clase 1: %d (ratio: %.1f:1)",
        conteo_res.get(0, 0), conteo_res.get(1, 0), conteo_res.get(0, 0) / max(conteo_res.get(1, 0), 1),
    )
    return X_res, y_res


def entrenar_modelo(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    logger.info("Ratio de desbalance post-SMOTE: %.2f", ratio)

    modelo = XGBClassifier(**XGB_PARAMS)
    logger.info("Entrenando XGBoost (eval_metric=aucpr, %d arboles)", modelo.n_estimators)
    modelo.fit(X_train, y_train)
    logger.info("Entrenamiento completado")
    return modelo


def _evaluar_umbral(
    y_test: pd.Series, y_proba: np.ndarray, threshold: float
) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "threshold": threshold,
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": cm.tolist(),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }


def _calcular_auc_roc(y_test: pd.Series, y_proba: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    auc_roc = auc(fpr, tpr)
    return auc_roc, fpr, tpr, thresholds


def _gini_from_auc(auc_roc: float) -> float:
    return 2.0 * auc_roc - 1.0


def evaluar_modelo(modelo: XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred_default = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1]

    auc_roc, fpr, tpr, _ = _calcular_auc_roc(y_test, y_proba)
    gini = _gini_from_auc(auc_roc)

    logger.info("=== AUC-ROC y Gini ===")
    logger.info("AUC-ROC: %.4f", auc_roc)
    logger.info("Gini:    %.4f", gini)

    metricas = {
        "auc_roc": auc_roc,
        "gini": gini,
    }

    for threshold in [THRESHOLD_DEFAULT, THRESHOLD_BUSINESS]:
        res = _evaluar_umbral(y_test, y_proba, threshold)
        umbral_key = str(threshold).replace(".", "_")
        metricas[f"umbral_{umbral_key}"] = res

        logger.info("=== Metricas del Modelo (umbral=%.2f) ===", threshold)
        logger.info("Accuracy:  %.4f", res["accuracy"])
        logger.info("Precision: %.4f", res["precision"])
        logger.info("Recall:    %.4f", res["recall"])
        logger.info("F1-Score:  %.4f", res["f1_score"])
        logger.info("Matriz de Confusion:\n%s", np.array(res["confusion_matrix"]))

        if threshold == THRESHOLD_DEFAULT:
            logger.info("Classification Report:\n%s", classification_report(y_test, y_pred_default, digits=4))

    return metricas, fpr, tpr


def _plot_confusion_matrix(cm: list, threshold: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))

    cm_arr = np.array(cm)
    ax.imshow(cm_arr, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(f"Matriz de Confusion (umbral={threshold:.2f})", fontsize=13, pad=15)
    ax.set_xlabel("Prediccion", fontsize=11)
    ax.set_ylabel("Real", fontsize=11)

    tick_marks = [0, 1]
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(["Legitima (0)", "Fraude (1)"])
    ax.set_yticklabels(["Legitima (0)", "Fraude (1)"])

    fmt = "d"
    thresh = cm_arr.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, format(cm_arr[i, j], fmt),
                ha="center", va="center",
                color="white" if cm_arr[i, j] > thresh else "black",
                fontsize=14, fontweight="bold",
            )

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Matriz de confusion guardada en %s", path)


def _plot_roc_curve(
    fpr: np.ndarray, tpr: np.ndarray, auc_roc: float, path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(
        fpr, tpr, color="darkorange", lw=2.5,
        label=f"XGBoost (AUC = {auc_roc:.4f})",
    )
    ax.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Clasificador aleatorio")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Tasa de Falsos Positivos (FPR)", fontsize=11)
    ax.set_ylabel("Tasa de Verdaderos Positivos (TPR)", fontsize=11)
    ax.set_title("Curva ROC - XGBoost Fraud Detector", fontsize=13, pad=15)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Curva ROC guardada en %s", path)


def _plot_correlation_matrix(X_encoded: pd.DataFrame, path: Path) -> None:
    num_cols = X_encoded.select_dtypes(include=[np.number]).columns
    if len(num_cols) < 2:
        logger.warning("Menos de 2 columnas numericas, no se genera matriz de correlacion")
        return

    corr = X_encoded[num_cols].corr()
    n_vars = len(num_cols)
    fig_height = max(5, n_vars * 0.6)
    fig_width = max(6, n_vars * 0.6)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(corr, interpolation="nearest", cmap=plt.cm.RdBu_r, vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax.set_xticks(range(n_vars))
    ax.set_yticks(range(n_vars))
    ax.set_xticklabels(num_cols, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(num_cols, fontsize=7)
    ax.set_title("Matriz de Correlacion - Features Numericas", fontsize=13, pad=15)

    for i in range(n_vars):
        for j in range(n_vars):
            val = corr.iloc[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color=color, fontsize=6, fontweight="bold")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Matriz de correlacion guardada en %s (%d variables)", path, n_vars)


def enriquecer_sqlite(
    df_test_raw: pd.DataFrame,
    y_test: pd.Series,
    y_proba: np.ndarray,
    metricas: dict,
) -> None:
    db_path = PROCESSED_DIR / "fraud_detection.db"
    if not db_path.exists():
        logger.warning("Base SQLite no encontrada en %s. Omitiendo enrichment.", db_path)
        return

    logger.info("Enriqueciendo base SQLite con predicciones y metricas...")
    conn = sqlite3.connect(str(db_path))

    test_ids = df_test_raw.index[: len(y_test)]
    preds_default = (y_proba >= THRESHOLD_DEFAULT).astype(int)
    preds_business = (y_proba >= THRESHOLD_BUSINESS).astype(int)

    df_pred = pd.DataFrame({
        "row_id": test_ids,
        "prob_fraude": y_proba,
        "pred_default": preds_default,
        "pred_business": preds_business,
        "real": y_test.values,
    })
    df_pred.to_sql("predicciones", conn, if_exists="replace", index=False)
    n_pred = len(df_pred)
    logger.info("Tabla 'predicciones': %d registros insertados", n_pred)

    ults = metricas["umbral_0_5"]
    ultb = metricas["umbral_0_25"]

    df_metricas = pd.DataFrame([{
        "timestamp": datetime.now().isoformat(),
        "auc_roc": round(metricas["auc_roc"], 6),
        "gini": round(metricas["gini"], 6),
        "accuracy_default": round(ults["accuracy"], 6),
        "precision_default": round(ults["precision"], 6),
        "recall_default": round(ults["recall"], 6),
        "f1_default": round(ults["f1_score"], 6),
        "accuracy_business": round(ultb["accuracy"], 6),
        "precision_business": round(ultb["precision"], 6),
        "recall_business": round(ultb["recall"], 6),
        "f1_business": round(ultb["f1_score"], 6),
    }])
    df_metricas.to_sql("modelo_metricas", conn, if_exists="replace", index=False)
    conn.close()
    logger.info("Tabla 'modelo_metricas': metricas historicas almacenadas")


def guardar_modelo(
    modelo: XGBClassifier,
    cat_means: dict,
    columnas: list,
    metricas: dict,
) -> Path:
    joblib.dump(modelo, MODEL_PATH)
    logger.info("Modelo guardado en %s (%.2f MB)", MODEL_PATH, MODEL_PATH.stat().st_size / (1024 * 1024))

    metadata = {
        "cat_means": cat_means,
        "feature_columns": columnas,
        "auc_roc": metricas.get("auc_roc"),
        "gini": metricas.get("gini"),
        "threshold_default": THRESHOLD_DEFAULT,
        "threshold_business": THRESHOLD_BUSINESS,
    }
    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Metadata guardada en %s (%d features)", META_PATH, len(columnas))
    return MODEL_PATH


def main() -> None:
    logger.info("=== INICIO PIPELINE: 05_model_training ===")
    try:
        df = cargar_datos()
        df_train, df_test = split_temporal(df)

        X_train, y_train, cat_means = preparar_features(df_train, entrenamiento=True)
        X_test, y_test, _ = preparar_features(df_test, cat_means)

        X_train = codificar_categoricas(X_train)
        X_test_encoded = codificar_categoricas(X_test)

        corr_path = REPORTS_DIR / "correlation_matrix.png"
        _plot_correlation_matrix(X_test_encoded, corr_path)

        X_train, y_train = balancear_smote(X_train, y_train)

        modelo = entrenar_modelo(X_train, y_train)
        metricas, fpr, tpr = evaluar_modelo(modelo, X_test_encoded, y_test)

        y_proba = modelo.predict_proba(X_test_encoded)[:, 1]
        enriquecer_sqlite(df_test, y_test, y_proba, metricas)

        for threshold in [THRESHOLD_DEFAULT, THRESHOLD_BUSINESS]:
            umbral_key = str(threshold).replace(".", "_")
            cm = metricas["umbral_" + umbral_key]["confusion_matrix"]
            cm_path = REPORTS_DIR / f"confusion_matrix_umbral_{umbral_key}.png"
            _plot_confusion_matrix(cm, threshold, cm_path)

        roc_path = REPORTS_DIR / "roc_curve.png"
        _plot_roc_curve(fpr, tpr, metricas["auc_roc"], roc_path)

        guardar_modelo(modelo, cat_means, list(X_train.columns), metricas)

        logger.info("=== RESUMEN FINAL ===")
        logger.info("AUC-ROC: %.4f | Gini: %.4f", metricas["auc_roc"], metricas["gini"])
        for threshold in [THRESHOLD_DEFAULT, THRESHOLD_BUSINESS]:
            umbral_key = str(threshold).replace(".", "_")
            m = metricas["umbral_" + umbral_key]
            logger.info(
                "Umbral=%.2f -> Recall: %.4f | Precision: %.4f | F1: %.4f",
                threshold, m["recall"], m["precision"], m["f1_score"],
            )
    except FileNotFoundError:
        logger.error("Archivo de datos no encontrado. Abortando entrenamiento.")
        raise
    except Exception:
        logger.exception("Error critico en entrenamiento. Abortando pipeline.")
        raise
    logger.info("=== FIN PIPELINE: 05_model_training ===")


if __name__ == "__main__":
    main()
