import logging
import sys
from pathlib import Path

import pandas as pd

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(funcName)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "pipeline.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

VALORES_PERMITIDOS_IS_FRAUD = {0, 1}


def cargar_datos_limpios() -> pd.DataFrame:
    input_path = PROCESSED_DIR / "02_cleaned_data.parquet"
    if not input_path.exists():
        raise FileNotFoundError(f"No se encontro {input_path}. Ejecute 02_cleaning.py primero.")

    logger.info("Cargando datos limpios desde %s", input_path.name)
    df = pd.read_parquet(input_path)
    logger.info("Dataset cargado: %d filas, %d columnas", len(df), len(df.columns))
    return df


def validacion_estructural(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("=== INICIO: Validacion Estructural ===")
    errores = []
    filas_antes = len(df)

    if "amt" in df.columns and not pd.api.types.is_float_dtype(df["amt"]):
        no_float = df[~df["amt"].apply(lambda x: isinstance(x, float))]
        errores.append(no_float)
        logger.warning("amt: %d registros no son Float", len(no_float))
    else:
        logger.info("amt: tipo Float OK")

    if "zip" in df.columns and not pd.api.types.is_integer_dtype(df["zip"]):
        no_int = df[~df["zip"].apply(lambda x: isinstance(x, (int, float)) and x == int(x))]
        errores.append(no_int)
        logger.warning("zip: %d registros no son Integer", len(no_int))
    else:
        logger.info("zip: tipo Integer OK")

    df_err = pd.concat(errores).drop_duplicates() if errores else pd.DataFrame()
    df_valido = df.drop(df_err.index) if not df_err.empty else df

    logger.info("Estructural: %d filas rechazadas de %d", len(df_err), filas_antes)
    logger.info("=== FIN: Validacion Estructural ===")
    return df_valido, df_err


def validacion_semantica(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("=== INICIO: Validacion Semantica ===")
    filas_antes = len(df)
    mascara = pd.Series(True, index=df.index)

    if "amt" in df.columns:
        mascara_amt = df["amt"] > 0
        invalidos_amt = (~mascara_amt).sum()
        if invalidos_amt > 0:
            logger.warning("amt <= 0: %d registros invalidos", invalidos_amt)
        else:
            logger.info("amt > 0: OK")
        mascara &= mascara_amt

    if "is_fraud" in df.columns:
        mascara_fraud = df["is_fraud"].isin(VALORES_PERMITIDOS_IS_FRAUD)
        invalidos_fraud = (~mascara_fraud).sum()
        if invalidos_fraud > 0:
            logger.warning("is_fraud no esta en {0,1}: %d registros invalidos", invalidos_fraud)
        else:
            logger.info("is_fraud en {0,1}: OK")
        mascara &= mascara_fraud

    df_valido = df[mascara]
    df_err = df[~mascara]

    logger.info("Semantica: %d filas rechazadas de %d", len(df_err), filas_antes)
    logger.info("=== FIN: Validacion Semantica ===")
    return df_valido, df_err


def guardar_validado(df: pd.DataFrame) -> Path:
    output_path = PROCESSED_DIR / "03_validated_data.parquet"
    df.to_parquet(output_path, index=False)
    logger.info("Dataset validado guardado en %s (%d filas)", output_path, len(df))
    return output_path


def main() -> pd.DataFrame:
    logger.info("=== INICIO PIPELINE: 03_validation ===")
    try:
        df = cargar_datos_limpios()

        df, err_estructural = validacion_estructural(df)
        df, err_semantica = validacion_semantica(df)

        total_rechazadas = len(err_estructural) + len(err_semantica)
        logger.info("Resumen validacion: %d filas aceptadas, %d rechazadas en total",
                    len(df), total_rechazadas)

        output = guardar_validado(df)
    except FileNotFoundError:
        logger.error("Archivo de entrada no encontrado. Abortando validacion.")
        raise
    except Exception:
        logger.exception("Error critico en etapa de validacion. Abortando pipeline.")
        raise
    logger.info("=== FIN PIPELINE: 03_validation === (%s)", output.name)
    return df


if __name__ == "__main__":
    main()
