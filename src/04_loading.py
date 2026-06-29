import sqlite3
from pathlib import Path

import pandas as pd

from config import COLUMNA_FECHA, PROCESSED_DIR
from logger_config import setup_logging

logger = setup_logging(__name__)

PROCESSED_DIR.mkdir(exist_ok=True)


def cargar_datos_validados() -> pd.DataFrame:
    input_path = PROCESSED_DIR / "03_validated_data.parquet"
    if not input_path.exists():
        raise FileNotFoundError(f"No se encontro {input_path}. Ejecute 03_validation.py primero.")

    logger.info("Cargando datos validados desde %s", input_path.name)
    df = pd.read_parquet(input_path)
    logger.info("Dataset cargado: %d filas, %d columnas", len(df), len(df.columns))
    return df


def exportar_parquet(df: pd.DataFrame) -> Path:
    output_path = PROCESSED_DIR / "04_produccion.parquet"
    df.to_parquet(output_path, index=False)
    tamano_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("Exportacion Parquet: %s (%.2f MB, %d filas)", output_path, tamano_mb, len(df))
    return output_path


def exportar_csv(df: pd.DataFrame) -> Path:
    output_path = PROCESSED_DIR / "04_produccion.csv"
    df.to_csv(output_path, index=False)
    tamano_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("Exportacion CSV: %s (%.2f MB, %d filas)", output_path, tamano_mb, len(df))
    return output_path


def simular_carga_sqlite(df: pd.DataFrame) -> None:
    db_path = PROCESSED_DIR / "fraud_detection.db"
    logger.info("Simulando carga en SQLite: %s", db_path)
    conn = sqlite3.connect(str(db_path))
    df.to_sql("transacciones", conn, if_exists="replace", index=False)
    count = conn.execute("SELECT COUNT(*) FROM transacciones").fetchone()[0]
    logger.info("SQLite: %d registros insertados en tabla 'transacciones'", count)
    conn.close()
    tamano_mb = db_path.stat().st_size / (1024 * 1024)
    logger.info("Base de datos SQLite: %s (%.2f MB)", db_path.name, tamano_mb)


def main() -> None:
    logger.info("=== INICIO PIPELINE: 04_loading ===")
    try:
        df = cargar_datos_validados()
        parquet_path = exportar_parquet(df)
        csv_path = exportar_csv(df)
        simular_carga_sqlite(df)

        logger.info("Pipeline completado exitosamente. Archivos generados:")
        logger.info("  - %s", parquet_path)
        logger.info("  - %s", csv_path)
    except FileNotFoundError:
        logger.error("Archivo de entrada no encontrado. Abortando carga.")
        raise
    except Exception:
        logger.exception("Error critico en etapa de carga. Abortando pipeline.")
        raise
    logger.info("=== FIN PIPELINE: 04_loading === (%d registros en produccion)", len(df))


if __name__ == "__main__":
    main()
