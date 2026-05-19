import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
IMAGE = "fraud-detection"

STAGES = [
    ("01_ingestion", "Carga de datos crudos"),
    ("02_cleaning", "Limpieza y enmascaramiento PII"),
    ("03_validation", "Validacion estructural y semantica"),
    ("04_loading", "Carga de datos finales"),
    ("05_model_training", "Entrenamiento XGBoost"),
]

DOCKER_RUN = [
    "docker", "run", "--rm", "--network", "host",
    "-v", f"{PROJECT_DIR / 'data'}:/app/data",
    "-v", f"{PROJECT_DIR / 'logs'}:/app/logs",
    "-v", f"{PROJECT_DIR / 'models'}:/app/models",
]


def run(cmd: list[str], desc: str) -> int:
    print(f"[{desc}] Ejecutando...", flush=True)
    result = subprocess.run(cmd, cwd=str(PROJECT_DIR))
    if result.returncode != 0:
        print(f"[{desc}] ERROR (codigo {result.returncode})", flush=True)
    else:
        print(f"[{desc}] OK", flush=True)
    return result.returncode


def main() -> int:
    print("=" * 50)
    print(" Pipeline DataOps - Fraud Detection")
    print("=" * 50)
    print(f"Directorio: {PROJECT_DIR}")
    print()

    print("[BUILD] Construyendo imagen Docker...", flush=True)
    build = subprocess.run(
        ["docker", "build", "-t", IMAGE, str(PROJECT_DIR)],
        cwd=str(PROJECT_DIR),
    )
    if build.returncode != 0:
        print("[BUILD] ERROR al construir la imagen", flush=True)
        return 1
    print("[BUILD] OK")
    print()

    for script, desc in STAGES:
        cmd = DOCKER_RUN + [IMAGE, "python", f"src/{script}.py"]
        rc = run(cmd, desc)
        if rc != 0:
            print(f"\nPipeline abortado en etapa {script}.", flush=True)
            return rc
        print()

    print("=" * 50)
    print(" Pipeline completado exitosamente")
    print(f" Logs:    {PROJECT_DIR / 'logs' / 'pipeline.log'}")
    print(f" Modelo:  {PROJECT_DIR / 'models' / 'xgboost_fraud_model.joblib'}")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    sys.exit(main())
