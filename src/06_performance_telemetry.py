#!/usr/bin/env python3
"""
Telemetria de Rendimiento - Pipeline DataOps
Captura CPU, RAM y tiempo de ejecucion por etapa.
Salida: logs/performance_metrics.json
"""
import json
import subprocess
import time
from pathlib import Path

import psutil

from config import LOGS_DIR, PROJECT_ROOT
from logger_config import setup_logging

logger = setup_logging(__name__)

STAGES = [
    {"script": "01_ingestion.py", "name": "Ingestion de datos crudos"},
    {"script": "02_cleaning.py", "name": "Limpieza y enmascaramiento PII"},
    {"script": "03_validation.py", "name": "Validacion estructural y semantica"},
    {"script": "04_loading.py", "name": "Carga de datos finales"},
    {"script": "05_model_training.py", "name": "Entrenamiento XGBoost"},
]


def get_system_metrics() -> dict:
    mem = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=0.3)
    return {
        "cpu_percent": cpu,
        "memory_percent": mem.percent,
        "memory_used_mb": round(mem.used / (1024 * 1024), 2),
        "memory_total_mb": round(mem.total / (1024 * 1024), 2),
        "memory_available_mb": round(mem.available / (1024 * 1024), 2),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
    }


def run_pipeline_with_telemetry() -> dict:
    metrics = {
        "pipeline_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "hostname": subprocess.run(
            ["hostname"], capture_output=True, text=True
        ).stdout.strip(),
        "stages": [],
        "system_baseline": get_system_metrics(),
    }

    total_start = time.time()

    for stage in STAGES:
        script_path = PROJECT_ROOT / "src" / stage["script"]
        if not script_path.exists():
            logger.error("Script no encontrado: %s", script_path)
            metrics["stages"].append({
                "script": stage["script"],
                "name": stage["name"],
                "error": "Script no encontrado",
            })
            continue

        logger.info("Ejecutando etapa: %s (%s)", stage["name"], stage["script"])
        stage_start = time.time()
        stage_metrics_before = get_system_metrics()

        result = subprocess.run(
            ["python", str(script_path)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )

        stage_end = time.time()
        stage_metrics_after = get_system_metrics()
        elapsed_ms = round((stage_end - stage_start) * 1000, 2)

        stage_record = {
            "script": stage["script"],
            "name": stage["name"],
            "elapsed_ms": elapsed_ms,
            "elapsed_seconds": round(elapsed_ms / 1000, 2),
            "exit_code": result.returncode,
            "cpu_before": stage_metrics_before["cpu_percent"],
            "cpu_after": stage_metrics_after["cpu_percent"],
            "memory_before_mb": stage_metrics_before["memory_used_mb"],
            "memory_after_mb": stage_metrics_after["memory_used_mb"],
            "memory_delta_mb": round(
                stage_metrics_after["memory_used_mb"]
                - stage_metrics_before["memory_used_mb"], 2
            ),
        }
        metrics["stages"].append(stage_record)

        status = "OK" if result.returncode == 0 else "ERROR"
        logger.info(
            "  -> %s (%sms, CPU: %.1f%%, RAM: %.0fMB)",
            status, f"{elapsed_ms:,.0f}",
            stage_metrics_after["cpu_percent"],
            stage_metrics_after["memory_used_mb"],
        )

    total_elapsed = round((time.time() - total_start) * 1000, 2)
    metrics["total_elapsed_ms"] = total_elapsed
    metrics["total_elapsed_seconds"] = round(total_elapsed / 1000, 2)
    metrics["system_final"] = get_system_metrics()

    output_path = LOGS_DIR / "performance_metrics.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Telemetria guardada en %s", output_path)
    logger.info("Tiempo total del pipeline: %.2fs", metrics["total_elapsed_seconds"])

    return metrics


if __name__ == "__main__":
    run_pipeline_with_telemetry()
