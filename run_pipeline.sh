#!/bin/bash
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE="fraud-detection"

echo "========================================"
echo " Pipeline DataOps - Fraud Detection"
echo "========================================"

echo "[1/6] Construyendo imagen Docker..."
docker build --network host -t "$IMAGE" "$PROJECT_DIR"

echo "[2/6] 01_ingestion - Carga de datos crudos..."
docker run --rm --network host \
    -v "$PROJECT_DIR/data:/app/data" \
    -v "$PROJECT_DIR/logs:/app/logs" \
    "$IMAGE" python src/01_ingestion.py

echo "[3/6] 02_cleaning - Limpieza y enmascaramiento PII..."
docker run --rm --network host \
    -v "$PROJECT_DIR/data:/app/data" \
    -v "$PROJECT_DIR/logs:/app/logs" \
    "$IMAGE" python src/02_cleaning.py

echo "[4/6] 03_validation - Validacion estructural y semantica..."
docker run --rm --network host \
    -v "$PROJECT_DIR/data:/app/data" \
    -v "$PROJECT_DIR/logs:/app/logs" \
    "$IMAGE" python src/03_validation.py

echo "[5/6] 04_loading - Carga de datos finales..."
docker run --rm --network host \
    -v "$PROJECT_DIR/data:/app/data" \
    -v "$PROJECT_DIR/logs:/app/logs" \
    "$IMAGE" python src/04_loading.py

echo "[6/6] 05_model_training - Entrenamiento XGBoost..."
docker run --rm --network host \
    -v "$PROJECT_DIR/data:/app/data" \
    -v "$PROJECT_DIR/logs:/app/logs" \
    -v "$PROJECT_DIR/models:/app/models" \
    "$IMAGE" python src/05_model_training.py

echo "========================================"
echo " Pipeline completado exitosamente"
echo " Logs: $PROJECT_DIR/logs/pipeline.log"
echo " Modelo: $PROJECT_DIR/models/xgboost_fraud_model.joblib"
echo "========================================"
