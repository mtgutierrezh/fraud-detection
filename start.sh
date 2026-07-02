#!/bin/bash
set -e

echo "[start.sh] Verificando archivos del modelo..."
if [ ! -f "models/xgboost_fraud_model.joblib" ]; then
    echo "ERROR: No se encuentra models/xgboost_fraud_model.joblib"
    echo "Ejecuta el pipeline primero o asegura que el archivo existe en el repositorio."
    exit 1
fi
if [ ! -f "models/model_metadata.json" ]; then
    echo "ERROR: No se encuentra models/model_metadata.json"
    exit 1
fi
echo "[start.sh] Modelo y metadatos encontrados. Iniciando Streamlit..."
exec streamlit run app.py --server.port="${PORT:-8501}" --server.address="0.0.0.0"
