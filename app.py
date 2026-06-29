import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODELS_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODELS_DIR / "xgboost_fraud_model.joblib"
META_PATH = MODELS_DIR / "model_metadata.json"
SAMPLE_PATH = Path(__file__).resolve().parent / "data" / "sample_test.csv"

THRESHOLD = 0.25
AUC_ROC = None
GINI = None

CATEGORIAS = [
    "food_dining", "gas_transport", "grocery_net", "grocery_pos",
    "health_fitness", "home", "kids_pets", "misc_net", "misc_pos",
    "personal_care", "shopping_net", "shopping_pos", "travel",
    "entertainment",
]

ESTADOS = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


@st.cache_resource
def cargar_modelo_y_metadata():
    global THRESHOLD, AUC_ROC, GINI

    if not MODEL_PATH.exists():
        st.error(f"Modelo no encontrado en {MODEL_PATH}. Ejecute 05_model_training.py primero.")
        st.stop()

    modelo = joblib.load(MODEL_PATH)

    if META_PATH.exists():
        with open(META_PATH) as f:
            metadata = json.load(f)
        cat_means = metadata.get("cat_means", {})
        feature_cols = metadata.get("feature_columns", [])
        THRESHOLD = metadata.get("threshold_business", 0.25)
        AUC_ROC = metadata.get("auc_roc")
        GINI = metadata.get("gini")
    else:
        cat_means = {}
        feature_cols = list(modelo.get_booster().feature_names) if modelo.get_booster().feature_names else []

    return modelo, cat_means, feature_cols


def feature_engineering_input(df: pd.DataFrame, cat_means: dict) -> pd.DataFrame:
    if all(c in df.columns for c in ["lat", "long", "merch_lat", "merch_long"]):
        df["dist_km"] = df.apply(
            lambda r: haversine_km(r["lat"], r["long"], r["merch_lat"], r["merch_long"]), axis=1
        )

    if "trans_date_trans_time" in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df["trans_date_trans_time"]):
            df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
        df["hora"] = df["trans_date_trans_time"].dt.hour
        df["dia_semana"] = df["trans_date_trans_time"].dt.dayofweek
    else:
        now = pd.Timestamp.now()
        df["hora"] = now.hour
        df["dia_semana"] = now.dayofweek

    if "category" in df.columns and "amt" in df.columns:
        df["amt_vs_cat_mean"] = df.apply(
            lambda r: r["amt"] - cat_means.get(r["category"], 0), axis=1
        )

    return df


def codificar_input(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype("category")
        if pd.api.types.is_categorical_dtype(df[col]):
            df[col] = df[col].cat.codes
    return df


def alinear_columnas(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    return df[feature_cols]


st.set_page_config(page_title="Fraud Detection", layout="centered")
st.title("Fraud Detection - Pipeline DataOps")
st.markdown("Simula una transaccion y el modelo XGBoost predice si es **fraudulenta** o **legitima**.")

modelo, cat_means, feature_cols = cargar_modelo_y_metadata()
if AUC_ROC is not None and GINI is not None:
    st.success(f"Modelo XGBoost cargado ({len(feature_cols)} features, umbral={THRESHOLD:.0%}, AUC={AUC_ROC:.4f}, Gini={GINI:.4f})")
else:
    st.success(f"Modelo XGBoost cargado ({len(feature_cols)} features, umbral={THRESHOLD:.0%})")

tab1, tab2 = st.tabs(["Formulario Manual", "Carga de CSV"])

with tab1:
    with st.form("transaccion_form"):
        st.subheader("Datos de la Transaccion")
        col1, col2 = st.columns(2)

        with col1:
            amt = st.number_input("Monto (USD)", min_value=0.01, value=100.0, step=10.0)
            category = st.selectbox("Categoria", CATEGORIAS)
            gender = st.selectbox("Genero", ["M", "F"])
            city_pop = st.number_input("Poblacion de la ciudad", min_value=0, value=50000, step=1000)

        with col2:
            state = st.selectbox("Estado", ESTADOS)
            zip_code = st.number_input("Codigo Postal", min_value=0, value=10001, step=1)
            lat = st.number_input("Latitud", value=40.71, step=0.01, format="%.4f")
            long = st.number_input("Longitud", value=-74.01, step=0.01, format="%.4f")

        submitted = st.form_submit_button("Predecir", type="primary")

    if submitted:
        with st.spinner("Calculando features y analizando transaccion..."):
            data = {
                "amt": amt, "category": category, "gender": gender,
                "state": state, "zip": zip_code,
                "lat": lat, "long": long, "city_pop": city_pop,
                "city": "Unknown", "job": "Unknown", "merchant": "Unknown",
                "merch_lat": lat + 0.005, "merch_long": long - 0.005,
            }

            try:
                df = pd.DataFrame([data])
                df = feature_engineering_input(df, cat_means)
                df = codificar_input(df)
                df = alinear_columnas(df, feature_cols)

                proba = modelo.predict_proba(df)[0]
                pred = 1 if proba[1] >= THRESHOLD else 0

                col_res1, col_res2, col_res3 = st.columns(3)
                with col_res1:
                    st.metric("Prob. Fraude", f"{proba[1]:.2%}")
                with col_res2:
                    st.metric("Prob. Legitima", f"{proba[0]:.2%}")
                with col_res3:
                    st.metric("Umbral", f"{THRESHOLD:.0%}")

                if pred == 1:
                    st.error(f"FRAUDE DETECTADO (prob: {proba[1]:.2%} >= umbral {THRESHOLD:.0%})")
                else:
                    st.success(f"Transaccion Legitima (prob: {proba[1]:.2%} < umbral {THRESHOLD:.0%})")

                with st.expander("Features generadas"):
                    st.dataframe(df, use_container_width=True)

            except Exception as e:
                st.error(f"Error en la prediccion: {e}")

with tab2:
    st.subheader("Carga masiva de transacciones (CSV)")

    if SAMPLE_PATH.exists():
        with open(SAMPLE_PATH, "rb") as f:
            st.download_button(
                label="Descargar sample_test.csv de referencia",
                data=f,
                file_name="sample_test.csv",
                mime="text/csv",
            )
    else:
        st.info("Archivo de referencia no disponible. El CSV debe contener las columnas: amt, category, gender, state, zip, lat, long, city_pop, merch_lat, merch_long.")

    uploaded = st.file_uploader("Seleccione archivo CSV", type="csv")
    if uploaded:
        try:
            df_input = pd.read_csv(uploaded)
            st.write(f"{len(df_input)} transacciones cargadas")

            df_feat = feature_engineering_input(df_input.copy(), cat_means)
            df_feat = codificar_input(df_feat)
            df_feat = alinear_columnas(df_feat, feature_cols)

            probas = modelo.predict_proba(df_feat)[:, 1]
            preds = (probas >= THRESHOLD).astype(int)
            df_input["prob_fraude"] = probas
            df_input["prediccion"] = ["FRAUDE" if p == 1 else "LEGITIMA" for p in preds]

            st.dataframe(df_input[["prediccion", "prob_fraude"] + list(df_input.columns[:-2])],
                         use_container_width=True)

            fraudes = preds.sum()
            st.metric("Fraudes detectados", fraudes)
            st.metric("Transacciones legitimas", len(preds) - fraudes)
        except Exception as e:
            st.error(f"Error al procesar archivo: {e}")
