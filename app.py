import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent / "models" / "xgboost_fraud_model.joblib"

FEATURES = [
    "category", "amt", "gender", "city", "state", "zip",
    "lat", "long", "city_pop", "job", "merchant",
    "merch_lat", "merch_long",
]

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


@st.cache_resource
def cargar_modelo():
    if not MODEL_PATH.exists():
        st.error(f"Modelo no encontrado en {MODEL_PATH}. Ejecute 05_model_training.py primero.")
        st.stop()
    return joblib.load(MODEL_PATH)


def preparar_input(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])

    df["category"] = pd.Categorical(df["category"], categories=CATEGORIAS)
    df["state"] = pd.Categorical(df["state"], categories=ESTADOS)
    df["gender"] = pd.Categorical(df["gender"], categories=["M", "F"])
    df["city"] = pd.Categorical(df["city"])
    df["job"] = pd.Categorical(df["job"])
    df["merchant"] = pd.Categorical(df["merchant"])

    df = pd.get_dummies(df, drop_first=True)

    for f in FEATURES:
        if f not in df.columns and f not in ["category", "state", "gender", "city", "job", "merchant"]:
            df[f] = 0.0

    columnas_modelo = modelo.get_booster().feature_names
    if columnas_modelo is not None:
        for col in columnas_modelo:
            if col not in df.columns:
                df[col] = 0
        df = df[columnas_modelo]

    return df


st.set_page_config(page_title="Fraud Detection", page_icon="🛡️", layout="centered")
st.title("🛡️ Detector de Fraude - Pipeline DataOps")
st.markdown("Simula una transaccion y el modelo XGBoost predice si es **fraudulenta** o **legitima**.")

modelo = cargar_modelo()
st.success("Modelo XGBoost cargado correctamente")

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

    submitted = st.form_submit_button("🔍 Predecir", type="primary")

if submitted:
    with st.spinner("Analizando transaccion..."):
        data = {
            "amt": amt,
            "category": category,
            "gender": gender,
            "state": state,
            "zip": zip_code,
            "lat": lat,
            "long": long,
            "city_pop": city_pop,
            "city": "Unknown",
            "job": "Unknown",
            "merchant": "Unknown",
            "merch_lat": lat + 0.01,
            "merch_long": long - 0.01,
        }

        try:
            X_input = preparar_input(data)
            pred = modelo.predict(X_input)[0]
            proba = modelo.predict_proba(X_input)[0]

            if pred == 1:
                st.error(f"🚨 **FRAUDE DETECTADO** (probabilidad: {proba[1]:.2%})")
            else:
                st.success(f"✅ **Transaccion Legitima** (probabilidad de fraude: {proba[1]:.2%})")

            st.metric("Probabilidad de Fraude", f"{proba[1]:.2%}")
            st.metric("Probabilidad de Legitima", f"{proba[0]:.2%}")
        except Exception as e:
            st.error(f"Error en la prediccion: {e}")
