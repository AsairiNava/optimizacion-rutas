import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import joblib
from xgboost import XGBRegressor

# Cargar el modelo y las columnas
model = joblib.load("model.pkl")
columnas_modelo = joblib.load("columns.pkl")

# ================================
# Simular datos de envíos
# ================================
def simular_envios(model, columnas_modelo, n_envios=5):
    envios = []
    for envio_id in range(1, n_envios + 1):
        rutas = pd.DataFrame({
            "envio_id": envio_id,
            "origen": np.random.choice(["México", "China", "India"]),
            "destino": np.random.choice(["Chile", "Alemania", "Canadá"]),
            "tipo_transporte": ["marítimo", "aéreo", "terrestre"],
            "distancia_km": np.random.randint(5000, 15000, 3),
            "clima": np.random.choice(["normal", "lluvia", "tormenta"]),
            "retraso_aduana_h": np.random.normal(10, 4, 3).round(1),
            "costo_usd": np.random.randint(300, 1000, 3)
        })
        envios.append(rutas)
    df = pd.concat(envios, ignore_index=True)

    clima_riesgo = {"normal": 0, "lluvia": 1, "nieve": 2, "tormenta": 3}
    velocidades = {"aéreo": 800, "marítimo": 200, "terrestre": 600}
    df["clima_riesgo"] = df["clima"].map(clima_riesgo)
    df["velocidad_estimada"] = df["tipo_transporte"].map(velocidades)
    df["tiempo_base_transporte"] = df["distancia_km"] / df["velocidad_estimada"]
    df["dia_envio"] = "Lun"
    df = pd.get_dummies(df, columns=["dia_envio"], drop_first=True)

    encoded = pd.get_dummies(df[["origen", "destino", "tipo_transporte"]], drop_first=True)
    X = pd.concat([df[["distancia_km", "retraso_aduana_h", "clima_riesgo", "tiempo_base_transporte"]], encoded], axis=1)
    for col in columnas_modelo:
        if col not in X.columns:
            X[col] = 0
    X = X[columnas_modelo]

    df["tiempo_estimado_modelo"] = model.predict(X.values)
    return df

# ================================
# Selección de mejor ruta
# ================================
def seleccionar_mejor_ruta_grupo(df, peso_tiempo, peso_costo, peso_riesgo):
    df = df.copy()
    df["score"] = (
        peso_tiempo * df["tiempo_estimado_modelo"].rank(method="min") +
        peso_costo * df["costo_usd"].rank(method="min") +
        peso_riesgo * df["retraso_aduana_h"].rank(method="min")
    )
    return df.sort_values("score").iloc[0]

# ================================
# Streamlit UI
# ================================
st.title("Optimización de rutas logísticas")
st.markdown("Selecciona los pesos que deseas darle a cada criterio de decisión:")

peso_tiempo = st.slider("Peso: Tiempo estimado", 0.0, 1.0, 0.5)
peso_costo = st.slider("Peso: Costo logístico", 0.0, 1.0, 0.3)
peso_riesgo = st.slider("Peso: Riesgo aduanal", 0.0, 1.0, 0.2)

total = peso_tiempo + peso_costo + peso_riesgo
peso_tiempo /= total
peso_costo /= total
peso_riesgo /= total

# Cachear simulación y resultados
@st.cache_data
def generar_envios(peso_tiempo, peso_costo, peso_riesgo):
    np.random.seed(42)
    return simular_envios(model, columnas_modelo, n_envios=5)

df_envios = generar_envios(peso_tiempo, peso_costo, peso_riesgo)

@st.cache_data
def obtener_mejores_rutas(df_envios, peso_tiempo, peso_costo, peso_riesgo):
    return df_envios.groupby("envio_id").apply(
        seleccionar_mejor_ruta_grupo,
        peso_tiempo=peso_tiempo,
        peso_costo=peso_costo,
        peso_riesgo=peso_riesgo
    ).reset_index(drop=True)

mejores_rutas = obtener_mejores_rutas(df_envios, peso_tiempo, peso_costo, peso_riesgo)

st.subheader("🏆 Mejores rutas seleccionadas")
st.dataframe(mejores_rutas[[
    "envio_id", "origen", "destino", "tipo_transporte",
    "tiempo_estimado_modelo", "costo_usd", "retraso_aduana_h", "score"
]])

csv = mejores_rutas.to_csv(index=False)
st.download_button("📅 Descargar rutas como CSV", csv, "mejores_rutas.csv", "text/csv")

# ================================
# Mapa cacheado
# ================================
@st.cache_resource(show_spinner=False)
def generar_mapa(mejores_rutas):
    m = folium.Map(location=[20, 0], zoom_start=2)
    coordenadas = {
        "México": [23.6345, -102.5528],
        "Chile": [-35.6751, -71.5430],
        "China": [35.8617, 104.1954],
        "Alemania": [51.1657, 10.4515],
        "India": [20.5937, 78.9629],
        "Canadá": [56.1304, -106.3468]
    }
    colores_transporte = {
        "aéreo": "#1f77b4",
        "marítimo": "#2ca02c",
        "terrestre": "#d62728"
    }
    for _, row in mejores_rutas.iterrows():
        origen_coord = coordenadas.get(row["origen"], [0, 0])
        destino_coord = coordenadas.get(row["destino"], [0, 0])
        color_ruta = colores_transporte.get(row["tipo_transporte"], "#000000")

        folium.Marker(origen_coord, popup=f"Origen: {row['origen']}").add_to(m)
        folium.Marker(destino_coord, popup=f"Destino: {row['destino']}").add_to(m)
        folium.PolyLine(
            [origen_coord, destino_coord],
            color=color_ruta,
            weight=4,
            tooltip=f"Envío {row['envio_id']} – {row['tipo_transporte']}<br>"
                    f"Tiempo: {round(row['tiempo_estimado_modelo'], 1)} h<br>"
                    f"Costo: ${row['costo_usd']}<br>"
                    f"Riesgo: {row['retraso_aduana_h']} h<br>"
                    f"Score: {round(row['score'], 2)}"
        ).add_to(m)
    return m

st.subheader("📜 Mapa de rutas seleccionadas")
st_folium(generar_mapa(mejores_rutas), width=700)
