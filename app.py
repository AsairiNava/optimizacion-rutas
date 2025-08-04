import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from xgboost import XGBRegressor
import joblib

# Cargar modelo y columnas
model = joblib.load("model.pkl")
columnas_modelo = joblib.load("columns.pkl")

# ==============================
# Funciones
# ==============================

def simular_envios(model, columnas_modelo, n_envios=5):
    envios = []
    for envio_id in range(1, n_envios + 1):
        rutas = pd.DataFrame({
            "envio_id": envio_id,
            "origen": np.random.choice(["México", "China", "India"], 3),
            "destino": np.random.choice(["Chile", "Alemania", "Canadá"], 3),
            "tipo_transporte": ["marítimo", "aéreo", "terrestre"],
            "distancia_km": np.random.randint(5000, 15000, 3),
            "clima": np.random.choice(["normal", "lluvia", "tormenta"], 3),
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

def seleccionar_mejor_ruta_grupo(df, peso_tiempo, peso_costo, peso_riesgo):
    df = df.copy()
    df["score"] = (
        peso_tiempo * df["tiempo_estimado_modelo"].rank(method="min") +
        peso_costo * df["costo_usd"].rank(method="min") +
        peso_riesgo * df["retraso_aduana_h"].rank(method="min")
    )
    return df.sort_values("score").iloc[0]

def obtener_mejores_rutas(df_envios, peso_tiempo, peso_costo, peso_riesgo):
    return df_envios.groupby("envio_id").apply(
        seleccionar_mejor_ruta_grupo,
        peso_tiempo=peso_tiempo,
        peso_costo=peso_costo,
        peso_riesgo=peso_riesgo
    ).reset_index(drop=True)

# ==============================
# App
# ==============================

st.title("Optimización de rutas logísticas")

# Sidebar
st.sidebar.subheader("🎛️ Ajustes de simulación")
st.sidebar.markdown("Asigna la importancia (total debe sumar 100):")
peso_tiempo = st.sidebar.slider("⏱️ Tiempo estimado", 0, 100, 50)
peso_costo = st.sidebar.slider("💰 Costo logístico", 0, 100, 30)
peso_riesgo = st.sidebar.slider("⚠️ Riesgo aduanal", 0, 100, 20)

# Normalizar pesos
total = peso_tiempo + peso_costo + peso_riesgo
if total > 0:
    peso_tiempo /= total
    peso_costo /= total
    peso_riesgo /= total

# Simular rutas y almacenar en session
if "rutas_simuladas" not in st.session_state or st.button("🔄 Simular rutas"):
    np.random.seed(42)
    df_envios = simular_envios(model, columnas_modelo, n_envios=5)
    st.session_state["rutas_simuladas"] = df_envios

# Obtener simulación actual
df_envios = st.session_state["rutas_simuladas"]

# Calcular mejores rutas con pesos actualizados
mejores_rutas = obtener_mejores_rutas(df_envios, peso_tiempo, peso_costo, peso_riesgo)

# Tabla
st.subheader("🏆 Mejores rutas seleccionadas")
st.dataframe(mejores_rutas[["envio_id", "origen", "destino", "tipo_transporte", "tiempo_estimado_modelo", "costo_usd", "retraso_aduana_h", "score"]])

# Descarga CSV
st.download_button(
    label="📥 Descargar CSV",
    data=mejores_rutas.to_csv(index=False),
    file_name="mejores_rutas.csv",
    mime="text/csv"
)

# Mapa
st.subheader("🗺️ Mapa de rutas seleccionadas")
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

m = folium.Map(location=[20, 0], zoom_start=2)
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

st_folium(m, width=700)
