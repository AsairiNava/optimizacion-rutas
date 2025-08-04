import streamlit as st
import pandas as pd
import numpy as np
import folium
import joblib
from streamlit_folium import st_folium

# Cargar modelo y columnas usadas en entrenamiento
model = joblib.load("model.pkl")
columnas_modelo = joblib.load("columns.pkl")

# =============================
# Cargar datos desde Google Sheets
# =============================
@st.cache_data
def cargar_datos_real():
    csv_url = "https://docs.google.com/spreadsheets/d/1q1tYM63bkdqZLroAOtgEh8NoRNiwZ9atsjb4WQ_NteE/export?format=csv"
    df = pd.read_csv(csv_url)

    # Enriquecer variables
    clima_riesgo = {"normal": 0, "lluvia": 1, "nieve": 2, "tormenta": 3}
    velocidades = {"aéreo": 800, "marítimo": 200, "terrestre": 600}
    df["clima_riesgo"] = df["clima"].map(clima_riesgo)
    df["velocidad_estimada"] = df["tipo_transporte"].map(velocidades)
    df["tiempo_base_transporte"] = df["distancia_km"] / df["velocidad_estimada"]

    # Dummy de día (fijo por ahora)
    df["dia_envio"] = "Lun"
    df = pd.get_dummies(df, columns=["dia_envio"], drop_first=True)

    # Codificación de variables categóricas
    encoded = pd.get_dummies(df[["origen", "destino", "tipo_transporte"]], drop_first=True)
    X = pd.concat([
        df[["distancia_km", "retraso_aduana_h", "clima_riesgo", "tiempo_base_transporte"]],
        encoded
    ], axis=1)

    # Alinear columnas al modelo
    for col in columnas_modelo:
        if col not in X.columns:
            X[col] = 0
    X = X[columnas_modelo].copy()
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)

    # Predicción
    df["tiempo_estimado_modelo"] = model.predict(X.values)
    return df

# =============================
# Selección de mejor ruta por envío
# =============================
def seleccionar_mejor_ruta_grupo(df, peso_tiempo, peso_costo, peso_riesgo):
    df = df.copy()
    df["score"] = (
        peso_tiempo * df["tiempo_estimado_modelo"].rank(method="min") +
        peso_costo * df["costo_usd"].rank(method="min") +
        peso_riesgo * df["retraso_aduana_h"].rank(method="min")
    )
    return df.sort_values("score").iloc[0]

# =============================
# Interfaz de la app
# =============================
st.title("📦 Optimización de Rutas Logísticas")
st.markdown("Ajusta los pesos según tus prioridades logísticas:")

peso_tiempo = st.slider("🔧 Peso: Tiempo estimado", 0, 10, 5)
peso_costo = st.slider("💰 Peso: Costo logístico", 0, 10, 3)
peso_riesgo = st.slider("⚠️ Peso: Riesgo aduanal", 0, 10, 2)

total = peso_tiempo + peso_costo + peso_riesgo
peso_tiempo /= total
peso_costo /= total
peso_riesgo /= total

# =============================
# Cargar datos y aplicar modelo
# =============================
df_envios = cargar_datos_real()

mejores_rutas = df_envios.groupby("envio_id").apply(
    seleccionar_mejor_ruta_grupo,
    peso_tiempo=peso_tiempo,
    peso_costo=peso_costo,
    peso_riesgo=peso_riesgo
).reset_index(drop=True)

# =============================
# Mostrar tabla de resultados
# =============================
st.subheader("🏁 Mejores rutas por envío")
st.dataframe(mejores_rutas[[
    "envio_id", "origen", "destino", "tipo_transporte",
    "tiempo_estimado_modelo", "costo_usd", "retraso_aduana_h", "score"
]])

# Botón para descargar
csv_export = mejores_rutas.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Descargar rutas optimizadas (CSV)",
    data=csv_export,
    file_name="mejores_rutas.csv",
    mime="text/csv"
)

# =============================
# Mapa interactivo
# =============================
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

st_folium(m, width=700, height=500)
