import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from xgboost import XGBRegressor
import joblib

# ================================
# Cargar el modelo y columnas entrenadas
# ================================
model = joblib.load("model.pkl")
columnas_modelo = joblib.load("columns.pkl")

# ================================
# Función para simular datos
# ================================
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

    # Enriquecer variables
    clima_riesgo = {"normal": 0, "lluvia": 1, "nieve": 2, "tormenta": 3}
    velocidades = {"aéreo": 800, "marítimo": 200, "terrestre": 600}
    df["clima_riesgo"] = df["clima"].map(clima_riesgo)
    df["velocidad_estimada"] = df["tipo_transporte"].map(velocidades)
    df["tiempo_base_transporte"] = df["distancia_km"] / df["velocidad_estimada"]
    df["dia_envio"] = "Lun"
    df = pd.get_dummies(df, columns=["dia_envio"], drop_first=True)

    # Codificar y alinear columnas
    encoded = pd.get_dummies(df[["origen", "destino", "tipo_transporte"]], drop_first=True)
    X = pd.concat([df[["distancia_km", "retraso_aduana_h", "clima_riesgo", "tiempo_base_transporte"]], encoded], axis=1)
    for col in columnas_modelo:
        if col not in X.columns:
            X[col] = 0
    X = X[columnas_modelo]

    # Predecir tiempo estimado
    df["tiempo_estimado_modelo"] = model.predict(X.values)
    return df

# ================================
# Selección de mejor ruta por envío
# ================================
def obtener_mejores_rutas(df, peso_tiempo, peso_costo, peso_riesgo):
    def seleccionar_mejor_ruta_grupo(df_grupo):
        df_grupo = df_grupo.copy()
        df_grupo["score"] = (
            peso_tiempo * df_grupo["tiempo_estimado_modelo"].rank(method="min") +
            peso_costo * df_grupo["costo_usd"].rank(method="min") +
            peso_riesgo * df_grupo["retraso_aduana_h"].rank(method="min")
        )
        return df_grupo.sort_values("score").iloc[0]

    return df.groupby("envio_id").apply(seleccionar_mejor_ruta_grupo).reset_index(drop=True)

# ================================
# App Streamlit
# ================================
st.set_page_config(page_title="Optimización de Rutas", layout="wide")
st.title("Optimización de rutas logísticas")

# Sidebar con sliders
st.sidebar.subheader("🎛️ Ajustes de simulación")
peso_tiempo = st.sidebar.slider("Peso: Tiempo estimado", 0.0, 1.0, 0.5)
peso_costo = st.sidebar.slider("Peso: Costo logístico", 0.0, 1.0, 0.3)
peso_riesgo = st.sidebar.slider("Peso: Riesgo aduanal", 0.0, 1.0, 0.2)

# Normalizar pesos
total = peso_tiempo + peso_costo + peso_riesgo
peso_tiempo /= total
peso_costo /= total
peso_riesgo /= total

# Botón para simular rutas
if "rutas_simuladas" not in st.session_state or st.button("🔄 Simular rutas"):
    st.session_state.rutas_simuladas = simular_envios(model, columnas_modelo, n_envios=5)

df_envios = st.session_state.rutas_simuladas
mejores_rutas = obtener_mejores_rutas(df_envios, peso_tiempo, peso_costo, peso_riesgo)

# Mostrar tabla
st.subheader("🏆 Mejores rutas seleccionadas")
st.dataframe(mejores_rutas[["envio_id", "origen", "destino", "tipo_transporte", "tiempo_estimado_modelo", "costo_usd", "retraso_aduana_h", "score"]])

# Botón para descargar CSV
csv = mejores_rutas.to_csv(index=False).encode('utf-8')
st.download_button(
    label="⬇️ Descargar mejores rutas en CSV",
    data=csv,
    file_name='mejores_rutas.csv',
    mime='text/csv'
)

# Visualizar en mapa
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
