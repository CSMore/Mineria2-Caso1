import streamlit as st
from utils.config_logging import setup_logging

setup_logging()
# Configuración de la página principal
st.set_page_config(page_title="FlowLab Analytics", layout="wide")

# Definición de las páginas usando las rutas de los archivos
pages = [
    st.Page("app_pages/pipeline.py", title="Pipeline", icon="⚙"),
    st.Page("app_pages/exploratorio.py", title="Análisis Exploratorio", icon="➖"),
    st.Page("app_pages/series_temporales.py", title="Series Temporales", icon="➖"),
    st.Page("app_pages/patron_estacional.py", title="Patrón Estacional", icon="➖"),

    st.Page("app_pages/agrupamiento.py", title="Agrupamiento (Clustering)", icon="➖"),
    st.Page("app_pages/reduccion.py", title="Reducción de Dimensionalidad", icon="➖"),

    st.Page("app_pages/aprendizaje.py", title="Regresión", icon="➖"),
    st.Page("app_pages/clasificacion.py", title="Clasificación", icon="➖"),

    st.Page("app_pages/results.py", title="Comparación de Modelo", icon="➖"),
    
    st.Page("app_pages/web_mining.py", title="Web Mining", icon="🕸️"),
]

# Páginas agrupadas en secciones
navigation_dict = {
    "⚙️ Datos": [pages[0], pages[1]],
    "⏰ Series Temporales": [pages[2], pages[3]],
    "💎 Aprendizaje No Supervisado": [pages[4], pages[5]],
    "💡 Aprendizaje Supervisado": [pages[6], pages[7]],
    "📈 Comparación": [pages[8]],
    "🕸️ Web Mining": [pages[9]]

} 

# Configuración de la navegación 
nav = st.navigation(navigation_dict, position="sidebar", expanded=True)
nav.run()
