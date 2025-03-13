import streamlit as st
from streamlit_option_menu import option_menu
import pipeline as pipeline  # Importar py Pipeline 
import results as results     # Importar py Results
import exploratorio as exploratorio  # Importar py Análisis Exploratorio
import aprendizaje as aprendizaje  # Importar py Aprendizaje Supervisado

st.set_page_config(
    page_title="Testing",
    layout="wide"
)
'''
if "pipeline_completed" not in st.session_state:
    st.session_state.pipeline_completed = False
if "dataset_loaded" not in st.session_state:
    st.session_state.dataset_loaded = False'''
if "page" not in st.session_state:
    st.session_state.page = "Pipeline"  # Página inicial predeterminada

# Función que controla la navegación (menú lateral)
def app_control():
    with st.sidebar:
        selected = option_menu(
            "Navegación",
            options=[
                "Pipeline",
                "Análisis Exploratorio",
                "Aprendizaje Supervisado",
                "Comparación de Modelo"
            ],
            icons=["gear", "bar-chart", "robot", "graph-up-arrow"],
            menu_icon="cast",
            default_index=0
        )
        st.session_state.page = selected
    return selected

def main():
    if "page" not in st.session_state:
        st.session_state.page = "Pipeline"  # Página inicial predeterminada

    app_control()

    #Navegacion
    if st.session_state.page == "Pipeline":
        pipeline.app().main()  # Llamada a la página Pipeline
        st.session_state.pipeline_completed = True

    elif st.session_state.page == "Análisis Exploratorio":
        #if st.session_state.pipeline_completed:
        exploratorio.app().main()  # Llamada al módulo Análisis Exploratorio
        #st.session_state.dataset_loaded = True  # Simular que el dataset fue cargado

    elif st.session_state.page == "Aprendizaje Supervisado":
        aprendizaje.app().main()  # Llamada al módulo Aprendizaje Supervisado

    elif st.session_state.page == "Comparación de Modelo":
        results.app().run()  # Llamada a la página Resultados

if __name__ == "__main__":
    main()

