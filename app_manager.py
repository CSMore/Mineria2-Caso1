import streamlit as st
import streamlit_antd_components as sac
import pipeline as pipeline  # Importar py Pipeline 
import results as results     # Importar py Results
import exploratorio as exploratorio  # Importar py Análisis Exploratorio
import aprendizaje as aprendizaje  # Importar py Aprendizaje Supervisado
import series_temporales as series_tiempo

st.set_page_config(
    page_title="Testing",
    layout="wide"
)


if "page" not in st.session_state:
    st.session_state.page = "Pipeline"  # Página inicial predeterminada


# Función que controla la navegación (menú lateral)
def app_control():
    with st.sidebar:
        selected = sac.menu([
            sac.MenuItem("Data", icon="gear",children=[
                sac.MenuItem("Pipeline", icon="bar-chart", description="Carga de datos"),
                sac.MenuItem("Análisis Exploratorio", icon="bar-chart", description="Explora y analiza el conjunto de datos"),
            ]),
            sac.MenuItem("Series Temporales", icon="bar-chart",children=[
                sac.MenuItem("Configuración inicial", icon="bar-chart", description="Carga de datos"),
                sac.MenuItem("Operaciones", icon="bar-chart", description="Carga de datos"),
            ]),
            sac.MenuItem("Aprendizaje Supervisado", icon="robot"),
            sac.MenuItem("Comparación de Modelo", icon="graph-up-arrow"),
        ] , variant='left-bar', color='#4682b4', open_all=True)#open_all=True, format_func=lambda x: x)  # Convierte el nombre en string directamente



        st.session_state.page = selected

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

    elif st.session_state.page == "Configuración inicial":
    #if st.session_state.pipeline_completed:
        series_tiempo.app().main()
        #st.session_state.dataset_loaded = True  # Simular que el dataset fue cargado

    elif st.session_state.page == "Operaciones":
    #if st.session_state.pipeline_completed:
        series_tiempo.app().main()  # Llamada al módulo Series Temporales
        #st.session_state.dataset_loaded = True  # Simular que el dataset fue cargado

    elif st.session_state.page == "Aprendizaje Supervisado":
        aprendizaje.app().main()  # Llamada al módulo Aprendizaje Supervisado

    elif st.session_state.page == "Comparación de Modelo":
        results.app().run()  # Llamada a la página Resultados

if __name__ == "__main__":
    main()

