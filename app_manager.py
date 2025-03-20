import streamlit as st
import streamlit_antd_components as sac
import pipeline as pipeline  # Importar py Pipeline 
import results as results     # Importar py Results
import exploratorio as exploratorio  # Importar py Análisis Exploratorio
import aprendizaje as aprendizaje  # Importar py Aprendizaje Supervisado
import series_temporales as series_tiempo

st.set_page_config(page_title="Testing", layout="wide")

if "page" not in st.session_state:
    st.session_state.page = "Pipeline"  # Página inicial predeterminada


# Función que controla la navegación (menú lateral)
def app_control():
    with st.sidebar:
        selected = sac.menu([
            sac.MenuItem("Pipeline", icon="gear",children=[
                sac.MenuItem("Pipeline", icon="bar-chart", description="Carga de datos"),
                sac.MenuItem("Análisis Exploratorio", icon="bar-chart", description="Explora y analiza el conjunto de datos"),
            ]),
            sac.MenuItem("Series Temporales", icon="bar-chart",children=[
                sac.MenuItem("Configuración inicial", icon="bar-chart", description="Carga de datos"),
                sac.MenuItem("Operaciones", icon="bar-chart", description="Carga de datos"),
            ]),
            sac.MenuItem("Aprendizaje Supervisado", icon="robot"),
            sac.MenuItem("Comparación de Modelo", icon="graph-up-arrow"),
        ] , variant='left-bar', color='#4682b4', open_all=True,)#open_all=True, format_func=lambda x: x)  # Convierte el nombre en string directamente

        if selected is not None:
            st.session_state.page = selected


def main():
    app_control()

    #Navegacion
    if st.session_state.page == "Pipeline":
        pipeline.app().main()  
        st.session_state.pipeline_completed = True

    elif st.session_state.page == "Análisis Exploratorio":
        exploratorio.app().main()  # Llamada al módulo Análisis Exploratorio

    elif st.session_state.page == "Configuración inicial":
        series_tiempo.app().main()

    elif st.session_state.page == "Operaciones":
        series_tiempo.app().main()  # Llamada al módulo Series Temporales

    elif st.session_state.page == "Aprendizaje Supervisado":
        aprendizaje.app().main()  

    elif st.session_state.page == "Comparación de Modelo":
        results.app().run()  

if __name__ == "__main__":
    main()

