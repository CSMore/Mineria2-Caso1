import streamlit as st
from streamlit_option_menu import option_menu
import pipeline as pipeline  # Importar py Pipeline 
import results as results     # Importar py Results

st.set_page_config(
    page_title="Testing",
    layout="wide"
)

def app_control():

    # Menú horizontal en la parte superior
    selected = option_menu(
        menu_title="",  # Sin título en el menú
        options=["Pipeline", "Resultados"],  # Opciones del menú
        icons=["sunrise", "bar-chart"],  # Iconos de las opciones
        menu_icon="cast",  # Icono principal del menú (opcional)
        default_index=0,  # Página inicial
        orientation="horizontal",  # Menú horizontal
        styles={
            "container": {"padding": "0", "margin": "0"},
            "icon": {"font-size": "16px"},
            "nav-link": {"font-size": "14px", "text-align": "center", "margin": "0px", "--hover-color": "#d9d9d9"},
            "nav-link-selected": {"background-color": "#184B44", "color": "white"},
        }
    ) 

    # Lógica de las opciones del menú
    if selected == "Pipeline":
        pipe_instance = pipeline.app()
        pipe_instance.main()
    elif selected == "Resultados":
        results_instance = results.app()
        results_instance.run()

def main():
    if "page" not in st.session_state:
        st.session_state.page = "app_manager"  # Página inicial predeterminada

    if st.session_state.page == "app_manager":
        app_control()  # Llamar directamente a "app_control"

if __name__ == "__main__":
    main()

