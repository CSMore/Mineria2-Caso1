import streamlit as st
import pandas as pd
import logging

# Configuración básica del log
def setup_logging():
    logging.basicConfig(filename="audit_log.txt", 
                        level=logging.INFO, 
                        format="%(asctime)s - %(message)s")

class EDA:
    def __init__(self):
        setup_logging()
        self.df = None

    def render_file_uploader(self):
        uploaded_file = st.file_uploader("Selecciona un archivo CSV:", type="csv")
        return uploaded_file

    def render_options_panel(self):
        col1, col2, col3 = st.columns(3)
        with col1:
            separator = st.selectbox("Separador", [",", ";", "\t"], index=0)
        with col2:
            decimal = st.selectbox("Decimal", [".", ","], index=0)
        with col3:
            encoding = st.selectbox("Codificación", ["utf-8", "latin1"], index=0)

        return separator, decimal, encoding

    def load_data(self, uploaded_file, separator, decimal, encoding):
        try:
            self.df = pd.read_csv(uploaded_file, sep=separator, decimal=decimal, encoding=encoding)
            logging.info("Archivo CSV cargado correctamente.")
            return True
        except Exception as e:
            logging.error(f"Error al cargar CSV: {e}")
            st.error(f"Error al cargar CSV: {e}")
            return False

    def show_preview(self):
        if self.df is not None:
            st.write("### Vista previa del Dataset")
            st.dataframe(self.df.head())

    def validate_and_save(self):
        if st.button("Cargar y Guardar Dataset"):
            if self.df is not None:
                st.session_state['data'] = self.df
                st.success("Datos guardados correctamente en sesión.")
                logging.info("Dataset guardado correctamente en sesión.")
            else:
                st.error("No hay dataset cargado para guardar.")

    def read_dataset(self):
        uploaded_file = self.render_file_uploader()
        if uploaded_file:
            separator, decimal, encoding = self.render_options_panel()
            if self.load_data(uploaded_file, separator, decimal, encoding):
                self.show_preview()
                self.validate_and_save()
        else:
            st.warning("Por favor, carga un dataset válido.")

class app:
    def main(self):
        st.markdown('<h2>Análisis de Datos 🔬</h2>', unsafe_allow_html=True)
        eda_instance = EDA()
        eda_instance.read_dataset()
