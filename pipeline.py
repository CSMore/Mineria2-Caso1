import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import logging

#Configuración de login para auditorias
logging.basicConfig(filename="audit_log.txt", level=logging.INFO, format="%(asctime)s - %(message)s")

class eda:
    def __init__(self):
        self.df = None
        self.preview_active = False

    def initialize_session_state(self):
        """Inicializa las variables de sesión si no existen."""
        if 'uploaded_file' not in st.session_state:
            st.session_state['uploaded_file'] = None
        if 'data_loaded' not in st.session_state:
            st.session_state['data_loaded'] = False

    def render_file_uploader(self):
        """Renderiza el file uploader y actualiza la sesión."""
        uploaded_file = st.file_uploader("Seleccione un archivo CSV", type=["csv"])
        if uploaded_file:
            st.session_state['uploaded_file'] = uploaded_file
            st.session_state['data_loaded'] = True
        return uploaded_file

    def render_options_panel(self):
        """Renderiza los selectores de configuración."""
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            separator = st.selectbox("Seleccione el separador", [",", ";", "\t"], index=0)
        with col2:
            decimal = st.selectbox("Seleccione el separador decimal", [".", ","], index=0)
        with col3:
            encoding = st.selectbox("Seleccione la codificación", ["utf-8", "latin1"], index=0)
        with col4:
            detect_headers = st.checkbox("Detectar automáticamente cabeceras", value=True)
        return separator, decimal, encoding, detect_headers

    def load_data(self, uploaded_file, separator, decimal, encoding, detect_headers):
        """Carga los datos según las configuraciones seleccionadas."""
        try:
            self.df = pd.read_csv(uploaded_file, sep=separator, decimal=decimal, encoding=encoding)
            return True
        except pd.errors.ParserError:
            logging.error("Error de análisis: el archivo está mal formado.")
            st.error("Error de análisis: el archivo está mal formado.")
            if detect_headers:
                # Intentar sin cabeceras
                self.df = pd.read_csv(uploaded_file, sep=separator, decimal=decimal, encoding=encoding, header=None)
                st.warning("El archivo no tenía cabeceras válidas. Se ha usado la primera fila como nombres de columnas.")
                return True
        except Exception as e:
            logging.error(f"Error al leer el archivo: {e}")
            st.error(f"Error al leer el archivo: {e}")
        return False

    def show_preview(self):
        """Muestra una vista previa del archivo cargado."""
        if st.button("👁️ Mostrar Vista Previa"):
            self.preview_active = True

        if self.preview_active and self.df is not None:
            with st.container():
                st.write("**Vista previa de los primeros registros**")
                st.dataframe(self.df.head(5))

                if st.button("❌ Cerrar Vista Previa"):
                    self.preview_active = False

    def process_data_loading(self, uploaded_file, separator, decimal, encoding):
        """Procesa la carga final de datos tras la validación."""
        if st.button("Cargar Datos"):
            if self.df is not None and self.validate_csv(separator, decimal, encoding):
                st.session_state['data'] = self.df  # Guardar en sesión
                st.success("Datos cargados correctamente.")
                logging.info(f"Datos cargados exitosamente desde el archivo: {uploaded_file.name}")
            else:
                st.error("No hay datos para cargar. Verifique el archivo y los parámetros.")

    def display_warnings(self):
        """Muestra advertencias si no se ha cargado un archivo."""
        if not st.session_state['data_loaded']:
            st.warning("Por favor, cargue un conjunto de datos válido.")
            logging.warning("No se ha cargado ningún archivo. Esperando que el usuario cargue uno.")

    def validate_csv(self, separator, decimal, encoding):
        """Valida que el archivo CSV cargado sea válido."""
        try:
            if self.df is not None and not self.df.empty:
                logging.info("Archivo validado correctamente con los parámetros seleccionados.")
                return True
            logging.warning("El archivo está vacío o no tiene datos válidos.")
        except Exception as e:
            logging.error(f"Error al validar el archivo: {e}")
        return False

    def read_dataset(self):
        """Controla el flujo general de carga y procesamiento del dataset."""
        self.initialize_session_state()
        uploaded_file = self.render_file_uploader()
        
        if uploaded_file:
            separator, decimal, encoding, detect_headers = self.render_options_panel()

            with st.spinner("Procesando archivo..."):
                if self.load_data(uploaded_file, separator, decimal, encoding, detect_headers):
                    self.show_preview()
                    self.process_data_loading(uploaded_file, separator, decimal, encoding)
        else:
            self.display_warnings()
        
class app:
    def main(self):
        st.markdown('<h3 class="custom-h3">Análisis de Datos  🔬</h3>', unsafe_allow_html=True)
        st.write("")
        eda_instance = eda()
        eda_instance.read_dataset()