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

    def check_data_quality(self):
        """Valida calidad de datos: duplicados, valores nulos y columnas constantes"""
        st.subheader("🔍 Calidad de los Datos")
        st.write(f"Filas con valores nulos: {self.df.isnull().sum().sum()}")
        st.write(f"Filas duplicadas: {self.df.duplicated().sum()}")

        # Identificar columnas con un solo valor
        constant_columns = [col for col in self.df.columns if self.df[col].nunique() == 1]
        if constant_columns:
            st.write(f"Columnas constantes (sin variabilidad): {constant_columns}")
        else:
            st.write("No hay columnas constantes.")

    def check_class_balance(self):
        """Detecta si el dataset de clasificación está balanceado"""
        st.subheader("Balance de Clases")
        class_column = st.selectbox("Selecciona la columna de clase (si aplica):", self.df.columns)
        class_counts = self.df[class_column].value_counts()
        st.bar_chart(class_counts)

    def detect_outliers(self):
        """Identifica valores atípicos usando el rango intercuartil (IQR)"""
        st.subheader("Detección de Valores Atípicos")
        numerical_cols = self.df.select_dtypes(include=["number"]).columns
        selected_col = st.selectbox("Selecciona una columna numérica:", numerical_cols)
        
        Q1 = self.df[selected_col].quantile(0.25)
        Q3 = self.df[selected_col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = self.df[(self.df[selected_col] < (Q1 - 1.5 * IQR)) | 
                           (self.df[selected_col] > (Q3 + 1.5 * IQR))]

        st.write(f"Valores atípicos detectados en {selected_col}: {len(outliers)} registros")

    def allow_column_selection(self):
        """Permite eliminar columnas irrelevantes"""
        st.subheader("Selección de Columnas")
        columns_to_drop = st.multiselect("Selecciona columnas a eliminar:", self.df.columns)
        if st.button("Eliminar Columnas"):
            self.df.drop(columns=columns_to_drop, inplace=True)
            st.success(f"Columnas eliminadas: {columns_to_drop}")

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
                self.check_data_quality()  # Nuevo: validaciones
                self.check_class_balance()  # Nuevo: balance de clases
                self.detect_outliers()  # Nuevo: detección de outliers
                self.allow_column_selection()  # Nuevo: eliminación de columnas
                self.validate_and_save()
        else:
            st.warning("Por favor, carga un dataset válido.")

class app:
    def main(self):
        st.markdown('<h2>Análisis de Datos </h2>', unsafe_allow_html=True)
        eda_instance = EDA()
        eda_instance.read_dataset()
