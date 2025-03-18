import streamlit as st
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error
from datetime import timedelta
def setup_logging():
    logging.basicConfig(filename="audit_log.txt", 
                        level=logging.INFO, 
                        format="%(asctime)s - %(message)s")
class SeriesTemporales:
    def __init__(self, df):
        setup_logging()
        self.original_df = df.copy()  # Keep a copy of the original dataframe
        self.df = df.copy()  # Working copy
        
        logging.info(f"DataFrame recibido con columnas: {list(self.df.columns)}")
        logging.info(f"Tipos de datos del DataFrame: {self.df.dtypes}")
        
        
        logging.info("Instancia de SeriesTemporales creada con datos cargados correctamente.")
    def head_df(self, n=5):
        if not self.__df.empty:
            return self.__df.head(n).astype(str)
        return "No se cargaron los datos :("
    def tail_df(self, n=5):
        if not self.__df.empty:
            return self.__df.tail(n).astype(str)
        return "No se cargaron los datos :("
    
    def validate_data_loaded(self):
        """Verifica si los datos están cargados en el pipeline."""
        if self.df is None:
            st.warning("No hay datos cargados. Carga primero un dataset válido.")
            logging.warning("Intento de realizar operaciones sin datos cargados.")
            return False
        return True
    @staticmethod
    def validate_date(df, date_column):
        """Función para validar y convertir la columna a tipo datetime."""
        if date_column not in df.columns:
            st.error(f"La columna '{date_column}' no está presente en los datos.")
            logging.error(f"La columna '{date_column}' no se encuentra en el DataFrame.")
            return None
        
        # Intentar convertir la columna a datetime, forzando errores a NaT
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce', dayfirst=True)
        if df[date_column].isnull().all():
            st.error(f"La columna '{date_column}' no tiene valores válidos de fecha.")
            return None
        
        # Eliminar filas con fechas inválidas (NaT)
        df = df.dropna(subset=[date_column])
        
        return df
    @staticmethod
    def validate_time(df, time_column):
        """Función para validar y convertir la columna a tipo timedelta."""
        if time_column not in df.columns:
            st.error(f"La columna '{time_column}' no está presente en los datos.")
            return None
        
        # Intentar convertir la columna a timedelta, forzando errores a NaT
        df[time_column] = pd.to_timedelta(df[time_column], errors='coerce')
        
        # Verificar si hay valores NaT en la columna después de la conversión
        if df[time_column].isnull().all():
            st.error(f"La columna '{time_column}' no tiene valores válidos de tiempo.")
            return None
        
        return df
    def render_option_selector(self):
        """Muestra la opción de seleccionar o crear."""
        col1, _ = st.columns(2)
        with col1:
            st.write("## Opciones")
            return st.radio("Selecciona una opción:", ("Seleccionar", "Crear"))
    def render_data_selector(self, option):
        """Muestra el selector de columna de datos."""
        if option == "Seleccionar":
            col2, _ = st.columns(2)
            with col2:
                st.write("## Datos")
                st.write("### Selecciona la columna con los datos temporales:")
                
                # First, check explicitly for a column named 'fecha' which is common in Spanish datasets
                if 'fecha' in self.df.columns:
                    # Test if it can be converted to datetime
                    test_series = pd.to_datetime(self.df['fecha'], errors='coerce')
                    valid_dates = test_series.notnull().sum()
                    
                    if valid_dates > 0:
                        st.success(f"Columna 'fecha' encontrada con {valid_dates} fechas válidas.")
                        logging.info(f"Columna 'fecha' identificada con {valid_dates} fechas válidas")
                        potential_date_columns = ['fecha']
                    else:
                        potential_date_columns = []
                else:
                    potential_date_columns = []
                
                # If 'fecha' wasn't found or doesn't contain valid dates, check all columns
                if not potential_date_columns:
                    # Print the first few rows of each string column to help with debugging
                    st.write("### Contenido de columnas de texto:")
                    for column in self.df.columns:
                        if pd.api.types.is_string_dtype(self.df[column]) or pd.api.types.is_object_dtype(self.df[column]):
                            st.write(f"Columna '{column}' - Primeros 5 valores: {self.df[column].head().tolist()}")
                    
                    # Now search for date columns more aggressively
                    for column in self.df.columns:
                        # Skip non-string/object columns
                        if not (pd.api.types.is_string_dtype(self.df[column]) or pd.api.types.is_object_dtype(self.df[column])):
                            continue
                            
                        # Try to convert to datetime
                        test_series = pd.to_datetime(self.df[column], errors='coerce')
                        valid_dates = test_series.notnull().sum()
                        
                        # If any dates are valid, add to potential columns
                        if valid_dates > 0:
                            potential_date_columns.append(column)
                            st.success(f"Columna '{column}' identificada con {valid_dates} fechas válidas")
                            logging.info(f"Columna '{column}' identificada con {valid_dates} fechas válidas")
                
                if potential_date_columns:
                    selected_column = st.selectbox("Selecciona una columna de fecha:", potential_date_columns)
                    
                    # When a column is selected, convert it and set as index
                    if st.button("Usar esta columna como índice temporal"):
                        # Convert the selected column to datetime
                        self.df[selected_column] = pd.to_datetime(self.df[selected_column], errors='coerce', dayfirst=False)
                        
                        # Print the first few converted dates for verification
                        st.write(f"Fechas convertidas (primeras 5): {self.df[selected_column].head().tolist()}")
                        
                        # Drop rows with invalid dates
                        valid_rows = self.df[selected_column].notnull().sum()
                        total_rows = len(self.df)
                        self.df = self.df.dropna(subset=[selected_column])
                        
                        # Set the selected column as index
                        self.df.set_index(selected_column, inplace=True)
                        
                        st.success(f"Columna '{selected_column}' configurada como índice temporal. {valid_rows} de {total_rows} filas son válidas.")
                        logging.info(f"Columna '{selected_column}' configurada como índice temporal")
                        
                    return selected_column
                else:
                    st.warning("No se encontraron columnas con formato de fecha.")
                    logging.warning("No se encontraron columnas con formato de fecha válido")
                    
                    # Debug information
                    st.write("### DEBUG - Información de columnas:")
                    for column in self.df.columns:
                        st.write(f"Columna: {column}, Tipo: {self.df[column].dtype}")
                        if pd.api.types.is_string_dtype(self.df[column]) or pd.api.types.is_object_dtype(self.df[column]):
                            st.write(f"Valores: {self.df[column].head().tolist()}")
                    
                    return None
        return None
    def render_create_data(self):
        """Muestra los controles para crear los datos si no se encuentran."""
        col2, _ = st.columns(2)
        with col2:
            st.write("## Datos")
            st.write("### Rango de fechas:")
            
            frecuencia = st.selectbox(
                "Los datos vienen de forma:",
                ("Anual", "Mensual", "Diaria", "Por días laborales", "Por hora", "Por minuto", "Por segundo")
            )
            tab1, tab2 = st.tabs(["Top Rows", "Last Rows"])
            with tab1:
                if 'fecha' in self.df.columns:
                    st.write(self.df[['fecha']].head())  # Muestra las primeras filas del dataframe
            with tab2:
                if 'fecha' in self.df.columns:
                    st.write(self.df[['fecha']].tail())  # Muestra las últimas filas del dataframe
            col1, col2 = st.columns(2)
            with col1:
                fecha_inicio = st.date_input("Fecha de inicio")
            with col2:
                fecha_fin = st.date_input("Fecha de fin")
            if st.button("Crear índice temporal"):
                # Crear el índice temporal según la frecuencia seleccionada
                freq_map = {
                    "Anual": "YE", 
                    "Mensual": "M", 
                    "Diaria": "D", 
                    "Por días laborales": "B", 
                    "Por hora": "H", 
                    "Por minuto": "T", 
                    "Por segundo": "S"
                }
                freq = freq_map.get(frecuencia)
                
                # Create date range
                date_range = pd.date_range(start=fecha_inicio, end=fecha_fin, freq=freq)
                
                # Create new dataframe with the date range as index
                if len(self.df.columns) > 0:
                    # If there's existing data, try to match it to the new date range
                    numeric_cols = self.df.select_dtypes(include=['number']).columns
                    if len(numeric_cols) > 0:
                        # Create a new dataframe with the date range and first numeric column
                        new_df = pd.DataFrame(index=date_range)
                        new_df[numeric_cols[0]] = np.nan  # Initialize with NaN
                        self.df = new_df
                    else:
                        # Just create a dataframe with the date range and a dummy column
                        self.df = pd.DataFrame(index=date_range, data={'valor': np.nan})
                else:
                    # Just create a dataframe with the date range and a dummy column
                    self.df = pd.DataFrame(index=date_range, data={'valor': np.nan})
                
                st.success(f"Índice temporal creado con frecuencia {frecuencia} desde {fecha_inicio} hasta {fecha_fin}.")
                logging.info(f"Índice temporal creado con frecuencia {frecuencia}")
                
            logging.info(f"Creación de datos con frecuencia {frecuencia} desde {fecha_inicio} hasta {fecha_fin}.")
            return frecuencia, fecha_inicio, fecha_fin
    def render_smoothing_selector(self):
        """Muestra el selector de suavizado."""
        col3, _ = st.columns(2)
        with col3:
            st.write("## Suavizado")
            st.write("### Cantidad de datos a promediar:")
            return st.slider("Selecciona el número de elementos a promediar:", 1, 10, 3)
    def render_options_panel(self):
        """Función principal que maneja el panel de opciones."""
        
        # Verificar si los datos están cargados
        if not self.validate_data_loaded():
            return None, None, None
        
        # Selección de opciones
        option = self.render_option_selector()
        
        # Inicializar variables
        datos = None
        suavizado = None
        
        # Si la opción es "Seleccionar", mostramos el selector de columnas de datos
        if option == "Seleccionar":
            datos = self.render_data_selector(option)
            # Selección de suavizado
            suavizado = self.render_smoothing_selector()
        # Si la opción es "Crear", pedimos los datos necesarios
        elif option == "Crear":
            frecuencia, fecha_inicio, fecha_fin = self.render_create_data()
            # Guardar los datos en una estructura que se pueda manejar con 3 valores de retorno
            datos = {
                'frecuencia': frecuencia,
                'fecha_inicio': fecha_inicio,
                'fecha_fin': fecha_fin
            }
            # Selección de suavizado
            suavizado = self.render_smoothing_selector()
        
        logging.info(f"Opciones seleccionadas: {option}, Datos: {datos}, Suavizado: {suavizado}")
        return option, datos, suavizado
class app:
    def main(self):
        st.title("Análisis de Series Temporales")
        
        if 'data' in st.session_state and st.session_state['data'] is not None:
            series_temp = SeriesTemporales(df=st.session_state['data'])
            # Resto del código
            option, datos, suavizado = series_temp.render_options_panel()
            if option is None:  # Verificar si no se han seleccionado opciones
                return
            if option == "Seleccionar" and datos is not None:
                st.write(f"Has seleccionado la columna: {datos}")
                
                # Here you could add additional analysis based on the selected time column
                if hasattr(series_temp, 'df') and series_temp.df is not None:
                    # Instead of checking is_all_dates, check if the index is a DatetimeIndex
                    if isinstance(series_temp.df.index, pd.DatetimeIndex):
                        st.write("### Vista previa de los datos temporales:")
                        st.write(series_temp.df.head())
                        
                        # Example: show a line chart of the time series data
                        if len(series_temp.df.columns) > 0:
                            first_col = series_temp.df.columns[0]
                            st.line_chart(series_temp.df[first_col])
                
            elif option == "Crear" and datos is not None:
                # Extraer los datos del diccionario
                if isinstance(datos, dict):  # Check if datos is a dictionary
                    frecuencia = datos.get('frecuencia')
                    fecha_inicio = datos.get('fecha_inicio')
                    fecha_fin = datos.get('fecha_fin')
                    
                    st.write(f"Frecuencia: {frecuencia}")
                    st.write(f"Fecha de inicio: {fecha_inicio}")
                    st.write(f"Fecha de fin: {fecha_fin}")
                    st.write(f"Suavizado: {suavizado}")
                else:
                    st.write(f"Datos seleccionados: {datos}")
                    st.write(f"Suavizado: {suavizado}")
                
                # Add visualization for created time series if needed
                
            # Continuar con el análisis de series temporales si es necesario
            # ...
        else:
            st.warning("No hay datos cargados. Carga un dataset en el Pipeline primero.")