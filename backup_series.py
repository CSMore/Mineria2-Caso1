import streamlit as st
import pandas as pd
import numpy as np
import logging
from datetime import timedelta

def setup_logging():
    logging.basicConfig(filename="audit_log.txt", level=logging.INFO, format="%(asctime)s - %(message)s")

class SeriesTemporales:
    def __init__(self, df):
        setup_logging()
        self.original_df = df.copy()
        self.df = df.copy()
        logging.info(f"DataFrame recibido con columnas: {list(self.df.columns)}")

    def preview_df(self, n=5, tail=False):
        """Muestra las primeras o últimas filas del dataframe."""
        if self.df is None or self.df.empty:
            return "No se cargaron los datos :("
        return self.df.tail(n).astype(str) if tail else self.df.head(n).astype(str)
    
    def render_data_preview(self):
        """Muestra pestañas con las primeras y últimas filas del dataframe."""
        tab1, tab2 = st.tabs(["Top Rows", "Last Rows"])
        with tab1:
            st.write(self.preview_df(n=5))
        with tab2:
            st.write(self.preview_df(n=5, tail=True))

    def render_date_selection(self):
        """Crea la selección de fechas con columnas en Streamlit."""
        col1, col2 = st.columns(2)
        with col1:
            fecha_inicio = st.date_input("Fecha de inicio")
        with col2:
            fecha_fin = st.date_input("Fecha de fin")
        return fecha_inicio, fecha_fin

    def validate_data_loaded(self):
        """Verifica si los datos están cargados en el pipeline."""
        if self.df is None or self.df.empty:
            st.warning("No hay datos cargados. Carga primero un dataset válido.")
            logging.warning("Intento de realizar operaciones sin datos cargados.")
            return False
        return True

    @staticmethod
    def validate_column(df, column, convert_func, error_msg):
        """Valida una columna para convertirla con una función y manejar errores."""
        if column not in df.columns:
            st.error(f"La columna '{column}' no está presente en los datos.")
            logging.error(f"La columna '{column}' no se encuentra en el DataFrame.")
            return None
        
        # Intentar convertir la columna
        df[column] = convert_func(df[column], errors='coerce')
        
        if df[column].isnull().all():
            st.error(error_msg)
            return None
        
        df = df.dropna(subset=[column])
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

    def render_data_selector(self):
        """Muestra el selector de columna de datos."""
        col2, _ = st.columns(2)
        with col2:
            st.write("## Datos")
            st.write("### Selecciona la columna con los datos temporales:")
            
            potential_date_columns = []
            
            # Revisar la columna 'fecha' primero
            if 'fecha' in self.df.columns:
                test_series = pd.to_datetime(self.df['fecha'], errors='coerce')
                valid_dates = test_series.notnull().sum()
                if valid_dates > 0:
                    potential_date_columns.append('fecha')
                    st.success(f"Columna 'fecha' encontrada con {valid_dates} fechas válidas.")
                    logging.info(f"Columna 'fecha' identificada con {valid_dates} fechas válidas")
            
            # Si no hay una columna 'fecha' válida, revisa todas las demás columnas
            if not potential_date_columns:
                st.write("### Contenido de columnas de texto:")
                for column in self.df.columns:
                    if pd.api.types.is_string_dtype(self.df[column]) or pd.api.types.is_object_dtype(self.df[column]):
                        st.write(f"Columna '{column}' - Primeros 5 valores: {self.df[column].head().tolist()}")
                for column in self.df.columns:
                    if pd.api.types.is_string_dtype(self.df[column]) or pd.api.types.is_object_dtype(self.df[column]):
                        test_series = pd.to_datetime(self.df[column], errors='coerce')
                        valid_dates = test_series.notnull().sum()
                        if valid_dates > 0:
                            potential_date_columns.append(column)
                            st.success(f"Columna '{column}' identificada con {valid_dates} fechas válidas")
                            logging.info(f"Columna '{column}' identificada con {valid_dates} fechas válidas")
            
            if potential_date_columns:
                selected_column = st.selectbox("Selecciona una columna de fecha:", potential_date_columns)
                if st.button("Usar esta columna como índice temporal"):
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
            
            self.render_data_preview()
            
            col1, col2 = st.columns(2)
            with col1:
                fecha_inicio = st.date_input("Fecha de inicio")
            with col2:
                fecha_fin = st.date_input("Fecha de fin")
            
            if st.button("Crear índice temporal"):
                freq_map = {
                    "Anual": "YE", "Mensual": "M", "Diaria": "D", "Por días laborales": "B", 
                    "Por hora": "H", "Por minuto": "T", "Por segundo": "S"
                }
                freq = freq_map.get(frecuencia)
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
                
            return frecuencia, fecha_inicio, fecha_fin

    def render_smoothing_selector(self):
        """Muestra el selector de suavizado."""
        col3, _ = st.columns(2)
        with col3:
            st.write("## Suavizado")
            st.write("### Cantidad de datos a promediar:")
            window_size = st.slider("Selecciona el número de elementos a promediar:", 1, 10, 3)
            
            if st.button("Aplicar suavizado"):
                self.apply_smoothing(window_size)
                self.display_time_series_data(show_smoothed=True)
                
            return window_size

    def apply_smoothing(self, window_size):
        """Apply moving average smoothing to the time series data."""
        if isinstance(self.df.index, pd.DatetimeIndex) and len(self.df.columns) > 0:
            # Apply rolling mean to numeric columns
            numeric_cols = self.df.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                smoothed_col_name = f"{col}_smoothed_{window_size}"
                self.df[smoothed_col_name] = self.df[col].rolling(window=window_size).mean()
            
            st.success(f"Suavizado aplicado con ventana de {window_size} períodos.")
            logging.info(f"Suavizado aplicado con ventana de {window_size}")
            return True
        return False

    def display_time_series_data(self, show_smoothed=False):
        """Display the time series data if it's properly formatted."""
        if isinstance(self.df.index, pd.DatetimeIndex):
            st.write("### Vista previa de los datos temporales:")
            st.write(self.df.head())
            
            # Show charts for the data
            if len(self.df.columns) > 0:
                # If smoothed data is available and requested, show both original and smoothed
                if show_smoothed:
                    smoothed_cols = [col for col in self.df.columns if '_smoothed_' in col]
                    if smoothed_cols:
                        st.write("### Datos originales y suavizados:")
                        # Create a new dataframe with only the columns we want to display
                        display_cols = []
                        # Add original columns first
                        numeric_cols = self.df.select_dtypes(include=['number']).columns
                        original_cols = [col for col in numeric_cols if '_smoothed_' not in col]
                        display_cols.extend(original_cols)
                        # Then add smoothed columns
                        display_cols.extend(smoothed_cols)
                        
                        # Display the chart with all data
                        st.line_chart(self.df[display_cols])
                        return True
                
                # If no smoothed data or not requested, just show original data
                numeric_cols = self.df.select_dtypes(include=['number']).columns
                original_cols = [col for col in numeric_cols if '_smoothed_' not in col]
                if original_cols:
                    st.write("### Datos originales:")
                    st.line_chart(self.df[original_cols])
                    return True
        return False
    
    def render_options_panel(self):
        """Función principal que maneja el panel de opciones."""
        if not self.validate_data_loaded():
            return None, None, None
        
        option = self.render_option_selector()
        datos = None
        suavizado = None
        
        if option == "Seleccionar":
            datos = self.render_data_selector()
            suavizado = self.render_smoothing_selector()
        elif option == "Crear":
            frecuencia, fecha_inicio, fecha_fin = self.render_create_data()
            datos = {'frecuencia': frecuencia, 'fecha_inicio': fecha_inicio, 'fecha_fin': fecha_fin}
            suavizado = self.render_smoothing_selector()
        
        logging.info(f"Opciones seleccionadas: {option}, Datos: {datos}, Suavizado: {suavizado}")
        return option, datos, suavizado

class app:
    def main(self):
        st.title("Análisis de Series Temporales")
        
        if 'data' in st.session_state and st.session_state['data'] is not None:
            series_temp = SeriesTemporales(df=st.session_state['data'])
            option, datos, suavizado = series_temp.render_options_panel()

            if option is None:  # Verificar si no se han seleccionado opciones
                return

            if option == "Seleccionar" and datos is not None:
                st.write(f"Has seleccionado la columna: {datos}")
                series_temp.display_time_series_data()

            elif option == "Crear" and datos is not None:
                if isinstance(datos, dict):
                    frecuencia = datos.get('frecuencia')
                    fecha_inicio = datos.get('fecha_inicio')
                    fecha_fin = datos.get('fecha_fin')
                    st.write(f"Frecuencia: {frecuencia}")
                    st.write(f"Fecha de inicio: {fecha_inicio}")
                    st.write(f"Fecha de fin: {fecha_fin}")
                    st.write(f"Suavizado: {suavizado}")
                    
                series_temp.display_time_series_data()
        else:
            st.warning("No hay datos cargados. Carga un dataset en el Pipeline primero.")






    def render_data_selector(self):
        """Renderiza el selector de columna de fechas en los datos."""
        potential_date_columns = [col for col in self.df.columns if pd.to_datetime(self.df[col], errors='coerce').notnull().sum() > 0]

        if potential_date_columns:
            return st.selectbox("Selecciona una columna de fecha:", potential_date_columns)
        else:
            st.warning("No se encontraron columnas con formato de fecha válido.")
            logging.warning("No se encontraron columnas con formato de fecha válido")
            return None
        
# ---------------------------------------------------------------------------------------------#

    def render_smoothing_button(self):
        """Muestra el botón de aplicar suavizado en una nueva fila completa."""
        if self.df is None or self.df.empty:
            return None

        # Crear un contenedor que ocupe todo el ancho de la pantalla
        with st.container():
            if st.button("Aplicar suavizado", key="apply_smoothing_button", use_container_width=True):
                if self.show_smoothed:
                    st.warning("El suavizado ya ha sido aplicado.")
                else:
                    if self.apply_smoothing(window_size=3):  # Ajusta el tamaño de la ventana como desees
                        st.success("Suavizado aplicado con éxito.")


    def render_options_panel(self):
        """Panel con las opciones organizadas en columnas."""
        if not self.validate_data_loaded():
            return None, None, None, None

        col1, col2, col3 = st.columns([1.7, 1.6, 1.5])  # Ajustamos tamaños para balance visual

        with col1: #Selección de fechas
            option, datos = self.render_option_selection()

        with col2: #Selección de la variable numérica
            selected_data_col = self.render_numeric_column_selector()
            if selected_data_col: 
                st.write(f"Columna de datos seleccionada: {selected_data_col}")

        with col3: #Configuración de suavizado
            suavizado = self.render_smoothing_selector()

        self.render_smoothing_button()

        self.display_time_series_data(show_smoothed=self.show_smoothed)
        return option, datos, selected_data_col, suavizado
    


    def select_date_column(self, potential_date_columns):
        """Muestra el selector de columna de fecha y permite usarla como índice temporal."""
        selected_column = st.selectbox("Selecciona una columna de fecha:", potential_date_columns)

        # Acción al seleccionar una columna
        if st.button("Usar como columna de fecha"):
            self.df[selected_column] = pd.to_datetime(self.df[selected_column], errors='coerce', dayfirst=False)
            
            # Eliminar filas con fechas no válidas
            valid_rows = self.df[selected_column].notnull().sum()
            total_rows = len(self.df)
            self.df = self.df.dropna(subset=[selected_column])
            
            # Establecer la columna seleccionada como índice
            self.df.set_index(selected_column, inplace=True)
            
            st.success(f"Columna '{selected_column}' configurada como índice temporal. {valid_rows} de {total_rows} filas son válidas.")
            logging.info(f"Columna '{selected_column}' configurada como índice temporal")
        
        return selected_column




    def display_time_series_data(self, show_smoothed=False):
        """Muestra los datos de la serie temporal."""
        if isinstance(self.df.index, pd.DatetimeIndex):
            st.write("### Vista previa de los datos temporales:")
            st.write(self.df.head())

            numeric_cols = self.df.select_dtypes(include=['number']).columns
            original_cols = [col for col in numeric_cols if '_smoothed_' not in col]

            if show_smoothed:
                smoothed_cols = [col for col in self.df.columns if '_smoothed_' in col]
                display_cols = original_cols + smoothed_cols
                st.write("### Datos originales y suavizados:")
            else:
                display_cols = original_cols
                st.write("### Datos originales:")

            if display_cols:
                st.line_chart(self.df[display_cols])
                return True
        return False