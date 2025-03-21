import streamlit as st
import pandas as pd
import numpy as np
from exploratorio import EDA
import logging

def setup_logging():
    """Configura el sistema de logging."""
    logging.basicConfig(filename="audit_log.txt", level=logging.INFO, format="%(asctime)s - %(message)s")


class SeriesTemporales:
    def __init__(self, df):
        setup_logging()
        self.original_df = df.copy()
        self.df = df.copy()
        #self.df = df  activar y quitar  el anterior.
        self.show_smoothed = False
        self.eda = EDA(self.df)
        logging.info(f"DataFrame recibido con columnas: {list(self.df.columns)}")
        #self.current_page = 0  # Control de paginación

    def validate_data_loaded(self):
        """Verifica si los datos están cargados en el pipeline."""
        if self.df is None or self.df.empty:
            st.warning("No hay datos cargados. Carga primero un dataset válido.")
            logging.warning("Intento de realizar operaciones sin datos cargados.")
            return False
        return True

    def display_top_bottom_tabs(self):
        """Muestra pestañas con las primeras y últimas filas del dataframe."""
        tab1, tab2 = st.tabs(["Top Rows", "Last Rows"])
        with tab1:
            st.write(self.eda.head_df(n=5))
        with tab2:
            st.write(self.eda.tail_df(n=5))               

    def smooth_params_panel(self):
        """Panel con los parámetros específicos para la configuración de suavizado."""
        if not self.validate_data_loaded():
            return None, None, None, None

        col1, col2, col3 = st.columns([1.7, 1.6, 1.5])  # Ajustamos tamaños para balance visual
        with col1: #Selección de fechas
            option, d_fecha = self.date_config_selection()

        with col2: #Selección de la variable numérica
            selected_data_col = self.render_numeric_column_selector()
            if selected_data_col: 
                st.write(f"Columna de datos seleccionada: {selected_data_col}")

        with col3: #Configuración de suavizado
            suavizado = self.render_smoothing_selector()

        self.render_smoothing_button( d_fecha, selected_data_col, suavizado)

        return option, d_fecha, selected_data_col, suavizado
    
    def modify_date_config(self):
        """Renderiza los controles para la creación de datos si no existen fechas."""
        frecuencia = st.selectbox(
            "Los datos vienen de forma:",
            ("Anual", "Mensual", "Diaria", "Por días laborales", "Por hora", "Por minuto", "Por segundo")
        )

        col1, col2 = st.columns(2)
        with col1:
            fecha_inicio = st.date_input("Fecha de inicio")
        with col2:
            fecha_fin = st.date_input("Fecha de fin")

        self.display_top_bottom_tabs()

        return {"frecuencia": frecuencia, "fecha_inicio": fecha_inicio, "fecha_fin": fecha_fin}

    def date_config_selection(self):
        """Configura la selección de fecha y la opción de modificación del formato de fecha."""
        st.write("### Fechas")

        date_selected = self.render_date_column_selector()
        
        if date_selected:
            modify_option = st.radio("¿Quieres modificar el formato de la fecha?", ("No", "Sí"))
            
            if modify_option == "No":
                st.write(f"Usando la fecha seleccionada: {date_selected}")
                # Aquí puedes convertir la columna a datetime y establecerla como índice
                self.df[date_selected] = pd.to_datetime(self.df[date_selected], errors='coerce')
                self.df.set_index(date_selected, inplace=True)
            elif modify_option == "Sí":
                date_config = self.modify_date_config()
                return modify_option, date_config
        else:
            # Si no se encontró ninguna columna de fecha, ofrecer crear una nueva
            modify_option = "Sí"
            date_config = self.modify_date_config()
            return modify_option, date_config
            
        return modify_option, date_selected

    def get_date_columns(self):
        """Identifica columnas que probablemente contienen fechas, excluyendo columnas puramente numéricas."""
        date_columns = []
        
        for column in self.df.columns:
            # Verifica si la columna ya es de tipo datetime
            if pd.api.types.is_datetime64_any_dtype(self.df[column]):
                date_columns.append(column)
                continue
                
            # Excluye columnas numéricas (int, float)
            if pd.api.types.is_numeric_dtype(self.df[column]):
                # Verifica si son años (entre 1900 y 2100)
                if self.df[column].min() >= 1900 and self.df[column].max() <= 2100:
                    # Podría ser una columna de años
                    pass
                else:
                    # Es una columna numérica normal, no fecha
                    continue
                    
            # Para columnas de texto, verifica si pueden convertirse a fechas
            if pd.api.types.is_string_dtype(self.df[column]) or pd.api.types.is_object_dtype(self.df[column]):
                # Verifica si hay valores vacíos o NaN
                non_null_values = self.df[column].dropna()
                if len(non_null_values) == 0:
                    continue
                    
                # Intenta convertir a datetime
                test_series = pd.to_datetime(non_null_values, errors='coerce')
                valid_dates = test_series.notnull().sum()
                
                # Si al menos el 70% son fechas válidas, considérala como columna de fecha
                if valid_dates / len(non_null_values) >= 0.7:
                    date_columns.append(column)
                    continue

                # Intento con formato español como '3-ene-23'
                test_series = non_null_values.apply(SeriesTemporales.parse_spanish_date)
                valid_dates = test_series.notnull().sum()
                
                # Si al menos el 70% son fechas válidas, considérala como columna de fecha
                if valid_dates / len(non_null_values) >= 0.7:
                    date_columns.append(column)
                    continue

            # Segundo intento: Intentar con formato específico '%d-%b-%y' ingles 
                try:
                    test_series = pd.to_datetime(non_null_values, format='%d-%b-%y', errors='coerce')
                    valid_dates = test_series.notnull().sum()
                    if valid_dates / len(non_null_values) >= 0.7:
                        date_columns.append(column)
                except Exception:
                    pass  # Si falla, simplemente ignoramos esta conversión
        
        return list(dict.fromkeys(date_columns))

    @staticmethod
    def parse_spanish_date(date_str):
        """
        Convierte fechas en formato español como '3-ene-23' a formato datetime.
        
        Args:
            date_str: String que contiene la fecha en formato día-mes-año (ej: '3-ene-23')
            
        Returns:
            Timestamp de pandas o NaT si no se puede convertir
        """
        if not isinstance(date_str, str):
            return pd.NaT
            
        try:
            # Diccionario para traducir abreviaturas de meses en español a números
            month_map = {
                'ene': '01', 'feb': '02', 'mar': '03', 'abr': '04',
                'may': '05', 'jun': '06', 'jul': '07', 'ago': '08',
                'sep': '09', 'oct': '10', 'nov': '11', 'dic': '12'
            }
            
            parts = date_str.split('-')
            if len(parts) != 3:
                return pd.NaT
                
            day, month_abbr, year = parts
            
            # Convertir mes de texto a número
            if month_abbr.lower() in month_map:
                month = month_map[month_abbr.lower()]
            else:
                return pd.NaT
                
            # Formatear año
            if len(year) == 2:
                year = '20' + year  # Asumimos años 2000+
                
            # Crear fecha en formato estándar
            return pd.to_datetime(f'{year}-{month}-{day.zfill(2)}')
        except:
            return pd.NaT




    def render_date_column_selector(self):
        """Muestra un selectbox con columnas de fecha disponibles."""
        date_columns = self.get_date_columns()
        
        if date_columns:
            selected_date_column = st.selectbox(
                "Selecciona una columna de fecha:",
                date_columns
            )
            return selected_date_column
        else:
            st.warning("No se encontraron columnas con formato de fecha válido.")
            return None



    def render_numeric_column_selector(self):
        """Renderiza la selección de la columna de datos numéricos."""
        st.write("### Datos")
        numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()

        if numeric_cols:
            return st.selectbox("Seleccione la columna con los datos:", numeric_cols)
        else:
            st.warning("No hay columnas numéricas disponibles.")
            return None

    def render_smoothing_selector(self):
        """Renderiza el selector de suavizado."""
        st.write("### Suavizado")
        window_size = st.slider("Selecciona el número de elementos a promediar:", 1, 10, 3)

        return window_size

    def render_smoothing_button(self, d_fecha, selected_data_col, suavizado ):
        """Muestra el botón de aplicar suavizado en una nueva fila completa."""
        if self.df is None or self.df.empty:
            return None

        # Crear un contenedor que ocupe todo el ancho de la pantalla
        with st.container():
            if st.button("Aplicar suavizado", key="apply_smoothing_button", use_container_width=True):
                if not d_fecha:
                    st.warning("Primero debes seleccionar una columna de fecha válida.")
                    return 
                    
                # Validar si se ha seleccionado una columna numérica
                if not selected_data_col:
                    st.warning("Debes seleccionar una columna numérica antes de aplicar el suavizado.")
                    return

                # Validar que se haya configurado un suavizado
                if not suavizado:
                    st.warning("Debes seleccionar un valor de suavizado antes de aplicar.")
                    return

                # Aplicar suavizado
                if self.apply_smoothing(window_size=suavizado):
                    self.show_smoothed = True
                    
                    # Generar la gráfica de los datos suavizados
                    self.display_time_series_data(show_smoothed=self.show_smoothed)






    def apply_smoothing_daily(self, date_column):
        """Genera una serie temporal interpolada a frecuencia diaria."""
        if date_column not in self.df.columns:
            st.warning("La columna de fecha seleccionada no es válida.")
            return
        
        self.df[date_column] = pd.to_datetime(self.df[date_column], errors='coerce')
        self.df.set_index(date_column, inplace=True)
        self.df = self.df.resample('D').ffill().reset_index()
        
        # Guardar en session_state para persistencia
        st.session_state['generated_data'] = self.df.copy()
        st.session_state['current_page'] = 0  # Reiniciar la paginación
        
        st.success("Suavizado aplicado y serie temporal generada correctamente.")
        self.display_paginated_table()

    def display_paginated_table(self):
        """Muestra la tabla de valores generados con paginación de 10 registros por página."""
        if 'generated_data' not in st.session_state:
            st.warning("No hay datos generados aún.")
            return
        
        df = st.session_state['generated_data']
        total_rows = len(df)
        rows_per_page = 10
        total_pages = (total_rows // rows_per_page) + (1 if total_rows % rows_per_page > 0 else 0)
        
        start_idx = st.session_state['current_page'] * rows_per_page
        end_idx = start_idx + rows_per_page
        
        st.write(f"Mostrando registros {start_idx + 1} - {min(end_idx, total_rows)} de {total_rows}")
        st.dataframe(df.iloc[start_idx:end_idx])
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button("⬅️ Anterior", disabled=st.session_state['current_page'] == 0):
                st.session_state['current_page'] -= 1
                st.rerun()
        with col3:
            if st.button("Siguiente ➡️", disabled=st.session_state['current_page'] >= total_pages - 1):
                st.session_state['current_page'] += 1
                st.rerun()
        
        st.download_button(
            "Descargar CSV", df.to_csv(index=False), "series_temporales.csv", "text/csv", key="download-csv"
        )





    def apply_smoothing(self, window_size):
        """Aplica el suavizado de media móvil a los datos de la serie temporal."""
        if isinstance(self.df.index, pd.DatetimeIndex) and len(self.df.columns) > 0:
            numeric_cols = self.df.select_dtypes(include=['number']).columns
            smoothed_data = self.df.copy()
            for col in numeric_cols:
                smoothed_col_name = f"{col}_smoothed_{window_size}"
                self.df[smoothed_col_name] = self.df[col].rolling(window=window_size).mean()

            st.success(f"Suavizado aplicado con ventana de {window_size} períodos.")
            logging.info(f"Suavizado aplicado con ventana de {window_size}")
            return True
        return False

    def display_time_series_data(self, show_smoothed=False):
        """Muestra los datos de la serie temporal."""
        if isinstance(self.df.index, pd.DatetimeIndex):

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

            if show_smoothed:
                smoothed_data = self.df.copy()  # Obtener los datos con suavizado aplicado
                self.save_smoothed_data(smoothed_data)

                return True
        return False

    def save_smoothed_data(self, smoothed_data):
        """Guarda los datos suavizados en un DataFrame y un archivo CSV."""
        numeric_cols = [col for col in smoothed_data.columns if '_smoothed_' in col]
        if numeric_cols:
            smoothed_data = smoothed_data[numeric_cols].copy()
            smoothed_data.index = self.df.index  # Asegurar que la fecha se mantenga como índice
            st.session_state['smoothed_data'] = smoothed_data
            smoothed_data.to_csv("smoothed_data.csv", index=True)
            st.success("Los datos suavizados han sido guardados correctamente.")
            logging.info("Datos suavizados guardados en 'smoothed_data.csv'")
            
            # Mostrar vista previa de los datos guardados
            #st.write("### Vista previa de los datos suavizados:")
            #st.dataframe(smoothed_data.tail())



class app:
    def main(self):

        st.markdown('<h2>Series Temporales </h2>', unsafe_allow_html=True)

        if 'data' in st.session_state and st.session_state['data'] is not None:
            series_temp = SeriesTemporales(df=st.session_state['data'])
            option, datos, selected_data_col, suavizado = series_temp.smooth_params_panel()

            if option is None:
                return
        else:
            st.warning("No hay datos cargados. Carga un dataset en el Pipeline primero.")