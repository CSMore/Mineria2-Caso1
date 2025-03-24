import streamlit as st
import pandas as pd
import numpy as np
from app_pages.exploratorio import EDA
import logging
class SeriesTemporales:
    def __init__(self, df):
        self.original_df = df.copy()
        self.df = df.copy()
        self.show_smoothed = False
        self.eda = EDA(self.df)
        logging.info(f"DataFrame recibido con columnas: {list(self.df.columns)}")
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
    
    def get_date_range_defaults(self, selected_date_column):
        """Calcula la fecha mínima y máxima de la columna seleccionada.
            Devuelve una tupla (default_start, default_end)
        """
        default_start = None
        default_end = None
        if selected_date_column and selected_date_column in self.df.columns:
            try:
                # Aplicar la función personalizada para convertir las fechas en formato español
                fechas = self.df[selected_date_column].apply(SeriesTemporales.parse_spanish_date)
                # Si la conversión falla completamente, intentar con pd.to_datetime
                if fechas.isna().all():
                    fechas = pd.to_datetime(self.df[selected_date_column], errors='coerce')
                default_start = fechas.min().date() if pd.notna(fechas.min()) else None
                default_end = fechas.max().date() if pd.notna(fechas.max()) else None
            except Exception as e:
                logging.warning(f"No se pudo calcular el rango de fechas: {e}")
        return default_start, default_end
    def render_date_config_ui(self, selected_date_column):
        """
        Renderiza los controles para configurar la fecha y la frecuencia.
        """
        default_start, default_end = self.get_date_range_defaults(selected_date_column)
        frecuencia = st.selectbox(
            "Los datos vienen de forma:",
            ("Anual", "Mensual", "Diaria", "Por días laborales", "Por hora", "Por minuto", "Por segundo")
        )
        col1, col2 = st.columns(2)
        with col1:
            fecha_inicio = st.date_input("Fecha de inicio", value=default_start)
        with col2:
            fecha_fin = st.date_input("Fecha de fin", value=default_end)
        self.display_top_bottom_tabs()
        return {"frecuencia": frecuencia, "fecha_inicio": fecha_inicio, "fecha_fin": fecha_fin}
    def render_date_config_selection_ui(self):
        """Renderiza los controles para seleccionar la columna de fecha y si se desea modificar su formato."""
        st.write("### Fechas")
        date_selected = self.render_date_column_selector()
        modify_option = None
        if date_selected:
            modify_option = st.radio("¿Quieres modificar el formato de la fecha?", ("No", "Sí"))
        else:
            st.warning("No se encontró una columna de fecha válida.")
        return date_selected, modify_option
    def apply_date_config_no_modification(self, date_selected):
        """
        Lógica para la configuración sin modificación:
        - Convierte la columna a datetime usando pd.to_datetime.
        - Establece la columna como índice.
        - Elimina índices duplicados.
        """
        st.caption(f"Usando la fecha seleccionada: {date_selected}")
        self.df[date_selected] = pd.to_datetime(self.df[date_selected], errors='coerce')
        self.df.set_index(date_selected, inplace=True)
        if self.df.index.duplicated().any():
            self.df = self.df[~self.df.index.duplicated(keep='first')]
        return date_selected
    def apply_date_config_modification(self, date_selected):
        """
        Lógica para la configuración con modificación:
        - Llama a render_date_config_ui para obtener la configuración visual (rango de fechas y frecuencia).
        - Aplica la conversión de fecha usando la función personalizada.
        - Establece la columna como índice y elimina duplicados.
        - Filtra el DataFrame según el rango seleccionado.
        """
        # Obtiene la configuración visual, que internamente calcula los defaults
        date_config = self.render_date_config_ui(date_selected)
        # Actualiza la columna de fechas usando la función personalizada
        self.df[date_selected] = self.df[date_selected].apply(SeriesTemporales.parse_spanish_date)
        self.df.set_index(date_selected, inplace=True)
        if self.df.index.duplicated().any():
            self.df = self.df[~self.df.index.duplicated(keep='first')]
        # Filtra el DataFrame según el rango de fechas seleccionado
        try:
            fecha_inicio = date_config["fecha_inicio"]
            fecha_fin = date_config["fecha_fin"]
            self.df = self.df.loc[f"{fecha_inicio}":f"{fecha_fin}"]
        except Exception as e:
            logging.warning("Error al filtrar por rango de fechas: " + str(e))
        return date_config
    def date_config_selection(self):
        """
        Función principal que configura la selección de fecha:
        - Renderiza los controles para seleccionar la columna de fecha y la opción para modificar el formato.
        - Según la opción elegida, aplica la lógica correspondiente.
        Retorna una tupla con (modify_option, configuración o la columna de fecha).
        """
        date_selected, modify_option = self.render_date_config_selection_ui()
        if date_selected:
            if modify_option == "No":
                result = self.apply_date_config_no_modification(date_selected)
            elif modify_option == "Sí":
                result = self.apply_date_config_modification(date_selected)
            else:
                result = date_selected  # Caso por defecto
            return modify_option, result
        else:
            # Si no se encontró una columna válida, se asume que se desea modificar y se procede
            modify_option = "Sí"
            result = self.apply_date_config_modification(None)
            return modify_option, result
    def get_date_columns(self):
        """Identifica columnas que probablemente contienen fechas, excluyendo columnas puramente numéricas."""
        date_columns = []
        
        for column in self.df.columns:
            if pd.api.types.is_datetime64_any_dtype(self.df[column]):
                date_columns.append(column)
                continue
            if pd.api.types.is_numeric_dtype(self.df[column]):
                if self.df[column].min() >= 1900 and self.df[column].max() <= 2100:
                    pass
                else:
                    continue
            if pd.api.types.is_string_dtype(self.df[column]) or pd.api.types.is_object_dtype(self.df[column]):
                non_null_values = self.df[column].dropna()
                if len(non_null_values) == 0:
                    continue
                test_series = pd.to_datetime(non_null_values, errors='coerce')
                valid_dates = test_series.notnull().sum()
                if valid_dates / len(non_null_values) >= 0.7:
                    date_columns.append(column)
                    continue
                test_series = non_null_values.apply(SeriesTemporales.parse_spanish_date)
                valid_dates = test_series.notnull().sum()
                if valid_dates / len(non_null_values) >= 0.7:
                    date_columns.append(column)
                    continue
                try:
                    test_series = pd.to_datetime(non_null_values, format='%d-%b-%y', errors='coerce')
                    valid_dates = test_series.notnull().sum()
                    if valid_dates / len(non_null_values) >= 0.7:
                        date_columns.append(column)
                except Exception:
                    pass
        
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
            month_map = {
                'ene': '01', 'feb': '02', 'mar': '03', 'abr': '04',
                'may': '05', 'jun': '06', 'jul': '07', 'ago': '08',
                'sep': '09', 'oct': '10', 'nov': '11', 'dic': '12'
            }
            parts = date_str.split('-')
            if len(parts) != 3:
                return pd.NaT
            day, month_abbr, year = parts
            if month_abbr.lower() in month_map:
                month = month_map[month_abbr.lower()]
            else:
                return pd.NaT
            if len(year) == 2:
                year = '20' + year
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
        window_size = st.slider("Selecciona el número de elementos a promediar:", 1, 10, 5)
        return window_size
    def render_smoothing_button(self, d_fecha, selected_data_col, suavizado):
        """Muestra el botón de aplicar suavizado en una nueva fila completa."""
        if self.df is None or self.df.empty:
            return None
        with st.container():
            if st.button("Aplicar suavizado", key="apply_smoothing_button", use_container_width=True):
                if not d_fecha:
                    st.warning("Primero debes seleccionar una columna de fecha válida.")
                    return 
                if not selected_data_col:
                    st.warning("Debes seleccionar una columna numérica antes de aplicar el suavizado.")
                    return
                if not suavizado:
                    st.warning("Debes seleccionar un valor de suavizado antes de aplicar.")
                    return
                if self.apply_smoothing(window_size=suavizado):
                    self.show_smoothed = True
                    self.display_time_series_data(show_smoothed=self.show_smoothed)
    def apply_smoothing(self, window_size):
        """Aplica el suavizado de media móvil a los datos de la serie temporal."""
        if isinstance(self.df.index, pd.DatetimeIndex) and len(self.df.columns) > 0:
            numeric_cols = self.df.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                smoothed_col_name = f"{col}_smoothed_{window_size}"
                self.df[smoothed_col_name] = self.df[col].rolling(window=window_size, center=True).mean()
            st.success(f"Suavizado aplicado con ventana de {window_size} períodos.")
            logging.info(f"Suavizado aplicado con ventana de {window_size}")
            return True
        return False
    def display_time_series_data(self, show_smoothed=False):
        """Muestra los datos de la serie temporal en forma de tabla con scroll,
        y actualiza solo este fragmento cuando se interactúa."""
        
        # Definimos el fragmento dentro de la función
        @st.fragment
        def time_series_fragment():
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
                    st.dataframe(self.df[display_cols], height=300)
                if show_smoothed:
                    # Guarda los datos suavizados en session_state y en CSV
                    smoothed_data = self.df.copy()
                    self.save_smoothed_data(smoothed_data)
                    col_left, col_right = st.columns([3, 1])
                    with col_left:
                        st.download_button(
                            "Descargar CSV",self.df[display_cols].to_csv(index=True),
                            "series_temporales.csv",
                            "text/csv",
                            key="download-csv"
                        )
                    with col_right:
                        if st.button("Siguiente"):
                            st.switch_page("app_pages/patron_estacional.py")  
        # Ejecutamos el fragmento, de modo que solo esta parte se refresca
        time_series_fragment()
        return False
    def save_smoothed_data(self, smoothed_data):
        """Guarda los datos suavizados en un DataFrame y un archivo CSV."""
        numeric_cols = [col for col in smoothed_data.columns if '_smoothed_' in col]
        if numeric_cols:
            smoothed_data = smoothed_data[numeric_cols].copy()
            smoothed_data.index = self.df.index
            st.session_state['smoothed_data'] = smoothed_data
            smoothed_data.to_csv("smoothed_data.csv", index=True)
            st.success("Los datos suavizados han sido guardados correctamente.")
            logging.info("Datos suavizados guardados en 'smoothed_data.csv'")
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

if __name__ == "__page__":
    app().main()
