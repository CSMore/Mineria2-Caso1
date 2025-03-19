import streamlit as st
import pandas as pd
import numpy as np
import logging

def setup_logging():
    """Configura el sistema de logging."""
    logging.basicConfig(filename="audit_log.txt", level=logging.INFO, format="%(asctime)s - %(message)s")


class SeriesTemporales:
    def __init__(self, df):
        setup_logging()
        self.original_df = df.copy()
        self.df = df.copy()
        self.show_smoothed = False
        logging.info(f"DataFrame recibido con columnas: {list(self.df.columns)}")

    def validate_data_loaded(self):
        """Verifica si los datos están cargados en el pipeline."""
        if self.df is None or self.df.empty:
            st.warning("No hay datos cargados. Carga primero un dataset válido.")
            logging.warning("Intento de realizar operaciones sin datos cargados.")
            return False
        return True

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

    def render_options_panel(self):
        """Panel con las opciones organizadas en columnas."""
        if not self.validate_data_loaded():
            return None, None, None, None

        col1, col2, col3 = st.columns([1.7, 1.6, 1.5])  # Ajustamos tamaños para balance visual

        with col1: #Selección de fechas
            option, d_fecha = self.render_option_selection()

        with col2: #Selección de la variable numérica
            selected_data_col = self.render_numeric_column_selector()
            if selected_data_col: 
                st.write(f"Columna de datos seleccionada: {selected_data_col}")

        with col3: #Configuración de suavizado
            suavizado = self.render_smoothing_selector()

        self.render_smoothing_button( d_fecha, selected_data_col, suavizado)

        #self.display_time_series_data(show_smoothed=self.show_smoothed)
        return option, d_fecha, selected_data_col, suavizado

    def render_option_selection(self):
        """Renderiza la opción de 'Seleccionar' o 'Crear' y la configuración asociada."""
        st.write("## Opciones")
        option = st.radio("Selecciona una opción:", ("Seleccionar", "Crear"))

        if option == "Seleccionar":
            datos = self.render_data_selector()
            if datos:  
                st.write(f"Columna de fecha seleccionada: {datos}")
        elif option == "Crear":
            datos = self.render_create_data()  # Retorna frecuencia, fecha_inicio y fecha_fin
            if datos:
                st.write(f"Frecuencia: {datos['frecuencia']}")
                st.write(f"Fecha de inicio: {datos['fecha_inicio']}")
                st.write(f"Fecha de fin: {datos['fecha_fin']}")            

        return option, datos

    def render_data_selector(self):
        """Renderiza el selector de columna de fechas en los datos."""
        potential_date_columns = self.get_potential_date_columns()

        if potential_date_columns:
            return self.select_date_column(potential_date_columns)
        else:
            st.warning("No se encontraron columnas con formato de fecha válido.")
            logging.warning("No se encontraron columnas con formato de fecha válido")
            return None

    def get_potential_date_columns(self):
        """Verifica las columnas que pueden ser de tipo fecha."""
        potential_date_columns = []

        potential_date_columns += self.get_valid_date_column('fecha')

        if not potential_date_columns: # si no hay una columna llamada "fecha", revisa las columnas de tipo datetime
            potential_date_columns += self.get_valid_datetime_columns()

        # Si aún no se encontró, revisa las columnas de texto y trata de convertirlas
        if not potential_date_columns:
            potential_date_columns += self.get_valid_date_columns_from_text()

        return potential_date_columns

    def get_valid_date_column(self, column_name):
        """Verifica si una columna específica es válida para fechas."""
        potential_date_columns = []
        if column_name in self.df.columns:
            test_series = pd.to_datetime(self.df[column_name], errors='coerce')
            valid_dates = test_series.notnull().sum()
            if valid_dates > 0:
                potential_date_columns.append(column_name)
                logging.info(f"Columna '{column_name}' identificada con {valid_dates} fechas válidas")
        return potential_date_columns

    def get_valid_datetime_columns(self):
        """Verifica si hay columnas de tipo datetime en el DataFrame."""
        potential_date_columns = []
        for column in self.df.columns:
            if pd.api.types.is_datetime64_any_dtype(self.df[column]):
                potential_date_columns.append(column)
                st.success(f"Columna '{column}' identificada como columna de fecha.")
                logging.info(f"Columna '{column}' identificada como columna de fecha")
        return potential_date_columns

    def get_valid_date_columns_from_text(self):
        """Verifica las columnas de texto y trata de convertirlas a fechas."""
        potential_date_columns = []
        st.write("### Revisando columnas de tipo texto...")
        for column in self.df.columns:
            if pd.api.types.is_string_dtype(self.df[column]) or pd.api.types.is_object_dtype(self.df[column]):
                test_series = pd.to_datetime(self.df[column], errors='coerce')
                valid_dates = test_series.notnull().sum()
                if valid_dates > 0:
                    potential_date_columns.append(column)
                    st.success(f"Columna '{column}' identificada con {valid_dates} fechas válidas")
                    logging.info(f"Columna '{column}' identificada con {valid_dates} fechas válidas")
        return potential_date_columns

    def select_date_column(self, potential_date_columns):
        """Muestra el selector de columna de fecha y permite usarla como índice temporal."""
        selected_column = st.selectbox("Selecciona una columna de fecha:", potential_date_columns)


        self.df[selected_column] = pd.to_datetime(self.df[selected_column], errors='coerce', dayfirst=False)
            
        # Eliminar filas con fechas no válidas
        valid_rows = self.df[selected_column].notnull().sum()
        total_rows = len(self.df)
        self.df = self.df.dropna(subset=[selected_column])
            
        # Establecer la columna seleccionada como índice
        self.df.set_index(selected_column, inplace=True)
        logging.info(f"Columna '{selected_column}' configurada como índice temporal")
        
        return selected_column

    def render_create_data(self):
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

        self.render_data_preview()

        return {"frecuencia": frecuencia, "fecha_inicio": fecha_inicio, "fecha_fin": fecha_fin}

    def render_numeric_column_selector(self):
        """Renderiza la selección de la columna de datos numéricos."""
        st.write("## Datos")
        st.write("### Seleccione la columna con los datos:")
        numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()

        if numeric_cols:
            return st.selectbox("Columnas disponibles:", numeric_cols)
        else:
            st.warning("No hay columnas numéricas disponibles.")
            return None

    def render_smoothing_selector(self):
        """Renderiza el selector de suavizado."""
        st.write("## Suavizado")
        st.write("### Cantidad de datos a promediar:")
        window_size = st.slider("Selecciona el número de elementos a promediar:", 1, 10, 3)

        return window_size

    def render_smoothing_button(self, d_fecha, selected_data_col, suavizado ):
        """Muestra el botón de aplicar suavizado en una nueva fila completa."""
        if self.df is None or self.df.empty:
            return None

        # Crear un contenedor que ocupe todo el ancho de la pantalla
        with st.container():
            if st.button("Aplicar suavizado", key="apply_smoothing_button", use_container_width=True):
                if not isinstance(self.df.index, pd.DatetimeIndex):
                    st.warning("Primero debes seleccionar una columna de fecha válida.")
                    selected_column = self.select_date_column(self.get_potential_date_columns())
                    if selected_column:  # Verifica si se seleccionó una columna
                        st.success(f"Columna '{selected_column}' configurada correctamente como índice temporal.")
                    else:
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


    def apply_smoothing(self, window_size):
        """Aplica el suavizado de media móvil a los datos de la serie temporal."""
        if isinstance(self.df.index, pd.DatetimeIndex) and len(self.df.columns) > 0:
            numeric_cols = self.df.select_dtypes(include=['number']).columns
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
            #st.write("### Vista previa de los datos temporales:")
            #st.write(self.df.head())

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


class app:
    def main(self):

        st.markdown('<h2>Series Temporales </h2>', unsafe_allow_html=True)

        if 'data' in st.session_state and st.session_state['data'] is not None:
            series_temp = SeriesTemporales(df=st.session_state['data'])
            option, datos, selected_data_col, suavizado = series_temp.render_options_panel()

            if option is None:
                return
        else:
            st.warning("No hay datos cargados. Carga un dataset en el Pipeline primero.")