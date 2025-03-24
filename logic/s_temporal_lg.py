import pandas as pd
import numpy as np
import logging
from utils.data_validation import validate_data_loaded

class SeriesTemporalesLogic:
    def __init__(self, df):
        self.original_df = df.copy()
        self.df = df.copy()
        logging.info(f"DataFrame recibido con columnas: {list(self.df.columns)}")

    def validate_data_loaded(self):
        return validate_data_loaded(self.df)

    def get_date_range_defaults(self, selected_date_column):
        """Calcula la fecha mínima y máxima de la columna seleccionada."""
        default_start = None
        default_end = None
        if selected_date_column and selected_date_column in self.df.columns:
            try:
                # Se intenta convertir la columna usando la función personalizada
                fechas = self.df[selected_date_column].apply(SeriesTemporalesLogic.parse_spanish_date)
                if fechas.isna().all():
                    fechas = pd.to_datetime(self.df[selected_date_column], errors='coerce')
                default_start = fechas.min().date() if pd.notna(fechas.min()) else None
                default_end = fechas.max().date() if pd.notna(fechas.max()) else None
            except Exception as e:
                logging.warning(f"No se pudo calcular el rango de fechas: {e}")
        return default_start, default_end

    @staticmethod
    def parse_spanish_date(date_str):
        """
        Convierte fechas en formato español como '3-ene-23' a formato datetime.
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

    def apply_date_config_no_modification(self, date_selected):
        """Convierte la columna a datetime, establece el índice y elimina duplicados."""
        self.df[date_selected] = pd.to_datetime(self.df[date_selected], errors='coerce')
        self.df.set_index(date_selected, inplace=True)
        if self.df.index.duplicated().any():
            self.df = self.df[~self.df.index.duplicated(keep='first')]
        return date_selected

    def apply_date_config_modification(self, date_selected, date_config):
        """
        Aplica la conversión personalizada y filtra el DataFrame según el rango de fechas.
        date_config es un diccionario con las claves "fecha_inicio" y "fecha_fin".
        """
        self.df[date_selected] = self.df[date_selected].apply(SeriesTemporalesLogic.parse_spanish_date)
        self.df.set_index(date_selected, inplace=True)
        if self.df.index.duplicated().any():
            self.df = self.df[~self.df.index.duplicated(keep='first')]
        try:
            fecha_inicio = date_config["fecha_inicio"]
            fecha_fin = date_config["fecha_fin"]
            self.df = self.df.loc[f"{fecha_inicio}":f"{fecha_fin}"]
        except Exception as e:
            logging.warning("Error al filtrar por rango de fechas: " + str(e))
        return date_config

    def get_date_columns(self):
        """Identifica columnas que probablemente contienen fechas."""
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
                test_series = non_null_values.apply(SeriesTemporalesLogic.parse_spanish_date)
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

    def apply_smoothing(self, window_size):
        """Aplica el suavizado de media móvil a las columnas numéricas."""
        if isinstance(self.df.index, pd.DatetimeIndex) and len(self.df.columns) > 0:
            numeric_cols = self.df.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                smoothed_col_name = f"{col}_smoothed_{window_size}"
                self.df[smoothed_col_name] = self.df[col].rolling(window=window_size, center=True).mean()
            logging.info(f"Suavizado aplicado con ventana de {window_size}")
            return True
        return False

    def save_smoothed_data(self):
        """Guarda los datos suavizados en un CSV y retorna el DataFrame suavizado."""
        smoothed_cols = [col for col in self.df.columns if '_smoothed_' in col]
        if smoothed_cols:
            smoothed_data = self.df[smoothed_cols].copy()
            smoothed_data.index = self.df.index
            smoothed_data.to_csv("smoothed_data.csv", index=True)
            logging.info("Datos suavizados guardados en 'smoothed_data.csv'")
            return smoothed_data
        return None

    def display_time_series_data_logic(self, show_smoothed=False):
        """
        Prepara el DataFrame para mostrar:
        - Si show_smoothed es True, se incluyen las columnas suavizadas.
        - Retorna el DataFrame con las columnas a mostrar.
        """
        if isinstance(self.df.index, pd.DatetimeIndex):
            numeric_cols = self.df.select_dtypes(include=['number']).columns
            original_cols = [col for col in numeric_cols if '_smoothed_' not in col]
            if show_smoothed:
                smoothed_cols = [col for col in self.df.columns if '_smoothed_' in col]
                display_cols = original_cols + smoothed_cols
            else:
                display_cols = original_cols
            return self.df[display_cols]
        return self.df
