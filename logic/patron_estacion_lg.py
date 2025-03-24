import logging
import pandas as pd
import plotly.graph_objects as go

def process_data(data: pd.DataFrame, seasonal_pattern: str, training_split: int):
    """
    Procesa los datos para generar el gráfico interactivo.
    - data: DataFrame con los datos suavizados (debe tener un índice de fecha).
    - seasonal_pattern: "Annual", "Monthly" o "Weekly".
    - training_split: Porcentaje de datos de entrenamiento.
    
    Retorna:
        fig: La figura de Plotly.
        resampled: DataFrame resampleado.
    """
    try:
        # Resampleo según el patrón seleccionado
        if seasonal_pattern == "Annual":
            resampled = data.resample('YE').mean()
        elif seasonal_pattern == "Monthly":
            resampled = data.resample('M').mean()
        elif seasonal_pattern == "Weekly":
            resampled = data.resample('W').mean()
        else:
            raise ValueError("Patrón estacional no reconocido.")

        total_points = len(resampled)
        if total_points == 0:
            raise ValueError("No hay datos disponibles luego del resampleo.")

        # Definir la cantidad de datos de entrenamiento
        train_count = int(total_points * training_split / 100)
        train_count = max(1, min(train_count, total_points - 1))
        train_data = resampled.iloc[:train_count]
        test_data = resampled.iloc[train_count:]

        # Seleccionar la primera columna numérica para graficar
        column_to_plot = resampled.select_dtypes(include='number').columns[0]

        # Crear trazas para la serie total, entrenamiento y prueba
        trace_total = go.Scatter(
            x=resampled.index,
            y=resampled[column_to_plot],
            mode='lines',
            name='Serie Resampleada',
            line=dict(dash='dot', color='gray')
        )

        trace_train = go.Scatter(
            x=train_data.index,
            y=train_data[column_to_plot],
            mode='lines+markers',
            name='Entrenamiento'
        )

        trace_test = go.Scatter(
            x=test_data.index,
            y=test_data[column_to_plot],
            mode='lines+markers',
            name='Prueba'
        )

        # Construir la figura
        fig = go.Figure(data=[trace_total, trace_train, trace_test])
        fig.update_layout(
            title=f"Patrón {seasonal_pattern} - Entrenamiento vs Prueba",
            xaxis_title="Fecha",
            yaxis_title=column_to_plot,
            hovermode="x unified"
        )

        logging.info("Gráfico interactivo generado correctamente usando Plotly.")
        return fig, resampled
    except Exception as e:
        logging.error("Error en process_data: " + str(e))
        raise e

class SeasonalPatternLogic:
    def __init__(self, data: pd.DataFrame):
        """
        Inicializa la lógica usando los datos suavizados.
        """
        self.data = data

    def validate_smoothed_data(self) -> bool:
        """
        Verifica que existan datos suavizados.
        """
        if self.data is None or self.data.empty:
            logging.warning("No se encontraron datos suavizados.")
            return False
        return True

    def process(self, seasonal_pattern: str, training_split: int):
        """
        Procesa los datos y retorna la figura y los datos resampleados.
        """
        # Es buena práctica trabajar con una copia de los datos
        return process_data(self.data.copy(), seasonal_pattern, training_split)
