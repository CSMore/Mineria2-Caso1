import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error
from datetime import timedelta

class SeriesTemporales:
    def __init__(self, df):
        self.df = df
        self.df['fecha'] = pd.to_datetime(self.df['fecha'])  # Asegurarse de que la fecha esté en formato datetime
        self.df.set_index('fecha', inplace=True)

    def analizar_datos(self):
        st.subheader("Análisis Exploratorio de la Serie Temporal")
        st.write("### Estadísticas Descriptivas:")
        st.write(self.df.describe())
        
        st.write("### Descomposición de la Serie Temporal")
        # Descomposición (tendencia, estacionalidad, residual)
        result = seasonal_decompose(self.df['valor'], model='additive', period=12)
        result.plot()
        st.pyplot(plt)

        st.write("### Visualización de la Serie Temporal:")
        st.line_chart(self.df['valor'])

    def entrenar_modelo(self):
        st.subheader("Entrenamiento de Modelo ARIMA")
        st.write("Configurando el modelo ARIMA...")

        # Se puede pedir al usuario elegir parámetros (p, d, q) para el modelo ARIMA
        p = st.slider('Parámetro p (orden de AR)', 0, 5, 1)
        d = st.slider('Parámetro d (diferenciación)', 0, 2, 1)
        q = st.slider('Parámetro q (orden de MA)', 0, 5, 1)

        # Dividir datos en entrenamiento y prueba (80-20)
        train_size = int(len(self.df) * 0.8)
        train, test = self.df['valor'][:train_size], self.df['valor'][train_size:]
        
        # Ajuste del modelo ARIMA
        model = ARIMA(train, order=(p, d, q))
        model_fit = model.fit()

        st.write(f"Modelo ARIMA ajustado: AR({p}), I({d}), MA({q})")

        # Predicciones
        predictions = model_fit.forecast(steps=len(test))
        st.write("### Predicciones vs Realidad")
        pred_df = pd.DataFrame({'Real': test, 'Predicción': predictions}, index=test.index)

        # Graficar las predicciones
        st.line_chart(pred_df)

        # Evaluación del modelo con MAE
        mae = mean_absolute_error(test, predictions)
        st.write(f"Error Absoluto Medio (MAE): {mae}")

        return model_fit, predictions

    def predecir(self, model_fit):
        st.subheader("Realizar Predicciones Futuras")
        # Hacer predicciones futuras
        forecast_steps = st.slider("¿Cuántos pasos hacia el futuro quieres predecir?", 1, 12, 6)
        
        forecast = model_fit.forecast(steps=forecast_steps)
        
        st.write(f"### Predicciones futuras ({forecast_steps} pasos):")
        future_dates = pd.date_range(start=self.df.index[-1] + timedelta(days=1), periods=forecast_steps, freq='M')
        forecast_df = pd.DataFrame(forecast, index=future_dates, columns=['Predicción'])

        st.write(forecast_df)
        st.line_chart(forecast_df)


class app:
    def main(self):
        st.title("Análisis de Series Temporales")
        
        # Obtener el dataset desde la sesión
        df = st.session_state.get('data', None)
        if df is not None:
            series_temp = SeriesTemporales(df)

            # Menú de opciones
            option = st.selectbox("Selecciona una opción", ["Análisis de Datos", "Entrenamiento de Modelo", "Predicciones Futuras"])

            if option == "Análisis de Datos":
                series_temp.analizar_datos()
            elif option == "Entrenamiento de Modelo":
                model_fit, predictions = series_temp.entrenar_modelo()
            elif option == "Predicciones Futuras":
                model_fit = st.session_state.get('model_fit', None)
                if model_fit:
                    series_temp.predecir(model_fit)
                else:
                    st.warning("Primero debes entrenar el modelo.")
        else:
            st.warning("No hay datos cargados. Carga primero un dataset válido.")
