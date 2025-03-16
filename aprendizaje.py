import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from CEvaluator import ModelEvaluator
import logging

logging.basicConfig(filename="audit_log.txt", level=logging.INFO, format="%(asctime)s - %(message)s")

class ModelTrainer:
    def __init__(self, df, target_column, test_size):
        self.df = df.select_dtypes(include=['number']).dropna()
        self.X = self.df.drop(columns=[target_column])
        self.y = self.df[target_column]
        self.test_size = test_size
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=42
        )

    def train_models(self, population_size, generations, mutation_rate):
        models = {
            "Regresión Lineal": LinearRegression(),
            "Árbol de Decisión": DecisionTreeRegressor(),
            "Random Forest": RandomForestRegressor(),
            "Gradient Boosting": GradientBoostingRegressor(),
            "Modelo Evaluador": ModelEvaluator(
                self.X_train, self.X_test, self.y_train, self.y_test,
                population_size=population_size,
                generations=generations,
                mutation_rate=mutation_rate
            )
        }

        resultados = {}

        for nombre, modelo in models.items():
            if nombre != "Modelo Evaluador":
                modelo.fit(self.X_train, self.y_train)
                preds = modelo.predict(self.X_test)
            else:
                modelo.fit(self.X_train.values, self.y_train.values)
                preds = modelo.predict(self.X_test.values)

            mse = mean_squared_error(self.y_test, preds)
            resultados[nombre] = mse
            logging.info(f"{nombre} entrenado con MSE: {mse}")

        return resultados

class app:
    def main(self):
        st.title("Aprendizaje Supervisado con Benchmark")

        if 'data' in st.session_state:
            df = st.session_state['data']
            columnas_numericas = df.select_dtypes(include=['number']).columns.tolist()

            target_column = st.selectbox("Selecciona la columna objetivo:", columnas_numericas)

            st.subheader("División Entrenamiento/Test")
            train_percentage = st.slider("Porcentaje para entrenamiento:", min_value=50, max_value=90, value=80, step=5)
            test_size = 1 - (train_percentage / 100)

            st.subheader("Parámetros del Modelo Evaluador")
            population_size = st.slider("Tamaño de población", 10, 200, 50, 10)
            generations = st.slider("Número de generaciones", 5, 100, 10, 5)
            mutation_rate = st.slider("Tasa de mutación", 0.01, 0.2, 0.05, 0.01)

            # Inicializa el trainer
            trainer = ModelTrainer(df, target_column, test_size)

            # Crear espacio en sesión para guardar benchmarks
            if "benchmarks" not in st.session_state:
                st.session_state["benchmarks"] = pd.DataFrame(columns=[
                    "Entrenamiento (%)", "Población", "Generaciones", "Mutación", 
                    "Regresión Lineal", "Árbol Decisión", "Random Forest", 
                    "Gradient Boosting", "Evaluador (Profesor)"
                ])

            if st.button("Ejecutar Benchmark"):
                with st.spinner("Ejecutando entrenamiento..."):
                    resultados = trainer.train_models(population_size, generations, mutation_rate)

                nuevo_benchmark = {
                    "Entrenamiento (%)": train_percentage,
                    "Población": population_size,
                    "Generaciones": generations,
                    "Mutación": mutation_rate,
                    "Regresión Lineal": resultados["Regresión Lineal"],
                    "Árbol Decisión": resultados["Árbol de Decisión"],
                    "Random Forest": resultados["Random Forest"],
                    "Gradient Boosting": resultados["Gradient Boosting"],
                    "Evaluador": resultados["Modelo Evaluador (Profesor)"]
                }

                # Añadir resultados al historial
                st.session_state["benchmarks"] = pd.concat([
                    st.session_state["benchmarks"],
                    pd.DataFrame([nuevo_benchmark])
                ], ignore_index=True)

                st.success("¡Benchmark ejecutado con éxito!")

            # Mostrar resultados acumulados en Streamlit
            if not st.session_state["benchmarks"].empty:
                st.subheader("Resultados acumulados del Benchmark:")
                st.dataframe(st.session_state["benchmarks"])

                # Gráfico interactivo comparativo
                st.subheader("Comparación visual del Benchmark:")
                modelo_a_comparar = st.selectbox("Selecciona modelo para comparar resultados:", [
                    "Regresión Lineal", "Árbol Decisión", "Random Forest", 
                    "Gradient Boosting", "Evaluador (Profesor)"
                ])

                st.line_chart(st.session_state["benchmarks"][modelo_a_comparar])

            if st.button("Limpiar resultados acumulados"):
                st.session_state["benchmarks"] = pd.DataFrame(columns=[
                    "Entrenamiento (%)", "Población", "Generaciones", "Mutación", 
                    "Regresión Lineal", "Árbol Decisión", "Random Forest", 
                    "Gradient Boosting", "Evaluador (Profesor)"
                ])
                st.info("Benchmark reiniciado correctamente.")
        else:
            st.warning("Primero debes cargar y procesar el dataset en el Pipeline.")

