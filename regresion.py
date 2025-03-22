import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, Ridge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import math

class Regresion:
    def __init__(self, df):
        self.df = df
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def preprocess_data(self):
        st.write("### Selección de columna objetivo")
        column_options = [col for col in self.df.columns if self.df[col].dtype != 'object']
    
        if not column_options:
            st.error("No hay columnas numéricas disponibles para aplicar regresión.")
            return False
    
        target_column = st.selectbox("Selecciona la variable objetivo (y):", column_options)

        if target_column not in self.df.columns:
            st.error("Columna objetivo no válida.")
            return False

        self.y = self.df[target_column]
        self.X = self.df.drop(columns=[target_column])
        self.X = self.X.select_dtypes(include=['number'])  # Mantener solo columnas numéricas


        if self.X.empty or self.y.empty:
            st.error("Los datos seleccionados no son válidos para regresión.")
            return False

        st.write("### División de datos")
        test_size = st.slider("Porcentaje para test:", 10, 50, 20)
    
        try:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size/100, random_state=42
            )
            st.success("Datos divididos exitosamente")
            return True
        except Exception as e:
            st.error(f"Error al dividir los datos: {e}")
            return False

        

    def train_model(self, model):
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)

        rmse = math.sqrt(mean_squared_error(self.y_test, predictions))
        mae = mean_absolute_error(self.y_test, predictions)
        er = np.sum(np.abs(self.y_test - predictions)) / np.sum(np.abs(self.y_test))

        st.write("### Métricas del modelo")
        st.metric("RMSE", f"{rmse:.4f}")
        st.metric("MAE", f"{mae:.4f}")
        st.metric("Error Relativo (ER)", f"{er:.4f}")

        fig, ax = plt.subplots()
        sns.scatterplot(x=self.y_test, y=predictions, ax=ax)
        ax.set_xlabel("Valores Reales")
        ax.set_ylabel("Predicciones")
        ax.set_title("Predicción vs Real")
        st.pyplot(fig)

    def run(self):
        if self.df is not None and not self.df.empty:
            st.markdown("<h2>Regresión Supervisada</h2>", unsafe_allow_html=True)
            if not self.preprocess_data():
                return

            st.write("### Selección del modelo")
            model_option = st.selectbox("Selecciona el modelo de regresión:", [
                "Regresión Lineal Simple (LinearRegression)",
                "Lasso", "LassoCV", "Ridge",
                "SVR", "Árbol de Decisión", "Random Forest", "Gradient Boosting"
            ])

            if st.button("Entrenar modelo"):
                if model_option == "Regresión Lineal Simple (LinearRegression)":
                    model = LinearRegression()
                elif model_option == "Lasso":
                    model = Lasso(alpha=0.1)
                elif model_option == "LassoCV":
                    model = LassoCV(alphas=np.logspace(-6, 6, 200), cv=10)
                elif model_option == "Ridge":
                    model = Ridge(alpha=1.0)
                elif model_option == "SVR":
                    model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
                elif model_option == "Árbol de Decisión":
                    model = DecisionTreeRegressor(max_depth=5, random_state=42)
                elif model_option == "Random Forest":
                    model = RandomForestRegressor(max_depth=5, random_state=42)
                elif model_option == "Gradient Boosting":
                    model = GradientBoostingRegressor(
                        n_estimators=500, max_depth=4,
                        min_samples_split=5, learning_rate=0.01,
                        loss='squared_error')
                else:
                    st.warning("Modelo no reconocido")
                    return

                self.train_model(model)
        else:
            st.warning("No hay datos cargados. Por favor, carga un dataset válido.")


class app:
    def main(self):
        st.markdown("<h2>Regresión</h2>", unsafe_allow_html=True)

        if 'data' in st.session_state and st.session_state['data'] is not None:
            regresion_app = Regresion(df=st.session_state['data'])
            regresion_app.run()
        else:
            st.warning("No hay datos cargados. Carga un dataset en el Pipeline primero.")
