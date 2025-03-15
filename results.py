import streamlit as st
import pandas as pd
from CEvaluator import ModelEvaluator
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

class app:
    def run(self):
        st.markdown('<h3 class="custom-h3">Comparación de Modelos 🧮</h3>', unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Carga tu dataset CSV", type="csv")

        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.write("Vista previa del dataset cargado:")
            st.dataframe(df.head())

            # Selección de columnas objetivo y características
            target_column = st.selectbox("Selecciona la columna objetivo:", df.columns)
            X = df.drop(columns=[target_column])
            y = df[target_column]

            # División de datos
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Instancia del evaluador
            evaluator = ModelEvaluator(X_train, X_test, y_train, y_test)

            # Selección del método (genético o exhaustivo)
            method = st.selectbox("Selecciona método de búsqueda de hiperparámetros:",
                                  ["Genético (más rápido)", "Exhaustivo (más lento)"])

            if st.button("Ejecutar búsqueda de hiperparámetros"):
                with st.spinner(f"Ejecutando método {method}... Esto puede tardar varios minutos."):
                    if method == "Genético (más rápido)":
                        results = evaluator.genetic_search()
                    else:
                        results = evaluator.exhaustive_search()

                    st.success(f"¡La búsqueda {method} se completó exitosamente!")

                    # Mostrar resultados claramente
                    for model_name, result in results.items():
                        st.subheader(f"Resultados del modelo: {model_name}")
                        st.write("Mejores parámetros encontrados:")
                        st.json(result['best_params'])

                        predictions = result['estimator'].predict(X_test)
                        rmse = mean_squared_error(y_test, predictions, squared=False)
                        r2 = r2_score(y_test, predictions)

                        st.write(f"RMSE: {rmse:.4f}")
                        st.write(f"R²: {r2:.4f}")
