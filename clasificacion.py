import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from exploratorio import EDA  # Si el archivo EDA tiene una clase Exploratorio


class Clasificacion:
    def __init__(self, df):
        self.df = df
        self.eda = EDA(self.df)
        self.model = None
        self.X = None
        self.y = None

    def preprocess_data(self):
        """Realiza el preprocesamiento de los datos"""
        st.write("### Preprocesamiento de Datos")
        st.write("Dividiendo datos en características y etiquetas...")
        X = self.df.dropna(axis=1, how='any')  # Elimina columnas con valores nulos
        
        # Mostrar un selectbox para que el usuario seleccione la columna objetivo
        column_options = [col for col in X.columns if col != 'ID']  # Puedes excluir columnas que no sean relevantes
        target_column = st.selectbox("Selecciona la columna objetivo", column_options)
        
        # Asignar la columna seleccionada como objetivo
        y = X.pop(target_column)

        # Escalado de características
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        self.X = X_scaled
        self.y = y
        return X_scaled, y

    def evaluate_model(self, model, X, y, cv=5):
        """Evaluar el modelo con validación cruzada y métrica AUC"""
        st.write(f"Evaluando modelo {model.__class__.__name__}...")
        auc_scores = cross_val_score(model, X, y, cv=StratifiedKFold(cv), scoring='roc_auc')
        mean_auc = auc_scores.mean()
        st.write(f"Promedio AUC: {mean_auc:.4f}")
        return mean_auc

    def train_model(self, model):
        """Entrena el modelo y evalúa su desempeño"""
        st.write(f"Entrenando el modelo {model.__class__.__name__}...")
        model.fit(self.X, self.y)
        y_pred = model.predict(self.X)
        accuracy = accuracy_score(self.y, y_pred)
        auc = roc_auc_score(self.y, model.predict_proba(self.X)[:, 1])
        
        st.write(f"Accuracy: {accuracy:.4f}")
        st.write(f"AUC: {auc:.4f}")
        
        # Mostrar la matriz de confusión
        cm = confusion_matrix(self.y, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)

    def display_classification_results(self, model):
        """Muestra los resultados de la clasificación"""
        st.write(f"### Resultados del Modelo {model.__class__.__name__}")
        self.train_model(model)

    def compare_models(self):
        """Compara el rendimiento de diferentes modelos"""
        models = {
            "Logistic Regression": LogisticRegression(),
            "Random Forest": RandomForestClassifier(),
            "SVM": SVC(probability=True),
            "Decision Tree": DecisionTreeClassifier(),
        }
        
        best_auc = 0
        best_model = None
        
        for name, model in models.items():
            st.write(f"Evaluando el modelo {name}")
            auc_score = self.evaluate_model(model, self.X, self.y)
            if auc_score > best_auc:
                best_auc = auc_score
                best_model = model
        
        st.write(f"El mejor modelo es: {best_model.__class__.__name__} con AUC de {best_auc:.4f}")
        self.display_classification_results(best_model)

    def run(self):
        """Ejecuta el flujo de clasificación"""
        if not self.df.empty:
            st.markdown('<h2>Clasificación de Datos</h2>', unsafe_allow_html=True)
            self.preprocess_data()

            # Opción para elegir el modelo
            model_option = st.selectbox("Selecciona el modelo de clasificación", ["Benchmarking", "Entrenar Modelo"])
            
            if model_option == "Benchmarking":
                self.compare_models()
            else:
                model = st.selectbox("Selecciona el modelo para entrenar", ["Logistic Regression", "Random Forest", "SVM", "Decision Tree"])
                if model == "Logistic Regression":
                    self.display_classification_results(LogisticRegression())
                elif model == "Random Forest":
                    self.display_classification_results(RandomForestClassifier())
                elif model == "SVM":
                    self.display_classification_results(SVC(probability=True))
                elif model == "Decision Tree":
                    self.display_classification_results(DecisionTreeClassifier())

        else:
            st.warning("No hay datos cargados. Carga un dataset primero.")


class app:
    def main(self):
        st.markdown('<h2>Clasificación de Datos</h2>', unsafe_allow_html=True)
        
        # Asumiendo que los datos están en st.session_state['data']
        if 'data' in st.session_state and st.session_state['data'] is not None:
            clasificador = Clasificacion(df=st.session_state['data'])
            clasificador.run()
        else:
            st.warning("No hay datos cargados. Carga un dataset en el Pipeline primero.")
