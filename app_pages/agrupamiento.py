# agrupamiento.py
import streamlit as st
import pandas as pd
import logging
from logic.agrupamiento_lg import kmeans_clustering, hac_clustering, gmm_clustering, compare_clustering_algorithms

class ClusteringUI:
    def __init__(self):
        # Verifica que exista el dataset cargado en sesión
        if "data" not in st.session_state or st.session_state["data"] is None:
            st.error("No se encontró ningún dataset. Por favor, carga un dataset desde Pipeline.")
            logging.error("Intento de clustering sin dataset cargado en sesión.")
            self.df = None
        else:
            self.df = st.session_state["data"]
            logging.info("Dataset cargado para clustering.")

    def display_ui(self):
        st.subheader("Vista previa del dataset")
        st.dataframe(self.df.head())

        st.markdown("### Selección de Algoritmo y Configuración de Parámetros")
        algoritmo = st.selectbox("Elige el algoritmo de agrupamiento:", 
                                 ["KMeans", "HAC", "Gaussian Mixture Model", "Comparar los 3"])

        # Parámetros comunes
        n_clusters = st.number_input("Número de clusters/componentes:", min_value=2, max_value=10, value=3, step=1)
        random_state = st.number_input("Semilla aleatoria:", value=42, step=1)

        # Parámetros específicos para KMeans
        if algoritmo == "KMeans" or algoritmo == "Comparar los 3":
            st.markdown("#### Parámetros para KMeans")
            max_iter = st.number_input("Máximo de iteraciones (KMeans):", min_value=100, max_value=1000, value=500, step=50)
            n_init = st.number_input("Número de inicializaciones (KMeans):", min_value=1, max_value=20, value=10, step=1)
        else:
            max_iter = 500
            n_init = 10

        # Parámetros específicos para HAC
        if algoritmo == "HAC" or algoritmo == "Comparar los 3":
            hac_method = st.selectbox("Método de enlace para HAC:", options=["ward", "single", "complete", "average"], index=0)
        else:
            hac_method = "ward"

        # Parámetros específicos para GMM
        if algoritmo == "Gaussian Mixture Model" or algoritmo == "Comparar los 3":
            covariance_type = st.selectbox("Tipo de covarianza para GMM:", options=["full", "tied", "diag", "spherical"], index=0)
        else:
            covariance_type = "full"

        if st.button("Ejecutar Agrupamiento"):
            logging.info(f"Ejecutar agrupamiento seleccionado: {algoritmo} con n_clusters={n_clusters}, random_state={random_state}")
            if algoritmo == "KMeans":
                labels, sil, db = kmeans_clustering(self.df, n_clusters=n_clusters, max_iter=max_iter, n_init=n_init, random_state=random_state)
                st.success(f"KMeans:\n - Silhouette Score: {sil:.4f}\n - Davies-Bouldin Index: {db:.4f}")
                logging.info("KMeans ejecutado correctamente.")
            elif algoritmo == "HAC":
                labels, sil, db = hac_clustering(self.df, n_clusters=n_clusters, method=hac_method)
                st.success(f"HAC ({hac_method.capitalize()}):\n - Silhouette Score: {sil:.4f}\n - Davies-Bouldin Index: {db:.4f}")
                logging.info("HAC ejecutado correctamente.")
            elif algoritmo == "Gaussian Mixture Model":
                labels, sil, db = gmm_clustering(self.df, n_components=n_clusters, covariance_type=covariance_type, random_state=random_state)
                st.success(f"GMM:\n - Silhouette Score: {sil:.4f}\n - Davies-Bouldin Index: {db:.4f}")
                logging.info("GMM ejecutado correctamente.")
            elif algoritmo == "Comparar los 3":
                results = compare_clustering_algorithms(
                    self.df, n_clusters=n_clusters, max_iter=max_iter, n_init=n_init, 
                    hac_method=hac_method, covariance_type=covariance_type, random_state=random_state
                )
                st.markdown("### Comparación de Algoritmos")
                for algo, metrics in results.items():
                    st.write(f"**{algo}**:")
                    st.write(f"- Silhouette Score: {metrics['Silhouette Score']:.4f}")
                    st.write(f"- Davies-Bouldin Index: {metrics['Davies-Bouldin Index']:.4f}")
                logging.info("Comparación de algoritmos ejecutada correctamente.")
            else:
                st.error("Algoritmo no reconocido.")
                logging.error("Algoritmo no reconocido en la interfaz de clustering.")

class app:
    def main(self):
        st.title("Agrupamiento (Clustering)")
        ui = ClusteringUI()
        if ui.df is not None:
            ui.display_ui()
        st.balloons()

if __name__ == "__page__":
    app().main()
