import streamlit as st
import pandas as pd
import logging
from logic.reduccion_lg import pca_reduction, tsne_reduction, umap_reduction, plot_components

class DimReductionUI:
    def __init__(self):
        # Verifica que exista un dataset cargado en la sesión
        if "data" not in st.session_state or st.session_state["data"] is None:
            st.error("No se encontró ningún dataset. Por favor, carga un dataset desde Pipeline.")
            logging.error("Intento de reducción de dimensiones sin dataset cargado en sesión.")
            self.df = None
        else:
            self.df = st.session_state["data"]
            logging.info("Dataset cargado para reducción de dimensiones.")

    def display_ui(self):
        st.subheader("Vista previa del dataset")
        st.dataframe(self.df.head())

        st.markdown("### Selección de Algoritmo y Configuración de Parámetros")
        algoritmo = st.selectbox("Elige el algoritmo de reducción de dimensiones:", 
                                 ["PCA", "TSNE", "UMAP", "Comparar los 3"])

        # Parámetro común: número de componentes (para visualización se recomienda 2)
        n_components = st.number_input("Número de componentes:", min_value=2, max_value=10, value=2, step=1)

        # Parámetros específicos para TSNE
        if algoritmo == "TSNE" or algoritmo == "Comparar los 3":
            st.markdown("#### Parámetros para TSNE")
            perplexity = st.number_input("Perplexity (TSNE):", min_value=5, max_value=50, value=30, step=1)
        else:
            perplexity = 30

        # Parámetros específicos para UMAP
        if algoritmo == "UMAP" or algoritmo == "Comparar los 3":
            st.markdown("#### Parámetros para UMAP")
            n_neighbors = st.number_input("n_neighbors (UMAP):", min_value=5, max_value=50, value=15, step=1)
            min_dist = st.number_input("min_dist (UMAP):", min_value=0.0, max_value=1.0, value=0.1, step=0.1)
        else:
            n_neighbors = 15
            min_dist = 0.1

        random_state = st.number_input("Semilla aleatoria:", value=42, step=1)

        if st.button("Ejecutar Reducción de Dimensiones"):
            logging.info(f"Ejecutar reducción de dimensiones: {algoritmo} con n_components={n_components}")
            if algoritmo == "PCA":
                components, _ = pca_reduction(self.df, n_components=n_components)
                fig = plot_components(components, "PCA")
                st.plotly_chart(fig)
            elif algoritmo == "TSNE":
                components = tsne_reduction(self.df, n_components=n_components, perplexity=perplexity, random_state=random_state)
                fig = plot_components(components, "TSNE")
                st.plotly_chart(fig)
            elif algoritmo == "UMAP":
                components = umap_reduction(self.df, n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
                fig = plot_components(components, "UMAP")
                st.plotly_chart(fig)
            elif algoritmo == "Comparar los 3":
                st.markdown("### Resultados de la reducción de dimensiones")
                # PCA
                components_pca, _ = pca_reduction(self.df, n_components=n_components)
                fig_pca = plot_components(components_pca, "PCA")
                st.plotly_chart(fig_pca)
                # TSNE
                components_tsne = tsne_reduction(self.df, n_components=n_components, perplexity=perplexity, random_state=random_state)
                fig_tsne = plot_components(components_tsne, "TSNE")
                st.plotly_chart(fig_tsne)
                # UMAP
                components_umap = umap_reduction(self.df, n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
                fig_umap = plot_components(components_umap, "UMAP")
                st.plotly_chart(fig_umap)
            else:
                st.error("Algoritmo no reconocido.")
                logging.error("Algoritmo no reconocido en la interfaz de reducción de dimensiones.")

class app:
    def main(self):
        st.title("Reducción de Dimensiones")
        ui = DimReductionUI()
        if ui.df is not None:
            ui.display_ui()
        st.balloons()

if __name__ == "__page__":
    app().main()
