import numpy as np
import pandas as pd
import logging
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
from logic.agrupamiento_lg import preprocess_for_clustering
import plotly.express as px

def pca_reduction(df, n_components=2):
    """
    Aplica PCA al DataFrame y retorna las componentes y el objeto PCA.
    """
    logging.info(f"Iniciando PCA con n_components={n_components}")
    df_encoded = preprocess_for_clustering(df)
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(df_encoded)
    logging.info("PCA completado")
    return components, pca

def tsne_reduction(df, n_components=2, perplexity=30, random_state=42):
    """
    Aplica TSNE al DataFrame y retorna las componentes.
    """
    logging.info(f"Iniciando TSNE con n_components={n_components}, perplexity={perplexity}, random_state={random_state}")
    df_encoded = preprocess_for_clustering(df)
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
    components = tsne.fit_transform(df_encoded)
    logging.info("TSNE completado")
    return components

def umap_reduction(df, n_components=2, n_neighbors=15, min_dist=0.1, random_state=42):
    """
    Aplica UMAP al DataFrame y retorna las componentes.
    """
    logging.info(f"Iniciando UMAP con n_components={n_components}, n_neighbors={n_neighbors}, min_dist={min_dist}, random_state={random_state}")
    df_encoded = preprocess_for_clustering(df)
    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    components = reducer.fit_transform(df_encoded)
    logging.info("UMAP completado")
    return components

def plot_components(components, algorithm_name):
    """
    Genera y retorna una figura interactiva con un scatter plot de las dos primeras componentes usando Plotly.
    """
    df_plot = pd.DataFrame(components, columns=['Componente 1', 'Componente 2'])
    fig = px.scatter(df_plot, x='Componente 1', y='Componente 2', title=f"Reducción de dimensiones con {algorithm_name}")
    return fig
