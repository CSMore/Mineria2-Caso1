# agrupamiento_lg.py
import numpy as np
import pandas as pd
import logging
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import LabelEncoder

def preprocess_for_clustering(df):
    """
    Convierte todas las columnas categóricas del DataFrame a valores numéricos usando LabelEncoder.
    """
    df_encoded = df.copy()
    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'object':
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
    return df_encoded

def kmeans_clustering(df, n_clusters=3, max_iter=500, n_init=10, random_state=42):
    """
    Aplica KMeans al DataFrame y retorna las etiquetas, el puntaje de Silhouette y el Davies-Bouldin.
    """
    logging.info(f"Iniciando KMeans con n_clusters={n_clusters}, max_iter={max_iter}, n_init={n_init}, random_state={random_state}")
    df_encoded = preprocess_for_clustering(df)
    kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter, n_init=n_init, random_state=random_state)
    labels = kmeans.fit_predict(df_encoded)
    sil_score = silhouette_score(df_encoded, labels)
    db_score = davies_bouldin_score(df_encoded, labels)
    logging.info(f"KMeans completado: Silhouette Score={sil_score:.4f}, Davies-Bouldin Index={db_score:.4f}")
    return labels, sil_score, db_score

def hac_clustering(df, n_clusters=3, method='ward'):
    """
    Aplica el método de Agrupamiento Jerárquico (HAC) usando el método especificado
    y retorna las etiquetas y métricas de evaluación.
    """
    logging.info(f"Iniciando HAC con n_clusters={n_clusters} y método={method}")
    df_encoded = preprocess_for_clustering(df)
    Z = linkage(df_encoded, method=method, metric='euclidean')
    labels = fcluster(Z, t=n_clusters, criterion='maxclust')
    labels = labels - 1  # Ajuste para que las etiquetas inicien en 0
    sil_score = silhouette_score(df_encoded, labels)
    db_score = davies_bouldin_score(df_encoded, labels)
    logging.info(f"HAC ({method}) completado: Silhouette Score={sil_score:.4f}, Davies-Bouldin Index={db_score:.4f}")
    return labels, sil_score, db_score

def gmm_clustering(df, n_components=3, covariance_type='full', random_state=42):
    """
    Aplica Gaussian Mixture Model (GMM) al DataFrame y retorna las etiquetas y métricas de evaluación.
    """
    logging.info(f"Iniciando GMM con n_components={n_components}, covariance_type={covariance_type}, random_state={random_state}")
    df_encoded = preprocess_for_clustering(df)
    gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=random_state)
    labels = gmm.fit_predict(df_encoded)
    sil_score = silhouette_score(df_encoded, labels)
    db_score = davies_bouldin_score(df_encoded, labels)
    logging.info(f"GMM completado: Silhouette Score={sil_score:.4f}, Davies-Bouldin Index={db_score:.4f}")
    return labels, sil_score, db_score

def compare_clustering_algorithms(df, n_clusters=3, max_iter=500, n_init=10, 
                                  hac_method='ward', covariance_type='full', random_state=42):
    """
    Compara KMeans, HAC y GMM utilizando métricas de evaluación.
    Retorna un diccionario con los resultados de cada algoritmo.
    """
    logging.info("Iniciando comparación de algoritmos de clustering")
    results = {}

    # KMeans
    k_labels, k_sil, k_db = kmeans_clustering(
        df, n_clusters=n_clusters, max_iter=max_iter, n_init=n_init, random_state=random_state
    )
    results['KMeans'] = {'Silhouette Score': k_sil, 'Davies-Bouldin Index': k_db}

    # HAC
    h_labels, h_sil, h_db = hac_clustering(df, n_clusters=n_clusters, method=hac_method)
    results[f'HAC ({hac_method.capitalize()})'] = {'Silhouette Score': h_sil, 'Davies-Bouldin Index': h_db}

    # GMM
    g_labels, g_sil, g_db = gmm_clustering(df, n_components=n_clusters, covariance_type=covariance_type, random_state=random_state)
    results['GMM'] = {'Silhouette Score': g_sil, 'Davies-Bouldin Index': g_db}

    logging.info("Comparación completada")
    return results
