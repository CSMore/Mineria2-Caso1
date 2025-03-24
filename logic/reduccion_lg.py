import logging
import pandas as pd

class reduccion():
    def __init__(self, df):
        self.__df = df

    @property
    def df(self):
        return self.__df

    @df.setter
    def df(self, p_df):
        self.__df = p_df

    def ACP(self, n_componentes):
        p_acp = ACPBasico(self.__df, n_componentes)
        self.__ploteoGraficosACP(p_acp, 1)
        self.__ploteoGraficosACP(p_acp, 2)
        self.__ploteoGraficosACP(p_acp, 3)

    def __ploteoGraficosACP(self, p_acp, tipo):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6), dpi=200)
        if tipo == 1:
            p_acp.plot_plano_principal()
        elif tipo == 2:
            p_acp.plot_circulo()
        elif tipo == 3:
            p_acp.plot_sobreposicion()
        ax.grid(False)
        plt.show()

    def HAC(self):
        p_hac = self.__df
        ward_res = ward(self.__df)
        average_res = average(self.__df)
        single_res = single(self.__df)
        complete_res = complete(self.__df)
        self.__ploteoGraficosHAC(p_hac, ward_res, 1)
        self.__ploteoGraficosHAC(p_hac, average_res, 2)
        self.__ploteoGraficosHAC(p_hac, single_res, 3)
        self.__ploteoGraficosHAC(p_hac, complete_res, 4)
        self.__clusterHAC(1)
        self.__clusterHAC(2)
        self.__clusterHAC(3)

    def __ploteoGraficosHAC(self, p_hac, res, tipo):
        fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=200)
        if tipo == 1:
            dendrogram(res, labels=self.__df.index.tolist(), ax=ax)
            print(f"Agregación de Ward:")
        elif tipo == 2:
            dendrogram(res, labels=self.__df.index.tolist(), ax=ax)
            print(f"Salto promedio:")
        elif tipo == 3:
            dendrogram(res, labels=self.__df.index.tolist(), ax=ax)
            print(f"Salto mínimo:")
        elif tipo == 4:
            dendrogram(res, labels=self.__df.index.tolist(), ax=ax)
            print(f"Salto máximo:")
        ax.grid(False)
        plt.show()

    def __clusterHAC(self, tipo):
        grupos = fcluster(linkage(self.__df, method='ward', metric='euclidean'), 3, criterion='maxclust')
        grupos = grupos - 1
        centros = np.array(pd.concat([AnalisisDatosExploratorio.centroide(0, self.__df, grupos),
                                      AnalisisDatosExploratorio.centroide(1, self.__df, grupos),
                                      AnalisisDatosExploratorio.centroide(2, self.__df, grupos)]))
        if tipo == 1:
            AnalisisDatosExploratorio.bar_plot(centros, self.__df.columns, scale=True)
        elif tipo == 2:
            AnalisisDatosExploratorio.bar_plot_detail(centros, self.__df.columns)
        elif tipo == 3:
            AnalisisDatosExploratorio.radar_plot(centros, self.__df.columns)
        plt.show()

    def Kmeans(self):
        self.__ploteoGraficosKMEDIAS(1)
        self.__ploteoGraficosKMEDIAS(2)

    def __ploteoGraficosKMEDIAS(self, tipo):
        if tipo == 1:
            self.__ploteoKmedias()
        elif tipo == 2:
            self.__ploteoKmedoids()

    def __ploteoKmedias(self):
        kmedias = KMeans(n_clusters=3, max_iter=500, n_init=150)
        kmedias.fit(self.__df)
        pca = PCA(n_components=2)
        componentes = pca.fit_transform(self.__df)
        fig, ax = plt.subplots(1, 1, figsize=(15, 8), dpi=200)
        colores = ['red', 'green', 'blue']
        colores_puntos = [colores[label] for label in kmedias.predict(self.__df)]
        ax.scatter(componentes[:, 0], componentes[:, 1], c=colores_puntos)
        ax.set_xlabel('componente 1')
        ax.set_ylabel('componente 2')
        ax.set_title('3 Cluster K-Medias')
        ax.grid(False)
        plt.show()

        centros = np.array(kmedias.cluster_centers_)
        AnalisisDatosExploratorio.bar_plot(centros, self.__df.columns)
        AnalisisDatosExploratorio.bar_plot_detail(centros, self.__df.columns)
        AnalisisDatosExploratorio.radar_plot(centros, self.__df.columns)
        plt.show()

    def __ploteoKmedoids(self):
        kmedoids = KMedoids(n_clusters=3, max_iter=500, metric='cityblock')
        kmedoids.fit(self.__df)
        pca = PCA(n_components=2)
        componentes = pca.fit_transform(self.__df)
        fig, ax = plt.subplots(1, 1, figsize=(15, 8), dpi=200)
        colores = ['red', 'green', 'blue']
        colores_puntos = [colores[label] for label in kmedoids.predict(self.__df)]
        ax.scatter(componentes[:, 0], componentes[:, 1], c=colores_puntos)
        ax.set_xlabel('componente 1')
        ax.set_ylabel('componente 2')
        ax.set_title('3 Cluster K-Medoids')
        ax.grid(False)
        plt.show()

        centros = np.array(kmedoids.cluster_centers_)
        AnalisisDatosExploratorio.bar_plot(centros, self.__df.columns)
        AnalisisDatosExploratorio.bar_plot_detail(centros, self.__df.columns)
        AnalisisDatosExploratorio.radar_plot(centros, self.__df.columns)
        plt.show()

    def TSNE(self, n_componentes):
        tsne = TSNE(n_componentes)
        componentes = tsne.fit_transform(self.__df)
        fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=200)
        ax.scatter(componentes[:, 0], componentes[:, 1])
        ax.set_xlabel('Componente 1')
        ax.set_ylabel('Componente 2')
        ax.set_title('T-SNE')
        ax.grid(False)
        plt.show()

    def UMAP(self, n_componentes, n_neighbors):
        modelo_umap = um.UMAP(n_componentes, n_neighbors)
        componentes = modelo_umap.fit_transform(self.__df)
        fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=200)
        ax.scatter(componentes[:, 0], componentes[:, 1])
        ax.set_xlabel('Componente 1')
        ax.set_ylabel('Componente 2')
        ax.set_title('UMAP')
        ax.grid(False)
        plt.show()

    def compare_algorithms(self):
        """Compare the performance of different clustering algorithms."""
        results = {}

        # KMeans
        kmeans = KMeans(n_clusters=3, max_iter=500, n_init=150)
        kmeans_labels = kmeans.fit_predict(self.__df)
        results['KMeans'] = {
            'Silhouette Score': silhouette_score(self.__df, kmeans_labels),
            'Davies-Bouldin Index': davies_bouldin_score(self.__df, kmeans_labels)
        }

        # HAC (Ward)
        hac_labels = fcluster(linkage(self.__df, method='ward', metric='euclidean'), 3, criterion='maxclust')
        results['HAC_Ward'] = {
            'Silhouette Score': silhouette_score(self.__df, hac_labels),
            'Davies-Bouldin Index': davies_bouldin_score(self.__df, hac_labels)
        }

        # Gaussian Mixture Model (GMM)
        gmm = GaussianMixture(n_components=3, covariance_type='full')
        gmm_labels = gmm.fit_predict(self.__df)
        results['GMM'] = {
            'Silhouette Score': silhouette_score(self.__df, gmm_labels),
            'Davies-Bouldin Index': davies_bouldin_score(self.__df, gmm_labels)
        }

        # Print the results
        for algorithm, metrics in results.items():
            print(f"\n{algorithm} Performance Metrics:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")