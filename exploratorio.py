import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

class EDA:
    def __init__(self, df=None):
        if df is not None:
            self.__df = df.convert_dtypes()
        else:
            self.__df = pd.DataFrame()

    def validate_column(self, col):
        if col not in self.__df.columns:
            st.error(f"La columna '{col}' no existe en el DataFrame.")
            return False

    def missing_values_info(self):
        if not self.__df.empty:
            return pd.DataFrame(self.__df.isnull().sum(), columns=["Cantidad de valores faltantes"])
        return "No se cargaron los datos :("

    def head_df(self, n=5):
        if not self.__df.empty:
            return self.__df.head(n).astype(str)
        return "No se cargaron los datos :("

    def tail_df(self, n=5):
        if not self.__df.empty:
            return self.__df.tail(n).astype(str)
        return "No se cargaron los datos :("

    def check_data_types(self):
        return self.__df.dtypes.astype(str)
    
    def summary_statistics(self):
        """Genera un resumen estadístico de las columnas numéricas."""
        if not self.__df.empty:
            return self.__df.describe().transpose()  # Transpone para mejor legibilidad
        else:
            return "No hay datos cargados para generar el resumen estadístico."


    def drop_irrelevant_columns(self, columns):
        self.__df.drop(columns=columns, inplace=True)

    def drop_missing_values(self):
        self.__df.dropna(inplace=True)

    def detect_outliers(self):
        num_df = self.__df.select_dtypes(include=['float64', 'int64'])
        if num_df.empty:
            return "No hay columnas numéricas en el DataFrame."

        Q1 = num_df.quantile(0.25)
        Q3 = num_df.quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((num_df < (Q1 - 1.5 * IQR)) | (num_df > (Q3 + 1.5 * IQR))).sum()
        Dicc_outliers = {col: outliers[col] for col in num_df.columns if outliers[col] > 0}

        return Dicc_outliers if Dicc_outliers else "No se detectaron valores atípicos en las columnas numéricas."

    def plot_scatter(self, col1, col2):
        if col1 not in self.__df.columns or col2 not in self.__df.columns:
            return st.error("Selecciona columnas válidas para el gráfico de dispersión.")
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.__df[col1], y=self.__df[col2])
        plt.title(f'Gráfico de Dispersión: {col1} vs {col2}')
        plt.xlabel(col1)
        plt.ylabel(col2)
        plt.grid()
        st.pyplot(plt)

    def plot_histogram(self, col):
        if col not in self.__df.columns:
            return st.error("Selecciona una columna válida para el histograma.")
        plt.figure(figsize=(10, 6))
        sns.histplot(self.__df[col], kde=True)
        plt.title(f'Histograma de {col}')
        plt.xlabel(col)
        plt.ylabel('Frecuencia')
        st.pyplot(plt)

    def plot_heatmap(self, corr_method='pearson'):
        num_df = self.__df.select_dtypes(include=['float64', 'int64'])
        if num_df.empty:
            return st.warning("No hay columnas numéricas para generar el mapa de calor.")

        corr_matrix = num_df.corr(method=corr_method)  # Matriz de correlación con el método seleccionado

        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, cmap="crest", annot=True, linewidths=0.5, cbar=True)  # Aquí solo pasamos corr_matrix
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        st.pyplot(plt)


    def plot_bar(self, col):
        if self.validate_column(col):
            return
        if self.__df[col].nunique() < 2:
                return st.warning("No hay suficientes categorías para generar un gráfico de barras.")
        plt.figure(figsize=(10, 6))
        sns.countplot(x=self.__df[col])
        plt.title(f'Gráfico de Barras: {col}')
        plt.xlabel(col)
        plt.ylabel('Frecuencia')
        st.pyplot(plt)

    def plot_violin(self, col):
        if self.validate_column(col):
            if self.__df[col].nunique() < 2:
                return st.warning("No hay suficientes valores distintos para generar un gráfico de violín.")
            plt.figure(figsize=(10, 6))
            sns.violinplot(x=self.__df[col])
            plt.title(f'Gráfico de Violín: {col}')
            plt.xlabel(col)
            st.pyplot(plt)

    def plot_line(self, col):
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=self.__df, x=self.__df.index, y=self.__df[col])
        plt.title(f'Gráfico de Líneas: {col}')
        plt.xlabel('Índice')
        plt.ylabel(col)
        st.pyplot(plt)

    def plot_pairplot(self):
        num_df = self.__df.select_dtypes(include=['float64', 'int64'])
        if num_df.empty:
            return st.warning("No hay columnas numéricas para generar el pairplot.")

        sns.pairplot(num_df)
        plt.suptitle("Pairplot de Variables Numéricas", y=1.02)
        st.pyplot(plt)

    def plot_boxplot(self, col):
        
        if col not in self.__df.columns:
            return st.error("Selecciona una columna válida para el boxplot.")
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=self.__df[col])
        plt.title(f'Boxplot de {col}')
        plt.xlabel(col)
        st.pyplot(plt)

    def __str__(self):
        return f"Clase EDA - DataFrame de la forma: {self.__df.shape}"

    def get_df(self):
        return self.__df.copy()

class app:
    def main(self):
        st.markdown('<h2>Exploración de Datos </h2>', unsafe_allow_html=True)

        if 'data' in st.session_state and st.session_state['data'] is not None:
            eda_instance = EDA(df=st.session_state['data'])

            st.subheader("Vista previa de datos cargados")
            st.dataframe(eda_instance.head_df())

            tab1, tab2, tab3 = st.tabs(["Top Rows", "Last Rows", "Tipos de Datos"])

            with tab1:
                st.write(eda_instance.head_df())

            with tab2:
                st.write(eda_instance.tail_df())

            with tab3:
                st.write(eda_instance.check_data_types())

            st.subheader("Resumen Estadístico")
            st.dataframe(eda_instance.summary_statistics())

            st.subheader("Normalización de Datos")
            norm_method = st.selectbox("Método de normalización", ["Ninguno", "MinMaxScaler", "StandardScaler"])
            if norm_method != "Ninguno":
                eda_instance.normalize_data(norm_method)
                st.success(f"Datos normalizados con {norm_method}") 
            
            st.subheader("Valores Faltantes")
            st.write(eda_instance.missing_values_info())

            st.subheader("Detección de Outliers")
            st.write(eda_instance.detect_outliers())
            
            st.subheader("Matriz de Correlación")
            corr_type = st.radio("Método de correlación:", ["pearson", "spearman", "kendall"], index=0)
            eda_instance.plot_heatmap(corr_method=corr_type)

            st.subheader("Visualización de Datos")
            columnas_numericas = eda_instance.get_df().select_dtypes(include=['float64', 'int64']).columns.tolist()
            tipo_grafico = st.selectbox("Selecciona un gráfico", ["Histograma", "Boxplot", "Barras", "Dispersión", "Línea", "Heatmap", "Pairplot"])

                      
            if not columnas_numericas:
                st.warning("No hay columnas numéricas disponibles en el dataset.")
                return  # Detener ejecución si no hay datos válidos
            
            if tipo_grafico == "Histograma":
                col = st.selectbox("Columna numérica", columnas_numericas)
                eda_instance.plot_histogram(col)

            elif tipo_grafico == "Boxplot":
                col = st.selectbox("Columna numérica para Boxplot", columnas_numericas)
                eda_instance.plot_boxplot(col)

            elif tipo_grafico == "Barras":
                col = st.selectbox("Columna para gráfico de barras", eda_instance.get_df().columns)
                eda_instance.plot_bar(col)

            elif tipo_grafico == "Dispersión":
                cols = st.multiselect("Dos columnas numéricas", columnas_numericas, max_selections=2)
                if len(cols) == 2:
                    eda_instance.plot_scatter(cols[0], cols[1])
                else:
                    st.warning("Selecciona exactamente dos columnas numéricas.")

            elif tipo_grafico == "Heatmap":
                eda_instance.plot_heatmap()

            elif tipo_grafico == "Pairplot":
                eda_instance.plot_pairplot()

            elif tipo_grafico == "Línea":
                col = st.selectbox("Columna numérica para gráfico de líneas", columnas_numericas)
                eda_instance.plot_line(col)
               
             # Agregar botón para limpiar los datos
            if st.button("Reiniciar Datos"):
                st.session_state['data'] = None
                st.rerun()  # Recargar la app

        else:
            st.warning("No hay datos cargados. Carga un dataset en el Pipeline primero.")
    
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                

