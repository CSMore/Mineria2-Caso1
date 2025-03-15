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
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.__df[col1], y=self.__df[col2])
        plt.title(f'Gráfico de Dispersión: {col1} vs {col2}')
        plt.xlabel(col1)
        plt.ylabel(col2)
        plt.grid()
        st.pyplot(plt)

    def plot_histogram(self, col):
        plt.figure(figsize=(10, 6))
        sns.histplot(self.__df[col], kde=True)
        plt.title(f'Histograma de {col}')
        plt.xlabel(col)
        plt.ylabel('Frecuencia')
        st.pyplot(plt)

    def plot_heatmap(self):
        num_df = self.__df.select_dtypes(include=['float64', 'int64'])
        if num_df.empty:
            return st.warning("No hay columnas numéricas para generar el mapa de calor.")

        plt.figure(figsize=(12, 10))
        sns.heatmap(num_df.corr(), cmap="crest", annot=True, linewidths=0.5, cbar=True)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        st.pyplot(plt)

    def plot_bar(self, col):
        plt.figure(figsize=(10, 6))
        sns.countplot(x=self.__df[col])
        plt.title(f'Gráfico de Barras: {col}')
        plt.xlabel(col)
        plt.ylabel('Frecuencia')
        st.pyplot(plt)

    def plot_violin(self, col):
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
        st.title("Exploración de Datos 🔍")

        if 'data' in st.session_state:
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

            st.subheader("Valores Faltantes")
            st.write(eda_instance.missing_values_info())

            st.subheader("Detección de Outliers")
            st.write(eda_instance.detect_outliers())

            columnas_numericas = eda_instance.get_df().select_dtypes(include=['float64', 'int64']).columns.tolist()

            tipo_grafico = st.selectbox("Selecciona un gráfico", ["Histograma", "Boxplot", "Barras", "Dispersión", "Línea", "Heatmap", "Pairplot"])

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
