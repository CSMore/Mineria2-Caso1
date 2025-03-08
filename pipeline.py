import streamlit as st
import json
import pandas as pd
import numpy as np
import seaborn as sns
import logging

#Configuración de login para auditorias
logging.basicConfig(filename="audit_log.txt", level=logging.INFO, format="%(asctime)s - %(message)s")

class eda:
     def __init__(self):
          self.df = None

     def read_dataset(self, path):
            """
            Cargar el dataset e identificar el separador.
            """
            separa = None
            try:
                if path.startswith("http"):
                    self.df = pd.read_csv(path, sep=";", decimal=".")
                else:
                    with open(path, 'r', encoding='utf-8') as file:
                        sample = file.read(800)  # Leer una muestra del archivo
                        st.write("Muestra del archivo:", sample)
                        if not sample.strip():
                            raise ValueError("El archivo está vacío o no contiene suficientes datos para detectar el separador.")

                        dialect = csv.Sniffer().sniff(sample)  
                        separa = dialect.delimiter
                        st.write("Separador detectado:", separa)

                    self.df = pd.read_csv(path, sep=separa, decimal=".",header=0, index_col=None)
            
                logging.info("Datos cargados con éxito. Separador detectado: '%s'" % separa)

            except FileNotFoundError:
                logging.error(f"No se encontró el archivo en la ruta: {path}")
                raise FileNotFoundError(f"No se encontró el archivo en la ruta: {path}")

            except UnicodeDecodeError:
                logging.error("Error de codificación: intenta usar una codificación distinta como 'latin1'.")
                raise UnicodeDecodeError("Error de codificación. Verifica el archivo y la codificación.")   
            except Exception as e:
                logging.error(f"Error al cargar el archivo: {path}: {str(e)}")
                raise



class app:
    def main(self):
        st.markdown('<h3 class="custom-h3">Análisis de Datos de Diabetes 🔬</h3>', unsafe_allow_html=True)
        st.write("")