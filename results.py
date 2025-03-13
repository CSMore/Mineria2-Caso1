import streamlit as st
import pandas as pd
import numpy as np
import logging

#Configuración de login para auditorias
logging.basicConfig(filename="audit_log.txt", level=logging.INFO, format="%(asctime)s - %(message)s")

class analysis:
    @staticmethod
    def load_data():
        #Cargar datos de la base de datos
        data = pd.read_csv("data.csv")




class app:
    def run(self):
        """Función principal para ejecutar la aplicación sin main()"""
        logging.info("Iniciando aplicación")
        st.markdown('<h3 class="custom-h3">Comparación de Modelo 🧮</h3>', unsafe_allow_html=True)
        st.write("")