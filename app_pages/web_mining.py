import streamlit as st
import pandas as pd
import time
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup

class app:
    def main(self):
        st.markdown("<h2>Web Mining y Scraping de MundoVegano</h2>", unsafe_allow_html=True)

        # --- Explicaciones tipo FAQ ---
        with st.expander("¿Qué es Web Mining?"):
            st.write("Web Mining o Minería Web es el proceso de descubrir patrones útiles, información o conocimientos de la Web a través de técnicas de minería de datos, inteligencia artificial y procesamiento de lenguaje natural.")

        with st.expander("¿Qué relación hay entre Web Mining y Web Scraping?"):
            st.write("Web Scraping es una de las técnicas usadas en Web Mining para extraer datos estructurados desde páginas web.")

        with st.expander("¿Se puede generalizar el scraping para cualquier página web?"):
            st.write("No. Cada página web tiene su propia estructura de HTML, por lo que los selectores o patrones para extraer datos cambian de un sitio a otro.")

        with st.expander("¿Qué página evaluamos en este proyecto?"):
            st.write("Evaluamos la tienda online de MundoVegano (Costa Rica), extrayendo productos de categorías como Sustitutos Cárnicos, Lácteos, Margarinas, entre otros.")

        st.markdown("---")

        st.markdown("<h2>Scraping de Mundo Vegano</h2>", unsafe_allow_html=True)
        st.write("""
        El **Web Mining** consiste en extraer y analizar información estructurada o no estructurada de sitios web.
        En este caso, aplicamos **scraping controlado** sobre la tienda [Mundo Vegano](https://mundovegano.cr/),
        para recolectar datos de productos, precios, descuentos y disponibilidad.
        
        ⚡ **Importante**: Cada sitio web es diferente. No existe una forma universal de scraping: hay que adaptarlo
        al HTML, estructura y comportamiento de cada página.
        """)

        if st.button("Ejecutar Scraping Mundo Vegano"):
            with st.spinner("Extrayendo datos... puede tardar unos segundos..."):
                df_scraping = scraping_mundovegano()

            if df_scraping is not None and not df_scraping.empty:
                st.success("Scraping finalizado exitosamente ✅")
                st.dataframe(df_scraping)

                csv = df_scraping.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Descargar resultados en CSV",
                    data=csv,
                    file_name='productos_mundovegano.csv',
                    mime='text/csv'
                )
            else:
                st.warning("No se extrajeron productos o hubo un error.")


@st.cache_data
def scraping_mundovegano():
    # Configuración de Selenium
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Para que no abra una ventana de Chrome
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")

    try:
        browser = webdriver.Chrome(options=chrome_options)
    except Exception as e:
        st.error(f"Error al iniciar el navegador: {e}")
        return None

    categorias_principales = ["Sustitutos Cárnicos", "Sustitutos Lácteos", "Veganesas y Margarinas"]
    datos_productos = []
    nombres_scrapeados = set()

    try:
        browser.get("https://mundovegano.cr/tienda/")
        time.sleep(3)

        for categoria in categorias_principales:
            categoria_elem = browser.find_element(By.LINK_TEXT, categoria)
            categoria_url = categoria_elem.get_attribute("href")

            subcategorias = []
            if categoria == "Sustitutos Cárnicos":
                subcat_elems = browser.find_elements(By.XPATH, f"//a[text()='{categoria}']/following-sibling::ul//a")
                subcategorias = [elem.text for elem in subcat_elems]

            for subcat in subcategorias:
                subcat_elem = browser.find_element(By.LINK_TEXT, subcat)
                subcat_url = subcat_elem.get_attribute("href")
                browser.get(subcat_url)
                time.sleep(2)

                soup_subcat = BeautifulSoup(browser.page_source, 'html.parser')
                productos = soup_subcat.find_all('a', string="Leer más")
                datos_productos.extend(extraer_productos(browser, productos, nombres_scrapeados, categoria, subcat))

            # Revisar productos directamente bajo categoría principal
            browser.get(categoria_url)
            time.sleep(2)
            soup_cat = BeautifulSoup(browser.page_source, 'html.parser')
            productos = soup_cat.find_all('a', string="Leer más")
            datos_productos.extend(extraer_productos(browser, productos, nombres_scrapeados, categoria, None))

    except Exception as e:
        st.error(f"Error durante el scraping: {e}")
    finally:
        browser.quit()

    return pd.DataFrame(datos_productos)

def extraer_productos(browser, productos, nombres_scrapeados, categoria, subcat):
    resultados = []
    for producto in productos:
        titulo_elem = producto.find_previous(['h4', 'h3', 'h2'])
        nombre_producto = titulo_elem.get_text().strip() if titulo_elem else "Nombre desconocido"

        if nombre_producto in nombres_scrapeados:
            continue

        url_producto = producto.get('href')
        if not url_producto or not url_producto.startswith("http"):
            continue

        browser.get(url_producto)
        time.sleep(2)
        soup_prod = BeautifulSoup(browser.page_source, 'html.parser')

        price_container = soup_prod.find('p', class_='price')
        if price_container:
            precio_ins = price_container.find('ins')
            precio_del = price_container.find('del')
            if precio_ins and precio_del:
                precio_descuento_text = precio_ins.get_text()
                precio_original_text = precio_del.get_text()
            else:
                precio_original_text = price_container.get_text()
                precio_descuento_text = None
        else:
            precio_original_text = None
            precio_descuento_text = None

        def convertir_precio(precio_text):
            if precio_text is None:
                return None
            valor = re.sub(r"[₡,]", "", precio_text).strip()
            try:
                return float(valor)
            except:
                return None

        precio_original = convertir_precio(precio_original_text)
        precio_descuento = convertir_precio(precio_descuento_text)

        estado_elem = soup_prod.find('p', class_='stock')
        estado = estado_elem.get_text().strip() if estado_elem else None

        resultados.append({
            "Nombre": nombre_producto,
            "Precio Original": precio_original,
            "Precio Descuento": precio_descuento,
            "Estado": estado,
            "Categoría": categoria,
            "Subcategoría": subcat
        })

        nombres_scrapeados.add(nombre_producto)
    return resultados

if __name__ == "__page__":
    app().main()


