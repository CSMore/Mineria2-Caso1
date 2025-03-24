import streamlit as st
import logging
from logic.patron_estacion_lg import SeasonalPatternLogic


class SeasonalPatternWidgets:

    def render_warning_no_data(self):
        st.warning("No hay datos suavizados disponibles. Aplica el suavizado en la sección anterior.")

    def render_configuration_panel(self):
        st.markdown("## Configuración del Patrón Estacional")
        col1, col2 = st.columns(2)
        with col1:
            self.seasonal_pattern = st.selectbox(
                "Seleccione el Patrón Estacional:", 
                ["Annual", "Monthly", "Weekly"], 
                key="seasonal_pattern"
            )
        with col2:    
            self.training_split = st.slider(
                "Entrenamiento - Prueba (%)", 
                min_value=1, max_value=100, value=80, 
                key="training_split"
            )

    def render_graph_section(self, sp_logic):
        # Ejecuta el procesamiento y muestra el resultado
        if st.button("Generar", key="generate_graph_button", use_container_width=True):
            try:
                fig, resampled = sp_logic.process(self.seasonal_pattern, self.training_split)
                self.render_data_preview(sp_logic.data)
                self.render_graph_info()
                st.plotly_chart(fig, use_container_width=True)
                self.render_download_button(resampled)
            except Exception as e:
                st.error("Error al generar el gráfico interactivo: " + str(e))

    def render_data_preview(self, data):
        st.write("#### Dataset de datos suavizados:")
        st.dataframe(data)

    def render_graph_info(self):
        st.write(f"### Generando gráfico interactivo para el patrón: {self.seasonal_pattern}")
        st.write(f"Porcentaje de entrenamiento: {self.training_split}%")

    def render_download_button(self, resampled):
        @st.fragment
        def download_fragment():
            st.download_button(
                "Descargar datos resampleados CSV", 
                resampled.to_csv(index=True), 
                "resampled_data.csv", 
                "text/csv"
            )
        download_fragment()


class app:
    def main(self):
        st.markdown("<h2>Pronóstico por Patrón Estacional</h2>", unsafe_allow_html=True)
        widgets = SeasonalPatternWidgets()
        # Cargar los datos suavizados desde la sesión
        data = st.session_state.get("smoothed_data", None)
        logic = SeasonalPatternLogic(data)
        
        if not logic.validate_smoothed_data():
            widgets.render_warning_no_data()
            return
        
        # Renderiza panel de configuración y sección gráfica
        widgets.render_configuration_panel()
        widgets.render_graph_section(logic)

if __name__ == "__page__":
    app().main()