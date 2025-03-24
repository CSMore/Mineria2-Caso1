import streamlit as st

class app:
    def main(self):
        color = st.color_picker("Pick A Color", "#00f900")
        st.write("The current color is", color)

if __name__ == "__page__":
    app().main()