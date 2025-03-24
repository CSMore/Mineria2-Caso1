import logging


def validate_data_loaded(data):
    """
    Verifica si el DataFrame 'data' está cargado y no está vacío.
    Retorna True si está cargado, False en caso contrario.
    """
    if data is None or data.empty:
        logging.warning("Intento de realizar operaciones sin datos cargados.")
        return False
    return True