import logging
def setup_logging():
    logging.basicConfig(
        filename="audit_log.txt",
        level=logging.INFO,
        format="%(asctime)s - %(message)s"
    )