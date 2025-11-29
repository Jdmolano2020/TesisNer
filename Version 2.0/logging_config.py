"""
Configuración de logging centralizada para el proyecto SROIE

Provee una función `get_logger(name)` y configura un logger raíz que
escribe a consola y a un archivo `logs/sroie.log`.
"""
import logging
import logging.handlers
import os

LOG_DIR = os.path.join(os.path.dirname(__file__), 'output', 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, 'sroie.log')

def configure_root_logger(level=logging.INFO):
    root = logging.getLogger()
    if root.handlers:
        return  # ya configurado

    root.setLevel(level)

    # Formatter
    fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # Rotating file handler
    fh = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=5, encoding='utf-8')
    fh.setLevel(level)
    fh.setFormatter(fmt)
    root.addHandler(fh)

def get_logger(name: str, level=logging.INFO):
    configure_root_logger(level)
    return logging.getLogger(name)
