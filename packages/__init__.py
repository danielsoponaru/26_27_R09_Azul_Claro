"""Paquete común del Reto 09 (Azul Claro).

Módulos:
    bbva_graficos   estilo de los gráficos (paleta BBVA), formato en español y guardado en Graficos/

Las funciones se pueden importar desde el módulo o directamente desde el paquete:

    from packages.bbva_graficos import configurar_graficos, guardar_mpl
    from packages import configurar_graficos, guardar_mpl
"""
from .bbva_graficos import (
    BBVA,
    CMAP_DIVERGENTE,
    CMAP_SECUENCIAL,
    MESES,
    PALETA,
    anotar_fuente,
    configurar_graficos,
    fechas_es,
    fmt_es,
    formato_es,
    guardar_mpl,
    guardar_plotly,
    imprimir,
    numeros_es,
    texto_es,
)

__all__ = [
    "BBVA", "PALETA", "CMAP_SECUENCIAL", "CMAP_DIVERGENTE", "MESES",
    "configurar_graficos", "fmt_es", "numeros_es", "fechas_es", "anotar_fuente",
    "texto_es", "formato_es", "imprimir",
    "guardar_mpl", "guardar_plotly",
]
