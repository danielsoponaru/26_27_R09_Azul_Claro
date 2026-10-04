"""Configuración global de gráficos (paleta BBVA) para los notebooks.

Uso en el notebook (primera celda):

    from packages.bbva_graficos import (
        configurar_graficos, guardar_mpl, guardar_plotly, fmt_es, BBVA, PALETA
    )
    configurar_graficos()
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns

# --- Paleta BBVA ---
BBVA = {
    "principal": "#004481",
    "marino": "#070E46",
    "medio": "#1973B8",
    "cielo": "#5BBEFF",
    "agua": "#2DCCCD",
    "gris": "#F4F4F4",
    "gris_dato": "#B8BEC6",
    "blanco": "#FFFFFF",
}
PALETA = [BBVA["principal"], BBVA["medio"], BBVA["cielo"], BBVA["agua"], BBVA["marino"]]

CARPETA_GRAFICOS = Path("Graficos")


def configurar_graficos(carpeta="Graficos"):
    """Aplica el estilo BBVA a matplotlib, seaborn y plotly y crea la carpeta de salida."""
    global CARPETA_GRAFICOS
    CARPETA_GRAFICOS = Path(carpeta)
    CARPETA_GRAFICOS.mkdir(exist_ok=True)

    # Seaborn primero, para que no pise los ajustes de matplotlib
    sns.set_theme(style="whitegrid")
    sns.set_palette(PALETA)

    mpl.rcParams.update({
        "figure.figsize": (10, 6),
        "figure.dpi": 100,
        "savefig.dpi": 200,
        "figure.facecolor": BBVA["blanco"],
        "axes.facecolor": BBVA["blanco"],
        "axes.edgecolor": BBVA["marino"],
        "axes.labelcolor": BBVA["marino"],
        "axes.titlecolor": BBVA["marino"],
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#E3E6EA",
        "grid.linewidth": 0.8,
        "xtick.color": BBVA["marino"],
        "ytick.color": BBVA["marino"],
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.frameon": False,
        "legend.fontsize": 11,
        "font.family": "sans-serif",
        "axes.prop_cycle": mpl.cycler(color=PALETA),
    })

    pio.templates["bbva"] = go.layout.Template(
        layout=dict(
            colorway=PALETA,
            font=dict(family="Arial, sans-serif", size=14, color=BBVA["marino"]),
            title=dict(font=dict(size=20, color=BBVA["marino"]), x=0.0),
            paper_bgcolor=BBVA["blanco"],
            plot_bgcolor=BBVA["blanco"],
            xaxis=dict(showgrid=False, linecolor=BBVA["marino"], ticks="outside"),
            yaxis=dict(gridcolor="#E3E6EA", zeroline=False),
            legend=dict(orientation="h", y=-0.2),
            margin=dict(l=60, r=30, t=70, b=60),
        )
    )
    pio.templates.default = "bbva"
    return CARPETA_GRAFICOS


def fmt_es(valor, decimales=0):
    """1234567.8 -> '1.234.568' (miles con punto, decimales con coma)."""
    texto = f"{valor:,.{decimales}f}"
    return texto.replace(",", "X").replace(".", ",").replace("X", ".")


def guardar_mpl(fig, nombre):
    """Guarda una figura de matplotlib/seaborn como PNG en la carpeta de gráficos."""
    CARPETA_GRAFICOS.mkdir(exist_ok=True)
    ruta = CARPETA_GRAFICOS / f"{nombre}.png"
    fig.savefig(ruta, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Guardado: {ruta}")
    return ruta


def guardar_plotly(fig, nombre, ancho=1200, alto=700):
    """Guarda una figura de plotly como PNG (requiere kaleido)."""
    CARPETA_GRAFICOS.mkdir(exist_ok=True)
    ruta = CARPETA_GRAFICOS / f"{nombre}.png"
    try:
        fig.write_image(str(ruta), width=ancho, height=alto, scale=2)
    except ValueError as e:
        raise RuntimeError(
            "No se pudo exportar con plotly. Instala kaleido: pip install -U kaleido"
        ) from e
    print(f"Guardado: {ruta}")
    return ruta
