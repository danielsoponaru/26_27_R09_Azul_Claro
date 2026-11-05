"""Estilo común de los gráficos del Reto 09 (paleta BBVA): configuración, formato en español y guardado.

Ruta en el repositorio: packages/bbva_graficos.py
Entrada: ninguna (no lee datos; solo da formato a los gráficos de los notebooks).
Salida: los gráficos se guardan en PNG en Graficos/.

Uso en el notebook (primera celda):

    from packages.bbva_graficos import (
        configurar_graficos, guardar_mpl, guardar_plotly, fmt_es, BBVA, PALETA
    )
    configurar_graficos()

Contenido:
    BBVA, PALETA          colores de la paleta y orden de uso cuando hay varias series
    CMAP_SECUENCIAL       escala continua para magnitudes (de gris a azul principal)
    CMAP_DIVERGENTE       escala centrada en 0 para correlaciones (marino - blanco - agua)
    MESES                 abreviaturas de los meses en español
    configurar_graficos   aplica el estilo a matplotlib, seaborn y plotly y crea Graficos/
    fmt_es                número en formato español para textos y anotaciones (1.234,5)
    numeros_es            números de un eje de matplotlib en formato español
    fechas_es             fechas del eje X como "mar 2025"
    anotar_fuente         escribe la fuente de los datos bajo el gráfico
    texto_es              números de un texto en formato español (1.234,5 y 12,5 %)
    formato_es            figura entera en formato español y, opcionalmente, su fuente debajo
    imprimir              print() con los números en formato español
    guardar_mpl           guarda una figura de matplotlib o seaborn en Graficos/
    cerrar_grafico        retoques finales, fuente, guardado y muestra de una figura de matplotlib
    guardar_plotly        guarda una figura de plotly en Graficos/ (necesita kaleido)

Por qué un módulo: si cada notebook define sus colores y tamaños, cada gráfico sale distinto.
Con la configuración en un único sitio, un cambio de estilo se hace aquí y llega a todos.
"""
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter, PercentFormatter, ScalarFormatter

# --- Paleta BBVA ---
BBVA = {
    "principal": "#004481",   # serie única o línea destacada
    "marino": "#070E46",      # texto, títulos y ejes
    "medio": "#1973B8",
    "cielo": "#5BBEFF",
    "agua": "#2DCCCD",        # acento y variaciones positivas
    "gris": "#F4F4F4",        # fondos y zonas de apoyo (casi blanco: no sirve para líneas)
    "gris_dato": "#B8BEC6",   # lo que no se resalta, líneas de referencia y categoría "Otros"
    "blanco": "#FFFFFF",      # fondo
}
# Orden de uso cuando hay varias series
PALETA = [BBVA["principal"], BBVA["medio"], BBVA["cielo"], BBVA["agua"], BBVA["marino"]]

# Escalas continuas: secuencial para magnitudes y divergente (centrada en 0) para correlaciones
CMAP_SECUENCIAL = LinearSegmentedColormap.from_list("bbva_secuencial", [BBVA["gris"], BBVA["principal"]])
CMAP_DIVERGENTE = LinearSegmentedColormap.from_list(
    "bbva_divergente", [BBVA["marino"], BBVA["blanco"], BBVA["agua"]])

MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]

CARPETA_GRAFICOS = Path("Graficos")


# --- Configuración ---
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


# --- Formato en español ---
def fmt_es(valor, decimales=0):
    """1234567.8 -> '1.234.568' (miles con punto, decimales con coma)."""
    texto = f"{valor:,.{decimales}f}"
    return texto.replace(",", "X").replace(".", ",").replace("X", ".")


class _NumerosEs(ScalarFormatter):
    """El formato por defecto de matplotlib (mismos decimales en cada marca), con los números en español."""

    def __call__(self, x, pos=None):
        texto = super().__call__(x, pos)
        decimales = len(texto.split(".")[1]) if "." in texto else 0
        return fmt_es(x, decimales).replace("-", "\N{MINUS SIGN}")


def numeros_es(eje):
    """Números de un eje (`ax.yaxis` o `ax.xaxis`) en formato español.

    No se fija el número de decimales a mano: con dos decimales fijos, una marca en 0,025
    saldría como 0,03 y el eje mentiría. Se usan los que matplotlib calcula para cada eje.
    """
    formato = _NumerosEs(useOffset=False)
    formato.set_scientific(False)
    eje.set_major_formatter(formato)


def fechas_es(ax):
    """Fechas del eje X como `mar 2025`, sin depender del idioma del sistema operativo."""
    def etiqueta(x, _):
        fecha = mdates.num2date(x)
        return f"{MESES[fecha.month - 1]} {fecha.year}"
    ax.xaxis.set_major_formatter(etiqueta)


def anotar_fuente(fig, texto):
    """Escribe la fuente de los datos bajo el gráfico. Se llama después de `plt.tight_layout()`."""
    fig.text(0.01, -0.01, f"Fuente: {texto}", ha="left", va="top", alpha=0.75)


def texto_es(texto):
    """Números de un texto en formato español: '1,234.5' -> '1.234,5' y '12.5%' -> '12,5 %'."""
    texto = re.sub(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+",
                   lambda m: m.group(0).replace(",", "\0").replace(".", ",").replace("\0", "."), texto)
    return re.sub(r"(?<=\d)%", " %", texto)


def formato_es(fig, fuente=None):
    """Pasa una figura entera al formato español: ejes numéricos y de porcentaje, etiquetas y textos.

    Con `fuente` escribe además "Fuente: ..." debajo de todo lo que ocupa la figura (leyendas y
    etiquetas fuera de los ejes incluidas), para que no se solape. Llamar justo antes de guardar_mpl().
    """
    for ax in fig.axes:
        for eje in (ax.xaxis, ax.yaxis):
            formato = eje.get_major_formatter()
            if isinstance(formato, PercentFormatter):
                eje.set_major_formatter(FuncFormatter(lambda v, pos, f=formato: texto_es(f(v, pos))))
            elif isinstance(formato, ScalarFormatter) and not isinstance(formato, _NumerosEs):
                numeros_es(eje)
    for texto in fig.findobj(mpl.text.Text):
        original = texto.get_text()
        nuevo = texto_es(original)
        if nuevo != original:
            texto.set_text(nuevo)
    if fuente:
        fig.canvas.draw()
        caja = fig.get_tightbbox(fig.canvas.get_renderer())      # en pulgadas
        ancho, alto = fig.get_size_inches()
        fig.text(caja.x0 / ancho, (caja.y0 - 0.08) / alto, f"Fuente: {fuente}", ha="left", va="top",
                 fontsize=11, color=BBVA["marino"])
    return fig


def imprimir(*valores, sep=" ", end="\n"):
    """Como print(), pero con los números en formato español (tablas incluidas)."""
    print(texto_es(sep.join(str(v) for v in valores)), end=end)


# --- Guardado ---
def guardar_mpl(fig, nombre):
    """Guarda una figura de matplotlib/seaborn como PNG en la carpeta de gráficos."""
    CARPETA_GRAFICOS.mkdir(exist_ok=True)
    ruta = CARPETA_GRAFICOS / f"{nombre}.png"
    fig.savefig(ruta, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Guardado: {ruta}")
    return ruta


def cerrar_grafico(fig, nombre, fuente=None, ajustar_titulos=True, formato_completo=False):
    """Último paso de un gráfico de matplotlib/seaborn: retoques, fuente, guardado y muestra.

    1. `ajustar_titulos`: si la figura tiene título general (`fig.suptitle`), lo pone a 16 pt en negrita y
       baja el de cada panel a 13 pt, porque a 16 pt (el tamaño global) no caben varios paneles en fila, y
       recoloca con `tight_layout()`. Con `False` no toca títulos ni la disposición (por ejemplo, cuando la
       figura ya se colocó a mano con `subplots_adjust`).
    2. Pasa a formato español los ejes numéricos que siguen con el formato por defecto. Con
       `formato_completo=True` usa `formato_es()`, que además traduce porcentajes y textos de la figura.
    3. Si se indica `fuente`, la escribe bajo el gráfico.
    4. Guarda el PNG en Graficos/ y muestra la figura.
    """
    if ajustar_titulos and fig._suptitle is not None:
        fig._suptitle.set(fontsize=16, fontweight="bold", color=BBVA["marino"])
        for ax in fig.axes:
            for posicion in ("left", "center", "right"):     # los títulos del estilo global van a la izquierda
                if ax.get_title(loc=posicion):
                    ax.set_title(ax.get_title(loc=posicion), loc=posicion, fontsize=13)
        fig.tight_layout()
    if formato_completo:
        formato_es(fig, fuente=fuente)
    else:
        for ax in fig.axes:
            for eje in (ax.xaxis, ax.yaxis):
                if type(eje.get_major_formatter()) is ScalarFormatter:
                    numeros_es(eje)
        if fuente:
            anotar_fuente(fig, fuente)
    ruta = guardar_mpl(fig, nombre)
    plt.show()
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
