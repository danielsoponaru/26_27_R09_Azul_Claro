"""Coste computacional: la red frente a Montecarlo y frente a la fórmula cerrada (issue #38).

Entradas:
    Datos/Transformados/rnn_dataset.npz       (rnn_01_datos.py): escalado de las entradas y test
    modelos/rnn_final_regresion.pt            red final y su tiempo de entrenamiento (rnn_03_hiperparametros.py)
    outputs/rnn_tabla_primas.csv              N_eq de cada fecha (rnn_04_prima.py)
    outputs/rnn_prima_resumen.json            N_eq global (rnn_04_prima.py)
Salidas:
    outputs/rnn_coste_computacional.csv       coste medio por valoración de cada método y puntos de equilibrio
    outputs/rnn_coste_por_fecha.csv           lo mismo, fecha a fecha
    outputs/rnn_coste_resumen.json            resumen y textos
    Graficos/rnn_coste_computacional.png
Ejecución: python rnn_05_coste.py

Qué se mide
    El objetivo del reto es reducir la complejidad computacional de valorar el producto. Se compara el
    tiempo de una valoración (una prima en una fecha, dada la historia observada) con cada método:
    - Montecarlo condicionado del equipo (prima_condicionada) con 100.000 continuaciones, la referencia.
    - Montecarlo con N_eq continuaciones: el que tiene el mismo error que la red (rnn_04_prima.py). Es
      la comparación justa en precisión: un Montecarlo de 100.000 continuaciones es mucho más preciso
      que la red, así que compararla solo con él la favorece. Se usa el N_eq de cada fecha y el global.
      Como mínimo 2 continuaciones, porque con 1 no hay error estándar.
    - Fórmula cerrada de BBVA (solo existe para la cesta geométrica): el límite de lo que se puede
      conseguir cuando hay fórmula.
    - Red con 1 historia (una valoración aislada) y con 10.000 historias en lote (el uso de la red en
      producción: muchas valoraciones a la vez, que es donde una red aprovecha el cálculo matricial).
    Montecarlo cuesta más cuanto antes es la fecha (simula los 20 - k pasos que faltan) y N_eq cambia
    mucho con la fecha, así que se mide fecha a fecha (k = 1..19), con la misma historia en todos los
    métodos: la primera de cada fecha en rnn_04_prima.py.

Cómo se mide
    time.perf_counter, mediana de 5 repeticiones después de una ejecución de calentamiento (la primera
    llamada a la red incluye la puesta en marcha de PyTorch). La mediana es robusta a un pico puntual de
    carga del ordenador. Es la CPU de un portátil: los tiempos absolutos dependen de la máquina, los
    cocientes entre métodos son más estables.

Punto de equilibrio
    La red tiene un coste fijo que Montecarlo no tiene: simular las 100.000 trayectorias de
    entrenamiento y entrenar la red final (su tiempo se lee de modelos/rnn_final_regresion.pt). El
    número de valoraciones a partir del cual compensa es
        coste fijo / (tiempo de Montecarlo por valoración - tiempo de la red por valoración).
    Si Montecarlo es más barato que la red, el denominador no es positivo y la red no compensa nunca:
    se dice tal cual.
    Se dan dos puntos de equilibrio, con dos costes fijos:
    - Solo el uso: simular los datos y entrenar la red final. Es lo que costaría volver a entrenarla,
      por ejemplo con otra cartera, ya sabiendo qué configuración usar.
    - El desarrollo completo: simular los datos, la búsqueda de hiperparámetros y la validación cruzada
      de rnn_03_hiperparametros.py. La red final es una de las configuraciones de la búsqueda, así que
      su entrenamiento ya está dentro y no se suma dos veces.
"""
import json
import os
import time
from pathlib import Path

# MKL en modo secuencial: evita la segunda copia de OpenMP que choca con la de PyTorch cuando se usa la
# fórmula cerrada (scipy) en el mismo proceso (ver rnn_04_prima.py). Tiene que fijarse antes de importar numpy
os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from packages import montecarlo as mc
from packages.bbva_graficos import BBVA, configurar_graficos, formato_es, guardar_mpl, imprimir, texto_es
from packages.formula_cerrada_bbva import BasketGeomAsian
from packages.rnn_modelos import DISPOSITIVO, RUTA_DATASET, RedRecurrente, evaluar

NUM_SIMS = 100_000            # continuaciones del Montecarlo de referencia
N_LOTE = 10_000               # historias por lote de la red
REPETICIONES = 5
FECHAS = range(1, mc.NUM_PASOS)

os.chdir(Path(__file__).resolve().parent)   # rutas relativas a la raíz del repo
configurar_graficos()
imprimir(f"Dispositivo de la red: {DISPOSITIVO}; hilos de PyTorch: {torch.get_num_threads()}")

d = np.load(RUTA_DATASET, allow_pickle=False)
media_x, desv_x = d["media_entradas"], d["desv_entradas"]
g = torch.load("modelos/rnn_final_regresion.pt", weights_only=True)
red = RedRecurrente(**g["config"])
red.load_state_dict(g["estado"])
tabla_primas = pd.read_csv("outputs/rnn_tabla_primas.csv", index_col="k")
resumen_primas = json.loads(Path("outputs/rnn_prima_resumen.json").read_text(encoding="utf-8"))
assert not resumen_primas["rapido"], "PARA: outputs/rnn_prima_resumen.json no es de la ejecución completa de rnn_04"
N_EQ_GLOBAL = resumen_primas["fechas"]["n_eq_red"]


def medir(funcion):
    """Mediana de REPETICIONES ejecuciones, en segundos, después de una de calentamiento."""
    funcion()
    tiempos = []
    for _ in range(REPETICIONES):
        inicio = time.perf_counter()
        funcion()
        tiempos.append(time.perf_counter() - inicio)
    return float(np.median(tiempos))


def entradas_historias(historias):
    """Historias (n, k, 3) de precios normalizados -> entradas escaladas (n, k, 4), como en rnn_01."""
    n, k, _ = historias.shape
    tiempo = np.broadcast_to(mc.TIEMPOS[:k] / mc.TTM, (n, k))[..., None]
    entradas = np.concatenate((mc.retornos_log(historias), tiempo), axis=2)
    return ((entradas - media_x) / desv_x).astype(np.float32)


def continuaciones(n_eq):
    return max(2, int(np.ceil(n_eq))) if np.isfinite(n_eq) else NUM_SIMS


# ========================== COSTE FIJO DE LA RED ==========================
t_datos = medir(lambda: mc.calcular_Y(mc.calcular_payoff(
    mc.simular_continuaciones(None, NUM_SIMS, np.random.default_rng(42)))))
t_entrenamiento = float(g["tiempo_entrenamiento_s"])
coste_fijo = t_datos + t_entrenamiento
busqueda = float(pd.read_csv("outputs/rnn_busqueda_hiperparametros.csv")["tiempo_entrenamiento_s"].sum())
validacion = float(pd.read_csv("outputs/rnn_validacion_cruzada.csv")
                   [["regresion_tiempo_s", "clasificacion_tiempo_s"]].sum().sum())
coste_desarrollo = t_datos + busqueda + validacion     # la búsqueda ya incluye el entrenamiento de la red final
imprimir(f"Coste fijo de uso: simular {NUM_SIMS:,} trayectorias {t_datos:.2f} s + entrenar la red final "
         f"{t_entrenamiento:.0f} s ({g['epocas']} épocas) = {coste_fijo:.0f} s.")
imprimir(f"Coste de desarrollo completo: simulación + búsqueda de hiperparámetros {busqueda / 60:.1f} min (incluye "
         f"la red final) + validación cruzada {validacion / 60:.1f} min = {coste_desarrollo / 60:.1f} min.")

# ============================== FECHA A FECHA ==============================
imprimir(f"\nTiempos por valoración (mediana de {REPETICIONES} repeticiones), fecha a fecha:")
filas = []
for k in FECHAS:
    historia = mc.simular_continuaciones(None, 1, np.random.default_rng(20_000 + 100 * k))[0, :k]
    lote = mc.simular_continuaciones(None, N_LOTE, np.random.default_rng(70_000 + k))[:, :k]
    x_1, x_lote = entradas_historias(historia[None]), entradas_historias(lote)
    n_eq = float(tabla_primas.loc[k, "n_eq_red"])
    n_fecha, n_global = continuaciones(n_eq), continuaciones(N_EQ_GLOBAL)
    historial_precios = np.hstack([np.ones((3, 1)), historia.T])
    rng = np.random.default_rng(80_000 + k)
    fila = {
        "k": k, "t_anios": k * mc.DT, "n_eq": n_eq, "continuaciones_n_eq": n_fecha,
        "mc_100000_s": medir(lambda: mc.prima_condicionada(historia, NUM_SIMS, rng)),
        "mc_n_eq_fecha_s": medir(lambda: mc.prima_condicionada(historia, n_fecha, rng)),
        "mc_n_eq_global_s": medir(lambda: mc.prima_condicionada(historia, n_global, rng)),
        "formula_s": medir(lambda: BasketGeomAsian(21, k, mc.R, 3, mc.SIGMA.reshape(-1, 1), mc.RHO, mc.TTM,
                                                   historial_precios, True)),
        "red_1_s": medir(lambda: evaluar(red, x_1, g["media_y"], g["desv_y"])),
        "red_lote_s": medir(lambda: evaluar(red, x_lote, g["media_y"], g["desv_y"])) / N_LOTE,
    }
    filas.append(fila)
    imprimir(f"  k = {k:2d}: Montecarlo 100,000 {1e3 * fila['mc_100000_s']:8.2f} ms | Montecarlo N_eq "
             f"({n_fecha:,}) {1e3 * fila['mc_n_eq_fecha_s']:7.3f} ms | red 1 historia {1e3 * fila['red_1_s']:6.3f} ms "
             f"| red en lote {1e6 * fila['red_lote_s']:6.2f} µs")
por_fecha = pd.DataFrame(filas).set_index("k")


def equilibrio(t_montecarlo, t_red, coste=coste_fijo):
    """Valoraciones a partir de las que compensa la red; NaN si Montecarlo es más barato (no compensa nunca)."""
    ahorro = t_montecarlo - t_red
    return coste / ahorro if ahorro > 0 else np.nan


for referencia in ("mc_100000", "mc_n_eq_fecha"):
    for uso in ("red_1", "red_lote"):
        por_fecha[f"cociente_{referencia}_{uso}"] = por_fecha[f"{referencia}_s"] / por_fecha[f"{uso}_s"]
        por_fecha[f"equilibrio_{referencia}_{uso}"] = [equilibrio(a, b) for a, b in
                                                       zip(por_fecha[f"{referencia}_s"], por_fecha[f"{uso}_s"])]

# ============================== MEDIA DE LAS FECHAS ==============================
metodos = {
    "Montecarlo, 100.000 continuaciones": "mc_100000_s",
    f"Montecarlo, N_eq global ({continuaciones(N_EQ_GLOBAL)} continuaciones)": "mc_n_eq_global_s",
    "Montecarlo, N_eq de cada fecha": "mc_n_eq_fecha_s",
    "Fórmula cerrada (cesta geométrica)": "formula_s",
    "Red, 1 historia": "red_1_s",
    f"Red, lote de {f'{N_LOTE:,}'.replace(',', '.')} historias": "red_lote_s",
}
medias = por_fecha[list(metodos.values())].mean()
global_ = pd.DataFrame({"metodo": list(metodos), "segundos_por_valoracion": [medias[c] for c in metodos.values()]})
for uso, columna in (("red_1", "red_1_s"), ("red_lote", "red_lote_s")):
    global_[f"veces_mas_lento_que_{uso}"] = global_["segundos_por_valoracion"] / medias[columna]
    global_[f"equilibrio_{uso}"] = [equilibrio(t, medias[columna]) for t in global_["segundos_por_valoracion"]]
    global_[f"equilibrio_desarrollo_{uso}"] = [equilibrio(t, medias[columna], coste_desarrollo)
                                               for t in global_["segundos_por_valoracion"]]
global_.loc[len(global_)] = {"metodo": "Coste fijo de uso (simulación + entrenamiento final), en segundos",
                             "segundos_por_valoracion": coste_fijo}
global_.loc[len(global_)] = {"metodo": "Coste de desarrollo completo (simulación + búsqueda + validación cruzada), "
                                       "en segundos", "segundos_por_valoracion": coste_desarrollo}

# ================================ TEXTOS ================================
mas_barato_1 = por_fecha.index[por_fecha["mc_n_eq_fecha_s"] <= por_fecha["red_1_s"]].tolist()
mas_barato_lote = por_fecha.index[por_fecha["mc_n_eq_fecha_s"] <= por_fecha["red_lote_s"]].tolist()
m = medias
texto = (
    f"Coste por valoración, media de las 19 fechas: Montecarlo de 100,000 continuaciones "
    f"{1e3 * m['mc_100000_s']:.1f} ms; red con 1 historia {1e3 * m['red_1_s']:.3f} ms "
    f"({m['mc_100000_s'] / m['red_1_s']:,.0f} veces menos) y en lote de {N_LOTE:,} historias "
    f"{1e6 * m['red_lote_s']:.2f} µs por valoración ({m['mc_100000_s'] / m['red_lote_s']:,.0f} veces menos); "
    f"fórmula cerrada {1e3 * m['formula_s']:.3f} ms. ")


def frase_equilibrio(referencia, coste):
    """Valoraciones a partir de las que compensa la red frente a `referencia`, de una en una y en lote."""
    eq_1, eq_lote = equilibrio(m[referencia], m["red_1_s"], coste), equilibrio(m[referencia], m["red_lote_s"], coste)
    return (f"de una en una, {'no compensa nunca, porque ese Montecarlo es más barato' if np.isnan(eq_1) else f'a partir de {eq_1:,.0f} valoraciones'}; "
            f"en lote, {'no compensa nunca' if np.isnan(eq_lote) else f'a partir de {eq_lote:,.0f}'}")


for coste, descripcion in ((coste_fijo, f"Con el coste fijo de uso ({coste_fijo:.0f} s: simular los datos y entrenar "
                                         "la red final)"),
                           (coste_desarrollo, f"Con el coste de desarrollo completo ({coste_desarrollo / 60:.0f} min: "
                                              "además, la búsqueda de hiperparámetros y la validación cruzada)")):
    texto += (f"{descripcion}, la red compensa frente al Montecarlo de 100,000 {frase_equilibrio('mc_100000_s', coste)}; "
              f"frente al Montecarlo de su misma precisión (N_eq global, {continuaciones(N_EQ_GLOBAL):,} "
              f"continuaciones, {1e3 * m['mc_n_eq_global_s']:.3f} ms), {frase_equilibrio('mc_n_eq_global_s', coste)}. ")
texto += ("Fecha a fecha, con el N_eq de cada fecha, el Montecarlo equivalente es más barato que la red con 1 historia "
          f"en {len(mas_barato_1)} de las 19 fechas" + (f" (k = {', '.join(map(str, mas_barato_1))})" if mas_barato_1 else "")
          + (f", y más barato que la red en lote en {len(mas_barato_lote)} (k = {', '.join(map(str, mas_barato_lote))})."
             if mas_barato_lote else ", y en ninguna es más barato que la red en lote."))
if mas_barato_1:
    texto += (" En esas fechas queda poca incertidumbre hasta el vencimiento y unas pocas continuaciones ya tienen "
              "el error de la red, así que valorar una sola historia con la red no compensa nunca.")

# ============================ GUARDADO E IMPRESIÓN ============================
global_.to_csv("outputs/rnn_coste_computacional.csv", index=False)
por_fecha.to_csv("outputs/rnn_coste_por_fecha.csv")
with open("outputs/rnn_coste_resumen.json", "w", encoding="utf-8") as archivo:
    json.dump({"coste_fijo_s": coste_fijo, "coste_desarrollo_s": coste_desarrollo, "simulacion_datos_s": t_datos,
               "entrenamiento_s": t_entrenamiento, "busqueda_hiperparametros_s": busqueda,
               "validacion_cruzada_s": validacion,
               "medias_s": {c: float(v) for c, v in medias.items()}, "n_eq_global": N_EQ_GLOBAL,
               "fechas_mc_equivalente_mas_barato_que_red_1": mas_barato_1,
               "fechas_mc_equivalente_mas_barato_que_red_lote": mas_barato_lote,
               "texto": texto_es(texto)}, archivo, indent=2, ensure_ascii=False)
imprimir("\nMedia de las 19 fechas:")
# print y no imprimir: las etiquetas ya llevan los miles en formato español y no se pueden volver a convertir
print(global_.to_string(index=False, float_format=lambda v: f"{v:.3g}".replace(".", ",")))
imprimir("\nFecha a fecha (cocientes = veces que Montecarlo tarda más que la red; equilibrio en valoraciones):")
columnas = ["n_eq", "continuaciones_n_eq", "cociente_mc_100000_red_1", "cociente_mc_100000_red_lote",
            "cociente_mc_n_eq_fecha_red_1", "cociente_mc_n_eq_fecha_red_lote", "equilibrio_mc_n_eq_fecha_red_1",
            "equilibrio_mc_n_eq_fecha_red_lote"]
imprimir(por_fecha[columnas].to_string(float_format=lambda v: f"{v:.3g}"))
imprimir("\n" + texto)

# ================================ GRÁFICO ================================
fig, (ax_b, ax_f) = plt.subplots(1, 2, figsize=(16, 6.5), gridspec_kw={"width_ratios": [1, 1.2]})
etiquetas = list(metodos)
valores = [medias[c] for c in metodos.values()]
colores = [BBVA["gris_dato"], BBVA["medio"], BBVA["cielo"], BBVA["marino"], BBVA["principal"], BBVA["agua"]]
orden = np.argsort(valores)[::-1]
barras = ax_b.barh([etiquetas[i] for i in orden], [valores[i] for i in orden], color=[colores[i] for i in orden])
for barra, i in zip(barras, orden):
    v = valores[i]
    texto_v = f"{v * 1e3:.3g} ms" if v >= 1e-3 else f"{v * 1e6:.3g} µs"
    ax_b.text(v * 1.15, barra.get_y() + barra.get_height() / 2, texto_v, va="center", fontsize=11, color=BBVA["marino"])
ax_b.set_xscale("log")
ax_b.set_xlim(right=max(valores) * 30)
ax_b.grid(False)
ax_b.set_xlabel("Segundos por valoración (escala logarítmica)")
ax_b.set_title("Media de las 19 fechas", fontsize=13)

tk = por_fecha["t_anios"]
for columna, etiqueta, color, estilo in (("mc_100000_s", "Montecarlo, 100.000 continuaciones", BBVA["gris_dato"], "-"),
                                         ("mc_n_eq_fecha_s", "Montecarlo, N_eq de la fecha", BBVA["cielo"], "-"),
                                         ("red_1_s", "Red, 1 historia", BBVA["principal"], "--"),
                                         ("red_lote_s", "Red, por valoración en lote", BBVA["agua"], "--")):
    ax_f.plot(tk, por_fecha[columna], color=color, linestyle=estilo, linewidth=2.5, marker="o", label=etiqueta)
ax_f.set_yscale("log")
ax_f.set_xlabel("Fecha de valoración t_k (años)")
ax_f.set_ylabel("Segundos por valoración (escala logarítmica)")
ax_f.set_title("Fecha a fecha", fontsize=13)
fig.suptitle("Gráfico 1. Coste de una valoración: red frente a Montecarlo y fórmula cerrada", x=0.01, ha="left",
             fontsize=16, fontweight="bold", color=BBVA["marino"])
fig.tight_layout()
formato_es(fig, fuente=f"mediana de {REPETICIONES} repeticiones en la CPU de un portátil; N_eq de rnn_04_prima.py.")
ax_f.legend(fontsize=10, loc="upper right", bbox_to_anchor=(1.0, 0.82))  # tras formato_es: miles ya en español
guardar_mpl(fig, "rnn_coste_computacional")
plt.close(fig)
