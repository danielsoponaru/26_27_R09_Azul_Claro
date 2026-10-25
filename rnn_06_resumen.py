"""Resumen de la parte de redes neuronales para el informe.

Entradas (las genera la ejecución completa de los scripts rnn_0X):
    Datos/Transformados/rnn_dataset.npz                  rnn_01_datos.py
    outputs/rnn_comparacion_regresion.csv
    outputs/rnn_comparacion_clasificacion.csv
    outputs/rnn_eleccion_arquitectura.json               rnn_02_arquitecturas.py
    outputs/rnn_busqueda_hiperparametros.csv
    outputs/rnn_configuracion_final.json                 rnn_03_hiperparametros.py
    outputs/rapido/rnn_busqueda_hiperparametros.csv
    outputs/rapido/rnn_configuracion_final.json          prueba --rapido de rnn_03 (lección del punto 10)
    outputs/rnn_tabla_primas.csv
    outputs/rnn_prima_resumen.json                       rnn_04_prima.py
    outputs/rnn_coste_resumen.json                       rnn_05_coste.py
Salida:
    outputs/rnn_resumen.txt                              el mismo resumen que se imprime (no se sube)
Ejecución: python rnn_06_resumen.py

Todos los números salen de los archivos de resultados: este script no entrena ni simula nada. Los textos que
ya escribieron los scripts anteriores, con sus números, se copian tal cual; las frases nuevas toman sus números
de los CSV y JSON y los escriben en formato español con fmt_es.
"""
import json
import os
import textwrap
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from packages import montecarlo as mc
from packages.bbva_graficos import fmt_es

os.chdir(Path(__file__).resolve().parent)   # rutas relativas a la raíz del repo
RUTA_SALIDA = Path("outputs/rnn_resumen.txt")


def leer_json(ruta):
    return json.loads(Path(ruta).read_text(encoding="utf-8"))


def pct(x, decimales=1):
    return f"{fmt_es(100 * x, decimales)} %"


eleccion = leer_json("outputs/rnn_eleccion_arquitectura.json")
config = leer_json("outputs/rnn_configuracion_final.json")
prima = leer_json("outputs/rnn_prima_resumen.json")
coste = leer_json("outputs/rnn_coste_resumen.json")
rapido = leer_json("outputs/rapido/rnn_configuracion_final.json")
assert not (eleccion["rapido"] or config["rapido"] or prima["rapido"]), "PARA: algún resultado es de una prueba --rapido"
assert rapido["rapido"], "PARA: outputs/rapido/rnn_configuracion_final.json no es de una prueba --rapido"
reg = pd.read_csv("outputs/rnn_comparacion_regresion.csv", index_col="modelo")
busqueda = pd.read_csv("outputs/rnn_busqueda_hiperparametros.csv", index_col="configuracion")
busqueda_rapida = pd.read_csv("outputs/rapido/rnn_busqueda_hiperparametros.csv", index_col="configuracion")
tabla_primas = pd.read_csv("outputs/rnn_tabla_primas.csv", index_col="k")
datos = np.load("Datos/Transformados/rnn_dataset.npz", allow_pickle=False)

secciones = []

# 1. Datos
payoff = datos["payoff"]
secciones.append(("Datos (rnn_01_datos.py)", [
    f"{fmt_es(len(payoff))} trayectorias independientes de Montecarlo (montecarlo_independiente.npz), "
    f"repartidas por trayectorias en {fmt_es(len(datos['idx_train']))} de entrenamiento, "
    f"{fmt_es(len(datos['idx_val']))} de validación y {fmt_es(len(datos['idx_test']))} de test. Prima en t0 por "
    f"Montecarlo: {fmt_es(payoff.mean() * np.exp(-mc.R * mc.TTM), 6)}; el {pct((payoff > 0).mean(), 3)} de las "
    f"trayectorias acaba dentro del dinero. Entrada de la red: los 3 retornos logarítmicos y el tiempo en cada uno "
    f"de los 20 pasos. Salidas: el payoff descontado a cada fecha (regresión) y si acaba dentro del dinero "
    f"(clasificación).",
]))

# 2. Estructura elegida y papel de la bidireccional
causal = eleccion["mejor_causal"]
bi = reg.loc["LSTM bidireccional"]
secciones.append(("Estructura elegida y papel de la bidireccional (rnn_02_arquitecturas.py)", [
    eleccion["texto_global"],
    eleccion["texto_causal"],
    f"Papel de la bidireccional: en el paso 20 su error está entre los mejores (RMSE de validación "
    f"{fmt_es(bi['val_rmse_paso20'], 4)}, frente a {fmt_es(reg.loc[causal, 'val_rmse_paso20'], 4)} de la {causal}), "
    f"pero su pérdida de los 20 pasos es {fmt_es(bi['val_perdida_20_pasos'], 4)}, frente a "
    f"{fmt_es(reg.loc[causal, 'val_perdida_20_pasos'], 4)} de la {causal}: en los pasos intermedios ya ha visto el "
    f"resto de la trayectoria, es decir, fechas futuras. Vale para el payoff al vencimiento, pero no para la prima "
    f"en cada fecha, que es lo que hay que valorar.",
]))

# 3. Mejora frente a la media, la lineal y el MLP
cfg = config["configuracion"]
nombre_final = f"{config['estructura']} {cfg['n_oculta']}x{cfg['n_capas']}"
r_final = config["test_regresion"]["rmse"]
r_media, r_lineal, r_mlp = (reg.loc[m, "test_rmse"] for m in ("Media de entrenamiento", "Regresión lineal",
                                                               "MLP [64, 32]"))
secciones.append(("Mejora frente a la media, la regresión lineal y el MLP (payoff al vencimiento, test)", [
    f"RMSE del payoff en test: media de entrenamiento {fmt_es(r_media, 4)}, regresión lineal {fmt_es(r_lineal, 4)}, "
    f"MLP {fmt_es(r_mlp, 4)} y red final ({nombre_final}) {fmt_es(r_final, 4)}. La red final reduce el error un "
    f"{pct(1 - r_final / r_media)} frente a la media y un {pct(1 - r_final / r_lineal)} frente a la regresión "
    f"lineal. En el vencimiento el MLP tiene un {pct(1 - r_mlp / r_final)} menos de error que la red final, por las "
    f"razones del punto 2; la red final se elige por la prima en cada fecha, que el MLP no puede dar.",
]))

# 4. Clasificación
secciones.append(("Clasificación: acabar o no dentro del dinero (rnn_02_arquitecturas.py)",
                  [eleccion["texto_clasificacion"]]))

# 5 y 6. Hiperparámetros, validación cruzada y test
secciones.append(("Hiperparámetros y validación cruzada (rnn_03_hiperparametros.py)",
                  [config["textos"]["fase1"], config["textos"]["fase2"], config["textos"]["validacion_cruzada"]]))
secciones.append(("Métricas de test de los modelos finales (rnn_03_hiperparametros.py)", [config["textos"]["test"]]))

# 7. Prima por fecha
filas = ["   k  t (años)  prima red  prima MC  error %  N_eq red"]
for k, f in tabla_primas.iterrows():
    n_eq = "-" if pd.isna(f["n_eq_red"]) else fmt_es(f["n_eq_red"], 0 if f["n_eq_red"] >= 10 else 2)
    filas.append(f"  {k:2d}  {fmt_es(f['t_anios'], 2):>8s}  {fmt_es(f['prima_red'], 4):>9s}  "
                 f"{fmt_es(f['prima_mc'], 4):>8s}  {fmt_es(f['error_pct_red'], 1):>7s}  {n_eq:>8s}")
secciones.append(("Prima en cada fecha: red frente a Montecarlo y frente a la fórmula (rnn_04_prima.py)",
                  [prima["textos"]["t0"], prima["textos"]["fechas"], prima["textos"]["referencia"],
                   "Tabla por fecha (k = 0 es t0; completa en outputs/rnn_tabla_primas.csv):\n" + "\n".join(filas)]))

# 8. Coste
secciones.append(("Coste computacional y puntos de equilibrio (rnn_05_coste.py)", [coste["texto"]]))

# 9. Hallazgo
secciones.append(("Hallazgo [H1]: el Montecarlo del notebook de BBVA no valora el payoff del reto", [
    "Qué es: la función MonteCarlo de MonteCarlo_y_Formula_cerrada_25_26.ipynb calcula el payoff como "
    "(G_CAT^(1/3) + G_MRK^(1/3) + G_MSFT^(1/3) - 3)+, que no es ni el payoff del enunciado, "
    "(G_CAT + G_MRK + G_MSFT - 3)+, ni la cesta geométrica que valora su fórmula cerrada, "
    "((G_CAT · G_MRK · G_MSFT)^(1/3) - 1)+. Además, su bucle de comparación con la fórmula le pasa la historia "
    "desplazada una fecha.",
    "Dónde se ve: en la línea del payoff, np.prod(rets, axis=(2))**(1/(num_assets * (num_asian_dates-1))), que eleva "
    "el producto temporal de cada activo a 1/60 antes de sumar los tres activos; y en simu2[1, :, :(i+1)], cuya "
    "primera columna es la fecha 1 y no la fecha 0.",
    "Cómo lo vimos: leyendo el código al copiar la fórmula cerrada a packages/formula_cerrada_bbva.py. Por eso no "
    "usamos ese bucle: nuestro control compara la cesta geométrica por Montecarlo condicionado con la fórmula, "
    "con la historia bien alineada (punto 7).",
]))

# 10. Lección de la prueba rápida
fase1_completa = busqueda[busqueda["fase"] == 1]["val_perdida_20_pasos"]
fase1_rapida = busqueda_rapida[busqueda_rapida["fase"] == 1]
agotadas = int((busqueda_rapida["epocas"] >= 20).sum())
secciones.append(("Lección del proceso: --rapido sirve para comprobar el código, no para decidir", [
    f"La prueba --rapido de rnn_03 (20.000 trayectorias, 20 épocas como máximo) eligió la "
    f"{rapido['busqueda']['fase1_elegida'].split(',')[0]}, la mayor ({fmt_es(fase1_rapida['parametros'].max())} "
    f"parámetros), porque ninguna configuración convergía: {agotadas} de las {len(busqueda_rapida)} agotaron las 20 "
    f"épocas, con la mejor época entre la {busqueda_rapida['mejor_epoca'].min()} y la "
    f"{busqueda_rapida['mejor_epoca'].max()}, y las redes grandes aprenden más por época. En la ejecución completa "
    f"ganó la {config['busqueda']['fase1_elegida'].split(',')[0]}, y las seis configuraciones de la fase 1 quedaron "
    f"a menos de {fmt_es(fase1_completa.max() - fase1_completa.min(), 4)} de pérdida entre sí.",
]))

# 11. Mejoras posibles
mlp = reg.loc["MLP [64, 32]"]
f = prima["fechas"]
secciones.append(("Mejoras posibles", [
    f"- Una GRU many-to-one entrenada solo con el payoff (salida solo en el paso 20), para compararla con el MLP en "
    f"igualdad de condiciones: en rnn_02 las recurrentes reparten su pérdida entre los 20 pasos y el MLP solo "
    f"aprende el payoff (RMSE de test {fmt_es(r_mlp, 4)} frente a {fmt_es(r_final, 4)} de la red final).",
    f"- Añadir como entrada la media geométrica parcial acumulada (log G_i hasta t_k): es la cantidad de la que "
    f"depende el payoff, y la red no tendría que reconstruirla sumando retornos. Debería bajar el error de la prima "
    f"en las fechas intermedias (ahora, {fmt_es(f['error_abs_medio_red'], 4)} de error absoluto medio, un "
    f"{fmt_es(f['error_pct_red'], 1)} % de la prima).",
    f"- Ejecutar en GPU: en la CPU del portátil, el entrenamiento final tardó {fmt_es(coste['entrenamiento_s'], 0)} s "
    f"y el desarrollo completo {fmt_es(coste['coste_desarrollo_s'] / 60, 0)} min. Bajarían, y con ellos los puntos "
    f"de equilibrio del coste.",
    f"- Más épocas para el MLP: en rnn_02 agotó las {mlp['epocas']:.0f}, con la mejor en la {mlp['mejor_epoca']:.0f}, "
    f"así que todavía podía mejorar.",
    f"- Corregir el sesgo de la red en t0 ({fmt_es(prima['t0']['sesgo_red'], 4)} ± "
    f"{fmt_es(1.96 * prima['t0']['ee_sesgo_red'], 4)} frente al payoff real de test), por ejemplo dando más peso en "
    f"la pérdida al paso 20, donde la etiqueta es exacta.",
    f"- Combinar red y Montecarlo según la fecha: en {len(coste['fechas_mc_equivalente_mas_barato_que_red_1'])} de "
    f"las 19 fechas, un Montecarlo con la precisión de la red es más barato que valorar una historia con la red.",
]))

# 12. Pendiente
secciones.append(("Pendiente", [
    "- Issue #39 (comparar con el dataset proporcionado): pendiente hasta que el profesor confirme qué dataset es.",
]))

# ============================ COMPOSICIÓN ============================
lineas = ["RESUMEN DE LA RED NEURONAL RECURRENTE - Reto 9 (BBVA), issues #30 a #38",
          f"Generado por rnn_06_resumen.py el {date.today().strftime('%d/%m/%Y')} a partir de outputs/.", ""]
for numero, (titulo, parrafos) in enumerate(secciones, start=1):
    lineas += [f"{numero}. {titulo}", "-" * min(100, len(titulo) + 4)]
    for parrafo in parrafos:
        if "\n" in parrafo:                 # tablas: se respetan las líneas
            lineas.append(parrafo)
        else:
            # Espacio no separable antes de % para que el ajuste de líneas no separe el número de su signo
            lineas.append(textwrap.fill(parrafo.replace(" %", " %"), width=100,
                                        subsequent_indent="  " if parrafo.startswith("- ") else ""))
        lineas.append("")
resumen = "\n".join(lineas)
RUTA_SALIDA.write_text(resumen, encoding="utf-8")
print(resumen)
print(f"Guardado: {RUTA_SALIDA}")
