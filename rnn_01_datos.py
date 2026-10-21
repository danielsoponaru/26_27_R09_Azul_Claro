"""Datos de la red recurrente: entradas, objetivos, división y escalado (issue #30).

Entradas:
    Datos/Transformados/montecarlo_independiente.npz   trayectorias de Montecarlo ("Monte carlo redes.ipynb")
    outputs/parametros_montecarlo.json                 sigma y rho de la cartera (seleccion_cartera.ipynb)
Salidas:
    Datos/Transformados/rnn_dataset.npz   entradas, objetivos, índices de la división y escalado (no se sube)
    Graficos/rnn_datos_payoff.png         distribución del payoff al vencimiento
Ejecución: python rnn_01_datos.py (las rutas son relativas a la raíz del repositorio).

De dónde salen los datos
    sigma y rho se estimaron en seleccion_cartera.ipynb sobre el CSV limpio
    (Datos/Transformados/dowjones_diario_2016_2025.csv): 2.513 rendimientos logarítmicos diarios de
    CAT, MRK y MSFT, del 05/01/2016 al 31/12/2025, anualizados con 252 sesiones. Se guardaron en
    outputs/parametros_montecarlo.json. packages/montecarlo.py no lee ese JSON: lleva los valores
    copiados como constantes (SIGMA y RHO). "Monte carlo redes.ipynb" simula con ese módulo y exporta
    el npz. El recorrido es JSON -> copia en montecarlo.py -> notebook -> npz, y por eso lo primero que
    hace este script es comprobar que los valores del módulo y los del JSON coinciden: si alguien
    actualizara el JSON y no el módulo, la red aprendería con una cartera que ya no es la nuestra.

Por qué trayectorias independientes
    El notebook exporta dos muestras. En montecarlo_condicionado.npz los 5 primeros pasos (1,25 años)
    son el mismo tramo común en las 100.000 trayectorias: en esos pasos la red no vería ninguna
    variación y solo aprendería una historia concreta, cuya prima condicionada es 0,154 frente a la
    prima incondicional de 0,295. En montecarlo_independiente.npz los 20 pasos son independientes
    entre trayectorias, que es lo que necesita una red que tiene que valorar cualquier historia.

Entrada de la red: forma (NUM_SIMS, 20, 4)
    Los 3 retornos logarítmicos de cada paso (retornos_log(X)) y un 4.º canal con el tiempo j/20.
    - El enunciado pide modelar la relación entre los retornos simulados y el payoff. Es una serie
      temporal multivariante de longitud fija (20 pasos trimestrales), así que no hacen falta
      ventanas deslizantes ni padding.
    - Retornos y no precios: los retornos tienen la misma escala en todos los pasos, mientras que los
      precios se dispersan con el tiempo. El payoff depende de los retornos acumulados,
      log G_k = (1/20) · suma_j (21 - j) · r_kj, así que la red tiene que ir acumulándolos paso a
      paso, que es lo que hace una red recurrente con su estado oculto.
    - Canal de tiempo: la prima en el paso j depende de cuánto falta para el vencimiento (descuento y
      fechas de la media que quedan por observar). Una red recurrente podría deducir la posición
      contando pasos, pero dársela explícita le ahorra aprender ese contador.

Objetivos
    - Regresión: Y (NUM_SIMS, 20), el payoff descontado a cada fecha t_k. En el paso 20 es el payoff.
    - Clasificación: itm = (payoff > 0), repetido en los 20 pasos. Los entregables piden errores de
      clasificación, y acabar dentro del dinero es la decisión de ejercer: en el paso t la red estima
      la probabilidad (neutral al riesgo) de ejercicio con lo observado hasta t.
    - Cesta geométrica: payoff_geo = max((G_CAT · G_MRK · G_MSFT)^(1/3) - 1, 0) e
      Y_geo = payoff_geo · exp(-R · (TTM - t_k)). Es el payoff que valora exactamente la fórmula
      cerrada de BBVA (packages/formula_cerrada_bbva.py); con él se entrena la red de contraste de
      rnn_04_prima.py, que así se puede medir contra un precio exacto.

División: 70 % entrenamiento, 15 % validación y 15 % test, por trayectorias (semilla 42)
    La unidad es la trayectoria completa: sus 20 pasos van siempre juntos. Las trayectorias son
    independientes entre sí, así que un reparto aleatorio da tres muestras de la misma distribución.
    La trayectoria 0 (simulación 1) va siempre a test: es la historia de referencia de
    Datos/Transformados/prima_por_instante.npz, con la que se compara la prima de la red, y no puede
    entrar ni en el entrenamiento ni en el cálculo del escalado.

Escalado (como en clase, calculado solo con entrenamiento)
    - Entradas: media y desviación de cada canal, con los 20 pasos de las trayectorias de entrenamiento.
    - Y: una sola media y una sola desviación para los 20 pasos. Así se conserva la relación entre
      pasos (cada Y[:, k] es el mismo payoff con otro descuento) y la pérdida sumada pesa todos los
      pasos en las mismas unidades. Lo mismo para Y_geo, con sus propios parámetros.
    - itm no se escala: es 0/1 y se entrena con BCEWithLogitsLoss.
    Validación y test no intervienen en el escalado para que no influyan en nada de lo que ve la red.
    En el npz se guardan los datos sin escalar y los parámetros; los scripts de entrenamiento aplican
    el escalado.
"""
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from packages import montecarlo as mc
from packages.bbva_graficos import BBVA, configurar_graficos, formato_es, guardar_mpl, imprimir

NUM_SIMS = 100_000                  # trayectorias de montecarlo_independiente.npz
SEMILLA = 42
PROP_TRAIN, PROP_VAL = 0.70, 0.15   # el 15 % restante es test
CANALES = ["ret_log_CAT", "ret_log_MRK", "ret_log_MSFT", "tiempo_j/20"]

os.chdir(Path(__file__).resolve().parent)   # rutas relativas a la raíz del repo, se lance desde donde se lance
RUTA_MC = Path("Datos/Transformados/montecarlo_independiente.npz")
RUTA_PARAMETROS = Path("outputs/parametros_montecarlo.json")
RUTA_SALIDA = Path("Datos/Transformados/rnn_dataset.npz")

# --- 1. sigma y rho: el módulo tiene que coincidir con el JSON ---
parametros = json.loads(RUTA_PARAMETROS.read_text(encoding="utf-8"))
assert np.array_equal(np.ravel(parametros["sigma"]), mc.SIGMA), \
    "PARA: SIGMA de packages/montecarlo.py no coincide con outputs/parametros_montecarlo.json"
assert np.array_equal(np.array(parametros["rho"]), mc.RHO), \
    "PARA: RHO de packages/montecarlo.py no coincide con outputs/parametros_montecarlo.json"
imprimir("sigma y rho de packages/montecarlo.py coinciden con outputs/parametros_montecarlo.json")

# --- 2. Carga del npz ---
assert RUTA_MC.exists(), f"PARA: falta {RUTA_MC}; hay que ejecutar antes 'Monte carlo redes.ipynb'"
datos = np.load(RUTA_MC, allow_pickle=False)
imprimir(f"\nClaves de {RUTA_MC}:")
for clave in datos.files:
    imprimir(f"  {clave:20s} {str(datos[clave].shape):16s} {datos[clave].dtype}")

X = datos["X"]
retornos = datos["retornos_log"]
G = datos["medias_geometricas"]
payoff = datos["payoff"]
Y = datos["Y"]
tiempos = datos["tiempos"]

# --- 3. Comprobaciones obligatorias ---
assert X.shape == (NUM_SIMS, mc.NUM_PASOS, 3), f"PARA: X tiene forma {X.shape}"
assert np.array_equal(datos["sigma"], mc.SIGMA) and np.array_equal(datos["rho"], mc.RHO), \
    "PARA: el npz se simuló con otros sigma o rho"
assert np.array_equal(retornos, mc.retornos_log(X)) and np.array_equal(G, mc.medias_geometricas(X)) \
    and np.array_equal(tiempos, mc.TIEMPOS), "PARA: el npz no es coherente con packages/montecarlo.py"
prima_t0 = payoff.mean() * np.exp(-mc.R * mc.TTM)
pct_itm = 100 * (payoff > 0).mean()
assert round(prima_t0, 6) == 0.295072, f"PARA: prima en t0 = {prima_t0:.6f}, se esperaba 0,295072"
assert round(pct_itm, 3) == 44.129, f"PARA: payoff > 0 en el {pct_itm:.3f} %, se esperaba 44,129 %"
assert np.array_equal(Y[:, -1], payoff), "PARA: Y[:, -1] no es el payoff"
imprimir(f"\nPrima en t0 (media del payoff descontado): {prima_t0:.6f}   (esperada 0.295072)")
imprimir(f"Trayectorias con payoff > 0 (itm): {pct_itm:.3f}%   (esperado 44.129%)")
imprimir("Y[:, -1] == payoff en las 100,000 trayectorias")

# --- 4. Entradas y objetivos ---
canal_tiempo = np.broadcast_to(tiempos / mc.TTM, (NUM_SIMS, mc.NUM_PASOS))[..., None]   # j/20
entradas = np.concatenate((retornos, canal_tiempo), axis=2)
itm = np.repeat((payoff > 0)[:, None], mc.NUM_PASOS, axis=1).astype(np.float32)
payoff_geo = np.maximum(np.prod(G, axis=1) ** (1 / 3) - 1, 0)
Y_geo = mc.calcular_Y(payoff_geo)

# --- 5. División por trayectorias; la 0 siempre a test ---
rng = np.random.default_rng(SEMILLA)
resto = rng.permutation(np.arange(1, NUM_SIMS))
n_train, n_val = int(PROP_TRAIN * NUM_SIMS), int(PROP_VAL * NUM_SIMS)
idx_train = np.sort(resto[:n_train])
idx_val = np.sort(resto[n_train:n_train + n_val])
idx_test = np.sort(np.concatenate(([0], resto[n_train + n_val:])))
assert 0 in idx_test and len(np.unique(np.concatenate((idx_train, idx_val, idx_test)))) == NUM_SIMS

# --- 6. Escalado con entrenamiento ---
media_entradas = entradas[idx_train].reshape(-1, len(CANALES)).mean(axis=0)
desv_entradas = entradas[idx_train].reshape(-1, len(CANALES)).std(axis=0)
media_Y, desv_Y = Y[idx_train].mean(), Y[idx_train].std()
media_Y_geo, desv_Y_geo = Y_geo[idx_train].mean(), Y_geo[idx_train].std()

np.savez(
    RUTA_SALIDA,
    entradas=entradas, Y=Y, itm=itm, payoff=payoff, payoff_geo=payoff_geo, Y_geo=Y_geo, X=X,
    tiempos=tiempos, canales=np.array(CANALES),
    idx_train=idx_train, idx_val=idx_val, idx_test=idx_test,
    media_entradas=media_entradas, desv_entradas=desv_entradas,
    media_Y=media_Y, desv_Y=desv_Y, media_Y_geo=media_Y_geo, desv_Y_geo=desv_Y_geo,
    semilla=SEMILLA,
)

# --- 7. Resumen ---
imprimir("\nFormas:")
for nombre, valor in (("entradas", entradas), ("Y", Y), ("itm", itm), ("payoff", payoff),
                      ("payoff_geo", payoff_geo), ("Y_geo", Y_geo), ("X", X)):
    imprimir(f"  {nombre:12s} {valor.shape}")

imprimir("\nDivisión por trayectorias:")
for nombre, idx in (("Entrenamiento", idx_train), ("Validación", idx_val), ("Test", idx_test)):
    imprimir(f"  {nombre:14s} {len(idx):>7,} trayectorias ({100 * len(idx) / NUM_SIMS:.0f}%), "
             f"payoff > 0 en el {100 * (payoff[idx] > 0).mean():.2f}%")

imprimir("\nEscalado de las entradas (solo entrenamiento):")
imprimir(pd.DataFrame({"Media": media_entradas, "Desviación": desv_entradas}, index=CANALES)
         .to_string(float_format=lambda v: f"{v:.6f}"))
imprimir(f"Y:     media {media_Y:.6f}, desviación {desv_Y:.6f}")
imprimir(f"Y_geo: media {media_Y_geo:.6f}, desviación {desv_Y_geo:.6f}")

imprimir("\nY por paso (las 100,000 trayectorias):")
tabla_pasos = pd.DataFrame({
    "Paso": np.arange(1, mc.NUM_PASOS + 1),
    "t (años)": tiempos,
    "Media de Y": Y.mean(axis=0),
    "Desviación de Y": Y.std(axis=0),
})
imprimir(tabla_pasos.to_string(index=False, formatters={
    "t (años)": "{:.2f}".format, "Media de Y": "{:.6f}".format, "Desviación de Y": "{:.6f}".format}))

imprimir(f"\nGuardado: {RUTA_SALIDA} ({RUTA_SALIDA.stat().st_size / 1e6:.0f} MB)")

# --- 8. Gráfico: payoff positivo, mediana y porcentaje de ceros ---
configurar_graficos()
positivos = payoff[payoff > 0]
mediana = np.median(positivos)
p99 = np.percentile(positivos, 99)

fig, ax = plt.subplots()
ax.hist(positivos, bins=30, range=(0, p99), color=BBVA["principal"], edgecolor=BBVA["blanco"])
ax.axvline(mediana, color=BBVA["agua"], linewidth=2.5)
ax.text(mediana, ax.get_ylim()[1] * 0.97, f"  Mediana del payoff positivo: {mediana:.3f}",
        color=BBVA["marino"], fontsize=12, fontweight="bold", va="top")
ax.text(0.98, 0.75,
        f"Payoff = 0 (fuera del dinero): {100 - pct_itm:.3f}% de las trayectorias\n"
        f"Payoff > 0 (dentro del dinero): {pct_itm:.3f}%, las del histograma\n"
        f"Eje cortado en el percentil 99 ({p99:.2f}); el máximo es {positivos.max():.2f}",
        transform=ax.transAxes, ha="right", va="top", multialignment="left", fontsize=11, color=BBVA["marino"],
        bbox=dict(boxstyle="round", fc=BBVA["gris"], ec=BBVA["gris_dato"]))
ax.set_title(f"Gráfico 1. El {100 - pct_itm:.1f}% de las trayectorias acaba sin pago")
ax.set_xlabel("Payoff al vencimiento (precios normalizados, S0 = 1)")
ax.set_ylabel("Nº de trayectorias")
fig.tight_layout()
formato_es(fig, fuente="100.000 trayectorias independientes de Montecarlo (semilla 42) con σ y ρ de 2016-2025.")
guardar_mpl(fig, "rnn_datos_payoff")
plt.close(fig)
