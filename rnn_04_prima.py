"""Prima en cada fecha: la red frente a Montecarlo y frente a la fórmula cerrada de BBVA (issue #37).

Entradas:
    Datos/Transformados/rnn_dataset.npz          (rnn_01_datos.py): escalado, test y payoffs
    Datos/Transformados/prima_por_instante.npz   ("Monte carlo redes.ipynb"): prima condicionada de la
                                                 trayectoria de referencia del equipo
    modelos/rnn_final_regresion.pt, modelos/rnn_final_clasificacion.pt   (rnn_03_hiperparametros.py)
    outputs/rnn_configuracion_final.json         configuración de la red final (rnn_03)
Salidas:
    outputs/rnn_tabla_primas.csv                 entregable "predicción del valor presente en el momento t":
                                                 una fila por fecha
    outputs/rnn_primas_por_historia.csv          el detalle de cada historia
    outputs/rnn_prima_resumen.json               prima en t0, errores globales y textos
    modelos/rnn_geometrica.pt                    red geométrica (no se sube)
    Graficos/rnn_prima_por_fecha.png, rnn_error_por_fecha.png, rnn_probabilidad_ejercicio.png,
             rnn_prima_trayectoria_referencia.png
Ejecución: python rnn_04_prima.py [--rapido]
    --rapido: la red geométrica se entrena con 20.000 trayectorias y 20 épocas como máximo, y el
    Montecarlo condicionado usa 20.000 continuaciones. Todo va a outputs/rapido/. Las redes finales
    se leen siempre de la ejecución completa de rnn_03.

Dos redes y dos referencias
    - Red principal: la red causal final de rnn_03 (payoff del enunciado). No hay fórmula para ese
      payoff, así que su referencia es el Montecarlo condicionado del equipo
      (prima_condicionada de packages/montecarlo.py), que tiene su propio error estándar.
    - Red geométrica: misma arquitectura e hiperparámetros, entrenada con Y_geo, el payoff de la cesta
      geométrica. La fórmula de BBVA es exacta solo para esa cesta (packages/formula_cerrada_bbva.py),
      así que con esta red se mide el error de una red contra un precio exacto, sin el ruido de
      Montecarlo en la referencia.

Prima en t0
    La red no tiene salida en t0: no ha visto ningún paso. Su prima en t0 es la media de su payoff
    predicho en las trayectorias de test, descontada a t0, es decir, un Montecarlo que usa la red como
    función de payoff. Se compara con 0,295072 (Montecarlo de 100.000 trayectorias) y además con la
    media del payoff real de esas mismas trayectorias de test: la segunda diferencia es solo error de
    la red, y la primera mezcla ese error con el de usar 15.000 trayectorias en vez de 100.000.
    Como las dos medias salen de las mismas trayectorias, la incertidumbre de la segunda diferencia es
    el error estándar de la diferencia pareada (desviación de los residuos / raíz de n), mucho menor
    que el error de muestreo de 15.000 trayectorias frente a la prima verdadera.
    La red geométrica, igual, frente a la fórmula en t0 (0,083164).
    Recorte a >= 0: cada payoff predicho se recorta a >= 0 porque un payoff no puede ser negativo. En
    las trayectorias que acaban fuera del dinero (el 56 %) la red predice valores alrededor de 0, y
    recortar los negativos sube la media. Por eso el script da también la prima sin recortar y cuánto
    del sesgo explica el recorte. En la ejecución completa explica solo una parte pequeña del sesgo de
    la red principal (+0,0011 de +0,0066) y en la red geométrica lo reduce, así que se mantiene la
    media de los payoffs recortados, coherente con el resto de predicciones del script, y la diferencia
    que queda se atribuye a un sesgo propio de la red.

Prima en t_k, k = 1..19
    Para cada k se generan 20 historias distintas, cada una con los k primeros pasos de
    simular_continuaciones(None, 1, default_rng(semilla propia)): 20 situaciones de mercado distintas
    en cada fecha. La red causal recibe esos k pasos (retornos logarítmicos y canal de tiempo, con el
    escalado de entrenamiento) y su salida en el último paso es su estimación de la prima en t_k.
    - Referencia de la red principal: prima_condicionada(historia, NUM_SIMS, rng), con NUM_SIMS
      continuaciones desde la historia.
    - Referencia de la red geométrica: BasketGeomAsian(21, k, R, 3, SIGMA, RHO, TTM, price_history,
      True) con price_history = np.hstack([np.ones((3, 1)), historia.T]): la columna 0 es la fecha 0
      (S0 = 1). No se copia el bucle de comparación del notebook de BBVA, que pasa la historia
      desplazada una fecha.
    - Control: con las mismas continuaciones se calcula la cesta geométrica por Montecarlo y se
      compara con la fórmula. Si el Montecarlo condicionado y la convención de price_history son
      correctos, deben coincidir dentro de ±1,96 errores estándar en torno al 95 % de las historias.
    - Clasificación: la probabilidad de ejercicio de la red frente a la proporción de continuaciones
      con payoff > 0.

Errores
    Por historia: diferencia con la referencia, y si cae dentro de ±1,96 errores estándar del
    Montecarlo. Si la red fuera tan precisa como un Montecarlo de NUM_SIMS continuaciones, en torno al
    95 % de sus errores caerían dentro. Por fecha: error absoluto medio y error relativo, calculado
    como error absoluto medio dividido entre la prima media de esa fecha. Se agrega así porque las
    historias muy fuera del dinero tienen primas casi nulas y un error relativo por historia se
    dispararía sin decir nada útil.
    La banda de ±1,96 errores estándar supone que la media de Montecarlo es aproximadamente normal.
    En historias muy fuera del dinero cerca del vencimiento, el payoff positivo es un suceso raro: con
    pocas continuaciones positivas la media es muy asimétrica y la banda deja de valer, y sin ninguna
    el Montecarlo da 0 con error estándar 0. Por eso los porcentajes dentro de la banda solo se
    calculan con las historias que tienen al menos 50 continuaciones con payoff positivo; las demás
    se cuentan aparte en el texto de resultados.

Tamaño de Montecarlo equivalente (N_eq), por fecha y global
    Un Montecarlo de N continuaciones estima la prima con varianza σ²/N, con σ la desviación del payoff
    descontado de las continuaciones de la historia (σ² = error estándar² · NUM_SIMS). El error
    cuadrático de la red frente al Montecarlo de referencia suma el error de la red y el ruido de ese
    Montecarlo, que es independiente: media(error²) = error real de la red + media(σ²/NUM_SIMS). N_eq es
    el número de continuaciones con el que un Montecarlo tendría el mismo error que la red:
        N_eq = media(σ²) / (media(error²) - media(σ²/NUM_SIMS)).
    Para la red geométrica la referencia es la fórmula, que es exacta, así que no hay ruido que restar:
    N_eq = media(σ²) / media(error²). N_eq cuenta todo el error de la red, sesgo incluido. Si el error
    de la red no supera el ruido de la referencia, no se puede estimar y queda vacío. En t0 no se
    define, porque la prima de la red sale de promediar su payoff en test y no de 20 historias.
    rnn_05_coste.py lo usa para comparar el coste de la red con el de un Montecarlo de su misma precisión.

Trayectoria de referencia del equipo
    prima_por_instante.npz trae la prima condicionada en k = 0..19 de la trayectoria 1 (índice 0, que
    rnn_01 dejó en test). Como la red es causal, una sola pasada por la trayectoria da su prima en
    todas las fechas a la vez.
"""
import argparse
import json
import os
import time
from pathlib import Path

# MKL (la librería de cálculo de numpy y scipy en este entorno) en modo secuencial. Si no, carga su propia
# copia de la librería OpenMP de Intel, que choca con la que trae PyTorch (OMP: Error #15) en cuanto la
# fórmula cerrada (scipy) se usa después de entrenar. Tiene que fijarse antes de importar numpy. Las
# operaciones de numpy de este script son pequeñas, y el Montecarlo da los mismos números (0,295072 en t0).
os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D
from torch import nn

from packages import montecarlo as mc
from packages.bbva_graficos import BBVA, configurar_graficos, formato_es, guardar_mpl, imprimir, texto_es
from packages.formula_cerrada_bbva import BasketGeomAsian
from packages.rnn_modelos import (DISPOSITIVO, RUTA_DATASET, RedRecurrente, cargar_datos, entrenar, evaluar,
                                  salida_bruta)

parser = argparse.ArgumentParser(description="Prima por fecha: red frente a Montecarlo y fórmula")
parser.add_argument("--rapido", action="store_true",
                    help="red geométrica con 20.000 trayectorias y 20 épocas; Montecarlo de 20.000 continuaciones")
args = parser.parse_args()

SEMILLA = 42
NUM_SIMS = 20_000 if args.rapido else 100_000      # continuaciones del Montecarlo condicionado
N_MAX = 20_000 if args.rapido else None
ENTRENAMIENTO = dict(batch=512, max_epocas=20 if args.rapido else 100, paciencia=10, min_delta=1e-6)
N_HISTORIAS = 20
FECHAS = range(1, mc.NUM_PASOS)                    # k = 1..19
Z95 = 1.96
MIN_POSITIVAS = 50          # continuaciones con payoff > 0 para que la banda de ±1,96 errores estándar valga

inicio_script = time.perf_counter()
os.chdir(Path(__file__).resolve().parent)   # rutas relativas a la raíz del repo
CARPETA_SALIDA = Path("outputs/rapido") if args.rapido else Path("outputs")
CARPETA_MODELOS = CARPETA_SALIDA if args.rapido else Path("modelos")
CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)
CARPETA_MODELOS.mkdir(exist_ok=True)
configurar_graficos(CARPETA_SALIDA if args.rapido else "Graficos")
FUENTE = (f"historias simuladas con packages/montecarlo.py; Montecarlo condicionado de "
          f"{'20.000' if args.rapido else '100.000'} continuaciones; fórmula de BBVA para la cesta geométrica.")

config = json.loads(Path("outputs/rnn_configuracion_final.json").read_text(encoding="utf-8"))
assert not config["rapido"], "PARA: outputs/rnn_configuracion_final.json no es de la ejecución completa de rnn_03"
imprimir(f"Modo: {'rápido' if args.rapido else 'completo'}; Montecarlo condicionado de {NUM_SIMS:,} continuaciones; "
         f"red: {config['estructura']} {config['configuracion']['n_oculta']}x{config['configuracion']['n_capas']}; "
         f"dispositivo: {DISPOSITIVO}")

d = np.load(RUTA_DATASET, allow_pickle=False)
media_x, desv_x = d["media_entradas"], d["desv_entradas"]


def n_equivalente(sigma2, error2, ruido_referencia):
    """Continuaciones de un Montecarlo con el mismo error que la red: media(σ²) / (media(error²) - ruido).

    Si el error de la red no supera el ruido de la referencia, no se puede estimar (NaN).
    """
    exceso = error2.mean() - ruido_referencia
    return float(sigma2.mean() / exceso) if exceso > 0 else np.nan


def cargar_red(ruta):
    g = torch.load(ruta, weights_only=True)
    modelo = RedRecurrente(**g["config"])
    modelo.load_state_dict(g["estado"])
    return modelo, g


def entradas_historias(historias):
    """Historias (n, k, 3) de precios normalizados -> entradas escaladas (n, k, 4), como en rnn_01."""
    n, k, _ = historias.shape
    tiempo = np.broadcast_to(mc.TIEMPOS[:k] / mc.TTM, (n, k))[..., None]
    entradas = np.concatenate((mc.retornos_log(historias), tiempo), axis=2)
    return ((entradas - media_x) / desv_x).astype(np.float32)


red, g_red = cargar_red("modelos/rnn_final_regresion.pt")
red_clf, _ = cargar_red("modelos/rnn_final_clasificacion.pt")

# ===================== RED GEOMÉTRICA =====================
datos_geo = cargar_datos("Y_geo", N_MAX)
cfg = config["configuracion"]
ruta_geo = CARPETA_MODELOS / "rnn_geometrica.pt"
firma = {"configuracion": cfg, "n_max": N_MAX, **ENTRENAMIENTO}
guardada = torch.load(ruta_geo, weights_only=True) if ruta_geo.exists() else None
if guardada is not None and guardada["firma"] == firma:
    red_geo = RedRecurrente(**guardada["config"])
    red_geo.load_state_dict(guardada["estado"])
    res_geo = guardada["res"]
    imprimir(f"\nRed geométrica recuperada de {ruta_geo}")
else:
    imprimir("\nEntrenando la red geométrica (misma configuración que la final, objetivo Y_geo)")
    config_red = {k: cfg[k] for k in ("tipo", "n_oculta", "n_capas", "dropout", "activacion_salida")}
    torch.manual_seed(SEMILLA)
    red_geo = RedRecurrente(**config_red, minimo_salida=-datos_geo["media_y"] / datos_geo["desv_y"])
    res_geo = entrenar(red_geo, datos_geo, nn.MSELoss(), lr=cfg["lr"], **ENTRENAMIENTO)
    torch.save({"config": config_red, "estado": red_geo.state_dict(), "media_y": datos_geo["media_y"],
                "desv_y": datos_geo["desv_y"], "res": res_geo, "firma": firma}, ruta_geo)
imprimir(f"  {res_geo['epocas']} épocas (mejor {res_geo['mejor_epoca']}), {res_geo['tiempo']:.0f} s")

# ========================= PRIMA EN t0 =========================
descuento_0 = np.exp(-mc.R * mc.TTM)
idx_test, x_test = datos_geo["idx_test"], datos_geo["x_test"]
n_test = len(idx_test)
payoff_test, geo_test = d["payoff"][idx_test], d["payoff_geo"][idx_test]
# Payoff predicho sin recortar y recortado a >= 0 (el recortado es lo que devuelve evaluar())
bruto_payoff = salida_bruta(red, x_test)[:, -1] * g_red["desv_y"] + g_red["media_y"]
bruto_geo = salida_bruta(red_geo, x_test)[:, -1] * datos_geo["desv_y"] + datos_geo["media_y"]
pred_payoff, pred_geo = np.maximum(bruto_payoff, 0), np.maximum(bruto_geo, 0)
formula_0 = float(BasketGeomAsian(21, 0, mc.R, 3, mc.SIGMA.reshape(-1, 1), mc.RHO, mc.TTM, np.ones((3, 1)), True))


def sesgo_pareado(prediccion, real):
    """Diferencia media con el payoff real de las mismas trayectorias y su error estándar pareado, en t0."""
    residuos = (prediccion - real) * descuento_0
    return float(residuos.mean()), float(residuos.std(ddof=1) / np.sqrt(len(residuos)))


t0 = {
    "red": float(pred_payoff.mean() * descuento_0),
    "red_sin_recorte": float(bruto_payoff.mean() * descuento_0),
    "real_test": float(payoff_test.mean() * descuento_0),
    "ee_real_test": float(payoff_test.std(ddof=1) * descuento_0 / np.sqrt(n_test)),
    "montecarlo_100000": float(d["payoff"].mean() * descuento_0),
    "ee_montecarlo_100000": float(d["payoff"].std(ddof=1) * descuento_0 / np.sqrt(len(d["payoff"]))),
    "red_geo": float(pred_geo.mean() * descuento_0),
    "red_geo_sin_recorte": float(bruto_geo.mean() * descuento_0),
    "real_geo_test": float(geo_test.mean() * descuento_0),
    "formula": formula_0,
    "prob_red": float(evaluar(red_clf, x_test, clasificacion=True)[:, -1].mean()),
    "prob_real_test": float((payoff_test > 0).mean()),
    "prob_montecarlo_100000": float((d["payoff"] > 0).mean()),
    "n_test": n_test,
}
t0["sesgo_red"], t0["ee_sesgo_red"] = sesgo_pareado(pred_payoff, payoff_test)
t0["sesgo_red_sin_recorte"], _ = sesgo_pareado(bruto_payoff, payoff_test)
t0["sesgo_red_geo"], t0["ee_sesgo_red_geo"] = sesgo_pareado(pred_geo, geo_test)
t0["sesgo_red_geo_sin_recorte"], _ = sesgo_pareado(bruto_geo, geo_test)
assert round(t0["montecarlo_100000"], 6) == 0.295072 and round(formula_0, 6) == 0.083164


def cambio_sesgo(con, sin):
    return f"lo {'reduce' if abs(sin) < abs(con) else 'aumenta'} de {con:+.6f} a {sin:+.6f}"


texto_t0 = (
    f"Prima en t0. Red principal: {t0['red']:.6f}, frente a {t0['montecarlo_100000']:.6f} del Montecarlo de "
    f"100,000 trayectorias. Frente al payoff real de las mismas {n_test:,} trayectorias de test "
    f"({t0['real_test']:.6f}), la diferencia es {t0['sesgo_red']:+.6f} ± {Z95 * t0['ee_sesgo_red']:.6f} (±1.96 "
    f"errores estándar de la diferencia pareada): como sale de las mismas trayectorias, es error de la red, y "
    f"{'es significativo' if abs(t0['sesgo_red']) > Z95 * t0['ee_sesgo_red'] else 'no es significativo'}. "
    f"Red geométrica: {t0['red_geo']:.6f}, frente a {t0['formula']:.6f} de la fórmula; frente a su payoff real "
    f"en test ({t0['real_geo_test']:.6f}), {t0['sesgo_red_geo']:+.6f} ± {Z95 * t0['ee_sesgo_red_geo']:.6f}. "
    f"Recorte a >= 0 de cada payoff predicho: en la red principal, quitarlo {cambio_sesgo(t0['sesgo_red'], t0['sesgo_red_sin_recorte'])} "
    f"(el recorte explica {t0['sesgo_red'] - t0['sesgo_red_sin_recorte']:+.6f}); en la geométrica, "
    f"{cambio_sesgo(t0['sesgo_red_geo'], t0['sesgo_red_geo_sin_recorte'])}. La prima en t0 usa la media "
    f"recortada (ver docstring). Probabilidad de ejercicio: {t0['prob_red']:.4f} según la red y "
    f"{t0['prob_real_test']:.4f} en test.")
imprimir("\n" + texto_t0)

# ===================== PRIMA EN CADA FECHA =====================
imprimir(f"\nPrimas en t_k con {N_HISTORIAS} historias por fecha y {NUM_SIMS:,} continuaciones por historia")
inicio_mc = time.perf_counter()
filas = []
for k in FECHAS:
    semillas = [20_000 + 100 * k + h for h in range(N_HISTORIAS)]
    historias = np.stack([mc.simular_continuaciones(None, 1, np.random.default_rng(s))[0, :k] for s in semillas])
    x = entradas_historias(historias)
    prima_red = evaluar(red, x, g_red["media_y"], g_red["desv_y"])[:, -1]
    prima_geo = evaluar(red_geo, x, datos_geo["media_y"], datos_geo["desv_y"])[:, -1]
    prob_red = evaluar(red_clf, x, clasificacion=True)[:, -1]
    descuento_k = np.exp(-mc.R * (mc.TTM - k * mc.DT))
    for h, (semilla, historia) in enumerate(zip(semillas, historias)):
        semilla_mc = 60_000 + 100 * k + h
        prima_mc, ee_mc = mc.prima_condicionada(historia, NUM_SIMS, np.random.default_rng(semilla_mc))
        # Las mismas continuaciones (misma semilla) para el control geométrico y la probabilidad de ejercicio
        X = mc.simular_continuaciones(historia, NUM_SIMS, np.random.default_rng(semilla_mc))
        payoff = mc.calcular_payoff(X)
        assert np.isclose(payoff.mean() * descuento_k, prima_mc, rtol=1e-12), "continuaciones distintas"
        cesta = np.maximum(np.prod(mc.medias_geometricas(X), axis=1) ** (1 / 3) - 1, 0) * descuento_k
        formula = float(BasketGeomAsian(21, k, mc.R, 3, mc.SIGMA.reshape(-1, 1), mc.RHO, mc.TTM,
                                        np.hstack([np.ones((3, 1)), historia.T]), True))
        ee_geo = cesta.std(ddof=1) / np.sqrt(NUM_SIMS)
        prob_mc = (payoff > 0).mean()
        n_positivas, n_positivas_geo = int((payoff > 0).sum()), int((cesta > 0).sum())
        filas.append({
            "k": k, "t_anios": k * mc.DT, "historia": h, "semilla_historia": semilla, "semilla_mc": semilla_mc,
            "prima_red": float(prima_red[h]), "prima_mc": float(prima_mc), "ee_mc": float(ee_mc),
            "error_red": float(prima_red[h] - prima_mc),
            "red_dentro_banda": bool(abs(prima_red[h] - prima_mc) <= Z95 * ee_mc),
            "prima_red_geo": float(prima_geo[h]), "prima_formula": formula,
            "error_red_geo": float(prima_geo[h] - formula),
            "geo_mc": float(cesta.mean()), "ee_geo_mc": float(ee_geo),
            "control_z": float((cesta.mean() - formula) / ee_geo) if ee_geo > 0 else np.nan,
            "n_positivas": n_positivas, "n_positivas_geo": n_positivas_geo,
            "prob_red": float(prob_red[h]), "prob_mc": float(prob_mc),
            "ee_prob_mc": float(np.sqrt(prob_mc * (1 - prob_mc) / NUM_SIMS)),
            "error_prob": float(prob_red[h] - prob_mc),
        })
    imprimir(f"  k = {k:2d} (t = {k * mc.DT:.2f} años) hecho; {time.perf_counter() - inicio_mc:.0f} s acumulados")
historias_df = pd.DataFrame(filas)
historias_df["control_dentro"] = historias_df["control_z"].abs() <= Z95
# Varianza del payoff descontado de las continuaciones de cada historia: σ² = error estándar² · NUM_SIMS
historias_df["sigma2_mc"] = historias_df["ee_mc"] ** 2 * NUM_SIMS
historias_df["sigma2_geo_mc"] = historias_df["ee_geo_mc"] ** 2 * NUM_SIMS
# La banda de ±1,96 errores estándar solo vale si la media de Montecarlo es aproximadamente normal
valida = historias_df["n_positivas"] >= MIN_POSITIVAS
valida_geo = historias_df["n_positivas_geo"] >= MIN_POSITIVAS

# Tabla por fecha (k = 0 sale de la prima en t0)
agregado = historias_df.groupby("k").agg(
    t_anios=("t_anios", "first"),
    prima_red=("prima_red", "mean"), prima_mc=("prima_mc", "mean"), ee_mc=("ee_mc", "mean"),
    error_abs_red=("error_red", lambda e: e.abs().mean()),
    prima_red_geo=("prima_red_geo", "mean"), prima_formula=("prima_formula", "mean"),
    error_abs_red_geo=("error_red_geo", lambda e: e.abs().mean()),
    geo_mc=("geo_mc", "mean"), ee_geo_mc=("ee_geo_mc", "mean"),
    prob_red=("prob_red", "mean"), prob_mc=("prob_mc", "mean"),
    error_abs_prob=("error_prob", lambda e: e.abs().mean()),
)
agregado["error_pct_red"] = 100 * agregado["error_abs_red"] / agregado["prima_mc"]
agregado["error_pct_red_geo"] = 100 * agregado["error_abs_red_geo"] / agregado["prima_formula"]
# Porcentajes dentro de la banda, solo con las historias en las que la banda es válida
agregado["red_dentro_banda_pct"] = 100 * historias_df[valida].groupby("k")["red_dentro_banda"].mean()
agregado["control_dentro_pct"] = 100 * historias_df[valida_geo].groupby("k")["control_dentro"].mean()
agregado["historias_validas_control"] = valida_geo.groupby(historias_df["k"]).sum()
# Tamaño de Montecarlo equivalente por fecha (la geométrica se mide contra la fórmula, sin ruido que restar)
grupos = dict(tuple(historias_df.groupby("k")))
agregado["n_eq_red"] = pd.Series({k: n_equivalente(g["sigma2_mc"], g["error_red"] ** 2, (g["ee_mc"] ** 2).mean())
                                  for k, g in grupos.items()})
agregado["n_eq_red_geo"] = pd.Series({k: n_equivalente(g["sigma2_geo_mc"], g["error_red_geo"] ** 2, 0.0)
                                      for k, g in grupos.items()})
fila_t0 = pd.DataFrame([{
    "t_anios": 0.0, "prima_red": t0["red"], "prima_mc": t0["montecarlo_100000"], "ee_mc": t0["ee_montecarlo_100000"],
    "error_abs_red": abs(t0["red"] - t0["montecarlo_100000"]),
    "error_pct_red": 100 * abs(t0["red"] - t0["montecarlo_100000"]) / t0["montecarlo_100000"],
    "prima_red_geo": t0["red_geo"], "prima_formula": t0["formula"],
    "error_abs_red_geo": abs(t0["red_geo"] - t0["formula"]),
    "error_pct_red_geo": 100 * abs(t0["red_geo"] - t0["formula"]) / t0["formula"],
    "prob_red": t0["prob_red"], "prob_mc": t0["prob_montecarlo_100000"],
    "error_abs_prob": abs(t0["prob_red"] - t0["prob_montecarlo_100000"]),
}], index=pd.Index([0], name="k"))
tabla = pd.concat([fila_t0, agregado])[[
    "t_anios", "prima_red", "prima_mc", "error_abs_red", "error_pct_red", "ee_mc", "red_dentro_banda_pct",
    "n_eq_red", "prima_red_geo", "prima_formula", "error_abs_red_geo", "error_pct_red_geo", "n_eq_red_geo",
    "geo_mc", "ee_geo_mc",
    "control_dentro_pct", "historias_validas_control", "prob_red", "prob_mc", "error_abs_prob"]]

# Resumen global (k = 1..19)
h_df = historias_df
resumen = {
    "error_abs_medio_red": float(h_df["error_red"].abs().mean()),
    "error_pct_red": float(100 * h_df["error_red"].abs().mean() / h_df["prima_mc"].mean()),
    "historias_validas": int(valida.sum()),
    "red_dentro_banda_pct": float(100 * h_df.loc[valida, "red_dentro_banda"].mean()),
    "ee_mc_medio": float(h_df.loc[valida, "ee_mc"].mean()),
    "error_abs_medio_red_geo": float(h_df["error_red_geo"].abs().mean()),
    "error_pct_red_geo": float(100 * h_df["error_red_geo"].abs().mean() / h_df["prima_formula"].mean()),
    "ee_geo_mc_medio": float(h_df.loc[valida_geo, "ee_geo_mc"].mean()),
    "red_geo_dentro_banda_mc_pct": float(100 * (h_df.loc[valida_geo, "error_red_geo"].abs()
                                                <= Z95 * h_df.loc[valida_geo, "ee_geo_mc"]).mean()),
    "historias_validas_geo": int(valida_geo.sum()),
    "control_dentro_pct": float(100 * h_df.loc[valida_geo, "control_dentro"].mean()),
    "control_pocas_positivas_n": int(((h_df["n_positivas_geo"] > 0) & ~valida_geo).sum()),
    "control_pocas_positivas_dentro_pct": float(100 * h_df.loc[(h_df["n_positivas_geo"] > 0) & ~valida_geo,
                                                                "control_dentro"].mean()),
    "control_sin_positivas_n": int((h_df["n_positivas_geo"] == 0).sum()),
    "control_sin_positivas_formula_max": float(h_df.loc[h_df["n_positivas_geo"] == 0, "prima_formula"].max()),
    "error_abs_medio_prob": float(h_df["error_prob"].abs().mean()),
    "n_eq_red": n_equivalente(h_df["sigma2_mc"], h_df["error_red"] ** 2, (h_df["ee_mc"] ** 2).mean()),
    "n_eq_red_geo": n_equivalente(h_df["sigma2_geo_mc"], h_df["error_red_geo"] ** 2, 0.0),
    "tiempo_montecarlo_s": time.perf_counter() - inicio_mc,
}
texto_fechas = (
    f"Primas en t_k (k = 1..19, {N_HISTORIAS} historias por fecha). Red principal frente al Montecarlo condicionado: "
    f"error absoluto medio {resumen['error_abs_medio_red']:.4f} ({resumen['error_pct_red']:.1f}% de la prima media). "
    f"En las {resumen['historias_validas']} historias en las que la banda es válida (al menos {MIN_POSITIVAS} "
    f"continuaciones con payoff positivo), el {resumen['red_dentro_banda_pct']:.0f}% de sus errores cae dentro de "
    f"±1.96 errores estándar del Montecarlo (error estándar medio {resumen['ee_mc_medio']:.4f}). Red geométrica "
    f"frente a la fórmula: error absoluto medio {resumen['error_abs_medio_red_geo']:.4f} "
    f"({resumen['error_pct_red_geo']:.1f}% de la prima media); el {resumen['red_geo_dentro_banda_mc_pct']:.0f}% cae "
    f"dentro de la banda de un Montecarlo de {NUM_SIMS:,} continuaciones de la misma cesta. Control: en las "
    f"{resumen['historias_validas_geo']} historias válidas, el Montecarlo de la cesta geométrica coincide con la "
    f"fórmula dentro de ±1.96 errores estándar en el {resumen['control_dentro_pct']:.0f}% (lo esperado, en torno al "
    f"95%). En las {resumen['control_pocas_positivas_n']} historias con entre 1 y {MIN_POSITIVAS - 1} continuaciones "
    f"positivas coincide en el {resumen['control_pocas_positivas_dentro_pct']:.0f}%: con un suceso tan raro la media "
    f"de Montecarlo es muy asimétrica y la banda normal no vale. En las {resumen['control_sin_positivas_n']} sin "
    f"ninguna, el Montecarlo da 0 y la fórmula como mucho {resumen['control_sin_positivas_formula_max']:.1e}: "
    f"coinciden en la práctica. Probabilidad de ejercicio: error absoluto medio de la red "
    f"{resumen['error_abs_medio_prob']:.4f}. Tamaño de Montecarlo equivalente: la red principal tiene el error de "
    f"un Montecarlo de unas {resumen['n_eq_red']:,.0f} continuaciones por prima y la geométrica, de unas "
    f"{resumen['n_eq_red_geo']:,.0f}, frente a las {NUM_SIMS:,} de la referencia.")

# ================ TRAYECTORIA DE REFERENCIA DEL EQUIPO ================
ref = np.load("Datos/Transformados/prima_por_instante.npz", allow_pickle=False)
assert np.array_equal(ref["historia"], d["X"][0]) and 0 in d["idx_test"], "PARA: la referencia no es la trayectoria 0"
prima_ref_red = np.maximum(salida_bruta(red, entradas_historias(ref["historia"][None]))[0]
                           * g_red["desv_y"] + g_red["media_y"], 0)            # t_1 .. t_20
red_ref = np.r_[t0["red"], prima_ref_red[:-1]]                                 # k = 0..19
error_ref = red_ref - ref["prima"]
dentro_ref = np.abs(error_ref) <= Z95 * ref["error_estandar"]
texto_ref = (f"Trayectoria de referencia del equipo (simulación 1): error absoluto medio de la red "
             f"{np.abs(error_ref).mean():.4f} en k = 0..19, con {dentro_ref.sum()} de 20 fechas dentro de ±1.96 "
             f"errores estándar del Montecarlo del equipo. En el vencimiento la red predice un payoff de "
             f"{prima_ref_red[-1]:.4f}, frente al real de {d['payoff'][0]:.4f}.")

# ============================ GUARDADO ============================
tabla.to_csv(CARPETA_SALIDA / "rnn_tabla_primas.csv")
historias_df.to_csv(CARPETA_SALIDA / "rnn_primas_por_historia.csv", index=False)
with open(CARPETA_SALIDA / "rnn_prima_resumen.json", "w", encoding="utf-8") as archivo:
    json.dump({"t0": t0, "fechas": resumen, "num_sims": NUM_SIMS, "n_historias": N_HISTORIAS,
               "red_geometrica": {"epocas": res_geo["epocas"], "tiempo_entrenamiento_s": res_geo["tiempo"]},
               "trayectoria_referencia": {"red": red_ref.tolist(), "montecarlo": ref["prima"].tolist(),
                                          "error_estandar": ref["error_estandar"].tolist()},
               "rapido": args.rapido,
               "textos": {"t0": texto_es(texto_t0), "fechas": texto_es(texto_fechas), "referencia": texto_es(texto_ref)}},
              archivo, indent=2, ensure_ascii=False)

imprimir("\nTabla por fecha (entregable: predicción del valor presente en t):")
imprimir(tabla.to_string(float_format=lambda v: f"{v:.4f}"))
imprimir("\n" + texto_fechas + "\n\n" + texto_ref)

# =============================== GRÁFICOS ===============================
t = tabla["t_anios"].to_numpy()

# Gráfico 1: prima por fecha
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for ax, (col_red, col_ref, etiqueta_ref, titulo) in zip(axes, [
        ("prima_red", "prima_mc", "Montecarlo condicionado", "Red principal (payoff del enunciado)"),
        ("prima_red_geo", "prima_formula", "Fórmula cerrada de BBVA", "Red geométrica (cesta geométrica)")]):
    ax.scatter(historias_df["t_anios"], historias_df[col_ref], s=12, color=BBVA["gris_dato"], alpha=0.7,
               label=f"{etiqueta_ref}, cada historia")
    ax.plot(t, tabla[col_ref], color=BBVA["marino"], linewidth=2.5, marker="o", label=f"{etiqueta_ref}, media")
    ax.plot(t, tabla[col_red], color=BBVA["agua"], linewidth=2.5, linestyle="--", marker="s", label="Red, media")
    ax.set_title(titulo, fontsize=13)
    ax.set_xlabel("Fecha de valoración t_k (años)")
    ax.set_ylabel("Prima en t_k (precios normalizados)")
    # Unas pocas historias muy dentro del dinero aplastarían las medias: se corta el eje y se avisa
    p99, maximo = np.percentile(historias_df[col_ref], 99), historias_df[col_ref].max()
    ax.set_ylim(0, p99 * 1.05)
    ax.text(0.98, 0.97, f"Eje cortado en el percentil 99 de las primas\npor historia ({p99:.3f}); el máximo es "
            f"{maximo:.3f}", transform=ax.transAxes, ha="right", va="top", multialignment="left", fontsize=10,
            color=BBVA["marino"], bbox=dict(boxstyle="round", fc=BBVA["gris"], ec=BBVA["gris_dato"]))
    ax.legend(fontsize=10, loc="upper left")
fig.suptitle(f"Gráfico 1. Prima por fecha: la red frente a su referencia ({N_HISTORIAS} historias por fecha)",
             x=0.01, ha="left", fontsize=16, fontweight="bold", color=BBVA["marino"])
fig.tight_layout()
formato_es(fig, fuente=FUENTE)
guardar_mpl(fig, "rnn_prima_por_fecha")
plt.close(fig)

# Gráfico 2: error por fecha frente a la precisión del Montecarlo
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for ax, (col_error, col_ee, titulo) in zip(axes, [
        ("error_red", "ee_mc", "Red principal menos Montecarlo condicionado"),
        ("error_red_geo", "ee_geo_mc", "Red geométrica menos fórmula cerrada")]):
    banda = Z95 * historias_df.groupby("k")[col_ee].mean()
    tk = banda.index * mc.DT
    ax.fill_between(tk, -banda, banda, color=BBVA["cielo"], alpha=0.35, linewidth=0,
                    label=f"±1.96 errores estándar de un Montecarlo de {NUM_SIMS:,} continuaciones")
    ax.scatter(historias_df["t_anios"], historias_df[col_error], s=14, color=BBVA["principal"], alpha=0.6,
               label="Error de la red en cada historia")
    ax.axhline(0, color=BBVA["marino"], linewidth=1)
    ax.set_title(titulo, fontsize=13)
    ax.set_xlabel("Fecha de valoración t_k (años)")
    ax.set_ylabel("Error de la prima (precios normalizados)")
    ax.legend(fontsize=10, loc="upper right")
fig.suptitle("Gráfico 2. Error de las redes por fecha frente a la precisión de Montecarlo", x=0.01, ha="left",
             fontsize=16, fontweight="bold", color=BBVA["marino"])
fig.tight_layout()
formato_es(fig, fuente=FUENTE)
guardar_mpl(fig, "rnn_error_por_fecha")
plt.close(fig)

# Gráfico 3: probabilidad de ejercicio en 3 trayectorias de ejemplo (dentro, fuera y en la frontera)
candidatas = np.stack([mc.simular_continuaciones(None, 1, np.random.default_rng(90_000 + i))[0] for i in range(50)])
suma_g = mc.medias_geometricas(candidatas).sum(axis=1)
payoff_c = np.maximum(suma_g - mc.STRIKE, 0)
positivos = np.where(payoff_c > 0)[0]
ejemplos = {
    "Acaba dentro del dinero": positivos[np.argmin(np.abs(payoff_c[positivos] - np.median(d["payoff"][d["payoff"] > 0])))],
    "Acaba en la frontera": int(np.argmin(np.abs(suma_g - mc.STRIKE))),
    "Acaba fuera del dinero": int(np.argmin(np.abs(suma_g - mc.STRIKE + 0.5))),
}
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)
for ax, (titulo, i) in zip(axes, ejemplos.items()):
    trayectoria = candidatas[i]
    prob_red_traj = 1 / (1 + np.exp(-salida_bruta(red_clf, entradas_historias(trayectoria[None]))[0]))
    prob_mc_traj = [(mc.calcular_payoff(mc.simular_continuaciones(trayectoria[:k], NUM_SIMS,
                                                                    np.random.default_rng(95_000 + 100 * i + k))) > 0).mean()
                    for k in FECHAS]
    tk = np.array(FECHAS) * mc.DT
    ax.plot(tk, prob_mc_traj, color=BBVA["marino"], linewidth=2.5, marker="o", label="Montecarlo condicionado")
    ax.plot(tk, prob_red_traj[:-1], color=BBVA["agua"], linewidth=2.5, linestyle="--", marker="s", label="Red")
    ax.axhline(0.5, color=BBVA["gris_dato"], linewidth=1, linestyle=":")
    ax.set_title(f"{titulo} (payoff final {payoff_c[i]:.3f})", fontsize=12)
    ax.set_xlabel("Fecha t_k (años)")
    ax.set_ylim(-0.03, 1.03)
axes[0].set_ylabel("Probabilidad de ejercicio")
axes[0].legend(fontsize=10, loc="center left")
fig.suptitle("Gráfico 3. Probabilidad de ejercicio a lo largo de tres trayectorias", x=0.01, ha="left",
             fontsize=16, fontweight="bold", color=BBVA["marino"])
fig.tight_layout()
formato_es(fig, fuente=FUENTE)
guardar_mpl(fig, "rnn_probabilidad_ejercicio")
plt.close(fig)

# Gráfico 4: trayectoria de referencia del equipo
tiempos_ref = ref["tiempo"]
fig, ax = plt.subplots(figsize=(11, 6))
ax.fill_between(tiempos_ref, ref["prima"] - Z95 * ref["error_estandar"], ref["prima"] + Z95 * ref["error_estandar"],
                color=BBVA["cielo"], alpha=0.4, linewidth=0)
ax.plot(tiempos_ref, ref["prima"], color=BBVA["marino"], linewidth=2.5, marker="o")
ax.plot(tiempos_ref, red_ref, color=BBVA["agua"], linewidth=2.5, linestyle="--", marker="s")
ax.scatter([mc.TTM], [d["payoff"][0]], color=BBVA["marino"], marker="*", s=250, zorder=4)
ax.scatter([mc.TTM], [prima_ref_red[-1]], color=BBVA["agua"], marker="*", s=250, zorder=4, edgecolor=BBVA["marino"])
ax.legend(handles=[
    Line2D([], [], color=BBVA["marino"], linewidth=2.5, marker="o", label="Montecarlo del equipo (prima_por_instante.npz)"),
    plt.Rectangle((0, 0), 1, 1, color=BBVA["cielo"], alpha=0.4,
                  label=f"±1.96 errores estándar (anchura media de la banda: "
                        f"{(2 * Z95 * ref['error_estandar']).mean():.4f})"),
    Line2D([], [], color=BBVA["agua"], linewidth=2.5, linestyle="--", marker="s", label="Red"),
    Line2D([], [], color=BBVA["marino"], marker="*", linestyle="", markersize=14, label="Payoff real y predicho en t = 5"),
], fontsize=10)
ax.set_xlabel("Fecha de valoración (años)")
ax.set_ylabel("Prima (precios normalizados)")
ax.set_ylim(bottom=0)
ax.set_title("Gráfico 4. Prima a lo largo de la trayectoria de referencia del equipo (simulación 1)", fontsize=14)
fig.tight_layout()
formato_es(fig, fuente="trayectoria 1 de montecarlo_independiente.npz; prima condicionada de prima_por_instante.npz.")
guardar_mpl(fig, "rnn_prima_trayectoria_referencia")
plt.close(fig)
imprimir(f"\nTiempo total: {(time.perf_counter() - inicio_script) / 60:.1f} min")
