"""Modelos de referencia y comparación de las cuatro estructuras recurrentes (issue #35).

Entrada:
    Datos/Transformados/rnn_dataset.npz   (rnn_01_datos.py), leído con packages/rnn_modelos.cargar_datos
Salidas:
    outputs/rnn_comparacion_regresion.csv       regresión del payoff al vencimiento: errores y eficiencia
    outputs/rnn_comparacion_clasificacion.csv   clasificación (acaba o no dentro del dinero)
    outputs/rnn_eleccion_arquitectura.json      redes elegidas y su justificación con los números
    modelos/rnn_arq_*.pt                        pesos de las redes (no se suben)
    Graficos/rnn_curvas_aprendizaje.png, rnn_comparacion_rmse.png, rnn_rmse_vs_parametros.png,
             rnn_payoff_real_vs_predicho.png, rnn_matriz_confusion.png
Ejecución: python rnn_02_arquitecturas.py [--rapido]
    --rapido: 20.000 trayectorias y como máximo 20 épocas, para probar. Todo lo que genera (CSV, JSON,
    gráficos y pesos) va a outputs/rapido/, que Git ignora, para no pisar los resultados buenos.

Modelos de referencia (payoff al vencimiento, con los 60 retornos aplanados como entrada)
    Como en los ejercicios tabulares de clase, una red tiene que ganar a la media y a un modelo lineal
    para justificar su coste.
    - Media de entrenamiento: el error que se comete sin mirar la trayectoria.
    - Regresión lineal (scikit-learn): el payoff es max(G_CAT + G_MRK + G_MSFT - 3, 0) y log G_k es
      lineal en los retornos, así que un modelo lineal ya captura buena parte de la relación. Lo que
      no puede capturar es el max(·, 0) y la exponencial.
    - MLP [64, 32] con ReLU (PyTorch): ve los mismos 60 retornos que la red recurrente, pero sin
      estructura temporal ni canal de tiempo. Responde a la pregunta de si hace falta la recurrencia
      o basta con una red densa. Si el MLP iguala a las recurrentes en el paso 20, la recurrencia no
      se justifica por el payoff al vencimiento sino por la prima en cada fecha: la red recurrente da
      una predicción en cada paso con lo observado hasta entonces, y el MLP necesita la trayectoria
      completa.
    Todas las predicciones de regresión, también las de las referencias, se recortan a >= 0 (la prima
    no puede ser negativa), para que la comparación sea justa.

Redes recurrentes (packages/rnn_modelos.py): RNN simple, LSTM, GRU y LSTM bidireccional, con 1 capa
de 64 neuronas y el mismo presupuesto de entrenamiento: MSE, lr 1e-3, lotes de 512, como máximo 100
épocas, paciencia 10 y semilla 42. Con el mismo presupuesto, las diferencias se deben a la
arquitectura y no a un ajuste distinto.

Métricas
    - Regresión, en test y sobre el payoff (paso 20), desescalado: MAE, RMSE y R². El paso 20 es el
      único en el que la etiqueta es exacta (en los demás es ruidosa, ver packages/rnn_modelos.py).
      El RMSE penaliza los errores grandes de la cola del payoff, el MAE es más robusto a ellos y el
      R² compara con la media.
    - Eficiencia (la rúbrica pide rendimiento y eficiencia): parámetros entrenables, tiempo de
      entrenamiento, épocas hasta el early stopping y tiempo de predicción por trayectoria.
    - Clasificación (payoff > 0, la decisión de ejercer), en test en el paso 20, con umbral 0,5:
      accuracy, F1 (clase positiva: ejercer) y matriz de confusión. Con un 56/44 de clases, la
      accuracy sola puede esconder que se falla mucho en una de ellas; el F1 y la matriz lo muestran.
      Referencias: clase mayoritaria y regresión logística (scikit-learn) sobre los 60 retornos.

Cómo se elige (siempre con validación; test solo se usa para informar)
    - Mejor modelo global (las cuatro recurrentes y las referencias): el menor RMSE del payoff en el
      paso 20 en validación. Es una comparación justa para todos, también para la bidireccional: en
      el paso 20 cualquier modelo ha visto la trayectoria completa. Con el criterio de clase de
      rendimiento frente a parámetros, si un modelo con menos parámetros queda a menos de un 2 % del
      mejor RMSE, se prefiere el pequeño.
    - Mejor causal (RNN simple, LSTM o GRU): la menor pérdida de validación de los 20 pasos. Es la que
      se usa para la prima en rnn_04_prima.py. Entre causales es una comparación justa: el ruido de
      las etiquetas intermedias es el mismo para todas, así que la diferencia es el error de la prima
      en cada t. La bidireccional queda fuera porque en el paso t usaría fechas futuras.
    - Clasificación: la menor log-loss en validación en el paso 20, con la misma regla del 2 %.
    Las dos elecciones de regresión se explican con los números de la ejecución al final del script
    (se imprimen y se guardan en outputs/rnn_eleccion_arquitectura.json).
"""
import argparse
import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, MaxNLocator, NullLocator, ScalarFormatter
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss, mean_absolute_error,
                             mean_squared_error, r2_score)
from torch import nn

from packages.bbva_graficos import (BBVA, CMAP_SECUENCIAL, configurar_graficos, formato_es, guardar_mpl,
                                    imprimir, texto_es)
from packages.rnn_modelos import (DISPOSITIVO, RUTA_DATASET, RedRecurrente, cargar_datos, contar_parametros,
                                  entrenar, evaluar)

parser = argparse.ArgumentParser(description="Referencias y comparación de estructuras recurrentes")
parser.add_argument("--rapido", action="store_true", help="20.000 trayectorias y 20 épocas como máximo")
args = parser.parse_args()

SEMILLA = 42
N_MAX = 20_000 if args.rapido else None
PRESUPUESTO = dict(lr=1e-3, batch=512, max_epocas=20 if args.rapido else 100, paciencia=10, min_delta=1e-6)
TOLERANCIA = 0.02          # rendimiento frente a parámetros: diferencias menores se resuelven por tamaño
ESTRUCTURAS = {
    "RNN simple": dict(tipo="rnn"),
    "LSTM": dict(tipo="lstm"),
    "GRU": dict(tipo="gru"),
    "LSTM bidireccional": dict(tipo="lstm", bidireccional=True),
}
ARCHIVOS = {"RNN simple": "rnn", "LSTM": "lstm", "GRU": "gru", "LSTM bidireccional": "lstm_bidireccional"}
EN_TEXTO = {"Media de entrenamiento": "la media de entrenamiento", "Regresión lineal": "la regresión lineal",
            "MLP [64, 32]": "el MLP"}

os.chdir(Path(__file__).resolve().parent)   # rutas relativas a la raíz del repo
CARPETA_SALIDA = Path("outputs/rapido") if args.rapido else Path("outputs")
CARPETA_MODELOS = CARPETA_SALIDA if args.rapido else Path("modelos")
CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)
CARPETA_MODELOS.mkdir(exist_ok=True)
configurar_graficos(CARPETA_SALIDA if args.rapido else "Graficos")
FUENTE = (f"{'20.000' if args.rapido else '100.000'} trayectorias de Montecarlo de rnn_dataset.npz "
          "(70 % entrenamiento, 15 % validación, 15 % test).")
pd.options.display.width = 250


def familia(nombre):
    return {"LSTM bidireccional": "bidireccional"}.get(nombre, "causal" if nombre in ESTRUCTURAS else "referencia")


def articulo(nombre):
    return "el" if nombre.startswith("MLP") else "la"


def elegir(tabla, columna, tolerancia):
    """El menor valor de `columna`; entre los que quedan a menos de `tolerancia` de él, el de menos parámetros."""
    candidatos = tabla[tabla[columna] <= tabla[columna].min() * (1 + tolerancia)]
    return candidatos.sort_values(["parametros", columna]).index[0]


def comparar(valor, referencia):
    """Diferencia relativa de error en palabras: 'un 71,0 % menor' o 'un 3,0 % mayor'."""
    return f"un {abs(1 - valor / referencia):.1%} {'menor' if valor <= referencia else 'mayor'}"


def guardar_modelo(modelo, config, datos, nombre):
    torch.save({"config": config, "estado": modelo.state_dict(),
                "media_y": datos["media_y"], "desv_y": datos["desv_y"]}, CARPETA_MODELOS / f"rnn_arq_{nombre}.pt")


def medir_prediccion(funcion, x):
    """Ejecuta la predicción y devuelve también su tiempo por trayectoria en microsegundos."""
    inicio = time.perf_counter()
    salida = funcion(x)
    return salida, (time.perf_counter() - inicio) / len(x) * 1e6


imprimir(f"Modo: {'rápido (20,000 trayectorias, 20 épocas como máximo)' if args.rapido else 'completo'}; "
         f"dispositivo: {DISPOSITIVO}")

# ============================== REGRESIÓN ==============================
datos = cargar_datos("Y", N_MAX)
payoff = np.load(RUTA_DATASET, allow_pickle=False)["payoff"]
payoff_c = {c: payoff[datos[f"idx_{c}"]] for c in ("train", "val", "test")}
plano = {c: datos[f"x_{c}"][..., :3].reshape(len(datos[f"x_{c}"]), -1) for c in ("train", "val", "test")}
imprimir(f"Trayectorias: {len(plano['train']):,} de entrenamiento, {len(plano['val']):,} de validación y "
         f"{len(plano['test']):,} de test")

filas_reg, pred_test_reg, historiales = [], {}, {}


def fila_regresion(nombre, parametros, pred_val, pred_test, t_pred, t_ent, res=None):
    pred_test_reg[nombre] = pred_test
    return {
        "modelo": nombre, "familia": familia(nombre), "parametros": parametros,
        "epocas": res["epocas"] if res else np.nan, "mejor_epoca": res["mejor_epoca"] if res else np.nan,
        "tiempo_entrenamiento_s": t_ent, "tiempo_prediccion_us": t_pred,
        "val_perdida_20_pasos": res["mejor_val"] if res and nombre in ESTRUCTURAS else np.nan,
        "val_rmse_paso20": np.sqrt(mean_squared_error(payoff_c["val"], pred_val)),
        "test_mae": mean_absolute_error(payoff_c["test"], pred_test),
        "test_rmse": np.sqrt(mean_squared_error(payoff_c["test"], pred_test)),
        "test_r2": r2_score(payoff_c["test"], pred_test),
    }


imprimir("\nRegresión: referencias")
inicio = time.perf_counter()
media = payoff_c["train"].mean()
t_ent = time.perf_counter() - inicio
pred_test, t_pred = medir_prediccion(lambda x: np.full(len(x), media), plano["test"])
filas_reg.append(fila_regresion("Media de entrenamiento", 1, np.full(len(plano["val"]), media), pred_test,
                                t_pred, t_ent))

inicio = time.perf_counter()
lineal = LinearRegression().fit(plano["train"], payoff_c["train"])
t_ent = time.perf_counter() - inicio
pred_test, t_pred = medir_prediccion(lambda x: np.maximum(lineal.predict(x), 0), plano["test"])
filas_reg.append(fila_regresion("Regresión lineal", plano["train"].shape[1] + 1,
                                np.maximum(lineal.predict(plano["val"]), 0), pred_test, t_pred, t_ent))

datos_mlp = {"x_train": plano["train"], "y_train": datos["y_train"][:, -1],
             "x_val": plano["val"], "y_val": datos["y_val"][:, -1],
             "media_y": datos["media_y"], "desv_y": datos["desv_y"]}
torch.manual_seed(SEMILLA)
mlp = nn.Sequential(nn.Linear(60, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1), nn.Flatten(0))
res = entrenar(mlp, datos_mlp, nn.MSELoss(), **PRESUPUESTO)
pred_test, t_pred = medir_prediccion(lambda x: evaluar(mlp, x, datos["media_y"], datos["desv_y"]), plano["test"])
filas_reg.append(fila_regresion("MLP [64, 32]", contar_parametros(mlp),
                                evaluar(mlp, plano["val"], datos["media_y"], datos["desv_y"]),
                                pred_test, t_pred, res["tiempo"], res))
guardar_modelo(mlp, {"capas": [60, 64, 32, 1]}, datos_mlp, "regresion_mlp")
imprimir(f"  MLP [64, 32]: {res['epocas']} épocas (mejor {res['mejor_epoca']}), {res['tiempo']:.1f} s")

imprimir("Regresión: redes recurrentes")
for nombre, config in ESTRUCTURAS.items():
    torch.manual_seed(SEMILLA)
    modelo = RedRecurrente(**config)
    res = entrenar(modelo, datos, nn.MSELoss(), **PRESUPUESTO)
    historiales[nombre] = res["historial"]
    pred_test, t_pred = medir_prediccion(
        lambda x: evaluar(modelo, x, datos["media_y"], datos["desv_y"])[:, -1], datos["x_test"])
    filas_reg.append(fila_regresion(nombre, contar_parametros(modelo),
                                    evaluar(modelo, datos["x_val"], datos["media_y"], datos["desv_y"])[:, -1],
                                    pred_test, t_pred, res["tiempo"], res))
    guardar_modelo(modelo, config, datos, f"regresion_{ARCHIVOS[nombre]}")
    imprimir(f"  {nombre}: {res['epocas']} épocas (mejor {res['mejor_epoca']}), {res['tiempo']:.1f} s")

tabla_reg = pd.DataFrame(filas_reg).set_index("modelo")

# ============================ CLASIFICACIÓN ============================
datos_itm = cargar_datos("itm", N_MAX)
itm_c = {c: datos_itm[f"y_{c}"][:, -1].astype(int) for c in ("train", "val", "test")}
filas_clf = []


def fila_clasificacion(nombre, parametros, prob_val, prob_test, t_pred, t_ent, res=None):
    clase_test = (prob_test >= 0.5).astype(int)
    vn, fp, fn, vp = confusion_matrix(itm_c["test"], clase_test, labels=[0, 1]).ravel()
    return {
        "modelo": nombre, "familia": familia(nombre), "parametros": parametros,
        "epocas": res["epocas"] if res else np.nan, "mejor_epoca": res["mejor_epoca"] if res else np.nan,
        "tiempo_entrenamiento_s": t_ent, "tiempo_prediccion_us": t_pred,
        "val_perdida_20_pasos": res["mejor_val"] if res else np.nan,
        "val_logloss_paso20": log_loss(itm_c["val"], prob_val, labels=[0, 1]),
        "val_accuracy_paso20": accuracy_score(itm_c["val"], prob_val >= 0.5),
        "test_accuracy": accuracy_score(itm_c["test"], clase_test),
        "test_f1": f1_score(itm_c["test"], clase_test, zero_division=0),
        "test_vn": vn, "test_fp": fp, "test_fn": fn, "test_vp": vp,
    }


imprimir("\nClasificación: referencias")
frecuencia = itm_c["train"].mean()           # probabilidad constante; la clase predicha es la mayoritaria
prob_test, t_pred = medir_prediccion(lambda x: np.full(len(x), frecuencia), plano["test"])
filas_clf.append(fila_clasificacion("Clase mayoritaria", 1, np.full(len(plano["val"]), frecuencia), prob_test,
                                    t_pred, 0.0))

inicio = time.perf_counter()
logistica = LogisticRegression(max_iter=1000).fit(plano["train"], itm_c["train"])
t_ent = time.perf_counter() - inicio
prob_test, t_pred = medir_prediccion(lambda x: logistica.predict_proba(x)[:, 1], plano["test"])
filas_clf.append(fila_clasificacion("Regresión logística", plano["train"].shape[1] + 1,
                                    logistica.predict_proba(plano["val"])[:, 1], prob_test, t_pred, t_ent))

imprimir("Clasificación: redes recurrentes")
for nombre, config in ESTRUCTURAS.items():
    torch.manual_seed(SEMILLA)
    modelo = RedRecurrente(**config)
    res = entrenar(modelo, datos_itm, nn.BCEWithLogitsLoss(), **PRESUPUESTO)
    prob_test, t_pred = medir_prediccion(lambda x: evaluar(modelo, x, clasificacion=True)[:, -1],
                                         datos_itm["x_test"])
    filas_clf.append(fila_clasificacion(nombre, contar_parametros(modelo),
                                        evaluar(modelo, datos_itm["x_val"], clasificacion=True)[:, -1],
                                        prob_test, t_pred, res["tiempo"], res))
    guardar_modelo(modelo, config, datos_itm, f"clasificacion_{ARCHIVOS[nombre]}")
    imprimir(f"  {nombre}: {res['epocas']} épocas (mejor {res['mejor_epoca']}), {res['tiempo']:.1f} s")

tabla_clf = pd.DataFrame(filas_clf).set_index("modelo")

# =============================== ELECCIÓN ===============================
mejor_global = elegir(tabla_reg, "val_rmse_paso20", TOLERANCIA)
causales = tabla_reg[tabla_reg["familia"] == "causal"].sort_values("val_perdida_20_pasos")
mejor_causal = causales.index[0]
mejor_clf = elegir(tabla_clf, "val_logloss_paso20", TOLERANCIA)
causales_clf = tabla_clf[tabla_clf["familia"] == "causal"].sort_values("val_perdida_20_pasos")
mejor_causal_clf = causales_clf.index[0]

g = tabla_reg.loc[mejor_global]
orden = tabla_reg.sort_values("val_rmse_paso20")
texto_global = (f"Mejor modelo global: {articulo(mejor_global)} {mejor_global}, por el RMSE del payoff en el paso 20 "
                f"en validación ({g['val_rmse_paso20']:.4f}). ")
if orden.index[0] != mejor_global:
    m = orden.iloc[0]
    texto_global += (f"{articulo(orden.index[0]).capitalize()} {orden.index[0]} tenía un RMSE algo menor "
                     f"({m['val_rmse_paso20']:.4f}, un {1 - m['val_rmse_paso20'] / g['val_rmse_paso20']:.1%} "
                     f"menos) pero con {m['parametros']:,.0f} parámetros frente a {g['parametros']:,.0f}: con "
                     f"menos de un {TOLERANCIA:.0%} de diferencia se prefiere el modelo más pequeño. ")
else:
    texto_global += (f"Le sigue {articulo(orden.index[1])} {orden.index[1]} ({orden.iloc[1]['val_rmse_paso20']:.4f}). ")
ref = tabla_reg["val_rmse_paso20"]
texto_global += "Su error es " + " y ".join(
    f"{comparar(g['val_rmse_paso20'], ref[r])} que el de {EN_TEXTO[r]} ({ref[r]:.4f})"
    for r in EN_TEXTO if r != mejor_global) + ". "
texto_global += (f"En test, que no se ha usado para elegir, da un RMSE de {g['test_rmse']:.4f} y un R² de "
                 f"{g['test_r2']:.4f}.")
if g["familia"] == "referencia":
    mejor_rec = tabla_reg[tabla_reg["familia"] != "referencia"].sort_values("val_rmse_paso20").index[0]
    r_rec = tabla_reg.loc[mejor_rec, "val_rmse_paso20"]
    texto_global += (f" La mejor recurrente en el paso 20 es la {mejor_rec} ({r_rec:.4f}, un error "
                     f"{comparar(r_rec, g['val_rmse_paso20'])}). Para el payoff al vencimiento, la "
                     f"recurrencia no aporta una mejora apreciable frente a una red densa que ve los mismos 60 "
                     f"retornos. Lo que justifica la red recurrente es la prima en cada fecha: da una predicción en "
                     f"cada paso con lo observado hasta entonces, mientras que {EN_TEXTO.get(mejor_global, mejor_global)}"
                     f" necesita la trayectoria completa.")
if g["familia"] == "bidireccional":
    texto_global += (" Al ser bidireccional, solo vale para el payoff al vencimiento, cuando la trayectoria "
                     "completa ya se conoce; para la prima en t se usa la mejor causal.")

c = tabla_reg.loc[mejor_causal]
otras = ", ".join(f"{fila['val_perdida_20_pasos']:.4f} de la {n}" for n, fila in causales.iloc[1:].iterrows())
texto_causal = (f"Mejor causal, la que se usa para la prima en cada fecha: la {mejor_causal}, con la menor pérdida "
                f"de validación en los 20 pasos ({c['val_perdida_20_pasos']:.4f}, frente a {otras}). Entre causales "
                f"el ruido de las etiquetas intermedias es el mismo, así que la diferencia mide el error de la prima "
                f"en cada t. Tiene {c['parametros']:,.0f} parámetros, se entrenó en {c['tiempo_entrenamiento_s']:.0f} s "
                f"({c['epocas']:.0f} épocas) y su RMSE del payoff en el paso 20 en validación es "
                f"{c['val_rmse_paso20']:.4f}.")

k = tabla_clf.loc[mejor_clf]
texto_clf = (f"Clasificación: la mejor por log-loss en el paso 20 en validación es {articulo(mejor_clf)} {mejor_clf} "
             f"({k['val_logloss_paso20']:.4f}); en test acierta el {k['test_accuracy']:.2%} con un F1 de "
             f"{k['test_f1']:.4f}, frente al {tabla_clf.loc['Clase mayoritaria', 'test_accuracy']:.2%} de la clase "
             f"mayoritaria y el {tabla_clf.loc['Regresión logística', 'test_accuracy']:.2%} de la regresión logística. "
             f"Entre causales, por la pérdida de los 20 pasos, gana la {mejor_causal_clf}")
if mejor_causal_clf == mejor_causal:
    texto_clf += ", la misma que en regresión: se puede usar la misma configuración para las dos tareas."
else:
    a = causales_clf.loc[mejor_causal_clf, "val_perdida_20_pasos"]
    b = causales_clf.loc[mejor_causal, "val_perdida_20_pasos"]
    texto_clf += (f", que no coincide con la de regresión ({mejor_causal}): pérdida de validación de "
                  f"{a:.4f} frente a {b:.4f} de la {mejor_causal}, un {1 - a / b:.1%} menor.")

tabla_reg.to_csv(CARPETA_SALIDA / "rnn_comparacion_regresion.csv")
tabla_clf.to_csv(CARPETA_SALIDA / "rnn_comparacion_clasificacion.csv")
with open(CARPETA_SALIDA / "rnn_eleccion_arquitectura.json", "w", encoding="utf-8") as archivo:
    json.dump({
        "mejor_global": mejor_global, "mejor_causal": mejor_causal,
        "tipo_causal": ESTRUCTURAS[mejor_causal]["tipo"],
        "mejor_clasificacion": mejor_clf, "mejor_causal_clasificacion": mejor_causal_clf,
        "tolerancia": TOLERANCIA, "rapido": args.rapido,
        "texto_global": texto_es(texto_global), "texto_causal": texto_es(texto_causal),
        "texto_clasificacion": texto_es(texto_clf),
    }, archivo, indent=2, ensure_ascii=False)

columnas_reg = ["parametros", "epocas", "tiempo_entrenamiento_s", "tiempo_prediccion_us", "val_perdida_20_pasos",
                "val_rmse_paso20", "test_mae", "test_rmse", "test_r2"]
columnas_clf = ["parametros", "epocas", "tiempo_entrenamiento_s", "val_perdida_20_pasos", "val_logloss_paso20",
                "test_accuracy", "test_f1", "test_vn", "test_fp", "test_fn", "test_vp"]
imprimir("\nRegresión del payoff al vencimiento (paso 20):")
imprimir(tabla_reg[columnas_reg].to_string(float_format=lambda v: f"{v:.4f}"))
imprimir("\nClasificación (paso 20, umbral 0.5):")
imprimir(tabla_clf[columnas_clf].to_string(float_format=lambda v: f"{v:.4f}"))
imprimir("\n" + texto_global + "\n\n" + texto_causal + "\n\n" + texto_clf)

# =============================== GRÁFICOS ===============================
COLOR_FAMILIA = {"referencia": BBVA["gris_dato"], "causal": BBVA["principal"], "bidireccional": BBVA["medio"]}


def color_modelo(nombre, tabla, elegido):
    return BBVA["agua"] if nombre == elegido else COLOR_FAMILIA[tabla.loc[nombre, "familia"]]


# Gráfico 1: curvas de aprendizaje
fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
for ax, (nombre, historial) in zip(axes.flat, historiales.items()):
    epocas = np.arange(1, len(historial["train"]) + 1)
    ax.plot(epocas, historial["train"], color=BBVA["medio"], linestyle="--", linewidth=2, label="Entrenamiento")
    ax.plot(epocas, historial["val"], color=BBVA["principal"], linewidth=2.2, label="Validación")
    mejor_epoca = tabla_reg.loc[nombre, "mejor_epoca"]
    ax.axvline(mejor_epoca, color=BBVA["agua"], linestyle=":", linewidth=2, label=f"Mejor época ({mejor_epoca:.0f})")
    ax.set_title(f"{nombre} ({tabla_reg.loc[nombre, 'parametros']:,.0f} parámetros)", fontsize=13)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Época")
    ax.set_ylabel("MSE escalado (20 pasos)")
    ax.legend(fontsize=10)
    if nombre == "LSTM bidireccional":
        ax.text(0.98, 0.55, "Ve la trayectoria completa en cada paso:\nsu pérdida no es comparable con las demás",
                transform=ax.transAxes, ha="right", fontsize=10, color=BBVA["marino"])
fig.suptitle("Gráfico 1. Curvas de aprendizaje de las cuatro estructuras (regresión)", x=0.01, ha="left",
             fontsize=16, fontweight="bold", color=BBVA["marino"])
fig.tight_layout()
formato_es(fig, fuente=FUENTE)
guardar_mpl(fig, "rnn_curvas_aprendizaje")
plt.close(fig)

# Gráfico 2: RMSE del payoff en test, ordenado (el mejor arriba)
orden_test = tabla_reg.sort_values("test_rmse", ascending=False)
fig, ax = plt.subplots(figsize=(10, 5.5))
barras = ax.barh(orden_test.index, orden_test["test_rmse"],
                 color=[color_modelo(n, tabla_reg, mejor_global) for n in orden_test.index])
for barra, valor in zip(barras, orden_test["test_rmse"]):
    ax.text(valor, barra.get_y() + barra.get_height() / 2, f" {valor:.4f}", va="center", fontsize=11,
            color=BBVA["marino"])
ax.set_xlim(0, orden_test["test_rmse"].max() * 1.15)
ax.grid(False)
ax.set_xlabel("RMSE del payoff al vencimiento en test")
ax.set_title("Gráfico 2. Error del payoff al vencimiento en test (menor es mejor)")
ax.legend(handles=[Patch(color=BBVA["agua"], label=f"Elegido por validación ({mejor_global})"),
                   Patch(color=BBVA["principal"], label="Recurrentes causales"),
                   Patch(color=BBVA["medio"], label="Recurrente bidireccional"),
                   Patch(color=BBVA["gris_dato"], label="Referencias")], loc="upper right")
fig.tight_layout()
formato_es(fig, fuente=FUENTE)
guardar_mpl(fig, "rnn_comparacion_rmse")
plt.close(fig)

# Gráfico 3: rendimiento frente a eficiencia (un marcador por modelo, para que no se pisen las etiquetas)
MARCADORES = {"Media de entrenamiento": "s", "Regresión lineal": "D", "MLP [64, 32]": "^", "RNN simple": "o",
              "LSTM": "P", "GRU": "X", "LSTM bidireccional": "*"}
fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), sharey=True)
for ax, columna, etiqueta in zip(axes, ["parametros", "tiempo_entrenamiento_s"],
                                 ["Parámetros entrenables (escala logarítmica)",
                                  "Tiempo de entrenamiento en segundos (escala logarítmica)"]):
    for nombre, fila in tabla_reg.iterrows():
        ax.scatter(fila[columna], fila["test_rmse"], s=260 if MARCADORES[nombre] == "*" else 150,
                   marker=MARCADORES[nombre], color=color_modelo(nombre, tabla_reg, mejor_global),
                   edgecolor=BBVA["marino"], alpha=0.85, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(etiqueta)
marcas = [m for m in (0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8)
          if tabla_reg["test_rmse"].min() * 0.9 <= m <= tabla_reg["test_rmse"].max() * 1.1]
axes[0].yaxis.set_major_locator(FixedLocator(marcas))
axes[0].yaxis.set_major_formatter(ScalarFormatter())
axes[0].yaxis.set_minor_locator(NullLocator())
axes[0].set_ylabel("RMSE del payoff en test (escala logarítmica)")
fig.legend(handles=[Line2D([], [], linestyle="", marker=MARCADORES[n], markersize=14 if MARCADORES[n] == "*" else 10,
                           color=color_modelo(n, tabla_reg, mejor_global), markeredgecolor=BBVA["marino"], label=n)
                    for n in tabla_reg.index],
           loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02), fontsize=11)
fig.suptitle("Gráfico 3. Rendimiento frente a eficiencia: error, tamaño y tiempo de entrenamiento", x=0.01,
             ha="left", fontsize=16, fontweight="bold", color=BBVA["marino"])
fig.tight_layout(rect=(0, 0.1, 1, 1))
formato_es(fig, fuente=FUENTE)
guardar_mpl(fig, "rnn_rmse_vs_parametros")
plt.close(fig)

# Gráfico 4: payoff real frente a predicho del mejor modelo global
pred = pred_test_reg[mejor_global]
limite = max(payoff_c["test"].max(), pred.max()) * 1.03
fig, ax = plt.subplots(figsize=(8, 7.5))
ax.scatter(payoff_c["test"], pred, s=6, alpha=0.4, color=BBVA["principal"], edgecolors="none")
ax.plot([0, limite], [0, limite], color=BBVA["marino"], linestyle="--", linewidth=1.5)
ax.text(limite * 0.62, limite * 0.55, "Predicción perfecta", rotation=45, color=BBVA["marino"], fontsize=11)
ax.text(0.03, 0.97, f"RMSE {g['test_rmse']:.4f}\nMAE {g['test_mae']:.4f}\nR² {g['test_r2']:.4f}",
        transform=ax.transAxes, va="top", fontsize=12, color=BBVA["marino"],
        bbox=dict(boxstyle="round", fc=BBVA["gris"], ec=BBVA["gris_dato"]))
ax.set(xlim=(0, limite), ylim=(0, limite))
ax.set_aspect("equal")
ax.set_xlabel("Payoff real")
ax.set_ylabel("Payoff predicho")
ax.set_title(f"Gráfico 4. Payoff real frente a predicho en test ({mejor_global})", fontsize=14)
fig.tight_layout()
formato_es(fig, fuente=FUENTE)
guardar_mpl(fig, "rnn_payoff_real_vs_predicho")
plt.close(fig)

# Gráfico 5: matriz de confusión del mejor clasificador
matriz = np.array([[k["test_vn"], k["test_fp"]], [k["test_fn"], k["test_vp"]]], dtype=float)
porcentaje = matriz / matriz.sum(axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(7.5, 6.5))
ax.imshow(matriz, cmap=CMAP_SECUENCIAL)
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{matriz[i, j]:,.0f}\n{porcentaje[i, j]:.1%} de los reales", ha="center", va="center",
                fontsize=13, fontweight="bold",
                color=BBVA["blanco"] if matriz[i, j] > matriz.max() / 2 else BBVA["marino"])
ax.set_xticks([0, 1], ["No ejercer", "Ejercer"])
ax.set_yticks([0, 1], ["No ejercer\n(payoff = 0)", "Ejercer\n(payoff > 0)"])
ax.set_xlabel("Predicción (probabilidad >= 0.5)")
ax.set_ylabel("Real")
ax.grid(False)
ax.set_title(f"Gráfico 5. Matriz de confusión en test ({mejor_clf})", fontsize=14)
fig.tight_layout()
formato_es(fig, fuente=FUENTE)
guardar_mpl(fig, "rnn_matriz_confusion")
plt.close(fig)
