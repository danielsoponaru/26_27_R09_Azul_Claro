"""Búsqueda de hiperparámetros, validación cruzada y modelos finales de la red causal (issue #36).

Entradas:
    Datos/Transformados/rnn_dataset.npz       (rnn_01_datos.py)
    outputs/rnn_eleccion_arquitectura.json    estructura causal elegida en la ejecución completa de
                                              rnn_02_arquitecturas.py
Salidas:
    outputs/rnn_busqueda_hiperparametros.csv  configuraciones probadas, con su pérdida y su intervalo
    outputs/rnn_validacion_cruzada.csv        métricas de cada bloque de la validación cruzada
    outputs/rnn_configuracion_final.json      configuración elegida, validación cruzada, test y textos
    modelos/rnn_final_regresion.pt            red final de regresión (no se sube)
    modelos/rnn_final_clasificacion.pt        red final de clasificación (no se sube)
    Graficos/rnn_hiperparametros.png
    outputs/rnn_03_progreso/                  resultados parciales para reanudar (ver "Reanudación")
Ejecución: python rnn_03_hiperparametros.py [--rapido]
    --rapido: 20.000 trayectorias y como máximo 20 épocas, para probar. Todo lo que genera va a
    outputs/rapido/, que Git ignora. La estructura se lee siempre de la ejecución completa de rnn_02,
    para que la prueba use la misma red que la ejecución completa.

Sobre qué red se busca
    La mejor causal de rnn_02_arquitecturas.py, porque es la que da la prima en cada fecha
    (rnn_04_prima.py). Se ajusta para la regresión y la misma configuración se usa en clasificación,
    con logits y BCEWithLogitsLoss: en rnn_02 la misma estructura ganó entre las causales en las dos
    tareas.

Orden de la búsqueda (el práctico de clase: primero la capacidad y después la optimización)
    1. Tamaño y profundidad: n_oculta {32, 64, 128} x n_capas {1, 2}, con lr 1e-3 y salida sin
       activación. Con 2 capas se añade un dropout de 0,2 entre ellas: el dropout de PyTorch solo
       actúa entre capas recurrentes, así que con 1 capa no haría nada. La capacidad va primero porque
       fija el coste de todo lo demás.
    2. Sobre la elegida en el punto 1: lr {1e-3, 3e-4} x activación de salida {"ninguna", "softplus"}.
       Un lr menor da pasos más finos cerca del mínimo a cambio de más épocas; la softplus obliga a que
       la prima no sea negativa (ver packages/rnn_modelos.py). La combinación de clase (lr 1e-3, sin
       activación) ya se entrenó en el punto 1 y se reutiliza: con la misma semilla daría lo mismo.

Criterio y regla de empate
    Se elige por la pérdida de validación de los 20 pasos, la misma con la que se eligió la estructura
    en rnn_02: entre redes causales, el ruido de las etiquetas intermedias es el mismo y la diferencia
    mide el error de la prima en cada fecha. Para cada configuración se calcula la diferencia con la
    mejor y su intervalo del 95 % con un bootstrap pareado (1.000 remuestreos de las mismas
    trayectorias de validación, semilla 42). Si el intervalo contiene el 0, es un empate y se elige la
    de menos parámetros (rendimiento frente a parámetros, como en clase). En el punto 2 todas tienen
    los mismos parámetros, así que entre empatadas gana la que menos se aparta de la configuración
    de clase: sin activación de salida, por ser la más simple (la softplus solo se justifica si mejora
    la pérdida de forma significativa), y con lr 1e-3. El texto de la elección da además las épocas
    y el tiempo de entrenamiento de cada lr: a igualdad de pérdida, el que converge antes es más barato.

Validación cruzada (5 bloques sobre entrenamiento + validación, el 85 % de las trayectorias)
    - Cada bloque se evalúa con una red entrenada en los otros cuatro. De esos cuatro se separa un
      15 % interno para el early stopping, así que el bloque evaluado no interviene ni en el ajuste
      ni en la parada.
    - El escalado se recalcula en cada bloque con sus datos de ajuste, para que el bloque evaluado no
      se filtre a través de la media y la desviación.
    - Test (el 15 % restante) no se toca: se reserva para la evaluación única de los modelos finales.
    - Se da media ± desviación de los 5 bloques: la media estima el rendimiento esperado y la
      desviación, cuánto depende de con qué trayectorias se entrena.
    - Regresión: pérdida de 20 pasos (MSE con la desviación de Y de rnn_dataset.npz, para que sea
      comparable con la búsqueda) y RMSE del payoff en el paso 20. Clasificación: accuracy y F1 en el
      paso 20, con umbral 0,5.

Modelos finales
    - Regresión: la configuración elegida, entrenada en la división estándar (70 % ajuste y 15 %
      parada). Es el mismo modelo que se entrenó en la búsqueda, así que no se vuelve a entrenar.
    - Clasificación: la misma configuración con BCEWithLogitsLoss, en la misma división.
    - Los dos se evalúan una sola vez en test. Su tiempo de entrenamiento lo usa rnn_05_coste.py.

Reanudación (la ejecución completa dura horas)
    Cada configuración de la búsqueda, cada mitad (regresión y clasificación) de cada bloque de la
    validación cruzada y el clasificador final se guardan en outputs/rnn_03_progreso/ nada más
    terminar. Se escribe primero en un archivo temporal y después se renombra, para que un corte a
    mitad de escritura no deje un archivo roto. Si el script se relanza, recupera lo guardado y sigue
    por donde iba. El resultado es el mismo que sin corte, porque cada entrenamiento fija su propia
    semilla y no depende de los anteriores.
    El progreso solo se reutiliza con la misma estructura, el mismo número de trayectorias y el mismo
    presupuesto de entrenamiento (firma.json). Por eso --rapido guarda el suyo en
    outputs/rapido/rnn_03_progreso/ y la ejecución completa nunca lo reutiliza. Para empezar de cero,
    se borra la carpeta de progreso.
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
from matplotlib.patches import Rectangle
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error,
                             r2_score)
from sklearn.model_selection import KFold
from torch import nn

from packages.bbva_graficos import (BBVA, CMAP_SECUENCIAL, configurar_graficos, formato_es, guardar_mpl,
                                    imprimir, texto_es)
from packages.rnn_modelos import (DISPOSITIVO, RUTA_DATASET, RedRecurrente, bootstrap_pareado, cargar_datos,
                                  contar_parametros, entrenar, evaluar, perdidas_20_pasos, salida_bruta)

parser = argparse.ArgumentParser(description="Búsqueda de hiperparámetros y validación cruzada")
parser.add_argument("--rapido", action="store_true", help="20.000 trayectorias y 20 épocas como máximo")
args = parser.parse_args()

SEMILLA = 42
N_MAX = 20_000 if args.rapido else None
ENTRENAMIENTO = dict(batch=512, max_epocas=20 if args.rapido else 100, paciencia=10, min_delta=1e-6)
TAMANOS, CAPAS = (32, 64, 128), (1, 2)
LRS, ACTIVACIONES = (1e-3, 3e-4), ("ninguna", "softplus")
LR_TEXTO = {1e-3: "1e-3", 3e-4: "3e-4"}
N_BLOQUES = 5
PROP_PARADA = 0.15          # parte interna de cada bloque para el early stopping

inicio_script = time.perf_counter()
os.chdir(Path(__file__).resolve().parent)   # rutas relativas a la raíz del repo
CARPETA_SALIDA = Path("outputs/rapido") if args.rapido else Path("outputs")
CARPETA_MODELOS = CARPETA_SALIDA if args.rapido else Path("modelos")
CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)
CARPETA_MODELOS.mkdir(exist_ok=True)
configurar_graficos(CARPETA_SALIDA if args.rapido else "Graficos")
FUENTE = (f"{'20.000' if args.rapido else '100.000'} trayectorias de Montecarlo de rnn_dataset.npz "
          "(validación: 15 % de las trayectorias).")
pd.options.display.width = 250

eleccion = json.loads(Path("outputs/rnn_eleccion_arquitectura.json").read_text(encoding="utf-8"))
assert not eleccion["rapido"], "PARA: outputs/rnn_eleccion_arquitectura.json no es de la ejecución completa de rnn_02"
TIPO, NOMBRE = eleccion["tipo_causal"], eleccion["mejor_causal"]
imprimir(f"Modo: {'rápido (20,000 trayectorias, 20 épocas como máximo)' if args.rapido else 'completo'}; "
         f"estructura: {NOMBRE} (elegida en rnn_02_arquitecturas.py); dispositivo: {DISPOSITIVO}")

# Progreso para reanudar: cada resultado se guarda nada más terminar (ver "Reanudación")
CARPETA_PROGRESO = CARPETA_SALIDA / "rnn_03_progreso"
CARPETA_PROGRESO.mkdir(exist_ok=True)


def guardar_atomico(guardar, ruta):
    """Escribe en un temporal y lo renombra: un corte a mitad de escritura no deja un archivo roto."""
    temporal = ruta.with_name(ruta.name + ".tmp")
    guardar(temporal)
    os.replace(temporal, ruta)


def guardar_json(objeto, ruta):
    guardar_atomico(lambda r: r.write_text(json.dumps(objeto, ensure_ascii=False, indent=2), encoding="utf-8"), ruta)


FIRMA = {"estructura": TIPO, "n_max": N_MAX, "semilla": SEMILLA, **ENTRENAMIENTO}
ruta_firma = CARPETA_PROGRESO / "firma.json"
if ruta_firma.exists():
    assert json.loads(ruta_firma.read_text(encoding="utf-8")) == FIRMA, \
        f"PARA: {CARPETA_PROGRESO} es de otra configuración; hay que borrarla para empezar de cero"
else:
    guardar_json(FIRMA, ruta_firma)
guardados = len(list(CARPETA_PROGRESO.glob("*.pt"))) + len(list(CARPETA_PROGRESO.glob("cv_*.json")))
imprimir(f"Progreso en {CARPETA_PROGRESO}: {guardados} resultados ya guardados")

datos = cargar_datos("Y", N_MAX)
payoff = np.load(RUTA_DATASET, allow_pickle=False)["payoff"]
payoff_val, payoff_test = payoff[datos["idx_val"]], payoff[datos["idx_test"]]


def configuracion(n_oculta, n_capas, lr=1e-3, activacion="ninguna"):
    return {"tipo": TIPO, "n_oculta": n_oculta, "n_capas": n_capas, "dropout": 0.2 if n_capas == 2 else 0.0,
            "lr": lr, "activacion_salida": activacion}


def etiqueta(config):
    return f"{config['n_oculta']}x{config['n_capas']}, lr {LR_TEXTO[config['lr']]}, {config['activacion_salida']}"


def crear_red(config, minimo_salida=0.0):
    """Red con los pesos iniciales de la semilla 42."""
    torch.manual_seed(SEMILLA)
    return RedRecurrente(config["tipo"], n_oculta=config["n_oculta"], n_capas=config["n_capas"],
                         dropout=config["dropout"], activacion_salida=config["activacion_salida"],
                         minimo_salida=minimo_salida)


filas, modelos, perdidas = [], {}, {}


def probar(fase, config):
    """Entrena una configuración de regresión en la división estándar y la guarda nada más terminar.

    Si ya se entrenó en esta ejecución, se reutiliza; si está guardada de una ejecución cortada, se recupera.
    """
    e = etiqueta(config)
    if e in modelos:
        imprimir(f"  {e}: ya entrenada en la fase 1, se reutiliza")
        return e
    ruta = CARPETA_PROGRESO / f"busqueda_{e.replace(', ', '_').replace(' ', '')}.pt"
    modelo = crear_red(config, -datos["media_y"] / datos["desv_y"])
    if ruta.exists():
        guardado = torch.load(ruta, weights_only=True)
        modelo.load_state_dict(guardado["estado"])
        res, fila, perdidas[e] = guardado["res"], guardado["fila"], guardado["perdidas"].numpy()
        imprimir(f"  {e}: recuperada de {ruta} (pérdida de 20 pasos {fila['val_perdida_20_pasos']:.4f})")
    else:
        res = entrenar(modelo, datos, nn.MSELoss(), lr=config["lr"], **ENTRENAMIENTO)
        perdidas[e] = perdidas_20_pasos(modelo, datos["x_val"], datos["y_val"])
        assert np.isclose(perdidas[e].mean(), res["mejor_val"], rtol=1e-3), "pérdida de 20 pasos incoherente"
        pred_val = evaluar(modelo, datos["x_val"], datos["media_y"], datos["desv_y"])[:, -1]
        fila = {"configuracion": e, "fase": fase, **config, "parametros": contar_parametros(modelo),
                "epocas": res["epocas"], "mejor_epoca": res["mejor_epoca"], "tiempo_entrenamiento_s": res["tiempo"],
                "val_perdida_20_pasos": res["mejor_val"],
                "val_rmse_paso20": float(np.sqrt(mean_squared_error(payoff_val, pred_val)))}
        guardar_atomico(lambda r: torch.save({"estado": modelo.state_dict(), "res": res, "fila": fila,
                                              "perdidas": torch.from_numpy(perdidas[e])}, r), ruta)
        imprimir(f"  {e}: {contar_parametros(modelo):,} parámetros, {res['epocas']} épocas (mejor "
                 f"{res['mejor_epoca']}), pérdida de 20 pasos {res['mejor_val']:.4f}, {res['tiempo']:.0f} s; guardada")
    modelos[e] = (modelo, res)
    filas.append(fila)
    return e


def elegir(tabla, candidatas, prefijo):
    """La menor pérdida de 20 pasos entre `candidatas` y la elegida con la regla de empate.

    Empatan con la mejor las que tienen un IC del 95 % bootstrap de la diferencia que contiene el 0.
    Entre empatadas: menos parámetros y, a igualdad, menos cambios sobre la configuración de clase.
    Añade a la tabla, con el prefijo de la fase, la diferencia, su intervalo y si hay empate.
    """
    mejor = tabla.loc[candidatas, "val_perdida_20_pasos"].idxmin()
    for columna in ("dif_vs_mejor", "ic95_inf", "ic95_sup"):
        tabla[prefijo + columna] = np.nan
    tabla[prefijo + "empate"] = pd.Series(pd.NA, index=tabla.index, dtype="boolean")
    for e in candidatas:
        inferior, superior = bootstrap_pareado(perdidas[e], perdidas[mejor])
        tabla.loc[e, prefijo + "dif_vs_mejor"] = (tabla.loc[e, "val_perdida_20_pasos"]
                                                 - tabla.loc[mejor, "val_perdida_20_pasos"])
        tabla.loc[e, prefijo + "ic95_inf"], tabla.loc[e, prefijo + "ic95_sup"] = inferior, superior
        tabla.loc[e, prefijo + "empate"] = bool(inferior <= 0 <= superior)
    empatadas = tabla.loc[candidatas][tabla.loc[candidatas, prefijo + "empate"].astype(bool)]
    cambios = (empatadas["lr"] != 1e-3).astype(int) + (empatadas["activacion_salida"] != "ninguna").astype(int)
    elegida = empatadas.assign(cambios=cambios).sort_values(["parametros", "cambios", "val_perdida_20_pasos"]).index[0]
    return mejor, elegida


def explicar(tabla, candidatas, mejor, elegida, prefijo):
    """La elección de una fase en palabras, con las diferencias y sus intervalos."""
    t = tabla.loc[candidatas].sort_values("val_perdida_20_pasos")
    texto = (f"Menor pérdida de 20 pasos en validación: {mejor} ({t.loc[mejor, 'val_perdida_20_pasos']:.4f}). "
             "Diferencia de las demás frente a ella, con su IC del 95 % por bootstrap pareado: "
             + "; ".join(f"{e}, {t.loc[e, prefijo + 'dif_vs_mejor']:+.4f} [{t.loc[e, prefijo + 'ic95_inf']:+.4f}, "
                         f"{t.loc[e, prefijo + 'ic95_sup']:+.4f}]{' (empate)' if t.loc[e, prefijo + 'empate'] else ''}"
                         for e in t.index if e != mejor) + ". ")
    if elegida == mejor:
        return texto + f"Se elige {mejor}."
    if t.loc[elegida, "parametros"] < t.loc[mejor, "parametros"]:
        return texto + (f"Es un empate y se elige {elegida} por tener menos parámetros "
                        f"({t.loc[elegida, 'parametros']:,.0f} frente a {t.loc[mejor, 'parametros']:,.0f}).")
    empatadas = t[t[prefijo + "empate"].astype(bool)]
    razones = []
    if t.loc[elegida, "activacion_salida"] == "ninguna" and (empatadas["activacion_salida"] != "ninguna").any():
        razones.append("sin activación de salida, por ser la más simple (la softplus solo se justificaría si "
                       "mejorara la pérdida de forma significativa)")
    otro_lr = empatadas[empatadas["lr"] != t.loc[elegida, "lr"]]
    if len(otro_lr):
        # Se compara con la empatada de otro lr y la misma activación, si la hay
        misma = otro_lr[otro_lr["activacion_salida"] == t.loc[elegida, "activacion_salida"]]
        o = (misma if len(misma) else otro_lr).iloc[0]
        e1, e2 = t.loc[elegida, "epocas"], o["epocas"]
        if e1 < e2:
            razones.append(
                f"lr {LR_TEXTO[t.loc[elegida, 'lr']]} frente a {LR_TEXTO[o['lr']]}, porque converge en "
                f"{'la mitad de' if e1 <= 0.55 * e2 else 'menos'} épocas ({e1:.0f} frente a {e2:.0f}"
                f"{', el máximo' if e2 >= ENTRENAMIENTO['max_epocas'] else ''}; "
                f"{t.loc[elegida, 'tiempo_entrenamiento_s']:.0f} s frente a {o['tiempo_entrenamiento_s']:.0f} s "
                f"de entrenamiento)")
        else:
            razones.append(f"el lr de clase ({LR_TEXTO[t.loc[elegida, 'lr']]})")
    return texto + (f"Empatan {len(empatadas)} configuraciones con los mismos parámetros ({'; '.join(empatadas.index)}), "
                    f"así que se elige {elegida}: " + "; y ".join(razones) + ".")


# ============================ BÚSQUEDA ============================
imprimir("\nFase 1: tamaño y capas (lr 1e-3, sin activación de salida)")
fase1 = [probar(1, configuracion(n, c)) for c in CAPAS for n in TAMANOS]
tabla = pd.DataFrame(filas).set_index("configuracion")
_, elegida1 = elegir(tabla, fase1, "f1_")
base = tabla.loc[elegida1]

imprimir(f"\nFase 2: lr y activación de salida sobre {elegida1}")
fase2 = [probar(2, configuracion(int(base["n_oculta"]), int(base["n_capas"]), lr, activacion))
         for lr in LRS for activacion in ACTIVACIONES]
tabla = pd.DataFrame(filas).set_index("configuracion")
mejor1, elegida1 = elegir(tabla, fase1, "f1_")
mejor2, elegida2 = elegir(tabla, fase2, "f2_")
tabla["elegida"] = tabla.index == elegida2
texto_fase1 = "Fase 1 (tamaño y capas): " + explicar(tabla, fase1, mejor1, elegida1, "f1_")
texto_fase2 = "Fase 2 (lr y activación de salida): " + explicar(tabla, fase2, mejor2, elegida2, "f2_")

if not args.rapido:
    previa = pd.read_csv("outputs/rnn_comparacion_regresion.csv", index_col="modelo").loc[NOMBRE, "val_perdida_20_pasos"]
    base_clase = etiqueta(configuracion(64, 1))
    imprimir(f"\nComprobación: {base_clase} reproduce la pérdida de la {NOMBRE} de rnn_02 ({previa:.4f}): "
             f"{'sí' if np.isclose(tabla.loc[base_clase, 'val_perdida_20_pasos'], previa, rtol=1e-6) else 'NO'}")

final = tabla.loc[elegida2]
config_final = {"tipo": TIPO, "n_oculta": int(final["n_oculta"]), "n_capas": int(final["n_capas"]),
                "dropout": float(final["dropout"]), "lr": float(final["lr"]),
                "activacion_salida": str(final["activacion_salida"])}
config_clf = {**config_final, "activacion_salida": "ninguna"}
modelo_reg, res_reg = modelos[elegida2]

# ======================= VALIDACIÓN CRUZADA =======================
d = np.load(RUTA_DATASET, allow_pickle=False)
entradas, Y, itm = d["entradas"], d["Y"], d["itm"]
desv_Y_global = float(d["desv_Y"])
idx_trval = np.sort(np.concatenate((datos["idx_train"], datos["idx_val"])))
imprimir(f"\nValidación cruzada de {N_BLOQUES} bloques sobre {len(idx_trval):,} trayectorias (entrenamiento + "
         f"validación), con {elegida2}")
filas_cv = []
for bloque, (pos_resto, pos_eval) in enumerate(KFold(N_BLOQUES, shuffle=True, random_state=SEMILLA)
                                               .split(idx_trval), start=1):
    resto = np.random.default_rng(SEMILLA + bloque).permutation(idx_trval[pos_resto])
    n_parada = round(PROP_PARADA * len(resto))
    parada, ajuste, evaluado = np.sort(resto[:n_parada]), np.sort(resto[n_parada:]), idx_trval[pos_eval]

    # Escalado con los datos de ajuste del bloque: el bloque evaluado no influye en la media ni en la desviación
    canales = entradas.shape[2]
    media_x = entradas[ajuste].reshape(-1, canales).mean(axis=0)
    desv_x = entradas[ajuste].reshape(-1, canales).std(axis=0)
    x = {c: ((entradas[i] - media_x) / desv_x).astype(np.float32)
         for c, i in (("ajuste", ajuste), ("parada", parada), ("evaluado", evaluado))}
    media_y, desv_y = float(Y[ajuste].mean()), float(Y[ajuste].std())

    # Regresión del bloque: se recupera si ya está guardada; si no, se entrena y se guarda al terminar
    ruta_r = CARPETA_PROGRESO / f"cv_bloque{bloque}_regresion.json"
    if ruta_r.exists():
        r = json.loads(ruta_r.read_text(encoding="utf-8"))
    else:
        modelo = crear_red(config_final, -media_y / desv_y)
        res = entrenar(modelo, {"x_train": x["ajuste"],
                                "y_train": ((Y[ajuste] - media_y) / desv_y).astype(np.float32),
                                "x_val": x["parada"], "y_val": ((Y[parada] - media_y) / desv_y).astype(np.float32)},
                       nn.MSELoss(), lr=config_final["lr"], **ENTRENAMIENTO)
        prima = salida_bruta(modelo, x["evaluado"]) * desv_y + media_y      # prima por paso, sin recortar
        r = {"regresion_epocas": res["epocas"], "regresion_tiempo_s": res["tiempo"],
             "perdida_20_pasos": float(np.mean(((prima - Y[evaluado]) / desv_Y_global) ** 2)),
             "rmse_paso20": float(np.sqrt(mean_squared_error(payoff[evaluado], np.maximum(prima[:, -1], 0))))}
        guardar_json(r, ruta_r)

    # Clasificación del bloque, igual
    ruta_c = CARPETA_PROGRESO / f"cv_bloque{bloque}_clasificacion.json"
    if ruta_c.exists():
        k = json.loads(ruta_c.read_text(encoding="utf-8"))
    else:
        modelo_c = crear_red(config_clf)
        res_c = entrenar(modelo_c, {"x_train": x["ajuste"], "y_train": itm[ajuste],
                                    "x_val": x["parada"], "y_val": itm[parada]},
                         nn.BCEWithLogitsLoss(), lr=config_final["lr"], **ENTRENAMIENTO)
        clase = (evaluar(modelo_c, x["evaluado"], clasificacion=True)[:, -1] >= 0.5).astype(int)
        itm_evaluado = itm[evaluado, -1].astype(int)
        k = {"clasificacion_epocas": res_c["epocas"], "clasificacion_tiempo_s": res_c["tiempo"],
             "accuracy": float(accuracy_score(itm_evaluado, clase)), "f1": float(f1_score(itm_evaluado, clase))}
        guardar_json(k, ruta_c)

    filas_cv.append({"bloque": bloque, "n_ajuste": len(ajuste), "n_parada": len(parada),
                     "n_evaluado": len(evaluado), **r, **k})
    imprimir(f"  Bloque {bloque}: pérdida de 20 pasos {r['perdida_20_pasos']:.4f}, RMSE paso 20 {r['rmse_paso20']:.4f}, "
             f"accuracy {k['accuracy']:.4f}, F1 {k['f1']:.4f} "
             f"({r['regresion_tiempo_s'] + k['clasificacion_tiempo_s']:.0f} s de entrenamiento; guardado)")
cv = pd.DataFrame(filas_cv).set_index("bloque")
resumen_cv = {m: (float(cv[m].mean()), float(cv[m].std(ddof=1)))
              for m in ("perdida_20_pasos", "rmse_paso20", "accuracy", "f1")}

# ======================== MODELOS FINALES ========================
imprimir("\nModelo final de clasificación (misma configuración, BCEWithLogitsLoss)")
datos_itm = cargar_datos("itm", N_MAX)
modelo_clf = crear_red(config_clf)
ruta_clf = CARPETA_PROGRESO / "final_clasificacion.pt"
if ruta_clf.exists():
    guardado = torch.load(ruta_clf, weights_only=True)
    modelo_clf.load_state_dict(guardado["estado"])
    res_clf = guardado["res"]
    imprimir(f"  recuperado de {ruta_clf}")
else:
    res_clf = entrenar(modelo_clf, datos_itm, nn.BCEWithLogitsLoss(), lr=config_final["lr"], **ENTRENAMIENTO)
    guardar_atomico(lambda r: torch.save({"estado": modelo_clf.state_dict(), "res": res_clf}, r), ruta_clf)
imprimir(f"  {res_clf['epocas']} épocas (mejor {res_clf['mejor_epoca']}), {res_clf['tiempo']:.0f} s")

# Evaluación única en test
pred_test = evaluar(modelo_reg, datos["x_test"], datos["media_y"], datos["desv_y"])[:, -1]
test_reg = {"perdida_20_pasos": float(perdidas_20_pasos(modelo_reg, datos["x_test"], datos["y_test"]).mean()),
            "mae": float(mean_absolute_error(payoff_test, pred_test)),
            "rmse": float(np.sqrt(mean_squared_error(payoff_test, pred_test))),
            "r2": float(r2_score(payoff_test, pred_test))}
itm_test = datos_itm["y_test"][:, -1].astype(int)
clase_test = (evaluar(modelo_clf, datos_itm["x_test"], clasificacion=True)[:, -1] >= 0.5).astype(int)
vn, fp, fn, vp = confusion_matrix(itm_test, clase_test, labels=[0, 1]).ravel()
test_clf = {"accuracy": float(accuracy_score(itm_test, clase_test)), "f1": float(f1_score(itm_test, clase_test)),
            "vn": int(vn), "fp": int(fp), "fn": int(fn), "vp": int(vp)}

for modelo, config, datos_m, res, nombre in ((modelo_reg, config_final, datos, res_reg, "regresion"),
                                              (modelo_clf, config_clf, datos_itm, res_clf, "clasificacion")):
    torch.save({"config": {k: config[k] for k in ("tipo", "n_oculta", "n_capas", "dropout", "activacion_salida")},
                "lr": config["lr"], "estado": modelo.state_dict(), "media_y": datos_m["media_y"],
                "desv_y": datos_m["desv_y"], "tiempo_entrenamiento_s": res["tiempo"], "epocas": res["epocas"]},
               CARPETA_MODELOS / f"rnn_final_{nombre}.pt")

texto_cv = (f"Validación cruzada de {N_BLOQUES} bloques ({elegida2}): pérdida de 20 pasos "
            f"{resumen_cv['perdida_20_pasos'][0]:.4f} ± {resumen_cv['perdida_20_pasos'][1]:.4f}, RMSE del payoff en "
            f"el paso 20 {resumen_cv['rmse_paso20'][0]:.4f} ± {resumen_cv['rmse_paso20'][1]:.4f}; en clasificación, "
            f"accuracy {resumen_cv['accuracy'][0]:.4f} ± {resumen_cv['accuracy'][1]:.4f} y F1 "
            f"{resumen_cv['f1'][0]:.4f} ± {resumen_cv['f1'][1]:.4f} (media ± desviación de los bloques).")
texto_test = (f"Test (una sola evaluación): regresión con pérdida de 20 pasos {test_reg['perdida_20_pasos']:.4f}, "
              f"RMSE del payoff {test_reg['rmse']:.4f}, MAE {test_reg['mae']:.4f} y R² {test_reg['r2']:.4f}; "
              f"clasificación con accuracy {test_clf['accuracy']:.4f} y F1 {test_clf['f1']:.4f} "
              f"({fp + fn:,} errores de {len(itm_test):,}).")

tabla.to_csv(CARPETA_SALIDA / "rnn_busqueda_hiperparametros.csv")
cv.to_csv(CARPETA_SALIDA / "rnn_validacion_cruzada.csv")
with open(CARPETA_SALIDA / "rnn_configuracion_final.json", "w", encoding="utf-8") as archivo:
    json.dump({
        "estructura": NOMBRE, "configuracion": config_final, "configuracion_clasificacion": config_clf,
        "busqueda": {"fase1_mejor": mejor1, "fase1_elegida": elegida1, "fase2_mejor": mejor2, "fase2_elegida": elegida2},
        "validacion_cruzada": {m: {"media": a, "desviacion": b} for m, (a, b) in resumen_cv.items()},
        "test_regresion": test_reg, "test_clasificacion": test_clf,
        "entrenamiento_final": {"regresion_tiempo_s": res_reg["tiempo"], "regresion_epocas": res_reg["epocas"],
                                "clasificacion_tiempo_s": res_clf["tiempo"], "clasificacion_epocas": res_clf["epocas"]},
        "rapido": args.rapido,
        "textos": {c: texto_es(t) for c, t in (("fase1", texto_fase1), ("fase2", texto_fase2),
                                                ("validacion_cruzada", texto_cv), ("test", texto_test))},
    }, archivo, indent=2, ensure_ascii=False)

columnas = ["fase", "parametros", "epocas", "tiempo_entrenamiento_s", "val_perdida_20_pasos", "val_rmse_paso20",
            "f1_dif_vs_mejor", "f1_ic95_inf", "f1_ic95_sup", "f1_empate", "f2_dif_vs_mejor", "f2_ic95_inf",
            "f2_ic95_sup", "f2_empate"]
imprimir("\nBúsqueda de hiperparámetros (validación):")
imprimir(tabla[columnas].to_string(float_format=lambda v: f"{v:.4f}"))
imprimir("\nValidación cruzada por bloques:")
imprimir(cv.to_string(float_format=lambda v: f"{v:.4f}"))
imprimir("\n" + "\n\n".join((texto_fase1, texto_fase2, texto_cv, texto_test)))

# ============================ GRÁFICO ============================
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for ax, columna, titulo in zip(axes, ["val_perdida_20_pasos", "val_rmse_paso20"],
                               ["Pérdida de validación de 20 pasos (criterio)",
                                "RMSE del payoff en el paso 20 (validación)"]):
    matriz = np.array([[tabla.loc[etiqueta(configuracion(n, c)), columna] for n in TAMANOS] for c in CAPAS])
    imagen = ax.imshow(matriz, cmap=CMAP_SECUENCIAL, aspect="auto")
    # La escala de color va del mínimo al máximo del panel: la barra y el rango del título evitan que
    # diferencias pequeñas parezcan grandes
    barra = fig.colorbar(imagen, ax=ax, fraction=0.05, pad=0.03)
    barra.set_label("Más oscuro = más error", color=BBVA["marino"])
    titulo = f"{titulo}\nrango de la escala: de {matriz.min():.4f} a {matriz.max():.4f}"
    for i, c in enumerate(CAPAS):
        for j, n in enumerate(TAMANOS):
            e = etiqueta(configuracion(n, c))
            oscuro = matriz[i, j] > matriz.min() + 0.5 * (matriz.max() - matriz.min())
            ax.text(j, i, f"{matriz[i, j]:.4f}\n{tabla.loc[e, 'parametros']:,.0f} parámetros", ha="center",
                    va="center", fontsize=12, color=BBVA["blanco"] if oscuro else BBVA["marino"],
                    fontweight="bold" if e == elegida1 else "normal")
    ax.add_patch(Rectangle((TAMANOS.index(int(base["n_oculta"])) - 0.5, CAPAS.index(int(base["n_capas"])) - 0.5),
                           1, 1, fill=False, edgecolor=BBVA["agua"], linewidth=4))
    ax.set_xticks(range(len(TAMANOS)), [f"{n} neuronas" for n in TAMANOS])
    ax.set_yticks(range(len(CAPAS)), [f"{c} capa{'s' if c > 1 else ''}" for c in CAPAS])
    ax.set_title(titulo, fontsize=13)
    ax.grid(False)
fig.suptitle(f"Gráfico 1. Tamaño y capas de la {NOMBRE} (recuadro: configuración elegida en la fase 1)", x=0.01,
             ha="left", fontsize=16, fontweight="bold", color=BBVA["marino"])
fig.tight_layout()
formato_es(fig, fuente=FUENTE)
guardar_mpl(fig, "rnn_hiperparametros")
plt.close(fig)
imprimir(f"\nTiempo total: {(time.perf_counter() - inicio_script) / 60:.1f} min")
