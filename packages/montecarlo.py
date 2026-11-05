"""Simulación de Montecarlo de la cartera CAT-MRK-MSFT y valoración de la call asiática del reto.

Ruta en el repositorio: packages/montecarlo.py
Entrada: ninguna. Los parámetros (sigma, rho) son los de outputs/parametros_montecarlo.json,
calculados en 04_Seleccion_cartera.ipynb sobre Datos/Transformados/dowjones_diario_2016_2025.csv.
Salida: arrays en memoria. Las exportaciones las hace el notebook o script que llama a estas funciones.

Modelo: movimiento browniano geométrico correlacionado con Cholesky, en medida neutral al riesgo (mu = r).
Payoff del enunciado: V_T = max(G_CAT + G_MRK + G_MSFT - 3, 0), con G_k la media geométrica de los
20 precios normalizados del activo k (sin incluir S0).

Por qué un módulo: el notebook de Montecarlo y los scripts de la red neuronal usan exactamente
las mismas funciones, así las trayectorias, el payoff y la prima no pueden divergir entre partes.
"""
import numpy as np

ACTIVOS = ["CAT", "MRK", "MSFT"]
SIGMA = np.array([0.304449, 0.226614, 0.267572])
RHO = np.array([
    [1.0, 0.249833, 0.394764],
    [0.249833, 1.0, 0.276564],
    [0.394764, 0.276564, 1.0],
])
R = 0.01
TTM = 5
NUM_PASOS = 20
DT = TTM / NUM_PASOS
TIEMPOS = np.arange(1, NUM_PASOS + 1) * DT     # 0,25 ... 5 (sin t = 0)
S0 = np.ones(3)
STRIKE = 3
L = np.linalg.cholesky(RHO)


def _incrementos(Z):
    """Incrementos logarítmicos del GBM a partir de normales estándar ya correlacionadas."""
    return (R - 0.5 * SIGMA**2) * DT + SIGMA * np.sqrt(DT) * Z


def simular_historia(k, rng):
    """Una única trayectoria de k pasos (precios normalizados, forma (k, 3)).

    Consume del generador exactamente lo mismo que el tramo común del notebook:
    rng.standard_normal((1, k, 3)).
    """
    Z = rng.standard_normal((1, k, 3)) @ L.T
    return (S0 * np.exp(np.cumsum(_incrementos(Z), axis=1)))[0]


def simular_continuaciones(historia, num_sims, rng):
    """Matriz X (num_sims, 20, 3) de precios normalizados.

    historia = None o de longitud 0: los 20 pasos son independientes entre simulaciones.
    historia de forma (k, 3): los k primeros pasos se fijan a esa historia en todas las
    simulaciones y los 20 - k restantes se simulan de forma independiente desde el último precio.
    """
    k = 0 if historia is None else len(historia)
    Z = rng.standard_normal((num_sims, NUM_PASOS - k, 3)) @ L.T
    punto_partida = S0 if k == 0 else historia[-1]
    futuro = punto_partida * np.exp(np.cumsum(_incrementos(Z), axis=1))
    if k == 0:
        return futuro
    comun = np.broadcast_to(historia, (num_sims, k, 3))
    return np.concatenate((comun, futuro), axis=1)


def medias_geometricas(X):
    """Media geométrica temporal de cada activo, forma (num_sims, 3)."""
    return np.exp(np.mean(np.log(X / S0), axis=1))


def calcular_payoff(X):
    """Payoff del enunciado: max(suma de las tres medias geométricas - strike, 0)."""
    return np.maximum(medias_geometricas(X).sum(axis=1) - STRIKE, 0)


def calcular_Y(payoff):
    """Payoff descontado a cada una de las 20 fechas, forma (num_sims, 20)."""
    return payoff[:, None] * np.exp(-R * (TTM - TIEMPOS))[None, :]


def retornos_log(X):
    """Retornos logarítmicos por paso (el primero respecto a S0), forma (num_sims, 20, 3)."""
    return np.diff(np.log(X / S0), axis=1, prepend=0.0)


def prima_condicionada(historia, num_sims, rng):
    """Prima en t_k dada una historia de k pasos (k = 0: prima inicial). Devuelve (prima, error estándar)."""
    k = 0 if historia is None else len(historia)
    payoff = calcular_payoff(simular_continuaciones(historia, num_sims, rng))
    descontados = payoff * np.exp(-R * (TTM - k * DT))
    return descontados.mean(), descontados.std(ddof=1) / np.sqrt(num_sims)
