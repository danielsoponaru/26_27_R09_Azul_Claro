"""Redes recurrentes del reto: modelo, carga de datos, entrenamiento y evaluación (issues #31-#34).

Ruta en el repositorio: packages/rnn_modelos.py
Entrada: Datos/Transformados/rnn_dataset.npz (lo genera rnn_01_datos.py), que lee cargar_datos().
Salida: modelos y predicciones en memoria. Los pesos los guardan los scripts rnn_0X que usan el módulo.
Prueba de formas (desde la raíz del repositorio): python -m packages.rnn_modelos

Por qué un módulo: los scripts de arquitecturas, hiperparámetros, prima y coste entrenan y evalúan
las mismas redes con el mismo escalado. Si cada script lo hiciera a su manera, las comparaciones
entre ellos dejarían de ser válidas.

Una salida en cada paso (many-to-many)
    La capa recurrente devuelve su estado oculto en los 20 pasos y la misma nn.Linear se aplica a
    todos, así que la salida tiene forma (lote, 20). La pérdida se calcula en cada paso y se suma,
    como en la diapositiva 7 de "Architectures adapted to data". MSELoss promedia en lugar de sumar:
    es la suma dividida entre 20 x lote, tiene el mismo óptimo y la escala del learning rate no
    depende del número de pasos.
    Por qué en cada paso: con MSE, la predicción óptima en el paso k es el valor esperado del payoff
    descontado dado lo observado hasta t_k, es decir, la prima en t_k. En el paso 20 el objetivo es el
    payoff, que es una función exacta de la trayectoria, así que ahí el error frente a Y sí mide la red.
    En los pasos 1-19 la etiqueta (el payoff descontado de una trayectoria concreta) es ruidosa: su
    error frente a Y no mide la red, porque ni la prima exacta lo tendría bajo. Esas primas se validan
    contra Montecarlo condicionado en rnn_04_prima.py.

Las cuatro estructuras de clase (1 capa de 64 neuronas, mismo presupuesto)
    - RNN simple: un único bloque con tanh. Es la más barata, pero sufre el desvanecimiento del
      gradiente cuando tiene que recordar muchos pasos.
    - LSTM: estado de celda y tres puertas (olvido, entrada y salida) para conservar información a lo
      largo de la secuencia. Aquí tiene que acumular los 20 retornos para formar las medias geométricas.
    - GRU: dos puertas (reset y update) y sin estado de celda separado: menos parámetros que la LSTM
      con el mismo tamaño oculto.
    - LSTM bidireccional (#34): una LSTM recorre la secuencia hacia delante y otra hacia atrás. Solo es
      válida para el payoff al vencimiento, porque entonces la trayectoria completa ya se conoce. Para
      la prima en t no, porque en el paso t usaría fechas futuras: estaría prediciendo con información
      que el mercado todavía no ha dado. Por lo mismo, su pérdida sobre los 20 pasos no es comparable
      con la de las causales (en los pasos intermedios ya "conoce" el payoff); las estructuras se
      comparan por el error en el paso 20.

Activación de salida (hiperparámetro, se decide en rnn_03_hiperparametros.py)
    - "ninguna": regresión sin activación, como en clase. La prima no puede ser negativa, así que las
      predicciones se recortan a >= 0 al evaluar.
    - "softplus": la "regresión estrictamente positiva" de clase. Se elige softplus y no ReLU porque el
      55,9 % de los payoffs son 0: con ReLU, la salida se quedaría en la zona plana con gradiente 0
      para buena parte de las trayectorias. Como Y se escala con (Y - media) / desv, el cero de la prima
      está en -media / desv; por eso la salida es minimo_salida + softplus(z), con
      minimo_salida = -media / desv, y la prima desescalada sale siempre >= 0. Así el escalado de Y es
      el mismo con las dos activaciones y sus pérdidas de validación se pueden comparar.
    - Clasificación: siempre "ninguna". La red devuelve logits y se entrena con BCEWithLogitsLoss, que
      ya incluye la sigmoide (aplicarla también en la red la duplicaría). evaluar() devuelve la
      probabilidad.

Entrenamiento (entrenar)
    Bucle de clase: DataLoader, zero_grad -> forward -> loss -> backward -> step, con model.train() y
    model.eval() + torch.no_grad() en validación.
    - Adam, el optimizador de los ejercicios de clase.
    - Gradient clipping a norma 1,0 entre backward y step: en las redes recurrentes el gradiente se
      multiplica paso a paso y puede explotar.
    - Early stopping sobre la pérdida de validación (paciencia 10, mejora mínima 1e-6, como máximo 100
      épocas). Se guarda el state_dict de la mejor época y se recarga al final: es el punto de la
      curva de validación donde la red generaliza mejor antes de sobreajustar.
    - El orden de los lotes sale de un generador con semilla 42, así que un mismo modelo entrenado dos
      veces da el mismo resultado.
    - GPU si torch.cuda.is_available(); si no, CPU.

Parámetros entrenables (contar_parametros)
    Por capa y dirección, PyTorch guarda dos vectores de bias, así que con entrada x y oculto h:
    RNN h·(x + h + 2), GRU 3·h·(x + h + 2) y LSTM 4·h·(x + h + 2). Con x = 4 y h = 64, la LSTM tiene
    4·64·70 = 17.920 y la Linear(64, 1) 65: 17.985 en total. No dependen de la longitud de la
    secuencia, porque los mismos pesos se reutilizan en cada paso.

Comparación entre modelos (perdidas_20_pasos y bootstrap_pareado)
    Para decidir si la diferencia de pérdida entre dos modelos es real o un empate, se calcula la
    pérdida de cada trayectoria de validación y un intervalo del 95 % de la diferencia con un
    bootstrap pareado. Si el intervalo contiene el 0, es un empate.
"""
import copy
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

DISPOSITIVO = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RUTA_DATASET = Path(__file__).resolve().parents[1] / "Datos" / "Transformados" / "rnn_dataset.npz"


class RedRecurrente(nn.Module):
    """Capa recurrente (rnn, lstm o gru) + Linear aplicada a todos los pasos: salida (lote, pasos)."""

    def __init__(self, tipo, n_entrada=4, n_oculta=64, n_capas=1, dropout=0.0, bidireccional=False,
                 activacion_salida="ninguna", minimo_salida=0.0):
        super().__init__()
        assert activacion_salida in ("ninguna", "softplus")
        capa = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[tipo]
        # El dropout de PyTorch solo actúa entre capas recurrentes: con 1 capa no hace nada
        self.recurrente = capa(n_entrada, n_oculta, n_capas, batch_first=True,
                               dropout=dropout if n_capas > 1 else 0.0, bidirectional=bidireccional)
        self.lineal = nn.Linear(n_oculta * (2 if bidireccional else 1), 1)
        self.activacion_salida = activacion_salida
        # Buffer y no parámetro: viaja con el state_dict pero no se entrena
        self.register_buffer("minimo_salida", torch.tensor(float(minimo_salida)))

    def forward(self, x):
        secuencia, _ = self.recurrente(x)                 # (lote, pasos, n_oculta · direcciones)
        salida = self.lineal(secuencia).squeeze(-1)       # (lote, pasos)
        if self.activacion_salida == "softplus":
            salida = self.minimo_salida + nn.functional.softplus(salida)
        return salida


def contar_parametros(modelo):
    """Número de parámetros entrenables."""
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad)


def cargar_datos(objetivo="Y", n_max=None):
    """Lee rnn_dataset.npz, escala con los parámetros de entrenamiento y devuelve los tres conjuntos.

    objetivo: "Y" (payoff descontado por paso), "Y_geo" (cesta geométrica) o "itm" (clasificación,
    sin escalar). n_max: trayectorias en total, repartidas 70/15/15 (con --rapido, 20.000); la
    trayectoria 0 sigue en test porque idx_test está ordenado.
    """
    d = np.load(RUTA_DATASET, allow_pickle=False)
    indices = {c: d[f"idx_{c}"] for c in ("train", "val", "test")}
    if n_max is not None:
        indices = {c: v[:round(n_max * p)] for (c, v), p in zip(indices.items(), (0.70, 0.15, 0.15))}
    entradas = ((d["entradas"] - d["media_entradas"]) / d["desv_entradas"]).astype(np.float32)
    if objetivo == "itm":
        media, desv, y = 0.0, 1.0, d["itm"].astype(np.float32)
    else:
        media, desv = float(d[f"media_{objetivo}"]), float(d[f"desv_{objetivo}"])
        y = ((d[objetivo] - media) / desv).astype(np.float32)
    datos = {"media_y": media, "desv_y": desv}
    for c, v in indices.items():
        datos[f"x_{c}"], datos[f"y_{c}"], datos[f"idx_{c}"] = entradas[v], y[v], v
    return datos


def entrenar(modelo, datos, perdida, lr=1e-3, batch=512, max_epocas=100, paciencia=10, min_delta=1e-6,
             semilla=42):
    """Adam + gradient clipping + early stopping. Devuelve historial, mejor época, épocas y tiempo (s)."""
    modelo.to(DISPOSITIVO)
    cargador = DataLoader(TensorDataset(torch.as_tensor(datos["x_train"]), torch.as_tensor(datos["y_train"])),
                          batch_size=batch, shuffle=True, generator=torch.Generator().manual_seed(semilla))
    cargador_val = DataLoader(TensorDataset(torch.as_tensor(datos["x_val"]), torch.as_tensor(datos["y_val"])),
                              batch_size=4096)
    optimizador = torch.optim.Adam(modelo.parameters(), lr=lr)
    historial = {"train": [], "val": []}
    mejor_val, mejor_epoca, mejor_estado, sin_mejora = float("inf"), 0, None, 0

    inicio = time.perf_counter()
    for epoca in range(1, max_epocas + 1):
        modelo.train()
        suma = 0.0
        for x, y in cargador:
            x, y = x.to(DISPOSITIVO), y.to(DISPOSITIVO)
            optimizador.zero_grad()
            valor_perdida = perdida(modelo(x), y)
            valor_perdida.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            optimizador.step()
            suma += valor_perdida.item() * len(x)
        historial["train"].append(suma / len(cargador.dataset))

        modelo.eval()
        with torch.no_grad():
            suma_val = sum(perdida(modelo(x.to(DISPOSITIVO)), y.to(DISPOSITIVO)).item() * len(x)
                           for x, y in cargador_val)
        historial["val"].append(suma_val / len(cargador_val.dataset))

        if historial["val"][-1] < mejor_val - min_delta:
            mejor_val, mejor_epoca, sin_mejora = historial["val"][-1], epoca, 0
            mejor_estado = copy.deepcopy(modelo.state_dict())
        else:
            sin_mejora += 1
            if sin_mejora >= paciencia:
                break
    tiempo = time.perf_counter() - inicio

    modelo.load_state_dict(mejor_estado)
    return {"historial": historial, "mejor_epoca": mejor_epoca, "mejor_val": mejor_val,
            "epocas": epoca, "tiempo": tiempo}


def evaluar(modelo, x, media_y=0.0, desv_y=1.0, clasificacion=False, batch=8192):
    """Predicciones desescaladas y recortadas a >= 0 (regresión) o probabilidades (clasificación)."""
    modelo.to(DISPOSITIVO).eval()
    with torch.no_grad():
        salida = torch.cat([modelo(torch.as_tensor(x[i:i + batch]).to(DISPOSITIVO)).cpu()
                            for i in range(0, len(x), batch)])
    if clasificacion:
        return torch.sigmoid(salida).numpy()
    return np.maximum(salida.numpy() * desv_y + media_y, 0)


def salida_bruta(modelo, x, batch=8192):
    """Salida de la red en la escala de entrenamiento, sin desescalar ni recortar: forma (n, pasos)."""
    modelo.to(DISPOSITIVO).eval()
    with torch.no_grad():
        return torch.cat([modelo(torch.as_tensor(x[i:i + batch]).to(DISPOSITIVO)).cpu()
                          for i in range(0, len(x), batch)]).numpy()


def perdidas_20_pasos(modelo, x, y):
    """MSE escalado de los 20 pasos de cada trayectoria. Su media es la pérdida de validación de entrenar()."""
    return ((salida_bruta(modelo, x) - y) ** 2).mean(axis=1)


def bootstrap_pareado(perdida_a, perdida_b, raiz=False, n_remuestreos=1000, semilla=42):
    """IC del 95 % de la diferencia de la pérdida media (a - b) remuestreando las mismas trayectorias.

    Es pareado porque los dos modelos se evalúan sobre las mismas trayectorias: la dificultad que
    comparten se cancela y el intervalo solo recoge la diferencia entre ellos. raiz=True compara RMSE
    (raíz de la media del error cuadrático). Con la misma semilla, todas las comparaciones usan los
    mismos remuestreos.
    """
    rng = np.random.default_rng(semilla)
    diferencias = np.empty(n_remuestreos)
    for b in range(n_remuestreos):
        i = rng.integers(0, len(perdida_a), len(perdida_a))
        media_a, media_b = perdida_a[i].mean(), perdida_b[i].mean()
        diferencias[b] = np.sqrt(media_a) - np.sqrt(media_b) if raiz else media_a - media_b
    return np.percentile(diferencias, [2.5, 97.5])


if __name__ == "__main__":
    # Prueba de formas: 1 época con 2.000 trayectorias (1.400 de entrenamiento) por cada estructura
    from packages.bbva_graficos import imprimir

    imprimir(f"Dispositivo: {DISPOSITIVO}")
    torch.manual_seed(42)
    lstm_clase = RedRecurrente("lstm", n_entrada=4, n_oculta=64)
    imprimir(f"Parámetros de LSTM(4, 64) + Linear(64, 1): {contar_parametros(lstm_clase):,} (esperados 17,985)")
    assert contar_parametros(lstm_clase) == 17_985

    datos = cargar_datos("Y", n_max=2_000)
    imprimir(f"x_train {datos['x_train'].shape}, y_train {datos['y_train'].shape}, "
             f"x_val {datos['x_val'].shape}, x_test {datos['x_test'].shape}; trayectoria 0 en test: "
             f"{datos['idx_test'][0] == 0}")

    estructuras = {
        "RNN simple": dict(tipo="rnn"),
        "LSTM": dict(tipo="lstm"),
        "GRU": dict(tipo="gru"),
        "LSTM bidireccional": dict(tipo="lstm", bidireccional=True),
    }
    imprimir("\nRegresión (MSE), 1 época:")
    for nombre, config in estructuras.items():
        torch.manual_seed(42)
        modelo = RedRecurrente(**config)
        res = entrenar(modelo, datos, nn.MSELoss(), max_epocas=1)
        salida = modelo(torch.as_tensor(datos["x_val"][:5]).to(DISPOSITIVO))
        pred = evaluar(modelo, datos["x_test"], datos["media_y"], datos["desv_y"])
        assert salida.shape == (5, 20) and pred.shape == (len(datos["x_test"]), 20) and pred.min() >= 0
        imprimir(f"  {nombre:19s} {contar_parametros(modelo):>7,} parámetros | salida {tuple(salida.shape)} | "
                 f"pérdida train {res['historial']['train'][0]:.4f}, val {res['mejor_val']:.4f} | "
                 f"{res['tiempo']:.2f} s")

    torch.manual_seed(42)
    modelo = RedRecurrente("lstm", activacion_salida="softplus",
                           minimo_salida=-datos["media_y"] / datos["desv_y"])
    entrenar(modelo, datos, nn.MSELoss(), max_epocas=1)
    with torch.no_grad():
        bruta = modelo(torch.as_tensor(datos["x_test"]).to(DISPOSITIVO)).cpu().numpy()
    minimo = (bruta * datos["desv_y"] + datos["media_y"]).min()
    assert minimo >= -1e-6
    imprimir(f"\nLSTM con softplus: mínimo de la prima desescalada sin recortar = {minimo:.6f} (>= 0)")

    datos_itm = cargar_datos("itm", n_max=2_000)
    torch.manual_seed(42)
    modelo = RedRecurrente("lstm")
    res = entrenar(modelo, datos_itm, nn.BCEWithLogitsLoss(), max_epocas=1)
    prob = evaluar(modelo, datos_itm["x_test"], clasificacion=True)
    assert prob.shape == (len(datos_itm["x_test"]), 20) and 0 <= prob.min() and prob.max() <= 1
    imprimir(f"LSTM de clasificación (BCEWithLogitsLoss): probabilidades {prob.shape} entre "
             f"{prob.min():.3f} y {prob.max():.3f}; pérdida val {res['mejor_val']:.4f}")
    print("Prueba de formas superada.")
