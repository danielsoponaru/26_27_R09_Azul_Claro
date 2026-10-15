"""Fórmula cerrada copiada sin cambios del notebook de BBVA.

Ruta en el repositorio: packages/formula_cerrada_bbva.py
Origen: las funciones Black y BasketGeomAsian del notebook MonteCarlo_y_Formula_cerrada_25_26.ipynb,
material que BBVA facilita para el reto (no se incluye en el repositorio). Se copian literalmente, con
sus imports, para que el precio exacto con el que comparamos la red sea el de BBVA y no una versión
reescrita por nosotros.
Entrada: ninguna; las funciones reciben los parámetros como argumentos. La comprobación del final lee
sigma y rho de outputs/parametros_montecarlo.json.
Salida: precios en memoria.

Qué valora la fórmula. BasketGeomAsian es el precio exacto, bajo Black-Scholes, de una call asiática
sobre la cesta geométrica con strike 1:
    ((G_CAT · G_MRK · G_MSFT)^(1/3) - 1)+
con G_k la media geométrica de los 20 precios normalizados del activo k (sin incluir S0). No es el
payoff del enunciado, que suma las tres medias:
    (G_CAT + G_MRK + G_MSFT - 3)+
El producto de lognormales correlacionadas es lognormal, así que la cesta geométrica tiene fórmula
cerrada; la suma no lo es, y por eso la prima del reto se calcula por Montecarlo.

Por qué 3 × fórmula es una cota inferior de la prima del reto. El payoff del enunciado es 3·(A - 1)+,
con A = (G_CAT + G_MRK + G_MSFT) / 3. Por la desigualdad entre la media aritmética y la geométrica,
A >= (G_CAT · G_MRK · G_MSFT)^(1/3) en cada trayectoria, y como x -> (x - 1)+ es no decreciente,
3·(A - 1)+ >= 3·((G_CAT · G_MRK · G_MSFT)^(1/3) - 1)+ trayectoria a trayectoria. Al descontar y tomar
la media se mantiene la desigualdad: prima del reto >= 3 × BasketGeomAsian.

Para qué la usamos. Como solo es exacta para la cesta geométrica, sirve para medir el error de una red
frente a un precio exacto: la red "geométrica" de rnn_04_prima.py se entrena con ese payoff y se compara
con esta fórmula, mientras que la red principal (payoff del enunciado) se compara con Montecarlo.
También es la referencia de coste de una valoración exacta en rnn_05_coste.py.

Argumentos, con la convención de BBVA:
- num_asian_dates = 21: las 20 fechas de observación más la fecha 0, que fija S0.
- value_date_index = k: índice de la fecha de valoración (0 en el origen).
- assets_vol: volatilidades en columna, forma (3, 1); assets_correl: matriz de correlaciones (3, 3).
- price_history: forma (3, k + 1), un activo por fila y una fecha por columna; la columna 0 es la
  fecha 0 (S0 = 1). En el bucle con el que el notebook de BBVA compara la fórmula con Montecarlo, la
  historia que se pasa (simu2[1, :, :(i+1)]) empieza en la fecha 1, es decir, llega desplazada una
  fecha. Por eso no copiamos ese bucle: nosotros pasamos np.hstack([np.ones((3, 1)), historia.T]).

Comprobación (desde la raíz del repositorio): python -m packages.formula_cerrada_bbva
Con sigma y rho de la cartera, la fórmula en t0 tiene que dar 0,083164; la misma cesta por Montecarlo,
con las 100.000 trayectorias de packages/montecarlo.py (semilla 42), tiene que coincidir dentro de
±1,96 errores estándar, y 3 × fórmula tiene que quedar por debajo de la prima del reto.
"""
import numpy as np
from scipy.stats import norm

N = norm.cdf

def Black(Forward, Strike, TTM, rate, Vol, IsCall):

  '''
  Inputs:
  -------
    Forward (float): Forward value
    Strike (float): strike price
    TTM (float): time to maturity in years
    rate (float): risk free rate
    Vol (float): volatility
    IsCall (bool): True if call option, False if put option
  Outputs:
  --------
    Option premium (float)
  '''

  if TTM >0:

    d1 = (np.log(Forward/Strike) + (Vol*Vol/2)*TTM)/(Vol*np.sqrt(TTM))
    d2 = (np.log(Forward/Strike) + (- Vol*Vol/2)*TTM)/(Vol*np.sqrt(TTM))

    if IsCall:

      return (Forward*N(d1)-Strike*N(d2))*np.exp(-rate*TTM)

    else:

      return (-Forward*N(-d1)+Strike*N(-d2))*np.exp(-rate*TTM)

  else:

    if IsCall:

      return np.maximum(Forward-Strike,0)

    else:

      return np.maximum(-Forward+Strike,0)


def BasketGeomAsian(num_asian_dates, value_date_index, risk_free_rate, num_assets, assets_vol, assets_correl, initial_maturity, price_history, IsCall):
  '''
  Inputs:
  -------
  * num_asian_dates (int): number of asian dates. Notice that this includes the initial date, which fixes the initial spot values.
  * value_date_index (int): index, within the array of asian dates indicating the value date.
  * risk_free_rate (float): risk free rate
  * num_assets (int): number of underlying assets.
  * assets_vol (array(float)): array of indiv assets vols.
  * assets_correl (array(float, float)): matrix of correlations
  * initial_maturity (float): maturity of the product as seen on initial spot fixing date.
  * price_history (array(float, float)): history of fixings of the underlyings up to value date. Assets per row, time steps per column.
  * IsCall (bool): True if call option, False if put option
  Outputs:
  --------
  * Option price (float)
  '''

  init_time_array = np.linspace(0, initial_maturity, num_asian_dates)

  pending_times_array = init_time_array[value_date_index+1:] - init_time_array[value_date_index]

  mu = np.sum(risk_free_rate - 0.5*assets_vol*assets_vol)*np.sum(pending_times_array) / (num_assets * (num_asian_dates-1))

  diag_vol = np.diag(assets_vol.reshape(-1))

  cov_matrix = np.matmul(diag_vol, np.matmul(assets_correl, diag_vol))

  xx, yy = np.meshgrid(pending_times_array, pending_times_array, sparse=True)
  z = np.minimum(xx, yy)

  V = np.sum(cov_matrix) * np.sum(z) / (num_assets*num_assets*(num_asian_dates-1)*(num_asian_dates-1))

  Forward = np.power(np.prod(price_history[:, 1:value_date_index+1] / price_history[:,0].reshape(-1,1)),1.0/(num_assets * (num_asian_dates-1)))

  Forward *= np.power(np.prod(price_history[:,value_date_index] / price_history[:,0]), (num_asian_dates-value_date_index-1)/(num_assets * (num_asian_dates-1)))

  Forward *= np.exp(mu + 0.5 * V)

  remaining_maturity = initial_maturity - init_time_array[value_date_index]


  return Black(Forward, 1.0, remaining_maturity, risk_free_rate,np.sqrt(V / remaining_maturity), IsCall)


if __name__ == "__main__":
    # Comprobación nuestra (no forma parte del código de BBVA) con los parámetros de la cartera
    import json
    from pathlib import Path

    from packages import montecarlo as mc
    from packages.bbva_graficos import imprimir

    ruta_parametros = Path(__file__).resolve().parents[1] / "outputs" / "parametros_montecarlo.json"
    parametros = json.loads(ruta_parametros.read_text(encoding="utf-8"))
    sigma = np.array(parametros["sigma"])
    rho = np.array(parametros["rho"])

    # 21 fechas (t0 y 20 trimestres), valoración en k = 0, r = 1 %, 3 activos y vencimiento a 5 años
    prima_formula = BasketGeomAsian(21, 0, 0.01, 3, sigma.reshape(-1, 1), rho, 5, np.ones((3, 1)), True)
    imprimir(f"Cesta geométrica en t0, fórmula de BBVA: {prima_formula:.6f}")
    assert round(prima_formula, 6) == 0.083164, "La copia no reproduce el valor esperado de 0,083164"

    # La misma cesta y el payoff del reto sobre las 100.000 trayectorias independientes del equipo
    X = mc.simular_continuaciones(None, 100_000, np.random.default_rng(42))
    G = mc.medias_geometricas(X)
    descuento = np.exp(-mc.R * mc.TTM)
    cesta = np.maximum(np.prod(G, axis=1) ** (1 / 3) - 1, 0) * descuento
    reto = mc.calcular_payoff(X) * descuento
    error_estandar = cesta.std(ddof=1) / np.sqrt(len(cesta))
    z = (cesta.mean() - prima_formula) / error_estandar
    imprimir(f"Cesta geométrica en t0, Montecarlo:      {cesta.mean():.6f} ± {error_estandar:.6f} (z = {z:.2f})")
    imprimir(f"3 × fórmula = {3 * prima_formula:.6f} <= prima del reto por Montecarlo = {reto.mean():.6f}")
    assert abs(z) < 1.96, "Montecarlo y fórmula no coinciden dentro de ±1,96 errores estándar"
    assert 3 * prima_formula <= reto.mean(), "No se cumple la cota inferior 3 × fórmula"
    print("Comprobación superada.")
