# Reto 09 · Azul Claro

Selección de una cartera de 3 acciones del Dow Jones (CAT, MRK, MSFT), valoración de su producto con Monte Carlo y una red neuronal, y un dashboard para explorarla.

## Estructura

```
01_…10_*.ipynb       Notebooks, numerados en el orden en que se ejecutan
packages/            Código común: estilo de los gráficos (bbva_graficos.py) y Monte Carlo (montecarlo.py)
Datos/Originales/    Datos crudos (no se suben a Git)
Datos/Transformados/ CSV limpio que generan los notebooks (no se sube a Git)
Graficos/            Gráficos PNG que generan los notebooks
outputs/             Tablas y parámetros que generan los notebooks
Dashboard Cartera/   Panel web (ver su README)
environment.yml      Entorno de Python
```

## Orden de ejecución

| Nº | Notebook | Qué hace |
|---|---|---|
| 01 | Limpieza de datos | Une los CSV crudos y Yahoo Finance y genera el CSV limpio |
| 02 | Análisis descriptivo | Mercado, rentabilidad y riesgo de las 26 empresas |
| 03 | Procesamiento de series temporales | Tendencia, estacionalidad y estacionariedad |
| 04 | Selección de cartera | Ranking de ternas y elección de CAT–MRK–MSFT |
| 05 | Preparación de series | Rentabilidades logarítmicas y test ADF de la cartera |
| 06 | Descomposición de activos | Tendencia y estacionalidad de los tres activos |
| 07 | Predicción de volatilidad | Modelos GARCH |
| 08 | Monte Carlo | Simulación de la cartera y prima del producto |
| 09 | Monte Carlo para redes | Muestras de simulación para entrenar la red |
| 10 | Mining | Red neuronal que predice el payoff (tarda mucho) |

Ejecútalos desde la raíz del repositorio. El 01 va primero (genera el CSV que usan todos los demás) y el 09 antes que el 10.

## Datos crudos

Crea la carpeta `Datos/Originales/DowJones/` y copia dentro los **26 CSV de Marketstack**, uno por empresa, con el nombre `<TICKER>_marketstack.csv` (por ejemplo `AAPL_marketstack.csv`). Cada CSV debe tener las columnas `open, high, low, last, close, volume, date, symbol, exchange`.

El notebook 01 descarga solo los datos de Yahoo Finance la primera vez (hace falta internet) y los guarda en `Datos/Originales/Yahoo/`.

## Entorno

```bash
conda env create -f environment.yml    # crea el entorno "reto09"
conda activate reto09
```

Para actualizarlo después de un cambio: `conda env update -f environment.yml --prune`. Exportar gráficos de Plotly a PNG necesita Chrome o Chromium instalado.
