# Azul Claro · Observatorio de cartera

Abre esta carpeta en VS Code y selecciona **Open with Live Server** sobre `index.html`. No requiere npm, servidor Python, conexión a internet ni claves de API. D3 7.9.0 está incluido con su licencia en `vendor/`.

## Qué incluye

- **Visión de cartera:** evolución base 100, indicadores, periodos, pesos interactivos, correlaciones y alternativas.
- **Selección:** universo de 26 acciones con filtros y ranking de 295 ternas con paginación.
- **Riesgo:** correlaciones y volatilidades móviles a 252 sesiones, 10.000 asignaciones y drawdown.
- **Gráficos del proyecto:** doce imágenes originales ampliables.
- Exportación CSV de las fechas, precios, índice y pesos actuales. Impresión de la vista abierta.

## Datos y actualización

`datos/proyecto.json` es una instantánea derivada de `Datos/Transformados/dowjones_diario_2016_2025.csv`: 64.557 filas, 26 empresas, del 04/01/2016 al 31/12/2025. El proceso de limpieza combina los CSV de Marketstack con Yahoo Finance según `01_Limpieza_de_datos.ipynb`. La instantánea incluye una huella SHA-256 del CSV fuente, visible en Metodología.

Para regenerarla desde la raíz del repositorio, con pandas y NumPy del entorno del proyecto:

```powershell
python "Dashboard Cartera/scripts/exportar_datos.py"
```

El generador lee los sectores y exclusiones del notebook local, reproduce el ranking y comprueba la selección CAT–MRK–MSFT y las 295 ternas. No ejecuta ni modifica notebooks. Las imágenes de `graficos/` son copias de las figuras originales; para actualizarlas hay que copiar los PNG correspondientes tras ejecutar los notebooks.

La carpeta incluye `datos/proyecto.json`, publicado junto al dashboard para que funcione directamente después de clonar. Para regenerar la instantánea se necesita el CSV transformado original del proyecto. No se requiere regenerarla para abrir el panel.

## Cómo interpretar

Se mantienen las fórmulas del notebook: μ logarítmica anualizada, σ muestral anualizada y tasa libre de riesgo del 1 %. El score de selección **no es** el Sharpe de cartera. Los periodos recalculan indicadores, evolución y matriz; ranking, perfiles optimizados y gráficos de riesgo utilizan siempre 2016–2025.

La curva de cartera acumula retornos simples con reajuste diario teórico a los pesos elegidos, sin costes, dividendos ni divisa. Los resultados son históricos, con sesgos de composición y supervivencia, y sin validación fuera de muestra. Los perfiles Máx. Sharpe y Mín. riesgo son los mejores entre 10.000 asignaciones aleatorias, no óptimos analíticos.

La interfaz utiliza la paleta de `packages/bbva_graficos.py`; no representa un producto oficial de la entidad. La autoría y revisión corresponden al equipo del proyecto.
