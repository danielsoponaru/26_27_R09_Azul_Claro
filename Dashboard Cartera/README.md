# Observatorio de cartera

Dashboard web de la cartera CAT–MRK–MSFT: evolución, selección frente a las 295 ternas válidas, riesgo y gráficos del proyecto.

## Cómo abrirlo

Hace falta servirlo por HTTP (si abres `index.html` con doble clic, el navegador no carga los datos). Elige una opción:

- **VS Code:** instala la extensión *Live Server*, clic derecho sobre `index.html` y **Open with Live Server**.
- **Terminal:** `cd "Dashboard Cartera"`, `python3 -m http.server 8000` y abre `http://localhost:8000`.

No necesita `npm`, internet ni claves (D3 va incluido en `vendor/`).

## Datos

`datos/proyecto.json` es una instantánea del CSV limpio del proyecto (26 empresas, 2016–2025). Para regenerarla, desde la raíz del repositorio y con el entorno del proyecto:

```bash
python "Dashboard Cartera/scripts/exportar_datos.py"
```

Las imágenes de `graficos/` son copias de las de `Graficos/`: si se regeneran los notebooks, hay que copiarlas a mano.

## Aviso

Resultados históricos, sin validación fuera de muestra ni costes. El score de selección no es el Sharpe de la cartera.
