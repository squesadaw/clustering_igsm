# IGSM 2025 — Clustering de Municipalidades para Identificar Brechas y Prioridades

Herramienta de análisis automatizado del Índice de Gestión de Servicios Municipales (IGSM) 2025.
Segmenta las 84 municipalidades de Costa Rica según el grado de madurez con que brindan sus servicios, e identifica brechas y prioridades de mejora.


## ¿Qué hace esta herramienta?

1. **Procesa** el CSV del IGSM automáticamente — limpieza, codificación y escalado de datos.
2. **Agrupa** las municipalidades mediante K-Means según su patrón de madurez en 10 servicios.
3. **Visualiza** los resultados en un dashboard interactivo con análisis de brechas y simulador de casos.


## Componentes

| Archivo | Descripción |
|---|---|
| `Clustering_igsm.py` | Script principal: procesa los datos y genera el clustering |
| `dashboard_igsm.py` | Dashboard interactivo para explorar los resultados |
| `IGSM 2025 - Detalle de cada municipalidad 2025.csv` | Datos fuente |

Para documentación detallada de cada componente ver:
- [README_dashboard.md](README_dashboard.md) — guía de uso del dashboard
- [README_clustering.md](README_clustering.md) — guía de uso del script


## Inicio rápido

**1. Instalar dependencias:**
```bash
pip install streamlit pandas numpy scikit-learn plotly pillow matplotlib
```

**2. Generar el análisis:**
```bash
python3 Clustering_igsm.py
```

**3. Abrir el dashboard:**
```bash
streamlit run dashboard_igsm.py
```

