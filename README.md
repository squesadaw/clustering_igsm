# IGSM 2025 — Clustering de Municipalidades para Identificar Brechas y Prioridades

Herramienta de análisis automatizado del Índice de Gestión de Servicios Municipales (IGSM) 2025.
Segmenta las 84 municipalidades de Costa Rica según el grado de madurez con que brindan sus servicios, e identifica brechas y prioridades de mejora.


## ¿Qué hace?

1. **Procesa** el CSV del IGSM automáticamente — limpieza, codificación y escalado de datos.
2. **Agrupa** las municipalidades mediante K-Means según su patrón de madurez en 10 servicios.
3. **Visualiza** los resultados en un dashboard interactivo con análisis de brechas y comparación entre municipalidades.


## Componentes

| Archivo | Descripción |
|---|---|
| `Clustering_igsm.py` | Script principal: procesa los datos y genera el clustering |
| `dashboard_igsm.py` | Dashboard interactivo para explorar los resultados |
| `IGSM 2025 - Detalle de cada municipalidad 2025.csv` | Datos fuente |


## Requisitos

- Python 3.9 o superior

```bash
pip install streamlit pandas numpy scikit-learn plotly matplotlib
```


## Inicio rápido

**1. Generar el análisis:**
```bash
python3 Clustering_igsm.py
```

**2. Abrir el dashboard:**
```bash
streamlit run dashboard_igsm.py
```


## Proceso de análisis

1. Limpieza y carga de datos
2. Conversión de grados de madurez a valores numéricos
3. Escalado de variables
4. Evaluación de distintos valores de k (2 a 5)
5. Selección automática del mejor k por silhouette score
6. Entrenamiento del modelo K-Means final
7. Reducción dimensional con PCA para visualización
8. Exportación de resultados


## Archivos generados

Los resultados se guardan en la carpeta `resultados/`:

| Archivo | Descripción |
|---|---|
| `igsm_clusters_municipalidades.csv` | Asignación de cada municipalidad a su cluster |
| `igsm_resumen_clusters.csv` | Promedios de madurez por cluster |
| `igsm_metricas_kmeans.csv` | Inercia y silhouette score por k |
| `igsm_provincias_por_cluster.csv` | Distribución de provincias por cluster |
| `igsm_madurez_total_por_cluster.csv` | Distribución del grado de madurez total por cluster |
| `igsm_pca_componentes.csv` | Coordenadas PCA por municipalidad |
| `igsm_elbow.png` | Gráfico del método del codo |
| `igsm_silhouette.png` | Gráfico de silhouette score |
| `igsm_clusters_pca.png` | Visualización de clusters en 2D |


## Dashboard

El dashboard permite explorar los resultados de forma interactiva:

- Resumen general del análisis
- Visualización de clusters en 2D (PCA)
- Análisis de brechas por servicio
- Comparación entre municipalidades
- Filtros por provincia, grupo y búsqueda
- Descarga de resultados en CSV

> Si se actualizan los datos fuente, es necesario ejecutar nuevamente `Clustering_igsm.py` antes de abrir el dashboard.
