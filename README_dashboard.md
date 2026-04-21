# Dashboard IGSM 2025

Este script ejecuta un dashboard que su objetivo es explorar los grupos generados, comparar municipalidades y analizar diferencias entre servicios.

---

## Requisitos

- Python 3.9 o superior
- Librerías necesarias:

pandas  
numpy  
streamlit  
plotly  

Instalación:

pip install pandas numpy streamlit plotly

---

## Archivo principal

dashboard_igsm.py

---

## Archivos necesarios

Deben estar en la misma carpeta:

igsm_metricas_kmeans.csv  
igsm_clusters_municipalidades.csv  
igsm_resumen_clusters.csv  
igsm_pca_componentes.csv  
IGSM 2025 - Detalle de cada municipalidad 2025.csv  

---

## Cómo ejecutarlo

Abrir una terminal en la carpeta del proyecto y ejecutar:

streamlit run dashboard_igsm.py

---

## Qué hace el dashboard

1. Carga los resultados del clustering
2. Aplica filtros por provincia, grupo y búsqueda
3. Muestra métricas generales del análisis
4. Visualiza los grupos en 2D (PCA)
5. Compara servicios entre grupos
6. Permite comparar municipalidades
7. Muestra tabla filtrada
8. Permite descargar resultados en CSV

---

## Funcionalidades principales

- Resumen general del análisis
- Visualización de clusters
- Brechas por servicio
- Comparación entre municipalidades
- Tabla y descarga de datos

---

## Nota

Este dashboard utiliza los archivos generados por el script de clustering.

Si se actualizan los datos, es necesario ejecutar nuevamente Clustering_igsm.py.
