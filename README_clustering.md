# Clustering IGSM 2025

Este script realiza un análisis de clustering utilizando K-Means sobre las municipalidades de Costa Rica, basado en los datos del IGSM 2025.

## Requisitos
- Python 3.9 o superior
- pandas
- numpy
- scikit-learn
- matplotlib

Instalar:
pip install pandas numpy scikit-learn matplotlib

## Archivo principal
Clustering_igsm.py

## Dataset
IGSM 2025 - Detalle de cada municipalidad 2025.csv

## Ejecución
python Clustering_igsm.py

## Proceso
- Limpieza de datos
- Conversión de madurez a valores numéricos
- Escalado
- Evaluación de k
- Selección por silhouette
- Entrenamiento K-Means
- PCA
- Exportación de resultados

## Archivos generados
- igsm_metricas_kmeans.csv
- igsm_clusters_municipalidades.csv
- igsm_resumen_clusters.csv
- igsm_provincias_por_cluster.csv
- igsm_madurez_total_por_cluster.csv
- igsm_pca_componentes.csv

Gráficos:
- igsm_elbow.png
- igsm_silhouette.png
- igsm_clusters_pca.png
