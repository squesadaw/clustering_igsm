
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# =========================================================
# CLUSTERING DEL IGSM
# Proyecto: Análisis de Datos
# =========================================================

ARCHIVO = "IGSM 2025 - Detalle de cada municipalidad 2025.csv"
RANDOM_STATE = 42
K_MIN = 2
K_MAX = 5





# =========================================
# 1. LIMPIEZA Y CARGA DE DATOS
# =========================================



def limpiar_valor(x):
    "Normaliza valores faltantes y texto limpio."
    if pd.isna(x):
        return np.nan

    s = str(x).strip()

    faltantes = {"", " ", "N/A", "NA", "n/a", "No aplica", "No Aplica", "no aplica"}
    if s in faltantes:
        return np.nan

    return s


def cargar_datos(ruta):
    df = pd.read_csv(ruta, dtype=str)
    df = df.apply(lambda col: col.map(limpiar_valor))
    return df





# =========================================
# 2. PREPARACION DE VARIABLES
# =========================================



def preparar_variables_madurez(df):
 
    columnas_madurez = [c for c in df.columns if "Grado de Madurez" in c]

    col_total = "Grado de Madurez Total 2025"
    columnas_servicio = [c for c in columnas_madurez if c != col_total]

    mapa_madurez = {
        "no brinda el servicio": 0,
        "inicial": 1,
        "basico": 2,
        "básico": 2,
        "intermedio": 3,
        "avanzado": 4,
        "optimizando": 5
    }

    X = df[columnas_servicio].copy()

    for col in X.columns:
        X[col] = X[col].map(lambda x: mapa_madurez.get(str(x).strip().lower(), np.nan) if pd.notna(x) else np.nan)

    return X, columnas_servicio, col_total





# =========================================
# 3. EVALUACION DEL NuMERO DE CLUSTERS
# =========================================


def evaluar_kmeans(X_scaled, k_min=2, k_max=5, random_state=42):
    resultados = []

    for k in range(k_min, k_max + 1):
        modelo = KMeans(n_clusters=k, n_init=20, random_state=random_state)
        etiquetas = modelo.fit_predict(X_scaled)

        resultados.append({
            "k": k,
            "inercia": modelo.inertia_,
            "silhouette": silhouette_score(X_scaled, etiquetas)
        })

    return pd.DataFrame(resultados)





# =========================================
# 4. MODELO FINAL
# =========================================

def entrenar_modelo_final(X_scaled, k_final, random_state=42):
    modelo = KMeans(n_clusters=k_final, n_init=20, random_state=random_state)
    etiquetas = modelo.fit_predict(X_scaled)
    return modelo, etiquetas




# =========================================
# 5. INTERPRETACION DE CLUSTERS
# =========================================


def nombre_cluster_por_nivel(valor_promedio):
    """Etiqueta para interpretar cluster."""
    if valor_promedio >= 2.7:
        return "Desempeño medio-alto"
    elif valor_promedio >= 2.0:
        return "Desempeño medio"
    else:
        return "Desempeño bajo"


def generar_resumen_clusters(df_original, X_madurez, etiquetas):
    base = df_original[["Provincia", "Cantón", "Municipalidad", "Grado de Madurez Total 2025"]].copy()
    base["cluster"] = etiquetas

    resumen = X_madurez.copy()
    resumen["cluster"] = etiquetas

    medias_cluster = resumen.groupby("cluster").mean().round(2)
    medias_cluster["promedio_general_cluster"] = medias_cluster.mean(axis=1).round(2)
    medias_cluster["perfil_cluster"] = medias_cluster["promedio_general_cluster"].apply(nombre_cluster_por_nivel)

    tamanos = base["cluster"].value_counts().sort_index().rename("cantidad_municipalidades")
    totales = base.groupby(["cluster", "Grado de Madurez Total 2025"]).size().unstack(fill_value=0)
    provincias = base.groupby(["cluster", "Provincia"]).size().unstack(fill_value=0)

    return base, medias_cluster, tamanos, totales, provincias




# =========================================
# 6. GRAFICOS
# =========================================


def graficar_metricas(df_metricas):
    plt.figure(figsize=(8, 5))
    plt.plot(df_metricas["k"], df_metricas["inercia"], marker="o")
    plt.xlabel("Número de clusters (k)")
    plt.ylabel("Inercia")
    plt.title("Método del codo")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("igsm_elbow.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(df_metricas["k"], df_metricas["silhouette"], marker="o")
    plt.xlabel("Número de clusters (k)")
    plt.ylabel("Silhouette score")
    plt.title("Calidad del clustering según silhouette")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("igsm_silhouette.png", dpi=200)
    plt.close()


def graficar_pca(X_scaled, etiquetas):
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    componentes = pca.fit_transform(X_scaled)

    df_pca = pd.DataFrame({
        "PC1": componentes[:, 0],
        "PC2": componentes[:, 1],
        "cluster": etiquetas
    })

    plt.figure(figsize=(8, 6))
    for cluster in sorted(df_pca["cluster"].unique()):
        datos = df_pca[df_pca["cluster"] == cluster]
        plt.scatter(datos["PC1"], datos["PC2"], label=f"Cluster {cluster}", alpha=0.8)

    plt.xlabel("Componente principal 1")
    plt.ylabel("Componente principal 2")
    plt.title("Municipalidades segmentadas (PCA 2D)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("igsm_clusters_pca.png", dpi=200)
    plt.close()

    return df_pca, pca.explained_variance_ratio_





# =========================================
# 7. PROCESO PRINCIPAL
# =========================================



def main():
    print("=== CLUSTERING IGSM ===")



    # 1. Carga de datos
    df = cargar_datos(ARCHIVO)
    print(f"Dimensión del dataset: {df.shape}")



    # 2. Preparar variables
    X_madurez, columnas_servicio, col_total = preparar_variables_madurez(df)

    print("\nVariables usadas para el modelo:")
    for col in columnas_servicio:
        print("-", col)



    # 3. Escalar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_madurez)



    # 4. Evaluar varios k
    metricas = evaluar_kmeans(X_scaled, k_min=K_MIN, k_max=K_MAX, random_state=RANDOM_STATE)
    metricas = metricas.sort_values("k").reset_index(drop=True)



    # Elegir automaticamente el mejor k por silhouette
    mejor_fila = metricas.loc[metricas["silhouette"].idxmax()]
    k_final = int(mejor_fila["k"])

    print("\nMétricas por número de clusters:")
    print(metricas.to_string(index=False))

    print(f"\nMejor k según silhouette: {k_final}")



    # 5. Modelo final
    modelo_final, etiquetas = entrenar_modelo_final(X_scaled, k_final, random_state=RANDOM_STATE)



    # 6. Resumenes
    asignaciones, medias_cluster, tamanos, totales, provincias = generar_resumen_clusters(df, X_madurez, etiquetas)

    print("\nTamaño de los clusters:")
    print(tamanos.to_string())

    print("\nDistribución del Grado de Madurez Total 2025 por cluster:")
    print(totales.to_string())

    print("\nPromedios de madurez por cluster:")
    print(medias_cluster.to_string())



    # 7. PCA para visualizacion
    df_pca, var_exp = graficar_pca(X_scaled, etiquetas)

    print("\nVarianza explicada por PCA:")
    print(f"PC1: {var_exp[0]:.4f}")
    print(f"PC2: {var_exp[1]:.4f}")
    print(f"Total PC1 + PC2: {var_exp.sum():.4f}")



    # 8. Guardar salidas
    metricas.to_csv("igsm_metricas_kmeans.csv", index=False)
    asignaciones.to_csv("igsm_clusters_municipalidades.csv", index=False)
    medias_cluster.to_csv("igsm_resumen_clusters.csv")
    totales.to_csv("igsm_madurez_total_por_cluster.csv")
    provincias.to_csv("igsm_provincias_por_cluster.csv")
    df_pca.to_csv("igsm_pca_componentes.csv", index=False)

    graficar_metricas(metricas)

    print("\nArchivos generados:")
    print("- igsm_metricas_kmeans.csv")
    print("- igsm_clusters_municipalidades.csv")
    print("- igsm_resumen_clusters.csv")
    print("- igsm_madurez_total_por_cluster.csv")
    print("- igsm_provincias_por_cluster.csv")
    print("- igsm_pca_componentes.csv")
    print("- igsm_elbow.png")
    print("- igsm_silhouette.png")
    print("- igsm_clusters_pca.png")


if __name__ == "__main__":
    main()
