import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Dashboard IGSM",
    layout="wide",
    initial_sidebar_state="expanded",
)



# DISENO
st.markdown(
    """
    <style>
    .small-box {
        background-color: #f7f7f7;
        border: 1px solid #e6e6e6;
        padding: 14px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .note-box {
        background-color: #eaf4ff;
        border: 1px solid #8bb8e8;
        color: #12324a;
        padding: 12px;
        border-radius: 10px;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    .note-box strong {
        color: #0b2239;
    }
    </style>
    """,
    unsafe_allow_html=True,
)



# RUTAS
def get_base_path():
    return os.path.dirname(os.path.abspath(__file__))


BASE_PATH = get_base_path()





# CARGA DE DATOS
@st.cache_data

def load_main_data():
    metricas = pd.read_csv(os.path.join(BASE_PATH, "igsm_metricas_kmeans.csv"))
    clusters = pd.read_csv(os.path.join(BASE_PATH, "igsm_clusters_municipalidades.csv"))
    resumen = pd.read_csv(os.path.join(BASE_PATH, "igsm_resumen_clusters.csv"))
    pca = pd.read_csv(os.path.join(BASE_PATH, "igsm_pca_componentes.csv"))
    return metricas, clusters, resumen, pca


@st.cache_data

def load_raw_data():
    df_raw = pd.read_csv(
        os.path.join(BASE_PATH, "IGSM 2025 - Detalle de cada municipalidad 2025.csv"),
        dtype=str,
    )
    return df_raw


metricas, clusters, resumen, pca = load_main_data()
df_raw = load_raw_data()






# FUNCIONES

def limpiar_valor(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    faltantes = {"", " ", "N/A", "NA", "n/a", "No aplica", "No Aplica", "no aplica"}
    if s in faltantes:
        return np.nan
    return s


@st.cache_data

def preparar_servicios(df):
    df = df.copy()
    df = df.apply(lambda col: col.map(limpiar_valor))

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
        "optimizando": 5,
    }

    X = df[columnas_servicio].copy()
    for col in X.columns:
        X[col] = X[col].map(
            lambda x: mapa_madurez.get(str(x).strip().lower(), np.nan) if pd.notna(x) else np.nan
        )

    return X, columnas_servicio


X_madurez, servicios_cols = preparar_servicios(df_raw)


def get_nombres_servicios():
    return {
        "Grado de Madurez Recolección, depósito y tratamiento de residuos 2025": "Residuos",
        "Grado de Madurez Aseo de Vías y Sitios Públicos 2025": "Aseo de vías",
        "Grado de Madurez Urbanismo 2025": "Urbanismo",
        "Grado de Madurez Red Vial Cantonal 2025": "Red vial",
        "Grado de Madurez Alcantarillado Pluvial 2025": "Alcantarillado",
        "Grado de Madurez Servicios Sociales y Complementarios 2025": "Servicios sociales",
        "Grado de Madurez Educativos, culturales y deportivos 2025": "Educativos",
        "Grado de Madurez Agua Potable 2025": "Agua potable",
        "Grado de Madurez Zona Marítimo Terrestre 2025": "Zona marítima",
        "Grado de Madurez Seguridad y Vigilancia en la comunidad 2025": "Seguridad",
    }


def madurez_total_num(valor):
    mapa = {
        "Inicial": 1,
        "Básico": 2,
        "Basico": 2,
        "Intermedio": 3,
        "Avanzado": 4,
        "Optimizando": 5,
    }
    if pd.isna(valor):
        return np.nan
    return mapa.get(str(valor).strip(), np.nan)


clusters["madurez_num"] = clusters["Grado de Madurez Total 2025"].map(madurez_total_num)

mejor = metricas.loc[metricas["silhouette"].idxmax()]
k_final = int(mejor["k"])
silhouette = float(mejor["silhouette"])



# Nombres claros
orden_promedio = resumen.sort_values("promedio_general_cluster").reset_index(drop=True)
cluster_bajo_id = int(orden_promedio.iloc[0]["cluster"])
cluster_alto_id = int(orden_promedio.iloc[-1]["cluster"])

nombres_cluster = {}
for _, fila in resumen.iterrows():
    cluster_id = int(fila["cluster"])
    promedio = float(fila["promedio_general_cluster"])
    if cluster_id == cluster_bajo_id:
        nombres_cluster[cluster_id] = f"Cluster {cluster_id} - Prioritario"
    elif promedio >= 2.0:
        nombres_cluster[cluster_id] = f"Cluster {cluster_id} - Mejor desempeño"
    else:
        nombres_cluster[cluster_id] = f"Cluster {cluster_id} - Desempeño medio"

clusters["grupo"] = clusters["cluster"].map(nombres_cluster)
pca["grupo"] = pca["cluster"].map(nombres_cluster)

resumen_simple = resumen[["cluster", "promedio_general_cluster", "perfil_cluster"]].copy()
tam = clusters.groupby("cluster").size().reset_index(name="cantidad")
resumen_simple = resumen_simple.merge(tam, on="cluster", how="left")
resumen_simple["grupo"] = resumen_simple["cluster"].map(nombres_cluster)





#unir datos de servicios con municipalidaddes
base_servicios = clusters[["Provincia", "Cantón", "Municipalidad", "Grado de Madurez Total 2025", "madurez_num", "cluster", "grupo"]].copy()
for col in servicios_cols:
    base_servicios[col] = X_madurez[col].values


nombres_servicios = get_nombres_servicios()


def calcular_brechas(df_resumen):
    cols = [c for c in df_resumen.columns if "Grado de Madurez" in c and c != "cluster"]
    alto = df_resumen[df_resumen["cluster"] == cluster_alto_id].iloc[0]
    bajo = df_resumen[df_resumen["cluster"] == cluster_bajo_id].iloc[0]
    filas = []
    for col in cols:
        filas.append(
            {
                "Servicio": nombres_servicios.get(col, col),
                "Cluster mejor": float(alto[col]),
                "Cluster prioritario": float(bajo[col]),
                "Brecha": float(alto[col]) - float(bajo[col]),
            }
        )
    return pd.DataFrame(filas).sort_values("Brecha", ascending=False)


brechas = calcular_brechas(resumen)


def recomendacion_cluster(cluster_id):
    if cluster_id == cluster_bajo_id:
        return "Priorizar acompañamiento, revisar servicios débiles y definir plan de mejora."
    if cluster_id == cluster_alto_id:
        return "Usar como referencia para comparar prácticas y metas."
    return "Dar seguimiento y buscar pasar al grupo de mejor desempeño."


resumen_simple["accion"] = resumen_simple["cluster"].apply(recomendacion_cluster)

servicios_debiles = brechas.head(3)["Servicio"].tolist()





# SIDEBAR
st.sidebar.title("Filtros")
provincias = ["Todas"] + sorted(clusters["Provincia"].dropna().unique().tolist())
grupos = ["Todos"] + list(resumen_simple.sort_values("cluster")["grupo"])

provincia_sel = st.sidebar.selectbox("Provincia", provincias)
grupo_sel = st.sidebar.selectbox("Grupo", grupos)
busqueda = st.sidebar.text_input("Buscar municipalidad")

st.sidebar.markdown("---")
st.sidebar.write("**Qué muestra este dashboard**")
st.sidebar.write("• Grupos de municipalidades")
st.sidebar.write("• Prioridades de mejora")
st.sidebar.write("• Comparación entre municipios")


datos = base_servicios.copy()
if provincia_sel != "Todas":
    datos = datos[datos["Provincia"] == provincia_sel]
if grupo_sel != "Todos":
    datos = datos[datos["grupo"] == grupo_sel]
if busqueda.strip():
    datos = datos[datos["Municipalidad"].str.contains(busqueda, case=False, na=False)]

pca_datos = pca.copy()
if grupo_sel != "Todos":
    pca_datos = pca_datos[pca_datos["grupo"] == grupo_sel]



# ENCABEZADO
st.title("Dashboard de segmentación del IGSM")
st.write(
    "Este dashboard organiza las municipalidades por nivel de desempeño y ayuda a ver cuáles requieren más atención."
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Municipalidades", len(clusters))
col2.metric("Grupos creados", k_final)
col3.metric("Calidad de agrupación", f"{silhouette:.3f}")
col4.metric("Grupo prioritario", nombres_cluster[cluster_bajo_id])

st.markdown(
    f"""
    <div class="note-box">
    <strong>Lectura rápida:</strong> el grupo más bajo es <strong>{nombres_cluster[cluster_bajo_id]}</strong>.
    Las mayores brechas están en <strong>{", ".join(servicios_debiles)}</strong>.
    </div>
    """,
    unsafe_allow_html=True,
)



# TABS
tab1, tab2, tab3, tab4 = st.tabs([
    "Resumen general",
    "Brechas por servicio",
    "Comparar municipalidades",
    "Detalle y descarga",
])


with tab1:
    c1, c2 = st.columns([1.4, 1])

    with c1:
        st.subheader("Mapa de municipalidades por grupo")
        fig_pca = px.scatter(
            pca_datos,
            x="PC1",
            y="PC2",
            color="grupo",
            hover_name=None,
            title="Municipalidades agrupadas según similitud",
        )
        fig_pca.update_traces(marker=dict(size=10, line=dict(width=0.5, color="white")))
        fig_pca.update_layout(legend_title_text="Grupo", height=500)
        st.plotly_chart(fig_pca, use_container_width=True)
        st.caption("Cada punto representa una municipalidad. Puntos cercanos tienden a parecerse más entre sí.")

    with c2:
        st.subheader("Resumen por grupo")
        tabla_resumen = resumen_simple[["grupo", "cantidad", "promedio_general_cluster", "perfil_cluster", "accion"]].copy()
        tabla_resumen.columns = [
            "Grupo",
            "Cantidad de municipalidades",
            "Promedio de servicios",
            "Perfil",
            "Acción sugerida",
        ]
        tabla_resumen["Promedio de servicios"] = tabla_resumen["Promedio de servicios"].round(2)
        st.dataframe(tabla_resumen, use_container_width=True, hide_index=True)

    st.markdown("---")

    b1, b2 = st.columns(2)
    with b1:
        st.subheader("Top 5 mejores municipalidades")
        top_mejores = (
            datos.sort_values(["madurez_num", "Municipalidad"], ascending=[False, True])
            [["Provincia", "Cantón", "Municipalidad", "Grado de Madurez Total 2025", "grupo"]]
            .head(5)
            .rename(columns={
                "Grado de Madurez Total 2025": "Madurez total 2025",
                "grupo": "Grupo",
            })
        )
        st.dataframe(top_mejores, use_container_width=True, hide_index=True)

    with b2:
        st.subheader("Top 5 municipalidades prioritarias")
        top_peores = (
            datos[datos["cluster"] == cluster_bajo_id]
            .sort_values(["madurez_num", "Municipalidad"], ascending=[True, True])
            [["Provincia", "Cantón", "Municipalidad", "Grado de Madurez Total 2025", "grupo"]]
            .head(5)
            .rename(columns={
                "Grado de Madurez Total 2025": "Madurez total 2025",
                "grupo": "Grupo",
            })
        )
        st.dataframe(top_peores, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Municipalidades por provincia y grupo")
    prov_cluster = clusters.groupby(["Provincia", "grupo"]).size().reset_index(name="cantidad")
    fig_bar = px.bar(
        prov_cluster,
        x="Provincia",
        y="cantidad",
        color="grupo",
        barmode="group",
        title="Distribución por provincia",
    )
    fig_bar.update_layout(height=430)
    st.plotly_chart(fig_bar, use_container_width=True)





with tab2:
    st.subheader("Brechas entre el grupo mejor y el grupo prioritario")
    st.write(
        "Aquí se ve en cuáles servicios existe mayor diferencia."
    )

    fig_brechas = go.Figure(
        data=[
            go.Bar(
                x=brechas["Servicio"],
                y=brechas["Brecha"],
                text=brechas["Brecha"].round(2),
                textposition="auto",
            )
        ]
    )
    fig_brechas.update_layout(
        title="Brecha de madurez por servicio",
        xaxis_title="Servicio",
        yaxis_title="Brecha",
        height=450,
    )
    st.plotly_chart(fig_brechas, use_container_width=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Tabla de brechas")
        st.dataframe(brechas.round(2), use_container_width=True, hide_index=True)
    with c2:
        st.subheader("Top 3 prioridades")
        top3 = brechas.head(3).reset_index(drop=True)
        for i, row in top3.iterrows():
            st.markdown(f"**{i+1}. {row['Servicio']}**")
            st.write(
                f"Grupo mejor: {row['Cluster mejor']:.2f} | Grupo prioritario: {row['Cluster prioritario']:.2f} | Brecha: {row['Brecha']:.2f}"
            )
            st.progress(float(row["Brecha"]) / float(brechas["Brecha"].max()))

    st.markdown("---")
    st.subheader("Comparación directa de servicios por grupo")
    cols_res = [c for c in resumen.columns if "Grado de Madurez" in c and c != "cluster"]
    filas = []
    for _, row in resumen.iterrows():
        for col in cols_res:
            filas.append(
                {
                    "Grupo": nombres_cluster[int(row["cluster"])],
                    "Servicio": nombres_servicios.get(col, col),
                    "Nivel": float(row[col]),
                }
            )
    df_long = pd.DataFrame(filas)
    fig_comp = px.bar(
        df_long,
        x="Servicio",
        y="Nivel",
        color="Grupo",
        barmode="group",
        title="Promedio por servicio en cada grupo",
    )
    fig_comp.update_layout(height=480)
    st.plotly_chart(fig_comp, use_container_width=True)





with tab3:
    st.subheader("Comparar dos municipalidades")
    lista_munis = sorted(clusters["Municipalidad"].dropna().unique().tolist())

    if len(lista_munis) >= 2:
        a, b = st.columns(2)
        muni_1 = a.selectbox("Municipalidad 1", lista_munis, index=0)
        muni_2 = b.selectbox("Municipalidad 2", lista_munis, index=1)

        comp = datos[datos["Municipalidad"].isin([muni_1, muni_2])].copy()
        tabla_comp = comp[["Provincia", "Cantón", "Municipalidad", "Grado de Madurez Total 2025", "grupo"]].copy()
        tabla_comp = tabla_comp.rename(columns={
            "Grado de Madurez Total 2025": "Madurez total 2025",
            "grupo": "Grupo",
        })
        st.dataframe(tabla_comp, use_container_width=True, hide_index=True)

        servicios_visibles = list(nombres_servicios.keys())
        radar_rows = []
        comp_serv = base_servicios[base_servicios["Municipalidad"].isin([muni_1, muni_2])].copy()
        for _, row in comp_serv.iterrows():
            for col in servicios_visibles:
                radar_rows.append(
                    {
                        "Municipalidad": row["Municipalidad"],
                        "Servicio": nombres_servicios.get(col, col),
                        "Nivel": row[col],
                    }
                )
        radar_df = pd.DataFrame(radar_rows)
        fig_radar = px.line_polar(
            radar_df,
            r="Nivel",
            theta="Servicio",
            color="Municipalidad",
            line_close=True,
            markers=True,
            title="Comparación de servicios entre dos municipalidades",
            color_discrete_sequence=["#0B84F3", "#F05A28"],
        )
        fig_radar.update_layout(height=550)
        st.plotly_chart(fig_radar, use_container_width=True)





with tab4:
    st.subheader("Detalle y descarga")
    detalle = datos[["Provincia", "Cantón", "Municipalidad", "Grado de Madurez Total 2025", "grupo"]].copy()
    detalle = detalle.rename(columns={
        "Grado de Madurez Total 2025": "Madurez total 2025",
        "grupo": "Grupo",
    })

    d1, d2, d3 = st.columns(3)
    d1.metric("Municipalidades visibles", len(detalle))
    d2.metric("Provincias visibles", detalle["Provincia"].nunique())
    d3.metric("Grupos visibles", detalle["Grupo"].nunique())

    st.markdown("### Tabla filtrada")
    st.dataframe(detalle, use_container_width=True, hide_index=True)

    st.markdown("### Descargar resultados")
    st.write("Descarga la tabla exactamente con los filtros que tengas aplicados en el dashboard.")

    csv = detalle.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Descargar tabla filtrada en CSV",
        data=csv,
        file_name="municipalidades_filtradas.csv",
        mime="text/csv",
        use_container_width=True,
    )


st.markdown("---")
st.caption(
    f"Dashboard IGSM | Basado en {len(clusters)} municipalidades | K={k_final} | Silhouette={silhouette:.3f}"
)
