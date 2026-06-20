import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import poisson

import statsmodels.api as sm
import statsmodels.formula.api as smf

import requests
import certifi
from io import StringIO


# =========================
# CONFIGURACIÓN DE LA APP
# =========================

st.set_page_config(
    page_title="Predicción de Marcadores",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ Predicción de marcadores internacionales")
st.caption("Modelo Poisson + ajuste tipo Dixon-Coles usando historial de partidos internacionales.")


# =========================
# CARGAR DATASET
# =========================

@st.cache_data
def cargar_datos():
    url = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
    
    response = requests.get(
        url,
        verify=certifi.where(),
        timeout=30
    )
    
    response.raise_for_status()
    
    csv_data = StringIO(response.text)
    
    df = pd.read_csv(csv_data)
    df["date"] = pd.to_datetime(df["date"])
    
    return df


# =========================
# ENTRENAR MODELO
# =========================

@st.cache_resource
def entrenar_modelo(fecha_inicio="2018-01-01", min_partidos=5, xi=0.0015):
    df = cargar_datos()

    df_recent = df[df["date"] >= fecha_inicio].copy()

    partidos_modelo = df_recent[
        [
            "date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "neutral"
        ]
    ].copy()

    conteo_home = partidos_modelo["home_team"].value_counts()
    conteo_away = partidos_modelo["away_team"].value_counts()
    conteo_total = conteo_home.add(conteo_away, fill_value=0)

    equipos_validos = conteo_total[conteo_total >= min_partidos].index.tolist()
    equipos_validos = sorted(equipos_validos)

    partidos_modelo = partidos_modelo[
        partidos_modelo["home_team"].isin(equipos_validos) &
        partidos_modelo["away_team"].isin(equipos_validos)
    ].copy()

    home_rows = partidos_modelo[
        [
            "date",
            "home_team",
            "away_team",
            "home_score",
            "neutral"
        ]
    ].copy()

    home_rows.columns = [
        "date",
        "team",
        "opponent",
        "goals",
        "neutral"
    ]

    home_rows["is_home"] = np.where(home_rows["neutral"] == True, 0, 1)

    away_rows = partidos_modelo[
        [
            "date",
            "away_team",
            "home_team",
            "away_score",
            "neutral"
        ]
    ].copy()

    away_rows.columns = [
        "date",
        "team",
        "opponent",
        "goals",
        "neutral"
    ]

    away_rows["is_home"] = 0

    model_data = pd.concat([home_rows, away_rows], ignore_index=True)

    fecha_max = model_data["date"].max()
    dias_desde_partido = (fecha_max - model_data["date"]).dt.days

    model_data["peso"] = np.exp(-xi * dias_desde_partido)

    formula = "goals ~ is_home + C(team) + C(opponent)"

    modelo_poisson = smf.glm(
        formula=formula,
        data=model_data,
        family=sm.families.Poisson(),
        freq_weights=model_data["peso"]
    ).fit(maxiter=100)

    info_modelo = {
        "partidos_usados": len(partidos_modelo),
        "equipos_usados": len(equipos_validos),
        "fecha_inicio": fecha_inicio,
        "min_partidos": min_partidos,
        "xi": xi
    }

    return modelo_poisson, equipos_validos, info_modelo


# =========================
# FUNCIONES DEL MODELO
# =========================

def goles_esperados_poisson(modelo_poisson, equipo_a, equipo_b, neutral=True):
    is_home_a = 0 if neutral else 1

    datos_prediccion = pd.DataFrame({
        "team": [equipo_a, equipo_b],
        "opponent": [equipo_b, equipo_a],
        "is_home": [is_home_a, 0]
    })

    predicciones = modelo_poisson.predict(datos_prediccion)

    goles_a = predicciones.iloc[0]
    goles_b = predicciones.iloc[1]

    return goles_a, goles_b


def ajuste_dc_individual(goles_a, goles_b, lambda_a, lambda_b, rho=-0.05):
    if goles_a == 0 and goles_b == 0:
        return 1 - lambda_a * lambda_b * rho
    elif goles_a == 0 and goles_b == 1:
        return 1 + lambda_a * rho
    elif goles_a == 1 and goles_b == 0:
        return 1 + lambda_b * rho
    elif goles_a == 1 and goles_b == 1:
        return 1 - rho
    else:
        return 1


def predecir_marcador_poisson_dc(
    modelo_poisson,
    equipo_a,
    equipo_b,
    neutral=True,
    max_goles=8,
    rho=-0.05
):
    lambda_a, lambda_b = goles_esperados_poisson(
        modelo_poisson,
        equipo_a,
        equipo_b,
        neutral=neutral
    )

    resultados = []

    for goles_a in range(max_goles + 1):
        for goles_b in range(max_goles + 1):
            prob_a = poisson.pmf(goles_a, lambda_a)
            prob_b = poisson.pmf(goles_b, lambda_b)

            ajuste = ajuste_dc_individual(
                goles_a,
                goles_b,
                lambda_a,
                lambda_b,
                rho
            )

            prob_total = prob_a * prob_b * ajuste

            if prob_total < 0 or not np.isfinite(prob_total):
                prob_total = 0

            resultados.append({
                "equipo_a": equipo_a,
                "equipo_b": equipo_b,
                "goles_a": goles_a,
                "goles_b": goles_b,
                "marcador": f"{equipo_a} {goles_a}-{goles_b} {equipo_b}",
                "probabilidad": prob_total
            })

    resultados_df = pd.DataFrame(resultados)

    suma_prob = resultados_df["probabilidad"].sum()

    resultados_df["probabilidad"] = resultados_df["probabilidad"] / suma_prob
    resultados_df = resultados_df.sort_values("probabilidad", ascending=False)

    return resultados_df, lambda_a, lambda_b


def resumen_resultado(resultados):
    gana_a = resultados[resultados["goles_a"] > resultados["goles_b"]]["probabilidad"].sum()
    empate = resultados[resultados["goles_a"] == resultados["goles_b"]]["probabilidad"].sum()
    gana_b = resultados[resultados["goles_a"] < resultados["goles_b"]]["probabilidad"].sum()

    resumen = pd.DataFrame({
        "Resultado": [
            f"Gana {resultados.iloc[0]['equipo_a']}",
            "Empate",
            f"Gana {resultados.iloc[0]['equipo_b']}"
        ],
        "Probabilidad_%": [
            gana_a * 100,
            empate * 100,
            gana_b * 100
        ]
    })

    return resumen


def graficar_top10(top10, equipo_a, equipo_b):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.barh(top10["marcador"], top10["probabilidad_%"])
    ax.set_xlabel("Probabilidad (%)")
    ax.set_title(f"Top 10 marcadores más probables: {equipo_a} vs {equipo_b}")
    ax.invert_yaxis()

    return fig


def mostrar_reporte_partido(
    modelo_poisson,
    equipo_a,
    equipo_b,
    neutral=True,
    max_goles=8,
    rho=-0.05,
    key_prefix="partido"
):
    resultados, xg_a, xg_b = predecir_marcador_poisson_dc(
        modelo_poisson,
        equipo_a,
        equipo_b,
        neutral=neutral,
        max_goles=max_goles,
        rho=rho
    )

    top10 = resultados.head(10).copy()
    top10["probabilidad_%"] = top10["probabilidad"] * 100

    marcador_mas_probable = top10.iloc[0]["marcador"]
    prob_marcador = top10.iloc[0]["probabilidad_%"]

    st.subheader(f"{equipo_a} vs {equipo_b}")

    m1, m2, m3 = st.columns(3)

    m1.metric(
        f"Goles esperados {equipo_a}",
        f"{xg_a:.2f}"
    )

    m2.metric(
        f"Goles esperados {equipo_b}",
        f"{xg_b:.2f}"
    )

    m3.metric(
        "Marcador más probable",
        marcador_mas_probable,
        f"{prob_marcador:.2f}%"
    )

    st.divider()

    col_tabla, col_resumen = st.columns([2, 1])

    with col_tabla:
        st.markdown("### Top 10 marcadores")

        tabla_top10 = top10[["marcador", "probabilidad_%"]].copy()
        tabla_top10["probabilidad_%"] = tabla_top10["probabilidad_%"].round(2)

        st.dataframe(
            tabla_top10,
            use_container_width=True,
            hide_index=True
        )

    with col_resumen:
        st.markdown("### Resultado probable")

        resumen = resumen_resultado(resultados)
        resumen["Probabilidad_%"] = resumen["Probabilidad_%"].round(2)

        st.dataframe(
            resumen,
            use_container_width=True,
            hide_index=True
        )

    fig = graficar_top10(top10, equipo_a, equipo_b)
    st.pyplot(fig)

    resultados_export = resultados.copy()
    resultados_export["probabilidad_%"] = resultados_export["probabilidad"] * 100

    nombre_archivo = f"prediccion_{equipo_a}_vs_{equipo_b}.csv"
    nombre_archivo = nombre_archivo.replace(" ", "_").replace("ç", "c")

    st.download_button(
        label="Descargar resultados CSV",
        data=resultados_export.to_csv(index=False).encode("utf-8"),
        file_name=nombre_archivo,
        mime="text/csv",
        key=f"download_{key_prefix}_{equipo_a}_{equipo_b}"
    )

    return resultados

# =========================
# SIDEBAR
# =========================

st.sidebar.header("Configuración del modelo")

fecha_inicio = st.sidebar.text_input(
    "Filtrar partidos desde:",
    value="2018-01-01"
)

min_partidos = st.sidebar.slider(
    "Mínimo de partidos por selección",
    min_value=3,
    max_value=20,
    value=5,
    step=1
)

xi = st.sidebar.slider(
    "Peso de partidos recientes",
    min_value=0.0000,
    max_value=0.0050,
    value=0.0015,
    step=0.0005,
    format="%.4f"
)

max_goles = st.sidebar.slider(
    "Máximo de goles a calcular",
    min_value=5,
    max_value=12,
    value=8,
    step=1
)

rho = st.sidebar.slider(
    "Ajuste Dixon-Coles",
    min_value=-0.10,
    max_value=0.10,
    value=-0.05,
    step=0.01
)


# =========================
# ENTRENAR
# =========================

with st.spinner("Cargando datos y entrenando modelo..."):
    modelo_poisson, equipos_validos, info_modelo = entrenar_modelo(
        fecha_inicio=fecha_inicio,
        min_partidos=min_partidos,
        xi=xi
    )


with st.sidebar.expander("Información del modelo"):
    st.write("Partidos usados:", info_modelo["partidos_usados"])
    st.write("Equipos usados:", info_modelo["equipos_usados"])
    st.write("Fecha inicio:", info_modelo["fecha_inicio"])


# =========================
# INTERFAZ PRINCIPAL
# =========================

st.divider()

partidos_rapidos_base = [
    ("Brazil", "Haiti"),
    ("Turkey", "Paraguay"),
    ("Netherlands", "Sweden"),
    ("Germany", "Ivory Coast"),
    ("Ecuador", "Curaçao"),
    ("Tunisia", "Japan"),
]

# Solo mostramos partidos cuyos equipos existan en el dataset
partidos_rapidos = [
    (a, b)
    for a, b in partidos_rapidos_base
    if a in equipos_validos and b in equipos_validos
]

tab_uno, tab_dia = st.tabs([
    "Calcular un partido",
    "Partidos del día"
])


# =========================
# TAB 1: UN PARTIDO
# =========================

with tab_uno:
    st.markdown("## Calcular un partido")

    opciones_rapidas = ["Personalizado"]

    for a, b in partidos_rapidos:
        opciones_rapidas.append(f"{a} vs {b}")

    partido_rapido = st.selectbox(
        "Partido rápido",
        opciones_rapidas
    )

    if partido_rapido != "Personalizado":
        equipo_a_rapido, equipo_b_rapido = partido_rapido.split(" vs ")
    else:
        equipo_a_rapido = "Mexico" if "Mexico" in equipos_validos else equipos_validos[0]
        equipo_b_rapido = "South Korea" if "South Korea" in equipos_validos else equipos_validos[1]

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        default_a = equipos_validos.index(equipo_a_rapido) if equipo_a_rapido in equipos_validos else 0

        equipo_a = st.selectbox(
            "Equipo A",
            equipos_validos,
            index=default_a
        )

    with col2:
        default_b = equipos_validos.index(equipo_b_rapido) if equipo_b_rapido in equipos_validos else 1

        equipo_b = st.selectbox(
            "Equipo B",
            equipos_validos,
            index=default_b
        )

    with col3:
        neutral_partido = st.toggle(
            "Cancha neutral",
            value=True,
            key="neutral_partido"
        )

    calcular = st.button(
        "Calcular marcador",
        type="primary",
        key="calcular_un_partido"
    )

    if calcular:
        if equipo_a == equipo_b:
            st.warning("Selecciona dos equipos diferentes.")
        else:
            mostrar_reporte_partido(
                modelo_poisson,
                equipo_a,
                equipo_b,
                neutral=neutral_partido,
                max_goles=max_goles,
                rho=rho,
                key_prefix="individual"
            )
    else:
        st.info("Selecciona dos equipos y da clic en Calcular marcador.")


# =========================
# TAB 2: PARTIDOS DEL DÍA
# =========================

with tab_dia:
    st.markdown("## Partidos rápidos del día")

    if len(partidos_rapidos) == 0:
        st.warning("No se encontraron partidos rápidos con nombres válidos en el dataset.")
    else:
        tabla_partidos = pd.DataFrame(
            partidos_rapidos,
            columns=["Equipo A", "Equipo B"]
        )

        st.dataframe(
            tabla_partidos,
            use_container_width=True,
            hide_index=True
        )

        calcular_todos = st.button(
            "Calcular todos los partidos del día",
            type="primary",
            key="calcular_todos"
        )

        if calcular_todos:
            resultados_exportar = []

            for i, (equipo_a_dia, equipo_b_dia) in enumerate(partidos_rapidos):
                with st.expander(
                    f"{equipo_a_dia} vs {equipo_b_dia}",
                    expanded=(i == 0)
                ):
                    resultados = mostrar_reporte_partido(
                        modelo_poisson,
                        equipo_a_dia,
                        equipo_b_dia,
                        neutral=True,
                        max_goles=max_goles,
                        rho=rho,
                        key_prefix=f"dia_{i}"
                    )

                    temp = resultados.copy()
                    temp["partido"] = f"{equipo_a_dia} vs {equipo_b_dia}"
                    temp["probabilidad_%"] = temp["probabilidad"] * 100

                    resultados_exportar.append(temp)

            if len(resultados_exportar) > 0:
                df_resultados_dia = pd.concat(
                    resultados_exportar,
                    ignore_index=True
                )

                st.divider()

                st.subheader("Descargar todos los resultados del día")

                st.download_button(
                    label="Descargar CSV con todos los partidos",
                    data=df_resultados_dia.to_csv(index=False).encode("utf-8"),
                    file_name="predicciones_partidos_del_dia.csv",
                    mime="text/csv",
                    key="download_todos_los_partidos"
                )
        else:
            st.info("Da clic en Calcular todos los partidos del día para generar los reportes.")