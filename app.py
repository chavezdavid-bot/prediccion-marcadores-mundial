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

from datetime import datetime
from zoneinfo import ZoneInfo

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

    # Agregamos resultados reales del Mundial 2026 si existe el archivo
    try:
        df_extra = pd.read_csv("resultados_mundial_2026.csv")
        df_extra["date"] = pd.to_datetime(df_extra["date"])

        df_extra["neutral"] = (
            df_extra["neutral"]
            .astype(str)
            .str.upper()
            .map({
                "TRUE": True,
                "FALSE": False
            })
        )

        # Usamos las mismas columnas del dataset principal
        df_extra = df_extra[df.columns]

        df = pd.concat(
            [df, df_extra],
            ignore_index=True
        )

        df = df.drop_duplicates(
            subset=["date", "home_team", "away_team"],
            keep="last"
        )

    except FileNotFoundError:
        pass
    
    return df


@st.cache_data
def cargar_calendario():
    calendario = pd.read_csv("partidos_mundial.csv")
    calendario["fecha"] = pd.to_datetime(calendario["fecha"]).dt.date
    return calendario
@st.cache_data
def cargar_ratings():
    try:
        ratings = pd.read_csv("ratings_equipos.csv")
    except FileNotFoundError:
        ratings = pd.DataFrame(columns=["equipo", "elo"])

    return ratings


def obtener_elo_equipo(equipo, ratings_df):
    if ratings_df is None:
        return 1500

    if ratings_df.empty:
        return 1500

    fila = ratings_df[ratings_df["equipo"] == equipo]

    if fila.empty:
        return 1500

    return float(fila["elo"].iloc[0])


@st.cache_data
def cargar_forma_mundial():
    """
    Carga resultados del Mundial 2026 y calcula forma reciente por equipo.
    La forma se suaviza para no sobre-reaccionar con pocos partidos.
    """
    try:
        resultados = pd.read_csv("resultados_mundial_2026.csv")
    except FileNotFoundError:
        return pd.DataFrame(columns=[
            "equipo",
            "partidos",
            "gf",
            "ga",
            "gf_prom",
            "ga_prom",
            "ataque_forma",
            "defensa_debilidad_forma"
        ])

    if resultados.empty:
        return pd.DataFrame(columns=[
            "equipo",
            "partidos",
            "gf",
            "ga",
            "gf_prom",
            "ga_prom",
            "ataque_forma",
            "defensa_debilidad_forma"
        ])

    home = resultados[["home_team", "home_score", "away_score"]].copy()
    home.columns = ["equipo", "gf", "ga"]

    away = resultados[["away_team", "away_score", "home_score"]].copy()
    away.columns = ["equipo", "gf", "ga"]

    forma_base = pd.concat([home, away], ignore_index=True)

    forma = forma_base.groupby("equipo").agg(
        partidos=("equipo", "count"),
        gf=("gf", "sum"),
        ga=("ga", "sum")
    ).reset_index()

    forma["gf_prom"] = forma["gf"] / forma["partidos"]
    forma["ga_prom"] = forma["ga"] / forma["partidos"]

    promedio_gf = forma_base["gf"].mean()
    promedio_ga = forma_base["ga"].mean()

    if promedio_gf <= 0 or not np.isfinite(promedio_gf):
        promedio_gf = 1.0

    if promedio_ga <= 0 or not np.isfinite(promedio_ga):
        promedio_ga = 1.0

    # Suavizado: con 1 partido se ajusta poco; con 3+ partidos pesa más.
    k_suavizado = 3
    forma["shrink"] = forma["partidos"] / (forma["partidos"] + k_suavizado)

    ataque_crudo = forma["gf_prom"] / promedio_gf
    defensa_debilidad_cruda = forma["ga_prom"] / promedio_ga

    forma["ataque_forma"] = 1 + forma["shrink"] * (ataque_crudo - 1)
    forma["defensa_debilidad_forma"] = 1 + forma["shrink"] * (defensa_debilidad_cruda - 1)

    # Evita extremos por muestras pequeñas o resultados muy raros.
    forma["ataque_forma"] = forma["ataque_forma"].clip(0.65, 1.60)
    forma["defensa_debilidad_forma"] = forma["defensa_debilidad_forma"].clip(0.65, 1.60)

    return forma


def obtener_forma_equipo(equipo, forma_df):
    if forma_df is None or forma_df.empty:
        return 1.0, 1.0, 0

    fila = forma_df[forma_df["equipo"] == equipo]

    if fila.empty:
        return 1.0, 1.0, 0

    ataque = float(fila["ataque_forma"].iloc[0])
    defensa_debilidad = float(fila["defensa_debilidad_forma"].iloc[0])
    partidos = int(fila["partidos"].iloc[0])

    return ataque, defensa_debilidad, partidos

# =========================
# ENTRENAR MODELO
# =========================

@st.cache_resource
def entrenar_modelo(fecha_inicio="2020-01-01", min_partidos=8, xi=0.0015):
    df = cargar_datos()

    fecha_inicio_dt = pd.to_datetime(fecha_inicio)

    df_recent = df[df["date"] >= fecha_inicio_dt].copy()

    # Para que la app sea más rápida en Streamlit Cloud,
    # usamos solo los partidos más recientes dentro del filtro.
    max_partidos_entrenamiento = 5000

    df_recent = df_recent.sort_values("date").tail(max_partidos_entrenamiento).copy()

    partidos_modelo = df_recent[
        [
            "date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "neutral",
            "tournament"
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
            "neutral",
            "tournament"
        ]
    ].copy()

    home_rows.columns = [
        "date",
        "team",
        "opponent",
        "goals",
        "neutral",
        "tournament"
    ]

    home_rows["is_home"] = np.where(home_rows["neutral"] == True, 0, 1)

    away_rows = partidos_modelo[
        [
            "date",
            "away_team",
            "home_team",
            "away_score",
            "neutral",
            "tournament"
        ]
    ].copy()

    away_rows.columns = [
        "date",
        "team",
        "opponent",
        "goals",
        "neutral",
        "tournament"
    ]

    away_rows["is_home"] = 0

    model_data = pd.concat([home_rows, away_rows], ignore_index=True)

    model_data["tournament"] = model_data["tournament"].fillna("")

    fecha_max = model_data["date"].max()

    dias_desde_partido = (fecha_max - model_data["date"]).dt.days

    model_data["peso"] = np.exp(-xi * dias_desde_partido)

    # Damos más peso a partidos de Copa Mundial
    peso_mundial = np.where(
        model_data["tournament"].astype(str).str.contains("FIFA World Cup", case=False, na=False),
        2.0,
        1.0
    )

    # Damos todavía más peso a partidos del Mundial 2026 ya jugados
    peso_mundial_2026 = np.where(
        model_data["date"] >= pd.to_datetime("2026-06-11"),
        4.0,
        1.0
    )

    model_data["peso"] = model_data["peso"] * peso_mundial * peso_mundial_2026

    formula = "goals ~ is_home + C(team) + C(opponent)"

    modelo_poisson = smf.glm(
        formula=formula,
        data=model_data,
        family=sm.families.Poisson(),
        freq_weights=model_data["peso"]
    ).fit(maxiter=50)

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

def goles_esperados_poisson(
    modelo_poisson,
    equipo_a,
    equipo_b,
    neutral=True,
    ratings_df=None,
    peso_elo=0.35,
    forma_df=None,
    peso_forma=0.30
):
    is_home_a = 0 if neutral else 1

    datos_prediccion = pd.DataFrame({
        "team": [equipo_a, equipo_b],
        "opponent": [equipo_b, equipo_a],
        "is_home": [is_home_a, 0]
    })

    predicciones = modelo_poisson.predict(datos_prediccion)

    goles_a = float(predicciones.iloc[0])
    goles_b = float(predicciones.iloc[1])

    # =========================
    # AJUSTE POR RATING / ELO
    # =========================

    elo_a = obtener_elo_equipo(equipo_a, ratings_df)
    elo_b = obtener_elo_equipo(equipo_b, ratings_df)

    diferencia_elo = elo_a - elo_b

    # Convertimos la diferencia Elo en un factor de ajuste suave.
    factor_a = 10 ** ((diferencia_elo / 400) * peso_elo)
    factor_b = 10 ** ((-diferencia_elo / 400) * peso_elo)

    goles_a_ajustado = goles_a * factor_a
    goles_b_ajustado = goles_b * factor_b

    # El ajuste Elo redistribuye los goles esperados, pero conserva el total original.
    total_original = goles_a + goles_b
    total_elo = goles_a_ajustado + goles_b_ajustado

    if total_elo > 0:
        escala = total_original / total_elo
        goles_a_ajustado *= escala
        goles_b_ajustado *= escala

    # =========================
    # AJUSTE POR FORMA DEL MUNDIAL ACTUAL
    # =========================

    ataque_a, defensa_debilidad_a, partidos_a = obtener_forma_equipo(equipo_a, forma_df)
    ataque_b, defensa_debilidad_b, partidos_b = obtener_forma_equipo(equipo_b, forma_df)

    # Ataque propio + fragilidad defensiva del rival.
    factor_forma_a = (ataque_a * defensa_debilidad_b) ** peso_forma
    factor_forma_b = (ataque_b * defensa_debilidad_a) ** peso_forma

    goles_a_forma = goles_a_ajustado * factor_forma_a
    goles_b_forma = goles_b_ajustado * factor_forma_b

    # La forma sí puede subir/bajar el total esperado, pero con límites.
    total_antes_forma = goles_a_ajustado + goles_b_ajustado
    total_despues_forma = goles_a_forma + goles_b_forma

    if total_antes_forma > 0 and total_despues_forma > 0:
        ratio_total = total_despues_forma / total_antes_forma
        ratio_total_limitado = np.clip(ratio_total, 0.75, 1.35)
        escala_total = ratio_total_limitado / ratio_total
        goles_a_forma *= escala_total
        goles_b_forma *= escala_total

    # Evita valores absurdos por cualquier combinación extrema.
    goles_a_forma = float(np.clip(goles_a_forma, 0.05, 6.5))
    goles_b_forma = float(np.clip(goles_b_forma, 0.05, 6.5))

    return goles_a_forma, goles_b_forma


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
    rho=-0.05,
    ratings_df=None,
    peso_elo=0.35,
    forma_df=None,
    peso_forma=0.30
):
    lambda_a, lambda_b = goles_esperados_poisson(
        modelo_poisson,
        equipo_a,
        equipo_b,
        neutral=neutral,
        ratings_df=ratings_df,
        peso_elo=peso_elo,
        forma_df=forma_df,
        peso_forma=peso_forma
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
    key_prefix="partido",
    ratings_df=None,
    peso_elo=0.35,
    forma_df=None,
    peso_forma=0.30
):
    resultados, xg_a, xg_b = predecir_marcador_poisson_dc(
        modelo_poisson,
        equipo_a,
        equipo_b,
        neutral=neutral,
        max_goles=max_goles,
        rho=rho,
        ratings_df=ratings_df,
        peso_elo=peso_elo,
        forma_df=forma_df,
        peso_forma=peso_forma
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
    
def obtener_partidos_por_fecha(calendario, fecha):
    partidos_fecha = calendario[calendario["fecha"] == fecha].copy()

    partidos = list(
        zip(
            partidos_fecha["equipo_a"],
            partidos_fecha["equipo_b"]
        )
    )

    return partidos, partidos_fecha


def filtrar_partidos_validos(partidos, equipos_validos):
    partidos_validos = []

    for equipo_a, equipo_b in partidos:
        if equipo_a in equipos_validos and equipo_b in equipos_validos:
            partidos_validos.append((equipo_a, equipo_b))

    return partidos_validos    

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

peso_elo = st.sidebar.slider(
    "Peso rating/Elo",
    min_value=0.0,
    max_value=1.0,
    value=0.35,
    step=0.05
)

peso_forma = st.sidebar.slider(
    "Peso forma Mundial actual",
    min_value=0.0,
    max_value=1.0,
    value=0.30,
    step=0.05
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

ratings_equipos = cargar_ratings()
forma_mundial = cargar_forma_mundial()


with st.sidebar.expander("Información del modelo"):
    st.write("Partidos usados:", info_modelo["partidos_usados"])
    st.write("Equipos usados:", info_modelo["equipos_usados"])
    st.write("Ratings cargados:", len(ratings_equipos))
    st.write("Equipos con forma Mundial:", len(forma_mundial))
    st.write("Fecha inicio:", info_modelo["fecha_inicio"])


# =========================
# INTERFAZ PRINCIPAL
# =========================

st.divider()

calendario_partidos = cargar_calendario()

hoy_mexico = datetime.now(
    ZoneInfo("America/Mexico_City")
).date()

partidos_rapidos_base, partidos_hoy_df = obtener_partidos_por_fecha(
    calendario_partidos,
    hoy_mexico
)

partidos_rapidos = filtrar_partidos_validos(
    partidos_rapidos_base,
    equipos_validos
)

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
                key_prefix="individual",
                ratings_df=ratings_equipos,
                peso_elo=peso_elo,
                forma_df=forma_mundial,
                peso_forma=peso_forma
            )
    else:
        st.info("Selecciona dos equipos y da clic en Calcular marcador.")


# =========================
# TAB 2: PARTIDOS DEL DÍA
# =========================

with tab_dia:
    st.markdown("## Partidos por fecha")

    fechas_disponibles = sorted(
        calendario_partidos["fecha"].unique()
    )

    fecha_seleccionada = st.date_input(
        "Selecciona una fecha",
        value=hoy_mexico
    )

    partidos_fecha_base, partidos_fecha_df = obtener_partidos_por_fecha(
        calendario_partidos,
        fecha_seleccionada
    )

    partidos_fecha = filtrar_partidos_validos(
        partidos_fecha_base,
        equipos_validos
    )

    if len(partidos_fecha_df) == 0:
        st.warning("No hay partidos registrados para esta fecha en el calendario CSV.")

        st.markdown("Fechas disponibles en el calendario:")

        fechas_df = pd.DataFrame({
            "Fechas disponibles": fechas_disponibles
        })

        st.dataframe(
            fechas_df,
            use_container_width=True,
            hide_index=True
        )

    elif len(partidos_fecha) == 0:
        st.warning("Hay partidos en el CSV, pero los nombres no coinciden con el dataset.")

        st.markdown("Partidos encontrados en el CSV:")

        st.dataframe(
            partidos_fecha_df,
            use_container_width=True,
            hide_index=True
        )

        st.info("Revisa que los nombres de los equipos coincidan con el dataset.")

    else:
        st.markdown(f"### Partidos para {fecha_seleccionada}")

        tabla_partidos = partidos_fecha_df[
            [
                "fecha",
                "equipo_a",
                "equipo_b",
                "grupo",
                "sede"
            ]
        ].copy()

        st.dataframe(
            tabla_partidos,
            use_container_width=True,
            hide_index=True
        )

        calcular_todos = st.button(
            "Calcular todos los partidos de esta fecha",
            type="primary",
            key="calcular_todos_fecha"
        )

        if calcular_todos:
            resultados_exportar = []

            for i, (equipo_a_dia, equipo_b_dia) in enumerate(partidos_fecha):
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
                        key_prefix=f"fecha_{fecha_seleccionada}_{i}",
                        ratings_df=ratings_equipos,
                        peso_elo=peso_elo,
                        forma_df=forma_mundial,
                        peso_forma=peso_forma
                    )

                    temp = resultados.copy()
                    temp["partido"] = f"{equipo_a_dia} vs {equipo_b_dia}"
                    temp["fecha"] = fecha_seleccionada
                    temp["probabilidad_%"] = temp["probabilidad"] * 100

                    resultados_exportar.append(temp)

            if len(resultados_exportar) > 0:
                df_resultados_fecha = pd.concat(
                    resultados_exportar,
                    ignore_index=True
                )

                st.divider()

                st.subheader("Descargar todos los resultados de la fecha")

                nombre_csv = f"predicciones_{fecha_seleccionada}.csv"

                st.download_button(
                    label="Descargar CSV con todos los partidos",
                    data=df_resultados_fecha.to_csv(index=False).encode("utf-8"),
                    file_name=nombre_csv,
                    mime="text/csv",
                    key=f"download_todos_{fecha_seleccionada}"
                )
        else:
            st.info("Da clic en Calcular todos los partidos de esta fecha para generar los reportes.")