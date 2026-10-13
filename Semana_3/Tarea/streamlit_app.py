"""
Laboratorio Final — Starbucks Rewards · Grupo 11
App para decidir a qué porcentaje de la base enviar el cupón de $5, a partir del T-learner del
laboratorio: curva Qini, tabla de políticas y validación contra el tipo real de cada cliente.

Construida a partir de streamlit_starter.py.

Correr con:  streamlit run streamlit_app.py
"""

from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Laboratorio Final — Starbucks Rewards", page_icon="☕", layout="wide")

GREEN = "#00704A"
ORANGE = "#C4512F"
GRAY = "#9A988E"
FEATURES = ["recency_days", "frequency", "monetary", "email_open_rate", "tenure_days"]

# Streamlit Community Cloud ejecuta la app desde la raíz del repositorio: las rutas se arman
# a partir de la carpeta de este archivo para que funcionen igual en local y en la nube.
CARPETA = Path(__file__).parent

MODELOS = {
    "Regresión logística": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=100, random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.05, min_samples_leaf=100, random_state=42
    ),
}

TIPOS = {
    "persuadable": "Persuadible",
    "sure_thing": "Compra segura",
    "lost_cause": "Causa perdida",
    "sleeping_dog": "Perro dormido",
}


@st.cache_data
def load_data():
    return pd.read_csv(CARPETA / "data" / "uplift_campaign.csv")


@st.cache_data
def load_verdad():
    ruta = CARPETA / "uplift_campaign_FULL.csv"
    if not ruta.exists():
        return None
    return pd.read_csv(ruta, usecols=["customer_id", "true_type", "true_uplift_prob", "true_value_of_contact"])


@st.cache_data
def entrenar_t_learner(df: pd.DataFrame, nombre_modelo: str, test_size: float, seed: int):
    train, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=df["treatment_group"])

    train_t = train[train["treatment_group"] == "treatment"]
    train_c = train[train["treatment_group"] == "control"]
    base = MODELOS[nombre_modelo]
    modelo_tratado = clone(base).fit(train_t[FEATURES], train_t["responded_60d"])
    modelo_control = clone(base).fit(train_c[FEATURES], train_c["responded_60d"])

    # Modelo de churn del Paso 1, solo para comparar su curva contra la del T-learner
    modelo_riesgo = LogisticRegression(max_iter=1000).fit(train[FEATURES], 1 - train["responded_60d"])

    test = test.copy()
    X_test = test[FEATURES]
    test["p_tratado"] = modelo_tratado.predict_proba(X_test)[:, 1]
    test["p_control"] = modelo_control.predict_proba(X_test)[:, 1]
    test["uplift_estimado"] = test["p_tratado"] - test["p_control"]
    test["prob_churn"] = modelo_riesgo.predict_proba(X_test)[:, 1]
    return train, test


def valor_de_la_politica(test_df, seleccionados):
    sel = test_df.loc[seleccionados]
    valor_tratado = sel.loc[sel["treatment_group"] == "treatment", "utilidad_neta_60d"].mean()
    valor_control = sel.loc[sel["treatment_group"] == "control", "margin_60d"].mean()
    return valor_tratado - valor_control, len(sel)


def curva_qini(test_df, score_col, steps=40, acumulada=False):
    # acumulada=False: la fórmula del laboratorio (multiplica por el total de la base).
    # acumulada=True: compras extra acumuladas en el top N (multiplica por N), la que se grafica.
    orden = test_df.sort_values(score_col, ascending=False).reset_index(drop=True)
    es_tratado = (orden["treatment_group"] == "treatment").values
    respondio = orden["responded_60d"].values

    acumulado_tratados = np.cumsum(np.where(es_tratado, respondio, 0))
    acumulado_control = np.cumsum(np.where(~es_tratado, respondio, 0))
    n_tratados_acum = np.cumsum(es_tratado)
    n_control_acum = np.cumsum(~es_tratado)

    n = len(orden)
    xs, ys = [0.0], [0.0]
    for i in np.linspace(1, n, steps).astype(int):
        nt, nc = n_tratados_acum[i - 1], n_control_acum[i - 1]
        tasa_t = acumulado_tratados[i - 1] / nt if nt > 0 else 0
        tasa_c = acumulado_control[i - 1] / nc if nc > 0 else 0
        xs.append(i / n)
        ys.append((tasa_t - tasa_c) * (i if acumulada else n))
    return np.array(xs), np.array(ys)


def coeficiente_qini(xs, ys, n):
    area_modelo = np.trapezoid(ys, xs)
    area_azar = np.trapezoid(np.linspace(0, ys[-1], len(xs)), xs)
    return (area_modelo - area_azar) / n


@st.cache_data
def tabla_de_politicas(test: pd.DataFrame, paso: int):
    orden = test.sort_values("uplift_estimado", ascending=False)
    filas = []
    for pct in range(paso, 101, paso):
        idx = orden.head(int(len(test) * pct / 100)).index
        valor_cliente, n = valor_de_la_politica(test, idx)
        filas.append({"% contactado": pct, "Clientes": n, "Valor por cliente ($)": valor_cliente,
                      "Valor total ($)": valor_cliente * n})
    tabla = pd.DataFrame(filas)
    tabla["Agrega el tramo ($)"] = tabla["Valor total ($)"].diff().fillna(tabla["Valor total ($)"])
    return tabla


def dinero(valor, decimales=0):
    texto = f"${abs(valor):,.{decimales}f}"
    return f"−{texto}" if valor < 0 else texto


# --- Título y carga de datos ---
st.title("Laboratorio Final — Starbucks Rewards")
st.caption("¿A qué porcentaje de la base conviene enviar el cupón de \\$5? Valor neto medido con el experimento "
           "(tratados contra control) sobre la base de prueba.")

df = load_data()

with st.sidebar:
    st.header("Parámetros")
    nombre_modelo = st.selectbox("Modelo base del T-learner", list(MODELOS))
    pct_contactar = st.slider("% de la base a contactar", min_value=10, max_value=100, value=20, step=10,
                              help="Clientes ordenados de mayor a menor uplift estimado.")

with st.spinner("Entrenando el T-learner…"):
    train, test = entrenar_t_learner(df, nombre_modelo, test_size=0.30, seed=42)

tabla = tabla_de_politicas(test, paso=10)
fila = tabla.loc[tabla["% contactado"] == pct_contactar].iloc[0]
mejor = tabla.loc[tabla["Valor total ($)"].idxmax()]

# --- Indicadores del % elegido ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("Clientes contactados", f"{int(fila['Clientes']):,} de {len(test):,}")
c2.metric("Valor neto total", dinero(fila["Valor total ($)"]))
c3.metric("Valor neto por cliente", dinero(fila["Valor por cliente ($)"], 2))
c4.metric("Corte con más valor total", f"{int(mejor['% contactado'])}% · {dinero(mejor['Valor total ($)'])}")

# --- Curva Qini y valor total por corte ---
xs_u, ys_u = curva_qini(test, "uplift_estimado", acumulada=True)
xs_r, ys_r = curva_qini(test, "prob_churn", acumulada=True)
qini_uplift = coeficiente_qini(*curva_qini(test, "uplift_estimado"), len(test))
qini_riesgo = coeficiente_qini(*curva_qini(test, "prob_churn"), len(test))

izq, der = st.columns(2)
with izq:
    st.subheader("Curva Qini")
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.plot(xs_u * 100, ys_u, color=GREEN, linewidth=3, label=f"T-learner · {nombre_modelo} (Qini={qini_uplift:+.3f})")
    ax.plot(xs_r * 100, ys_r, color=ORANGE, linewidth=3, label=f"Riesgo/churn (Qini={qini_riesgo:+.3f})")
    ax.plot([0, 100], [0, ys_u[-1]], color=GRAY, linestyle="--", label="Al azar")
    ax.axvline(pct_contactar, color="#333333", linewidth=1, linestyle=":")
    ax.set_xlabel("% de la base contactada (de prueba)")
    ax.set_ylabel("Compras extra acumuladas")
    ax.legend(loc="lower right", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Compras que ocurren solo gracias al cupón. Coeficiente Qini con la fórmula del laboratorio.")

with der:
    st.subheader("Valor neto total por corte")
    fig, ax = plt.subplots(figsize=(7, 4.6))
    colores = [GREEN if p == pct_contactar else GRAY for p in tabla["% contactado"]]
    ax.bar(tabla["% contactado"], tabla["Valor total ($)"], width=7, color=colores)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("% de la base contactada (ordenada por uplift estimado)")
    ax.set_ylabel("Valor neto total ($)")
    ax.set_xticks(range(10, 101, 10))
    ax.yaxis.set_major_formatter(lambda v, _: dinero(v))
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Valor neto: margen menos el cupón (\\$5, si compró) y el envío (\\$0.05), contra el margen del control.")

# --- Tabla de políticas ---
st.subheader("Tabla de políticas")


def resaltar(row):
    estilo = f"background-color: {GREEN}; color: white" if row["% contactado"] == pct_contactar else ""
    return [estilo] * len(row)


st.dataframe(
    tabla.style.apply(resaltar, axis=1).format({
        "% contactado": "{:d}%",
        "Clientes": "{:,}",
        "Valor por cliente ($)": "{:+,.2f}",
        "Valor total ($)": "{:+,.0f}",
        "Agrega el tramo ($)": "{:+,.0f}",
    }),
    hide_index=True,
    width="stretch",
    height=390,
)

# --- Validación contra la verdad ---
verdad = load_verdad()
if verdad is not None:
    with st.expander("Validación contra el tipo real de cada cliente"):
        datos = test[["customer_id", "uplift_estimado"]].merge(verdad, on="customer_id")
        elegidos = datos.sort_values("uplift_estimado", ascending=False).head(int(fila["Clientes"]))
        persuadibles = (datos["true_type"] == "persuadable").sum()

        v1, v2, v3 = st.columns(3)
        v1.metric("Valor real del grupo elegido", dinero(elegidos["true_value_of_contact"].sum()))
        v2.metric("Persuadibles alcanzados",
                  f"{(elegidos['true_type'] == 'persuadable').sum():,} de {persuadibles:,}")
        v3.metric("Correlación con el uplift real",
                  f"{datos['uplift_estimado'].corr(datos['true_uplift_prob']):.3f}")

        composicion = (elegidos["true_type"].map(TIPOS).value_counts(normalize=True)
                       .reindex(TIPOS.values()).fillna(0).mul(100).round(1))
        st.dataframe(composicion.rename("% del grupo elegido").to_frame(), width="stretch")
        st.caption("El tipo real viene del simulador del laboratorio; en una campaña real no se conoce.")
