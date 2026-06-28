import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Yield Map | PRJ4", page_icon="📊", layout="wide")

DATA_DIR = Path(__file__).parent / "data"

RATING_ORDER = [
    "AAA", "AA+", "AA", "AA-",
    "A+", "A", "A-",
    "BBB+", "BBB", "BBB-",
    "BB+", "BB", "BB-",
    "B+", "B", "B-",
    "CCC", "D",
]

RATING_COLORS = {
    "AAA": "#0d47a1", "AA+": "#1565c0", "AA": "#1976d2", "AA-": "#1e88e5",
    "A+":  "#2e7d32", "A":   "#388e3c", "A-":  "#43a047",
    "BBB+":"#f57f17", "BBB": "#f9a825", "BBB-":"#fbc02d",
    "BB+": "#e65100", "BB":  "#f4511e", "BB-": "#ff7043",
    "B+":  "#880e4f", "B":   "#ad1457", "B-":  "#c2185b",
    "CCC": "#4e342e", "D":   "#212121",
}


def clean_rating(series: pd.Series) -> pd.Series:
    """Берём первый рейтинг если вида A-/BBB+, нормализуем."""
    s = series.str.split("/").str[0].str.strip()
    s = s.str.extract(r"^([A-D][A-Za-z+\-]*)")[0]
    return s


@st.cache_data(ttl=3600)
def load_fixed() -> pd.DataFrame:
    path = DATA_DIR / "bonds.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["maturity"]   = pd.to_datetime(df["maturity"],   errors="coerce")
    df["issue_date"] = pd.to_datetime(df["issue_date"], errors="coerce")
    df["rating_clean"] = clean_rating(df["rating"].fillna(""))
    df["rating_order"] = df["rating_clean"].map({r: i for i, r in enumerate(RATING_ORDER)})
    for col in ["volume", "price", "g_spread", "yield_pct", "duration", "coupon", "spread"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_data(ttl=3600)
def load_floaters() -> pd.DataFrame:
    path = DATA_DIR / "floaters.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["maturity"]   = pd.to_datetime(df["maturity"],   errors="coerce")
    df["issue_date"] = pd.to_datetime(df["issue_date"], errors="coerce")
    df["years_to_maturity"] = (
        (df["maturity"] - pd.Timestamp.now()).dt.days / 365
    ).clip(lower=0).round(2)
    df["rating_clean"] = clean_rating(df["rating"].fillna(""))
    df["rating_order"] = df["rating_clean"].map({r: i for i, r in enumerate(RATING_ORDER)})
    for col in ["volume", "price", "spread_ks", "yield_total"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return pd.DataFrame(df)


# ══════════════════════════════════════════════════
# ШАПКА — переключатель типа облигаций
# ══════════════════════════════════════════════════
st.title("📊 Карта доходностей облигаций")

bond_type = st.radio(
    label="Тип облигаций",
    options=["Фиксированный купон", "Плавающий купон (флоутеры)"],
    horizontal=True,
    label_visibility="collapsed",
)
st.markdown("---")

IS_FIXED = bond_type == "Фиксированный купон"

# ══════════════════════════════════════════════════
# ЗАГРУЗКА ДАННЫХ
# ══════════════════════════════════════════════════
if IS_FIXED:
    df = load_fixed()
    x_col    = "duration"
    y_col    = "yield_pct"
    x_label  = "Дюрация (лет)"
    y_label  = "Доходность к погашению (%)"
    size_col = "volume"
    extra_hover = {"g_spread": True, "coupon": True, "option_type": True}
else:
    df = load_floaters()
    x_col    = "years_to_maturity"
    y_col    = "spread_ks"
    x_label  = "До погашения (лет)"
    y_label  = "Спред к КС (б.п.)"
    size_col = "volume"
    extra_hover = {"yield_total": True, "base_rate": True}

if df.empty:
    st.error("Данные не найдены. Запустите fetch.py.")
    st.stop()

updated = df["updated"].iloc[0] if "updated" in df.columns else "—"

# ══════════════════════════════════════════════════
# САЙДБАР — фильтры
# ══════════════════════════════════════════════════
st.sidebar.title("Фильтры")
st.sidebar.caption(f"Данные от: **{updated}** | {len(df)} выпусков")

# Валюта
currencies = sorted(df["currency"].dropna().unique().tolist())
if "RUB" in currencies:
    currencies = ["RUB"] + [c for c in currencies if c != "RUB"]
sel_currency = st.sidebar.multiselect("Валюта", currencies, default=["RUB"])

# Рейтинг
avail_ratings = [r for r in RATING_ORDER if r in df["rating_clean"].dropna().unique()]
sel_ratings = st.sidebar.multiselect("Рейтинг", avail_ratings, default=avail_ratings)

# Сектор
all_sectors = sorted(df["sector"].dropna().unique().tolist())
sel_sectors = st.sidebar.multiselect("Сектор", all_sectors, default=all_sectors)

# Диапазон по X и Y
x_vals = df[x_col].dropna()
y_vals = df[y_col].dropna()

x_min, x_max = float(x_vals.min()), float(x_vals.max())
y_min, y_max = float(y_vals.min()), float(y_vals.max())

sel_x = st.sidebar.slider(x_label, x_min, x_max, (x_min, min(x_max, 10.0)), 0.25)
sel_y = st.sidebar.slider(y_label, y_min, y_max, (y_min, y_max), 0.5 if IS_FIXED else 5.0)

# ══════════════════════════════════════════════════
# ПРИМЕНЕНИЕ ФИЛЬТРОВ
# ══════════════════════════════════════════════════
mask = (
    df["currency"].isin(sel_currency) &
    df["rating_clean"].isin(sel_ratings) &
    df["sector"].isin(sel_sectors) &
    df[x_col].between(sel_x[0], sel_x[1]) &
    df[y_col].between(sel_y[0], sel_y[1])
)
fdf = df[mask].dropna(subset=[x_col, y_col, "rating_clean"]).copy()
fdf = fdf.sort_values("rating_order")

st.caption(f"Выбрано: **{len(fdf)}** из {len(df)} выпусков")

if fdf.empty:
    st.info("Нет данных по выбранным фильтрам.")
    st.stop()

# ══════════════════════════════════════════════════
# TAB-ы
# ══════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["🗺 Карта доходностей", "📈 Кривые по рейтингам", "📋 Таблица"])


# ───────────────────────────────────────────────
# TAB 1 — SCATTER
# ───────────────────────────────────────────────
with tab1:
    st.subheader(f"{'Дюрация × Доходность' if IS_FIXED else 'Срок × Спред к КС'}")

    hover = {"issuer": True, "rating": True, size_col: True,
             x_col: ":.2f", y_col: ":.2f",
             "rating_clean": False, "rating_order": False}
    hover.update(extra_hover)

    fig = px.scatter(
        fdf,
        x=x_col,
        y=y_col,
        color="rating_clean",
        color_discrete_map=RATING_COLORS,
        size=size_col,
        size_max=32,
        hover_name="ticker",
        hover_data=hover,
        labels={"rating_clean": "Рейтинг", x_col: x_label, y_col: y_label,
                size_col: "Объём (млн ₽)"},
        category_orders={"rating_clean": RATING_ORDER},
        template="plotly_white",
        height=580,
    )
    fig.update_traces(marker=dict(opacity=0.8, line=dict(width=0.5, color="white")))
    fig.update_layout(legend=dict(title="Рейтинг", orientation="v"))
    st.plotly_chart(fig, use_container_width=True)

    # Сводная таблица по рейтингам
    agg = {
        "Выпусков":    (x_col, "count"),
        f"{y_label} среднее": (y_col, "mean"),
        f"{y_label} медиана": (y_col, "median"),
        f"{x_label} медиана": (x_col, "median"),
    }
    if IS_FIXED:
        agg["G-спред б.п. медиана"] = ("g_spread", "median")

    grp = (
        fdf.groupby("rating_clean")
        .agg(**agg)
        .reindex([r for r in RATING_ORDER if r in fdf["rating_clean"].unique()])
        .dropna(how="all")
        .round(2)
    )
    grp.index.name = "Рейтинг"
    st.dataframe(grp, use_container_width=True)


# ───────────────────────────────────────────────
# TAB 2 — YIELD CURVES BY RATING
# ───────────────────────────────────────────────
with tab2:
    st.subheader("Медианные кривые доходности по рейтингам")

    bins   = [0, 0.5, 1, 1.5, 2, 3, 4, 5, 7, 10, 15, 20]
    mids   = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 4.5, 6.0, 8.5, 12.5, 17.5]

    plot2 = fdf.copy()
    plot2["x_mid"] = pd.cut(plot2[x_col], bins=bins, labels=mids, right=True).astype(float)

    fig2 = go.Figure()
    for rating in [r for r in RATING_ORDER if r in plot2["rating_clean"].unique()]:
        sub = (
            plot2[plot2["rating_clean"] == rating]
            .groupby("x_mid")[y_col]
            .median()
            .reset_index()
            .sort_values("x_mid")
            .dropna()
        )
        if len(sub) < 2:
            continue
        fig2.add_trace(go.Scatter(
            x=sub["x_mid"], y=sub[y_col],
            mode="lines+markers",
            name=rating,
            line=dict(color=RATING_COLORS.get(rating, "#999"), width=2.5),
            marker=dict(size=7),
        ))

    fig2.update_layout(
        xaxis_title=x_label,
        yaxis_title=f"Медиана: {y_label}",
        legend_title="Рейтинг",
        hovermode="x unified",
        template="plotly_white",
        height=520,
    )
    st.plotly_chart(fig2, use_container_width=True)

    if IS_FIXED:
        st.subheader("G-спред по рейтингам (медиана, б.п.)")
        gs = (
            fdf.groupby("rating_clean")["g_spread"]
            .median()
            .reindex([r for r in RATING_ORDER if r in fdf["rating_clean"].unique()])
            .dropna()
            .reset_index()
        )
        gs.columns = ["rating_clean", "g_spread"]
        fig3 = px.bar(
            gs, x="rating_clean", y="g_spread",
            color="rating_clean", color_discrete_map=RATING_COLORS,
            labels={"rating_clean": "Рейтинг", "g_spread": "G-спред (б.п.)"},
            template="plotly_white", height=360,
        )
        fig3.update_layout(showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)


# ───────────────────────────────────────────────
# TAB 3 — ТАБЛИЦА
# ───────────────────────────────────────────────
with tab3:
    st.subheader("Список выпусков")

    if IS_FIXED:
        show = ["isin", "ticker", "issuer", "sector", "rating",
                "yield_pct", "g_spread", "duration", "coupon",
                "maturity", "volume", "price", "option_type", "liquidity"]
        rename = {
            "isin": "ISIN", "ticker": "Тикер", "issuer": "Эмитент",
            "sector": "Сектор", "rating": "Рейтинг",
            "yield_pct": "Дох.%", "g_spread": "G-спред б.п.",
            "duration": "Дюрация", "coupon": "Купон%",
            "maturity": "Погашение", "volume": "Объём млн",
            "price": "Цена", "option_type": "Оферта", "liquidity": "Ликвидность",
        }
    else:
        show = ["isin", "ticker", "issuer", "sector", "rating",
                "yield_total", "spread_ks", "years_to_maturity",
                "maturity", "volume", "price", "base_rate", "liquidity"]
        rename = {
            "isin": "ISIN", "ticker": "Тикер", "issuer": "Эмитент",
            "sector": "Сектор", "rating": "Рейтинг",
            "yield_total": "Итог.дох.%", "spread_ks": "Спред к КС б.п.",
            "years_to_maturity": "Лет до погаш.",
            "maturity": "Погашение", "volume": "Объём млн",
            "price": "Цена", "base_rate": "База", "liquidity": "Ликвидность",
        }

    cols = [c for c in show if c in fdf.columns]
    tbl = fdf[cols].rename(columns=rename).reset_index(drop=True)
    st.dataframe(tbl, use_container_width=True, height=560)

    csv_bytes = fdf[cols].to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "⬇ Скачать CSV", csv_bytes,
        file_name=f"bonds_{bond_type[:3]}_{updated}.csv",
        mime="text/csv",
    )

st.markdown("---")
st.caption("Автор проекта: Ширманов К.А. | tg: [shirman7](tg://resolve?domain=shirman7)")
