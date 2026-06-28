import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Bond Market | PRJ4", page_icon="📊", layout="wide")

DATA_DIR = Path(__file__).parent / "data"

RATING_ORDER = [
    "AAA", "AA+", "AA", "AA-", "A+", "A", "A-",
    "BBB+", "BBB", "BBB-", "BB+", "BB", "BB-",
    "B+", "B", "B-", "CCC+", "CCC", "CCC-", "D",
]

RATING_COLORS = {
    "AAA": "#1a7abf", "AA+": "#1a7abf", "AA": "#1a7abf", "AA-": "#2196f3",
    "A+":  "#4caf50", "A":   "#4caf50", "A-":  "#8bc34a",
    "BBB+":"#ff9800", "BBB": "#ff9800", "BBB-":"#ffc107",
    "BB+": "#ff5722", "BB":  "#ff5722", "BB-": "#e91e63",
    "B+":  "#9c27b0", "B":   "#9c27b0", "B-":  "#673ab7",
    "CCC+":"#795548", "CCC": "#795548", "CCC-":"#9e9e9e", "D": "#424242",
}


@st.cache_data(ttl=3600)
def load_bonds() -> pd.DataFrame:
    path = DATA_DIR / "bonds.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["maturity"] = pd.to_datetime(df["maturity"], errors="coerce")
    df["issue_date"] = pd.to_datetime(df["issue_date"], errors="coerce")
    # Нормализуем рейтинг: берём первый если вида "A-/BBB+"
    df["rating_clean"] = df["rating"].str.split("/").str[0].str.strip()
    df["rating_clean"] = df["rating_clean"].str.extract(r"^([A-D][A-Z+\-]*)")
    df["rating_order"] = df["rating_clean"].map(
        {r: i for i, r in enumerate(RATING_ORDER)}
    )
    df["color"] = df["rating_clean"].map(RATING_COLORS).fillna("#9e9e9e")
    return df


@st.cache_data(ttl=3600)
def load_floaters() -> pd.DataFrame:
    path = DATA_DIR / "floaters.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["maturity"] = pd.to_datetime(df["maturity"], errors="coerce")
    df["years_to_maturity"] = (
        (df["maturity"] - pd.Timestamp.now()).dt.days / 365
    ).clip(lower=0).round(2)
    df["rating_clean"] = df["rating"].str.split("/").str[0].str.strip()
    df["rating_clean"] = df["rating_clean"].str.extract(r"^([A-D][A-Z+\-]*)")
    df["color"] = df["rating_clean"].map(RATING_COLORS).fillna("#9e9e9e")
    return df


df = load_bonds()
df_fl = load_floaters()

if df.empty:
    st.error("Файл data/bonds.csv не найден. Запустите fetch.py для загрузки данных.")
    st.stop()

updated = df["updated"].iloc[0] if "updated" in df.columns else "—"

# ─── SIDEBAR ───
st.sidebar.title("Фильтры")
st.sidebar.caption(f"Данные от: **{updated}**  |  {len(df)} выпусков")

currencies = ["RUB"] + [c for c in df["currency"].dropna().unique() if c != "RUB"]
sel_currency = st.sidebar.multiselect("Валюта", currencies, default=["RUB"])

all_ratings = [r for r in RATING_ORDER if r in df["rating_clean"].dropna().unique()]
sel_ratings = st.sidebar.multiselect("Рейтинг", all_ratings, default=all_ratings)

all_sectors = sorted(df["sector"].dropna().unique())
sel_sectors = st.sidebar.multiselect("Сектор", all_sectors, default=all_sectors)

dur_min, dur_max = 0.0, float(df["duration"].dropna().max() or 10.0)
sel_dur = st.sidebar.slider("Дюрация (лет)", dur_min, dur_max, (dur_min, min(dur_max, 10.0)), 0.25)

yld_min = float(df["yield_pct"].dropna().min() or 0)
yld_max = float(df["yield_pct"].dropna().max() or 50)
sel_yld = st.sidebar.slider("Доходность (%)", yld_min, yld_max, (yld_min, yld_max), 0.5)

# Применяем фильтры
mask = (
    df["currency"].isin(sel_currency) &
    df["rating_clean"].isin(sel_ratings) &
    df["sector"].isin(sel_sectors) &
    df["duration"].between(sel_dur[0], sel_dur[1]) &
    df["yield_pct"].between(sel_yld[0], sel_yld[1])
)
fdf = df[mask].copy()

st.title("📊 Рынок облигаций")
st.caption(f"Источник: bondresearch.ru | Данные: **{updated}** | Выбрано: **{len(fdf)}** из {len(df)} выпусков")

tab1, tab2, tab3, tab4 = st.tabs(["🗺 Карта рынка", "📈 Yield Map", "🌊 Флоутеры", "📋 Таблица"])


# ═══════════════════════════════════════════════
# TAB 1 — КАРТА РЫНКА (Дюрация × Доходность)
# ═══════════════════════════════════════════════
with tab1:
    st.subheader("Карта рынка: Дюрация × Доходность")
    if fdf.empty:
        st.info("Нет данных по выбранным фильтрам.")
    else:
        plot_df = fdf.dropna(subset=["duration", "yield_pct", "rating_clean"]).copy()
        plot_df = plot_df.sort_values("rating_order")

        fig = px.scatter(
            plot_df,
            x="duration",
            y="yield_pct",
            color="rating_clean",
            color_discrete_map=RATING_COLORS,
            size="volume",
            size_max=30,
            hover_name="ticker",
            hover_data={
                "issuer": True,
                "rating": True,
                "coupon": True,
                "g_spread": True,
                "maturity": True,
                "volume": True,
                "duration": ":.2f",
                "yield_pct": ":.2f",
                "rating_clean": False,
                "color": False,
            },
            labels={
                "duration": "Дюрация (лет)",
                "yield_pct": "Доходность (%)",
                "rating_clean": "Рейтинг",
                "volume": "Объём (млн ₽)",
            },
            category_orders={"rating_clean": RATING_ORDER},
            template="plotly_white",
            height=560,
        )
        fig.update_layout(legend=dict(orientation="v", title="Рейтинг"))
        st.plotly_chart(fig, use_container_width=True)

        # Метрики по рейтинговым группам
        st.markdown("**Средняя доходность по рейтинговым группам**")
        grp = (
            plot_df.groupby("rating_clean")
            .agg(
                выпусков=("isin", "count"),
                дох_средн=("yield_pct", "mean"),
                дох_медиана=("yield_pct", "median"),
                дюрация=("duration", "mean"),
                g_спред=("g_spread", "mean"),
            )
            .reindex([r for r in RATING_ORDER if r in plot_df["rating_clean"].unique()])
            .dropna(how="all")
            .round(2)
        )
        grp.index.name = "Рейтинг"
        grp.columns = ["Выпусков", "Дох. средн. %", "Дох. медиана %", "Дюрация лет", "G-спред б.п."]
        st.dataframe(grp, use_container_width=True)


# ═══════════════════════════════════════════════
# TAB 2 — YIELD MAP (кривые по рейтингам)
# ═══════════════════════════════════════════════
with tab2:
    st.subheader("Yield Map: кривые доходности по рейтингам")

    if fdf.empty:
        st.info("Нет данных по выбранным фильтрам.")
    else:
        plot_df2 = fdf.dropna(subset=["duration", "yield_pct", "rating_clean"]).copy()

        # Группируем по рейтингу и бакетам дюрации
        bins = [0, 0.5, 1, 1.5, 2, 3, 4, 5, 7, 10, 15, 20]
        labels = ["0-0.5", "0.5-1", "1-1.5", "1.5-2", "2-3", "3-4", "4-5", "5-7", "7-10", "10-15", "15+"]
        plot_df2["dur_bucket"] = pd.cut(plot_df2["duration"], bins=bins, labels=labels, right=True)
        plot_df2["dur_mid"] = pd.cut(plot_df2["duration"], bins=bins,
                                     labels=[0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 4.5, 6, 8.5, 12.5, 17.5],
                                     right=True).astype(float)

        fig2 = go.Figure()
        for rating in [r for r in RATING_ORDER if r in plot_df2["rating_clean"].unique()]:
            sub = (
                plot_df2[plot_df2["rating_clean"] == rating]
                .groupby("dur_mid")["yield_pct"]
                .median()
                .reset_index()
                .sort_values("dur_mid")
            )
            if len(sub) < 2:
                continue
            fig2.add_trace(go.Scatter(
                x=sub["dur_mid"],
                y=sub["yield_pct"],
                mode="lines+markers",
                name=rating,
                line=dict(color=RATING_COLORS.get(rating, "#999"), width=2),
                marker=dict(size=6),
            ))

        fig2.update_layout(
            xaxis_title="Дюрация (лет)",
            yaxis_title="Медианная доходность (%)",
            legend_title="Рейтинг",
            hovermode="x unified",
            template="plotly_white",
            height=500,
        )
        st.plotly_chart(fig2, use_container_width=True)

        # G-спред по рейтингам
        st.subheader("G-спред по рейтингам (медиана, б.п.)")
        gs = (
            fdf.dropna(subset=["g_spread", "rating_clean"])
            .groupby("rating_clean")["g_spread"]
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
            template="plotly_white", height=380,
        )
        fig3.update_layout(showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)


# ═══════════════════════════════════════════════
# TAB 3 — ФЛОУТЕРЫ
# ═══════════════════════════════════════════════
with tab3:
    st.subheader("Флоутеры (КС + спред)")
    if df_fl.empty:
        st.info("Файл data/floaters.csv не найден или пуст.")
    else:
        fl_ratings = [r for r in RATING_ORDER if r in df_fl["rating_clean"].dropna().unique()]
        sel_fl_ratings = st.multiselect("Рейтинг (флоутеры)", fl_ratings, default=fl_ratings, key="fl_r")
        ffl = df_fl[df_fl["rating_clean"].isin(sel_fl_ratings)].copy()

        fig_fl = px.scatter(
            ffl.dropna(subset=["years_to_maturity", "spread_ks"]),
            x="years_to_maturity", y="spread_ks",
            color="rating_clean",
            color_discrete_map=RATING_COLORS,
            size="volume", size_max=28,
            hover_name="ticker",
            hover_data={"issuer": True, "rating": True, "yield_total": True, "maturity": True},
            labels={
                "years_to_maturity": "До погашения (лет)",
                "spread_ks": "Спред к КС (б.п.)",
                "rating_clean": "Рейтинг",
            },
            category_orders={"rating_clean": RATING_ORDER},
            template="plotly_white", height=480,
        )
        st.plotly_chart(fig_fl, use_container_width=True)

        grp_fl = (
            ffl.dropna(subset=["spread_ks", "rating_clean"])
            .groupby("rating_clean")
            .agg(выпусков=("isin", "count"), спред_медиана=("spread_ks", "median"))
            .reindex([r for r in RATING_ORDER if r in ffl["rating_clean"].unique()])
            .dropna(how="all").round(1)
        )
        grp_fl.index.name = "Рейтинг"
        grp_fl.columns = ["Выпусков", "Спред к КС медиана (б.п.)"]
        st.dataframe(grp_fl, use_container_width=True)


# ═══════════════════════════════════════════════
# TAB 4 — ТАБЛИЦА
# ═══════════════════════════════════════════════
with tab4:
    st.subheader("Таблица выпусков")
    show_cols = ["isin", "ticker", "issuer", "sector", "rating", "currency",
                 "yield_pct", "g_spread", "duration", "coupon", "maturity",
                 "volume", "price", "option_type", "liquidity"]
    show_cols = [c for c in show_cols if c in fdf.columns]
    tbl = fdf[show_cols].copy()
    tbl.columns = [
        c.replace("yield_pct", "Дох.%")
         .replace("g_spread", "G-спред")
         .replace("duration", "Дюрация")
         .replace("coupon", "Купон%")
         .replace("maturity", "Погашение")
         .replace("volume", "Объём млн")
         .replace("price", "Цена")
         .replace("option_type", "Оферта")
         .replace("liquidity", "Ликвидность")
         .replace("issuer", "Эмитент")
         .replace("sector", "Сектор")
         .replace("rating", "Рейтинг")
         .replace("currency", "Валюта")
         .replace("ticker", "Тикер")
         .replace("isin", "ISIN")
        for c in tbl.columns
    ]
    st.dataframe(tbl.reset_index(drop=True), use_container_width=True, height=600)

    csv_bytes = fdf[show_cols].to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇ Скачать CSV", csv_bytes,
                       file_name=f"bonds_{updated}.csv", mime="text/csv")

st.markdown("---")
st.caption("Автор проекта: Ширманов К.А. | tg: [shirman7](tg://resolve?domain=shirman7)")
