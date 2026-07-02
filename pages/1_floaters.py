"""
1_floaters.py — Модуль анализа облигаций с плавающим купоном (флоутеров)
PRJ4 | Карта спредов (по мотивам проекта Matchbox)
Запуск: streamlit run fixed.py (эта страница подключается автоматически из pages/)
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import common
from common import FONT_FAMILY, ISSUER_PREFIX

# ════════════════════════════════════════════════════════════════
# КОНФИГУРАЦИЯ СТРАНИЦЫ
# ════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Плавающий купон | PRJ4",
    page_icon="📈",
    layout="wide",
)

COLOR_PLACEMENT = "#C9A227"   # Спред при размещении — охра
COLOR_CURRENT   = "#820411"   # Текущий спред — тёмно-бордовый (единый стиль с первой страницей)
COLOR_LINE      = "#999999"   # Соединительные линии между точками


# ════════════════════════════════════════════════════════════════
# ИНИЦИАЛИЗАЦИЯ SESSION STATE
# ════════════════════════════════════════════════════════════════
_DEFAULTS: dict = {
    "deleted_isins_float": set(),  # ISIN, удалённые из таблицы флоутеров
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ════════════════════════════════════════════════════════════════
# ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
# ════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def load_floaters() -> pd.DataFrame:
    """
    Загружает data/floaters.csv — файл создаётся fetch.py (GitHub Actions, каждое утро).
    Кэшируется на 1 час: повторные вызовы не перечитывают диск.
    """
    path = common.DATA_DIR / "floaters.csv"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path, encoding="utf-8-sig")

    # Числовые колонки
    for col in ["volume", "price", "spread_ks", "yield_total", "coupon_freq"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Даты
    for col in ["maturity", "issue_date", "next_coupon"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Нормализованный рейтинг и числовой порядок для сортировки
    df["rating_clean"] = common.clean_rating(df["rating"].fillna(""))
    df["rating_order"] = df["rating_clean"].map(
        {r: i for i, r in enumerate(common.RATING_ORDER)}
    )

    # Атомарные секторы (без склеенных через запятую дублей)
    df["sector_set"] = common.sector_set(df["sector"])

    # ── Пересчёт спреда по методике Matchbox ─────────────────────
    # spread_ks — спред к КС по условиям выпуска (фиксированный), в п.п.
    # Переводим в базисные пункты, как это делает Matchbox: *100
    df["placement_spread"] = df["spread_ks"] * 100

    # Срок до погашения в годах на дату последнего обновления данных
    _as_of = (
        pd.to_datetime(df["updated"].iloc[0])
        if "updated" in df.columns and not df.empty
        else pd.Timestamp.now()
    )
    df["years_left"] = (df["maturity"] - _as_of).dt.days / 365.25

    # Текущий спред = контрактный спред + ценовая надбавка/скидка, приведённая
    # к годовым бп, + поправка на то, что купон считается от номинала, а не
    # от текущей цены. Формула идентична Matchbox (app_test.py).
    _price_effect = (100 - df["price"]) * 100 / df["years_left"]
    _coupon_adj = df["placement_spread"] / df["price"] * 100 - df["placement_spread"]
    df["current_spread"] = df["placement_spread"] + _coupon_adj + _price_effect

    _invalid = (
        (df["years_left"] <= 0)
        | df["price"].isna() | (df["price"] == 0)
        | df["placement_spread"].isna()
    )
    df.loc[_invalid, "current_spread"] = np.nan
    df["delta_spread"] = df["current_spread"] - df["placement_spread"]

    return df


# ════════════════════════════════════════════════════════════════
# ГЛАВНАЯ СТРАНИЦА
# ════════════════════════════════════════════════════════════════

df_full = load_floaters()

st.title("📈 Плавающий купон — Карта спредов")
st.caption("Данные обновляются каждое утро. Методика пересчёта спреда — по проекту Matchbox.")
st.page_link("fixed.py", label="Перейти к карте фиксированного купона", icon="📊")

if df_full.empty:
    st.error("⚠️ Файл `data/floaters.csv` не найден. Запустите `python fetch.py`.")
    st.stop()

updated     = str(df_full["updated"].iloc[0]) if "updated" in df_full.columns else "—"
total_count = len(df_full)


# ════════════════════════════════════════════════════════════════
# САЙДБАР: ОБНОВЛЕНИЕ ДАННЫХ + ФИЛЬТРЫ
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    common.render_refresh_button(df_full, [load_floaters])

    st.header("Фильтры")
    st.caption(f"Данные от **{updated}** | Всего: **{total_count}** выпусков")
    st.divider()

    # ── Свои ISIN ───────────────────────────────────────────────
    isin_input = st.text_area(
        "Свои ISIN (по одному на строку)",
        height=100,
        help="Оставьте пустым, чтобы не фильтровать по ISIN",
    )
    input_isins: list[str] = [l.strip() for l in isin_input.splitlines() if l.strip()]

    # ── Рейтинг ─────────────────────────────────────────────────
    _avail_ratings = [r for r in common.RATING_ORDER if r in df_full["rating_clean"].dropna().unique()]
    sel_ratings: list[str] = st.multiselect("Рейтинг", _avail_ratings, default=[])

    # ── Сектор (атомарные значения) ────────────────────────────
    _avail_sectors = sorted({s for _set in df_full["sector_set"] for s in _set})
    sel_sectors: list[str] = st.multiselect("Сектор", _avail_sectors, default=[])

    # ── Тикер / Эмитент ─────────────────────────────────────────
    _all_tickers = sorted(df_full["ticker"].dropna().unique().tolist())
    _all_issuers = sorted(df_full["issuer"].dropna().unique().tolist())
    _ticker_issuer_options = _all_tickers + [ISSUER_PREFIX + i for i in _all_issuers]
    sel_tickers: list[str] = st.multiselect(
        "Тикер / Эмитент (опционально)",
        _ticker_issuer_options,
        default=[],
        help="Выбор эмитента подгружает все его выпуски одним кликом.",
    )

    # ── Дата размещения ─────────────────────────────────────────
    _dates = df_full["issue_date"].dropna()
    sel_date_range = None
    if not _dates.empty:
        _d_min, _d_max = _dates.min().date(), _dates.max().date()
        sel_date_range = st.date_input(
            "Дата размещения",
            value=(_d_min, _d_max),
            min_value=_d_min,
            max_value=_d_max,
        )

    # ── Срок до погашения ───────────────────────────────────────
    _years_vals = df_full["years_left"].dropna()
    _years_vals = _years_vals[_years_vals > 0]
    sel_years = None
    if not _years_vals.empty:
        _y_min, _y_max = round(float(_years_vals.min()), 1), round(float(_years_vals.max()), 1)
        sel_years = st.slider(
            "Срок до погашения (лет)",
            min_value=_y_min,
            max_value=_y_max,
            value=(_y_min, min(_y_max, 10.0)),
            step=0.1,
        )

    st.divider()
    st.subheader("Настройки графика")
    show_labels: bool = st.checkbox(
        "Показывать подписи (Тикер, Рейтинг, Δ спреда)",
        value=False,
        help="Подпись у каждой точки текущего спреда",
    )


# ════════════════════════════════════════════════════════════════
# ПРИМЕНЕНИЕ ФИЛЬТРОВ
# ════════════════════════════════════════════════════════════════
mask = pd.Series(True, index=df_full.index)

if st.session_state.deleted_isins_float:
    mask &= ~df_full["isin"].isin(st.session_state.deleted_isins_float)

if input_isins:
    mask &= df_full["isin"].isin(input_isins)

if sel_ratings:
    mask &= df_full["rating_clean"].isin(sel_ratings)

if sel_sectors:
    _sel_sector_set = set(sel_sectors)
    mask &= df_full["sector_set"].apply(lambda s: bool(s & _sel_sector_set))

if sel_tickers:
    _sel_issuers = {t[len(ISSUER_PREFIX):] for t in sel_tickers if t.startswith(ISSUER_PREFIX)}
    _sel_tickers_only = {t for t in sel_tickers if not t.startswith(ISSUER_PREFIX)}
    _ticker_mask = pd.Series(False, index=df_full.index)
    if _sel_tickers_only:
        _ticker_mask |= df_full["ticker"].isin(_sel_tickers_only)
    if _sel_issuers:
        _ticker_mask |= df_full["issuer"].isin(_sel_issuers)
    mask &= _ticker_mask

if sel_date_range and len(sel_date_range) == 2:
    _start, _end = sel_date_range
    mask &= df_full["issue_date"].dt.date.between(_start, _end)

if sel_years:
    mask &= df_full["years_left"].between(sel_years[0], sel_years[1], inclusive="both")

fdf = df_full[mask].dropna(subset=["issue_date", "placement_spread", "current_spread"]).copy()
fdf = fdf.sort_values("issue_date")


# ════════════════════════════════════════════════════════════════
# СТАТИСТИКА НАД ГРАФИКОМ
# ════════════════════════════════════════════════════════════════
_m1, _m2, _m3, _m4 = st.columns(4)
_m1.metric("Выпусков", f"{len(fdf)}")
_m2.metric(
    "Спред при размещении, бп",
    f"{fdf['placement_spread'].mean():.0f}" if not fdf.empty else "—",
)
_m3.metric(
    "Текущий спред, бп",
    f"{fdf['current_spread'].mean():.0f}" if not fdf.empty else "—",
)
_m4.metric(
    "Δ спред, бп",
    f"{fdf['delta_spread'].mean():+.0f}" if not fdf.empty else "—",
)

if fdf.empty:
    st.info("Нет данных по заданным фильтрам.")
    st.stop()


# ════════════════════════════════════════════════════════════════
# ПОСТРОЕНИЕ ГРАФИКА PLOTLY
# ════════════════════════════════════════════════════════════════
fig = go.Figure()


def _make_hover(row: pd.Series, spread_col: str, label: str) -> str:
    """Формирует HTML-подсказку при наведении на точку."""
    mat = str(row["maturity"])[:10] if pd.notna(row.get("maturity")) else "—"
    return (
        f"<b>{row['ticker']}</b><br>"
        f"Эмитент: {row.get('issuer') or '—'}<br>"
        f"Рейтинг: {row.get('rating') or '—'}<br>"
        f"{label}: {row[spread_col]:.0f} бп<br>"
        f"Цена: {row.get('price') or '—'} пп<br>"
        f"Погашение: {mat}"
    )


_hover_placement = fdf.apply(lambda r: _make_hover(r, "placement_spread", "Спред при размещении"), axis=1).tolist()
_hover_current   = fdf.apply(lambda r: _make_hover(r, "current_spread", "Текущий спред"), axis=1).tolist()

# ── Линии между точками: спред при размещении → текущий спред ───
# Один трейс на все пары (x, None-разрывы) — быстрее, чем трейс на бумагу.
_line_x: list = []
_line_y: list = []
for _, _row in fdf.iterrows():
    _line_x += [_row["issue_date"], _row["issue_date"], None]
    _line_y += [_row["placement_spread"], _row["current_spread"], None]

fig.add_trace(go.Scatter(
    x=_line_x,
    y=_line_y,
    mode="lines",
    name="Изменение спреда",
    line=dict(color=COLOR_LINE, width=1.5),
    hoverinfo="skip",
))

# ── Точки: спред при размещении ──────────────────────────────────
fig.add_trace(go.Scatter(
    x=fdf["issue_date"],
    y=fdf["placement_spread"],
    mode="markers",
    name="Спред при размещении",
    marker=dict(color=COLOR_PLACEMENT, size=9, opacity=0.9, line=dict(width=0.5, color="white")),
    hovertemplate="%{customdata}<extra></extra>",
    customdata=_hover_placement,
))

# ── Точки: текущий спред ─────────────────────────────────────────
_labels: list[str] | None = None
_mode = "markers"
if show_labels:
    _labels = fdf.apply(
        lambda r: f"{r['ticker']}, {r['rating'] if pd.notna(r['rating']) else r['rating_clean']}, {r['delta_spread']:+.0f}",
        axis=1,
    ).tolist()
    _mode = "markers+text"

fig.add_trace(go.Scatter(
    x=fdf["issue_date"],
    y=fdf["current_spread"],
    mode=_mode,
    name="Текущий спред",
    text=_labels,
    textposition="top center",
    textfont=dict(family=FONT_FAMILY, size=10, color="black"),
    marker=dict(color=COLOR_CURRENT, size=9, opacity=0.9, line=dict(width=0.5, color="white")),
    hovertemplate="%{customdata}<extra></extra>",
    customdata=_hover_current,
))

fig.update_layout(
    title=dict(
        text="Карта спредов флоутеров",
        font=dict(family=FONT_FAMILY, size=18),
        x=0.5,
        xanchor="center",
    ),
    xaxis=dict(
        title=dict(text="<b>Дата размещения</b>", font=dict(family=FONT_FAMILY, size=14)),
        tickfont=dict(family=FONT_FAMILY, size=12),
        gridcolor="#eeeeee",
        zeroline=False,
    ),
    yaxis=dict(
        title=dict(text="<b>Спред к КС, бп</b>", font=dict(family=FONT_FAMILY, size=14)),
        tickfont=dict(family=FONT_FAMILY, size=12),
        gridcolor="#eeeeee",
        zeroline=False,
    ),
    legend=dict(
        font=dict(family=FONT_FAMILY, size=12),
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#cccccc",
        borderwidth=1,
        orientation="v",
        yanchor="top",
        y=1.0,
        xanchor="left",
        x=0.0,
    ),
    hovermode="closest",
    template="plotly_white",
    height=620,
    margin=dict(l=60, r=30, t=90, b=60),
    font=dict(family=FONT_FAMILY),
)

st.plotly_chart(
    fig,
    use_container_width=True,
    config={
        "toImageButtonOptions": {
            "format": "png",
            "filename": f"floater_map_{updated}",
            "width": 1600,
            "height": 900,
            "scale": 1,
        },
        "displaylogo": False,
    },
)


# ════════════════════════════════════════════════════════════════
# ТАБЛИЦА ДАННЫХ
# ════════════════════════════════════════════════════════════════
st.subheader("Список выпусков")
st.caption(
    "Отметьте выпуски в колонке **«🗑»** и нажмите **«Применить удаление»** "
    "— точки исчезнут с графика. Кнопка **«Восстановить»** отменяет удаление."
)

_TABLE_COLS: dict[str, str] = {
    "ticker":            "Тикер",
    "rating":            "Рейтинг",
    "currency":          "Валюта",
    "price":             "Цена, пп",
    "placement_spread":  "Спред при размещении, бп",
    "current_spread":    "Текущий спред, бп",
    "delta_spread":      "Δ спред, бп",
    "yield_total":       "Доходность, %",
    "coupon_freq":       "Выплаты/год",
    "years_left":        "До погашения, лет",
    "maturity":          "Погашение",
    "issue_date":        "Размещение",
    "option_type":       "Опцион",
    "isin":              "ISIN",
    "issuer":            "Эмитент",
    "sector":            "Сектор",
    "liquidity":         "Ликвидность",
}

_show_cols = [c for c in _TABLE_COLS if c in fdf.columns]
_tbl = fdf[_show_cols].rename(columns=_TABLE_COLS).copy()

for _dc in ("Погашение", "Размещение"):
    if _dc in _tbl.columns:
        _tbl[_dc] = _tbl[_dc].apply(
            lambda x: x.strftime("%d.%m.%Y") if pd.notna(x) else ""
        )

_tbl.insert(0, "🗑", False)

_edited = st.data_editor(
    _tbl,
    use_container_width=True,
    height=480,
    hide_index=True,
    column_config={
        "🗑": st.column_config.CheckboxColumn(
            label="Удалить",
            help="Отметьте строку для удаления с графика",
            default=False,
            width="small",
        ),
        "Цена, пп":                 st.column_config.NumberColumn(format="%.2f"),
        "Спред при размещении, бп": st.column_config.NumberColumn(format="%.0f"),
        "Текущий спред, бп":        st.column_config.NumberColumn(format="%.0f"),
        "Δ спред, бп":              st.column_config.NumberColumn(format="%+.0f"),
        "Доходность, %":            st.column_config.NumberColumn(format="%.2f"),
        "Выплаты/год":              st.column_config.NumberColumn(format="%d"),
        "До погашения, лет":        st.column_config.NumberColumn(format="%.2f"),
        "Рейтинг":                  st.column_config.TextColumn(width="medium"),
    },
    disabled=[c for c in _tbl.columns if c != "🗑"],
    key="floaters_editor",
)

# ── Кнопки управления таблицей ──────────────────────────────────
_bc1, _bc2, _bc3 = st.columns([2, 2, 3])

with _bc1:
    if st.button("🗑 Применить удаление", use_container_width=True, key="del_float"):
        _marked = _edited.loc[_edited["🗑"] == True]
        _new_del: set[str] = set()
        if "ISIN" in _marked.columns:
            _new_del = set(_marked["ISIN"].tolist())

        if _new_del:
            st.session_state.deleted_isins_float |= _new_del
            st.success(f"Удалено {len(_new_del)} выпусков с графика.")
            st.rerun()
        else:
            st.info("Не выбрано ни одного выпуска для удаления.")

with _bc2:
    if st.button("↩ Восстановить все", use_container_width=True, key="restore_float"):
        st.session_state.deleted_isins_float = set()
        st.rerun()

with _bc3:
    _csv = fdf[_show_cols].to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "⬇ Скачать CSV",
        data=_csv,
        file_name=f"floaters_{updated}.csv",
        mime="text/csv",
        use_container_width=True,
        key="download_float",
    )

if st.session_state.deleted_isins_float:
    st.caption(
        f"Скрыто с графика: **{len(st.session_state.deleted_isins_float)}** выпусков. "
        "Нажмите «Восстановить все» для отмены."
    )

# ════════════════════════════════════════════════════════════════
# ПОДВАЛ
# ════════════════════════════════════════════════════════════════
st.markdown("---")
st.caption(
    "Спред при размещении — спред к КС по условиям выпуска. Текущий спред "
    "пересчитан из цены и срока до погашения (методика проекта Matchbox). "
    "Расчёт не учитывает амортизацию, если она есть у выпуска."
)
st.caption(
    "PRJ4 | Автор: Ширманов К.А. | tg: [shirman7](tg://resolve?domain=shirman7)"
)
