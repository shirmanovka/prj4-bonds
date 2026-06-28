"""
fixed.py — Модуль анализа облигаций с фиксированным купоном
PRJ4 | Карта доходностей
Запуск: streamlit run fixed.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from pathlib import Path

# ════════════════════════════════════════════════════════════════
# КОНФИГУРАЦИЯ СТРАНИЦЫ
# ════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Фиксированный купон | PRJ4",
    page_icon="📊",
    layout="wide",
)

DATA_DIR = Path(__file__).parent / "data"

# Порядок рейтингов от высшего к низшему (AAA → D)
RATING_ORDER = [
    "AAA", "AA+", "AA", "AA-",
    "A+",  "A",   "A-",
    "BBB+", "BBB", "BBB-",
    "BB+",  "BB",  "BB-",
    "B+",   "B",   "B-",
    "CCC", "CC", "C", "D",
]

# Три типа эмитентов для фильтра
ISSUER_TYPES = ["ОФЗ", "Корпоративные", "Субфедеральные"]

# Цветовая схема (строго по ТЗ)
COLOR_BONDS  = "#820411"   # Основные бумаги — тёмно-бордовый
COLOR_CUSTOM = "#FFA500"   # Пользовательские точки — оранжевый
COLOR_GCURVE = "#1565C0"   # G-кривая — синий
FONT_FAMILY  = "Calibri"   # Шрифт всех текстов на графике


# ════════════════════════════════════════════════════════════════
# ИНИЦИАЛИЗАЦИЯ SESSION STATE
# Хранит состояние между рендерами страницы
# ════════════════════════════════════════════════════════════════
_DEFAULTS: dict = {
    "custom_points": [],    # Пользовательские точки [{duration, yield_pct, name}]
    "deleted_isins":  set(), # ISIN, удалённые из таблицы → исчезают с графика
    "show_labels":    True,  # Видимость подписей на графике
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ════════════════════════════════════════════════════════════════
# ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
# ════════════════════════════════════════════════════════════════

def _clean_rating(series: pd.Series) -> pd.Series:
    """
    Нормализует рейтинг: берёт первый компонент из составного вида
    'AA (AA-)' или 'A-/BBB+', убирает пробелы.
    Пример: 'AA (AA-)' → 'AA', 'A-/BBB+' → 'A-'
    """
    s = series.str.split(r"[/\s\(]").str[0].str.strip()
    s = s.str.extract(r"^([A-D][A-Za-z+\-]*)")[0]
    return s


@st.cache_data(ttl=3600)
def load_fixed() -> pd.DataFrame:
    """
    Загружает data/bonds.csv — файл создаётся fetch.py (GitHub Actions, каждое утро).
    Кэшируется на 1 час: повторные вызовы не перечитывают диск.
    """
    path = DATA_DIR / "bonds.csv"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path, encoding="utf-8-sig")

    # Числовые колонки
    for col in ["volume", "price", "g_spread", "yield_pct", "current_yield",
                "duration", "spread", "coupon", "face_value"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Даты
    for col in ["maturity", "issue_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Нормализованный рейтинг и числовой порядок для сортировки
    df["rating_clean"] = _clean_rating(df["rating"].fillna(""))
    df["rating_order"] = df["rating_clean"].map(
        {r: i for i, r in enumerate(RATING_ORDER)}
    )

    # Классификация типа эмитента по тикеру и сектору
    def _classify(row: pd.Series) -> str:
        ticker = str(row.get("ticker") or "")
        sector = str(row.get("sector") or "")
        # "ОФЗ" в тикере ("ОФЗ 26207") или в секторе ("ОФЗ, Cуверенный долг (РФ)")
        if "ОФЗ" in ticker or "ОФЗ" in sector:
            return "ОФЗ"
        # "Субфедеральный долг" — муниципальные
        if "убфедеральный" in sector:
            return "Субфедеральные"
        return "Корпоративные"

    df["issuer_type"] = df.apply(_classify, axis=1)

    # Наличие опциона: текст для таблицы
    df["has_option"] = df["option_type"].apply(
        lambda x: "Да" if str(x) not in ("Отсутствует", "nan", "") else "Нет"
    )

    return df


# ════════════════════════════════════════════════════════════════
# G-КРИВАЯ: ПОЛИНОМИАЛЬНАЯ АППРОКСИМАЦИЯ ПО ТОЧКАМ ОФЗ
# ════════════════════════════════════════════════════════════════

def _build_gcurve(
    df_ofz: pd.DataFrame, x_min: float, x_max: float
) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    Аппроксимирует G-кривую по точкам ОФЗ полиномом степени до 3.
    Если ОФЗ < 3 точек — возвращает синтетическую кривую нормальной формы.

    Returns:
        (x_arr, y_arr, is_synthetic) — массивы для линии и флаг синтетики.
    """
    x_plot = np.linspace(max(0.05, x_min), max(x_max, 0.1), 300)

    data = df_ofz.dropna(subset=["duration", "yield_pct"])
    data = data[
        (data["duration"] > 0) &
        (data["yield_pct"] > 0) &
        (data["yield_pct"] < 100)  # фильтр технических выбросов
    ]

    if len(data) >= 3:
        deg = min(3, len(data) - 1)
        coeffs = np.polyfit(data["duration"].values, data["yield_pct"].values, deg)
        y_plot = np.polyval(coeffs, x_plot)
        return x_plot, y_plot, False

    # Синтетическая кривая: типичная нормальная форма для ОФЗ ~14.5%
    y_plot = 14.5 - 0.5 * x_plot + 0.02 * x_plot ** 2
    return x_plot, y_plot, True


# ════════════════════════════════════════════════════════════════
# ПОЗИЦИИ ПОДПИСЕЙ: ПРОСТОЙ АЛГОРИТМ ПРОТИВ НАЛОЖЕНИЙ
# ════════════════════════════════════════════════════════════════

def _assign_text_positions(x_arr: np.ndarray, y_arr: np.ndarray) -> list[str]:
    """
    Назначает позицию подписи каждой точке, минимизируя наложения.
    Алгоритм: точки в плотных областях (нормализованное расстояние < 0.04)
    получают позиции по циклу из 6 вариантов вместо одного 'top center'.
    """
    n = len(x_arr)
    if n == 0:
        return []

    OPTS = [
        "top center", "bottom center",
        "top right",  "bottom left",
        "top left",   "bottom right",
    ]

    # Нормализуем к [0, 1] для равномерного сравнения по осям
    xr = float(np.max(x_arr) - np.min(x_arr)) or 1.0
    yr = float(np.max(y_arr) - np.min(y_arr)) or 1.0
    xn = (x_arr - np.min(x_arr)) / xr
    yn = (y_arr - np.min(y_arr)) / yr

    positions: list[str] = []
    opt_counter: dict[int, int] = {}  # ключ ячейки сетки → счётчик

    for i in range(n):
        # Ячейка сетки 20×20 для данной точки
        cell = int(xn[i] * 20) * 100 + int(yn[i] * 20)
        cnt = opt_counter.get(cell, 0)
        positions.append(OPTS[cnt % len(OPTS)])
        opt_counter[cell] = cnt + 1

    return positions


# ════════════════════════════════════════════════════════════════
# ЭКСПОРТ В PNG (требует kaleido)
# ════════════════════════════════════════════════════════════════

def _fig_to_png(fig: go.Figure) -> bytes | None:
    """Конвертирует Plotly-фигуру в PNG. Возвращает None если kaleido не установлен."""
    try:
        return fig.to_image(format="png", width=1600, height=800, scale=2)
    except Exception:
        return None


# ════════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════
# ГЛАВНАЯ СТРАНИЦА
# ════════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════

st.title("📊 Фиксированный купон — Карта доходностей")
st.caption("Источник: bondresearch.ru | Обновляется каждое утро (GitHub Actions / fetch.py)")

# Загрузка данных
df_full = load_fixed()
if df_full.empty:
    st.error("⚠️ Файл `data/bonds.csv` не найден. Запустите `python fetch.py`.")
    st.stop()

updated     = str(df_full["updated"].iloc[0]) if "updated" in df_full.columns else "—"
total_count = len(df_full)

# ОФЗ для G-кривой — из полного датасета (независимо от фильтров пользователя)
df_ofz = df_full[df_full["issuer_type"] == "ОФЗ"].copy()


# ════════════════════════════════════════════════════════════════
# САЙДБАР: ФИЛЬТРЫ + НАСТРОЙКИ ГРАФИКА + КОНСТРУКТОР СДЕЛКИ
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("Фильтры")
    st.caption(f"Данные от **{updated}** | Всего: **{total_count}** выпусков")
    st.divider()

    # ── Валюта ──────────────────────────────────────────────────
    _currencies = sorted(df_full["currency"].dropna().unique().tolist())
    if "RUB" in _currencies:
        _currencies = ["RUB"] + [c for c in _currencies if c != "RUB"]
    sel_currency: list[str] = st.multiselect(
        "Валюта",
        _currencies,
        default=["RUB"],
        help="Валюта номинала",
    )

    # ── Рейтинг (отсортирован AAA → D) ──────────────────────────
    _avail_ratings = [r for r in RATING_ORDER if r in df_full["rating_clean"].dropna().unique()]
    sel_ratings: list[str] = st.multiselect(
        "Рейтинг",
        _avail_ratings,
        default=_avail_ratings,
        help="Кредитный рейтинг (AAA → D)",
    )

    # ── Тип эмитента ────────────────────────────────────────────
    sel_issuer_types: list[str] = st.multiselect(
        "Тип эмитента",
        ISSUER_TYPES,
        default=ISSUER_TYPES,
    )

    # ── Дюрация: двойной слайдер, шаг 0.1 ──────────────────────
    _dur_vals = df_full["duration"].dropna()
    _dur_min  = round(float(_dur_vals.min()), 1)
    _dur_max  = round(float(_dur_vals.max()), 1)
    sel_dur: tuple[float, float] = st.slider(
        "Дюрация (лет)",
        min_value=_dur_min,
        max_value=_dur_max,
        value=(_dur_min, min(_dur_max, 10.0)),
        step=0.1,
        help="Диапазон дюрации для отображения",
    )

    # ── ISIN / Тикер (опционально) ──────────────────────────────
    _all_tickers = sorted(df_full["ticker"].dropna().unique().tolist())
    sel_tickers: list[str] = st.multiselect(
        "ISIN / Тикер (опционально)",
        _all_tickers,
        default=[],
        help="Оставьте пустым для отображения всех",
    )

    st.divider()
    st.subheader("Настройки графика")

    # ── Подписи точек: чекбокс ──────────────────────────────────
    show_labels: bool = st.checkbox(
        "Показывать подписи (Тикер, Рейтинг)",
        value=st.session_state.show_labels,
        key="show_labels",
        help="Подпись у каждой точки: 'Тикер, Рейтинг'",
    )

    # ── G-кривая ────────────────────────────────────────────────
    show_gcurve: bool = st.checkbox(
        "G-кривая (кривая ОФЗ)",
        value=True,
        help="Полиномиальная аппроксимация по точкам ОФЗ",
    )

    # ── Размер точек ────────────────────────────────────────────
    marker_size: int = st.slider(
        "Размер точек",
        min_value=3,
        max_value=18,
        value=7,
        step=1,
    )

    st.divider()

    # ────────────────────────────────────────────────────────────
    # КОНСТРУКТОР СДЕЛКИ
    # Добавляет пользовательские точки на график
    # ────────────────────────────────────────────────────────────
    st.subheader("🔧 Конструктор сделки")
    st.caption("Добавьте целевую точку на график")

    c_dur  = st.number_input("Дюрация", min_value=0.0, max_value=30.0,
                              value=2.0, step=0.1, format="%.1f")
    c_yld  = st.number_input("Доходность, %", min_value=0.0, max_value=60.0,
                              value=15.0, step=0.05, format="%.2f")
    c_name = st.text_input("Название точки", value="Моя цель", max_chars=30)

    if st.button("➕ Добавить точку", use_container_width=True, type="primary"):
        if c_name.strip():
            st.session_state.custom_points.append(
                {"duration": c_dur, "yield_pct": c_yld, "name": c_name.strip()}
            )
            st.success(f"Точка «{c_name.strip()}» добавлена!")
        else:
            st.warning("Введите название точки.")

    # Список добавленных точек с кнопкой удаления каждой
    if st.session_state.custom_points:
        st.caption(f"Добавлено точек: **{len(st.session_state.custom_points)}**")
        _to_remove: list[int] = []
        for _i, _pt in enumerate(st.session_state.custom_points):
            _c1, _c2 = st.columns([5, 1])
            _c1.caption(f"**{_pt['name']}** · {_pt['duration']}л, {_pt['yield_pct']}%")
            if _c2.button("✕", key=f"rm_pt_{_i}", help="Удалить точку"):
                _to_remove.append(_i)
        if _to_remove:
            for _idx in reversed(_to_remove):
                st.session_state.custom_points.pop(_idx)
            st.rerun()

        if st.button("🗑 Очистить все точки", use_container_width=True):
            st.session_state.custom_points = []
            st.rerun()


# ════════════════════════════════════════════════════════════════
# ПРИМЕНЕНИЕ ФИЛЬТРОВ
# ════════════════════════════════════════════════════════════════

mask = pd.Series(True, index=df_full.index)

# Исключаем ISIN, удалённые через таблицу
if st.session_state.deleted_isins:
    mask &= ~df_full["isin"].isin(st.session_state.deleted_isins)

if sel_currency:
    mask &= df_full["currency"].isin(sel_currency)

if sel_ratings:
    mask &= df_full["rating_clean"].isin(sel_ratings)

if sel_issuer_types:
    mask &= df_full["issuer_type"].isin(sel_issuer_types)

mask &= df_full["duration"].between(sel_dur[0], sel_dur[1], inclusive="both")

if sel_tickers:
    mask &= df_full["ticker"].isin(sel_tickers)

# Убираем строки без координат графика
fdf = df_full[mask].dropna(subset=["duration", "yield_pct"]).copy()
fdf = fdf.sort_values("rating_order")


# ════════════════════════════════════════════════════════════════
# СТАТИСТИКА НАД ГРАФИКОМ
# ════════════════════════════════════════════════════════════════
_m1, _m2, _m3, _m4 = st.columns(4)
_m1.metric("Выпусков", f"{len(fdf)}")
_m2.metric(
    "Ср. доходность",
    f"{fdf['yield_pct'].mean():.2f}%" if not fdf.empty else "—",
)
_m3.metric(
    "Ср. дюрация",
    f"{fdf['duration'].mean():.2f} л" if not fdf.empty else "—",
)
_m4.metric(
    "Ср. G-спред",
    f"{fdf['g_spread'].mean():.0f} бп" if (not fdf.empty and fdf['g_spread'].notna().any()) else "—",
)

if fdf.empty:
    st.info("Нет данных по заданным фильтрам.")
    st.stop()


# ════════════════════════════════════════════════════════════════
# ПОСТРОЕНИЕ ГРАФИКА PLOTLY
# ════════════════════════════════════════════════════════════════

fig = go.Figure()

# ── Hover-подсказка ─────────────────────────────────────────────
def _make_hover(row: pd.Series) -> str:
    """Формирует HTML-подсказку при наведении на точку."""
    g = f"{int(row['g_spread'])} бп" if pd.notna(row.get("g_spread")) else "—"
    v = f"{int(row['volume'])} млн"  if pd.notna(row.get("volume"))   else "—"
    mat = str(row["maturity"])[:10]  if pd.notna(row.get("maturity")) else "—"
    return (
        f"<b>{row['ticker']}</b><br>"
        f"Эмитент: {row.get('issuer') or '—'}<br>"
        f"Рейтинг: {row.get('rating') or '—'}<br>"
        f"Доходность: {row['yield_pct']:.2f}%<br>"
        f"Дюрация: {row['duration']:.2f} л<br>"
        f"G-спред: {g}<br>"
        f"Купон: {row.get('coupon') or '—'}%<br>"
        f"Объём: {v}<br>"
        f"Цена: {row.get('price') or '—'} пп<br>"
        f"Опцион: {row.get('option_type') or '—'}<br>"
        f"Погашение: {mat}"
    )

_hover_texts = fdf.apply(_make_hover, axis=1).tolist()

# ── Подписи: алгоритм расстановки позиций ───────────────────────
_tp = _assign_text_positions(
    fdf["duration"].values,
    fdf["yield_pct"].values,
)
_labels: list[str] | None
if show_labels:
    _labels = fdf.apply(
        lambda r: f"{r['ticker']}<br><span style='font-size:7px'>{r['rating_clean']}</span>",
        axis=1,
    ).tolist()
    _mode = "markers+text"
else:
    _labels = None
    _mode   = "markers"

# ── Основные точки (фиксированный купон) ────────────────────────
fig.add_trace(go.Scatter(
    x=fdf["duration"],
    y=fdf["yield_pct"],
    mode=_mode,
    name="Фиксированный купон",
    text=_labels,
    textposition=_tp if show_labels else None,
    textfont=dict(family=FONT_FAMILY, size=8, color="#5a0010"),
    marker=dict(
        color=COLOR_BONDS,
        size=marker_size,
        opacity=0.85,
        line=dict(width=0.5, color="white"),
    ),
    hovertemplate="%{customdata}<extra></extra>",
    customdata=_hover_texts,
))

# ── G-кривая ────────────────────────────────────────────────────
if show_gcurve:
    _x_lo = float(fdf["duration"].min()) if not fdf.empty else 0.1
    _x_hi = float(fdf["duration"].max()) if not fdf.empty else 15.0
    _gx, _gy, _is_synth = _build_gcurve(df_ofz, _x_lo, _x_hi)

    _gcurve_name = (
        "G-кривая (синтетич.)" if _is_synth
        else f"G-кривая ОФЗ ({len(df_ofz)} вып.)"
    )
    fig.add_trace(go.Scatter(
        x=_gx,
        y=_gy,
        mode="lines",
        name=_gcurve_name,
        line=dict(color=COLOR_GCURVE, width=2.0, dash="dash"),
        hovertemplate="G-кривая: %{y:.2f}%<br>Дюрация: %{x:.2f} л<extra></extra>",
    ))
    if _is_synth:
        st.caption(
            "⚠️ G-кривая синтетическая — реальных точек ОФЗ в выборке недостаточно."
        )

# ── Пользовательские точки (конструктор сделки) ─────────────────
if st.session_state.custom_points:
    _cp = pd.DataFrame(st.session_state.custom_points)
    fig.add_trace(go.Scatter(
        x=_cp["duration"],
        y=_cp["yield_pct"],
        mode="markers+text",
        name="Мои точки",
        text=_cp["name"],
        textposition="top center",
        textfont=dict(family=FONT_FAMILY, size=10, color=COLOR_CUSTOM),
        marker=dict(
            color=COLOR_CUSTOM,
            size=marker_size + 5,
            symbol="star",
            opacity=1.0,
            line=dict(width=1, color="white"),
        ),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Дюрация: %{x:.2f} л<br>"
            "Доходность: %{y:.2f}%"
            "<extra></extra>"
        ),
    ))

# ── Оформление графика (шрифт Calibri везде) ────────────────────
fig.update_layout(
    title=dict(
        text="Дюрация × Доходность к погашению",
        font=dict(family=FONT_FAMILY, size=18),
        x=0.0,
    ),
    xaxis=dict(
        title=dict(text="Дюрация (лет)", font=dict(family=FONT_FAMILY, size=13)),
        tickfont=dict(family=FONT_FAMILY, size=11),
        gridcolor="#eeeeee",
        zeroline=False,
    ),
    yaxis=dict(
        title=dict(text="Доходность к погашению (%)", font=dict(family=FONT_FAMILY, size=13)),
        tickfont=dict(family=FONT_FAMILY, size=11),
        ticksuffix="%",
        gridcolor="#eeeeee",
        zeroline=False,
    ),
    legend=dict(
        font=dict(family=FONT_FAMILY, size=11),
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#cccccc",
        borderwidth=1,
        orientation="h",
        yanchor="bottom",
        y=1.01,
        xanchor="left",
        x=0.0,
    ),
    hovermode="closest",
    template="plotly_white",
    height=620,
    margin=dict(l=60, r=30, t=90, b=60),
    font=dict(family=FONT_FAMILY),
)


# ════════════════════════════════════════════════════════════════
# КНОПКА ЭКСПОРТА PNG + ОТОБРАЖЕНИЕ ГРАФИКА
# ════════════════════════════════════════════════════════════════
_btn_col, _ = st.columns([2, 5])
_png = _fig_to_png(fig)
with _btn_col:
    if _png:
        st.download_button(
            "⬇ Скачать график (PNG)",
            data=_png,
            file_name=f"yield_map_{updated}.png",
            mime="image/png",
        )
    else:
        st.caption("_PNG: установите `pip install kaleido`_")

st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════
# ТАБЛИЦА ДАННЫХ (st.data_editor с удалением строк)
# ════════════════════════════════════════════════════════════════
st.subheader("Список выпусков")
st.caption(
    "Отметьте выпуски в колонке **«🗑»** и нажмите **«Применить удаление»** "
    "— точки исчезнут с графика. Кнопка **«Восстановить»** отменяет удаление."
)

# Маппинг: внутреннее имя колонки → отображаемое название
_TABLE_COLS: dict[str, str] = {
    "ticker":        "Тикер",
    "rating":        "Рейтинг",
    "currency":      "Валюта",
    "volume":        "Объем, млн",
    "price":         "Цена, пп",
    "g_spread":      "G-спред, бп",
    "yield_pct":     "Дох-ть, %",
    "current_yield": "Тек. дох-ть, %",
    "duration":      "Дюрация",
    "coupon":        "Купон, %",
    "has_option":    "Опцион",
    "maturity":      "Погашение",
    "issue_date":    "Размещение",
    "isin":          "ISIN",
    "issuer":        "Эмитент",
    "sector":        "Сектор",
    "option_type":   "Тип опциона",
    "issuer_type":   "Тип эмитента",
    "liquidity":     "Ликвидность",
}

_show_cols = [c for c in _TABLE_COLS if c in fdf.columns]
_tbl = fdf[_show_cols].rename(columns=_TABLE_COLS).copy()

# Форматируем даты в читаемый вид
for _dc in ("Погашение", "Размещение"):
    if _dc in _tbl.columns:
        _tbl[_dc] = _tbl[_dc].apply(
            lambda x: x.strftime("%d.%m.%Y") if pd.notna(x) else ""
        )

# Добавляем чекбокс «Удалить» первой колонкой
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
        "Дох-ть, %":      st.column_config.NumberColumn(format="%.2f"),
        "Тек. дох-ть, %": st.column_config.NumberColumn(format="%.2f"),
        "Дюрация":        st.column_config.NumberColumn(format="%.2f"),
        "Купон, %":       st.column_config.NumberColumn(format="%.2f"),
        "G-спред, бп":    st.column_config.NumberColumn(format="%.0f"),
        "Цена, пп":       st.column_config.NumberColumn(format="%.2f"),
        "Объем, млн":     st.column_config.NumberColumn(format="%.0f"),
    },
    # Все колонки только для чтения, кроме чекбокса удаления
    disabled=[c for c in _tbl.columns if c != "🗑"],
    key="bonds_editor",
)

# ── Кнопки управления таблицей ──────────────────────────────────
_bc1, _bc2, _bc3 = st.columns([2, 2, 3])

with _bc1:
    if st.button("🗑 Применить удаление", use_container_width=True):
        # Определяем ISIN отмеченных строк
        _marked = _edited.loc[_edited["🗑"] == True]
        _new_del: set[str] = set()
        if "ISIN" in _marked.columns:
            _new_del = set(_marked["ISIN"].tolist())

        if _new_del:
            st.session_state.deleted_isins |= _new_del
            st.success(f"Удалено {len(_new_del)} выпусков с графика.")
            st.rerun()
        else:
            st.info("Не выбрано ни одного выпуска для удаления.")

with _bc2:
    if st.button("↩ Восстановить все", use_container_width=True):
        st.session_state.deleted_isins = set()
        st.rerun()

with _bc3:
    _csv = fdf[_show_cols].to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "⬇ Скачать CSV",
        data=_csv,
        file_name=f"fixed_bonds_{updated}.csv",
        mime="text/csv",
        use_container_width=True,
    )

if st.session_state.deleted_isins:
    st.caption(
        f"Скрыто с графика: **{len(st.session_state.deleted_isins)}** выпусков. "
        "Нажмите «Восстановить все» для отмены."
    )

# ════════════════════════════════════════════════════════════════
# ПОДВАЛ
# ════════════════════════════════════════════════════════════════
st.markdown("---")
st.caption(
    "PRJ4 | Автор: Ширманов К.А. | tg: [shirman7](tg://resolve?domain=shirman7)"
)
