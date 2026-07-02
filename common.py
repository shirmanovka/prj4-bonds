"""
common.py — Общие константы и утилиты для страниц PRJ4
Используется fixed.py и pages/*.py, чтобы не дублировать логику
рейтингов, секторов и обновления данных между страницами.
"""

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

import fetch as _fetch

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

FONT_FAMILY = "Calibri"   # Шрифт всех текстов на графиках

# Префикс для опций-эмитентов в комбинированном фильтре "Тикер / Эмитент"
ISSUER_PREFIX = "Эмитент: "


def clean_rating(series: pd.Series) -> pd.Series:
    """
    Нормализует рейтинг: берёт первый компонент из составного вида
    'AA (AA-)' или 'A-/BBB+', убирает пробелы.
    Пример: 'AA (AA-)' → 'AA', 'A-/BBB+' → 'A-'
    """
    s = series.str.split(r"[/\s\(]").str[0].str.strip()
    s = s.str.extract(r"^([A-D][A-Za-z+\-]*)")[0]
    return s


def sector_set(series: pd.Series) -> pd.Series:
    """
    Разбивает поле sector (может содержать несколько значений через запятую,
    например "IT, МФО") на множество атомарных секторов — чтобы фильтр по
    одному сектору находил такие бумаги и не показывал склеенные строки
    как отдельные "секторы".
    """
    return series.fillna("").apply(
        lambda s: frozenset(p.strip() for p in s.split(",") if p.strip())
    )


def get_last_update_display(df: pd.DataFrame) -> str:
    """
    Определяет время последнего успешного обновления данных для отображения.
    Приоритет: data/last_update.txt (точный timestamp) → mtime bonds.csv → колонка 'updated'.
    """
    ts_path = DATA_DIR / "last_update.txt"
    if ts_path.exists():
        try:
            dt = datetime.fromisoformat(ts_path.read_text(encoding="utf-8").strip())
            return dt.strftime("%d.%m.%Y %H:%M")
        except ValueError:
            pass

    bonds_path = DATA_DIR / "bonds.csv"
    if bonds_path.exists():
        dt = datetime.fromtimestamp(bonds_path.stat().st_mtime)
        return dt.strftime("%d.%m.%Y %H:%M")

    if "updated" in df.columns and not df.empty:
        return f"{df['updated'].iloc[0]} 00:00"

    return "—"


def render_refresh_button(df_for_timestamp: pd.DataFrame, cache_clear_fns: list) -> None:
    """
    Рисует в сайдбаре кнопку принудительного обновления данных + время
    последнего обновления. По клику вызывает fetch.main() (качает и
    фиксированные, и плавающие облигации разом) и сбрасывает переданные
    кэш-функции (@st.cache_data), затем перезапускает страницу.
    """
    if st.button("🔄 Обновить данные", use_container_width=True):
        with st.spinner("Загружаю данные..."):
            try:
                _fetch.main()
            except Exception as e:
                st.error(f"Не удалось обновить данные: {e}")
            else:
                for _fn in cache_clear_fns:
                    _fn.clear()
                st.rerun()
    st.caption(f"Обновлено: **{get_last_update_display(df_for_timestamp)}**")
    st.divider()
