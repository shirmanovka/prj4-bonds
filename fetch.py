"""
Скачивает данные облигаций с bondresearch.ru и сохраняет в data/bonds.csv и data/floaters.csv.
Запускается GitHub Actions каждое утро или вручную.
"""
import re
import sys
import requests
import pandas as pd
from datetime import date, datetime
from pathlib import Path

HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.bondresearch.ru/dashboard/"}

# Колонки по индексу в массиве
FIXED_COLS = {
    "isin":          39,
    "ticker":        22,
    "rating":         2,
    "currency":       3,
    "volume":         4,
    "price":          5,
    "g_spread":       6,
    "yield_pct":      7,
    "current_yield":  8,
    "duration":       9,
    "spread":        10,
    "coupon":        11,
    "option_date":   12,
    "maturity":      13,
    "issue_date":    14,
    "option_type":   24,
    "issuer":        26,
    "sector":        27,
    "liquidity":     31,
    "face_value":    32,
    "coupon_freq":   33,
    "next_coupon":   34,
    "guarantee":     38,
}

FLOATER_COLS = {
    "isin":          1,   # ISIN (plain text)
    "rating":        3,
    "currency":      4,
    "volume":        5,
    "price":         6,
    "spread_ks":     7,   # спред к КС (б.п.)
    "yield_total":  12,   # КС + спред (итоговая доходность %)
    "maturity":     14,
    "issue_date":   15,
    "ticker":       20,
    "option_type":  22,
    "issuer":       24,
    "sector":       25,
    "liquidity":    29,
    "coupon_freq":  30,
    "next_coupon":  31,
    "base_rate":    35,   # 'КС'
}


def fetch_json(url: str) -> list:
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    # Данные могут быть в ключе 'demo' или в корне
    if isinstance(data, dict):
        for key in data:
            if isinstance(data[key], list):
                return data[key]
    return data


def extract_isin_from_link(html: str) -> str | None:
    m = re.search(r'/bond/((?:RU|XS|US)[0-9A-Z]+)', str(html))
    return m.group(1) if m else None


def parse_fixed(rows: list) -> pd.DataFrame:
    records = []
    for row in rows:
        if not isinstance(row, list) or len(row) < 40:
            continue
        isin = row[FIXED_COLS["isin"]] or extract_isin_from_link(row[1])
        if not isin:
            continue
        rec = {"isin": isin}
        for col, idx in FIXED_COLS.items():
            if col == "isin":
                continue
            rec[col] = row[idx] if idx < len(row) else None
        records.append(rec)
    return pd.DataFrame(records)


def parse_floaters(rows: list) -> pd.DataFrame:
    records = []
    for row in rows:
        if not isinstance(row, list) or len(row) < 15:
            continue
        isin = row[FLOATER_COLS["isin"]] if len(row) > FLOATER_COLS["isin"] else None
        if not isin or not str(isin).startswith(("RU", "XS")):
            continue
        rec = {"isin": str(isin)}
        for col, idx in FLOATER_COLS.items():
            if col == "isin":
                continue
            rec[col] = row[idx] if idx < len(row) else None
        records.append(rec)
    return pd.DataFrame(records)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = ["volume", "price", "g_spread", "yield_pct", "current_yield",
                "duration", "spread", "coupon", "face_value", "coupon_freq",
                "spread_ks", "yield_total"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # option_date приходит с префиксом "К: 2026-12-20" или "П: 2026-12-08" —
    # извлекаем только дату перед парсингом
    if "option_date" in df.columns:
        df["option_date"] = (
            df["option_date"].astype(str)
            .str.extract(r"(\d{4}-\d{2}-\d{2})")[0]
        )
    date_cols = ["maturity", "issue_date", "next_coupon", "option_date"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
    df = df[df["isin"].notna() & (df["isin"] != "")]
    df = df.drop_duplicates("isin")
    return df


def fetch_fixed_bonds(out_dir: Path, today: str) -> int:
    """Скачивает фиксированные облигации и сохраняет data/bonds.csv. Кидает исключение при ошибке."""
    rows = fetch_json("https://www.bondresearch.ru/boards/base_test.json")
    df_fixed = clean(parse_fixed(rows))
    df_fixed["updated"] = today
    df_fixed["bond_type"] = "fixed"
    df_fixed.to_csv(out_dir / "bonds.csv", index=False, encoding="utf-8-sig")
    return len(df_fixed)


def fetch_floaters(out_dir: Path, today: str) -> int:
    """Скачивает флоутеры и сохраняет data/floaters.csv. Кидает исключение при ошибке."""
    rows = fetch_json("https://www.bondresearch.ru/boards/pig_floaters_mk.json")
    df_fl = clean(parse_floaters(rows))
    df_fl["updated"] = today
    df_fl["bond_type"] = "floater"
    df_fl.to_csv(out_dir / "floaters.csv", index=False, encoding="utf-8-sig")
    return len(df_fl)


def write_timestamp(out_dir: Path) -> str:
    """Фиксирует момент успешного обновления данных (для отображения в интерфейсе)."""
    ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    (out_dir / "last_update.txt").write_text(ts, encoding="utf-8")
    return ts


def main():
    """
    Обновляет data/bonds.csv и data/floaters.csv.
    Ошибка при загрузке фиксированных облигаций — критична (пробрасывается вызывающему).
    Ошибка при загрузке флоутеров — не критична, только предупреждение.
    """
    out_dir = Path(__file__).parent / "data"
    out_dir.mkdir(exist_ok=True)
    today = str(date.today())

    print("Загружаю фиксированные облигации...")
    n_fixed = fetch_fixed_bonds(out_dir, today)
    print(f"  Сохранено: {n_fixed} строк → data/bonds.csv")

    print("Загружаю флоутеры...")
    try:
        n_fl = fetch_floaters(out_dir, today)
        print(f"  Сохранено: {n_fl} строк → data/floaters.csv")
    except Exception as e:
        print(f"  ПРЕДУПРЕЖДЕНИЕ флоутеры: {e}", file=sys.stderr)

    write_timestamp(out_dir)
    print("Готово.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ОШИБКА фиксированные: {e}", file=sys.stderr)
        sys.exit(1)
