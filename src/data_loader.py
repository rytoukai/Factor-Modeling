"""Load Bloomberg-style Excel sheets into tidy long-form pandas data,
handling the date/ticker formatting gotchas and the point-in-time
reporting-lag alignment. See Main.ipynb sections 1-4 for how this is used."""

from pathlib import Path

import pandas as pd


def excel_serial_to_date(series: pd.Series) -> pd.Series:
    """Coerce a DATE column to real Timestamps whether it came in as an
    Excel serial number (no date format applied) or was already parsed."""
    if pd.api.types.is_numeric_dtype(series):
        parsed = pd.to_datetime(series, unit="D", origin="1899-12-30")
    else:
        parsed = pd.to_datetime(series)
    # pin to a fixed resolution — pandas can otherwise hand back datetime64[s]
    # vs datetime64[us] depending on the code path taken above, and
    # merge_asof refuses to join keys with mismatched resolutions
    return parsed.astype("datetime64[ns]")


def clean_ticker(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper()


def load_field_long(path: Path, sheet_name: str, value_name: str) -> pd.DataFrame:
    """Load a DATE x ticker sheet into tidy long form: date, ticker, <value_name>."""
    df = pd.read_excel(path, sheet_name=sheet_name)
    date_col = df.columns[0]
    df = df.rename(columns={date_col: "date"})
    df["date"] = excel_serial_to_date(df["date"])

    long = df.melt(id_vars="date", var_name="ticker", value_name=value_name)
    long["ticker"] = clean_ticker(long["ticker"])
    long = long.dropna(subset=[value_name])
    return long.sort_values(["ticker", "date"]).reset_index(drop=True)


def load_sector_map(path: Path, sheet_name: str) -> pd.DataFrame:
    """GICS sheet is transposed: rows are attributes (GICS, GICS Sub-Industry),
    columns are tickers. Flip it to one row per ticker."""
    raw = pd.read_excel(path, sheet_name=sheet_name, index_col=0)
    sector = raw.T.reset_index().rename(columns={"index": "ticker"})
    sector.columns = [str(c).strip() for c in sector.columns]
    sector["ticker"] = clean_ticker(sector["ticker"])
    return sector


def ffill_with_reporting_lag(
    price_long: pd.DataFrame,
    fundamental_long: pd.DataFrame,
    value_name: str,
    lag_days: int,
) -> pd.DataFrame:
    """Attach a fundamental to each price date using as-of (backward)
    forward-fill, but only once it would actually have been public:
    available_date = report/period date + lag_days."""
    fundamental_long = fundamental_long.copy()
    fundamental_long["available_date"] = (
        fundamental_long["date"] + pd.Timedelta(days=lag_days)
    ).astype("datetime64[ns]")

    merged_parts = []
    for ticker, price_grp in price_long.groupby("ticker"):
        fund_grp = (
            fundamental_long.loc[fundamental_long["ticker"] == ticker, ["available_date", value_name]]
            .sort_values("available_date")
        )
        price_grp = price_grp.sort_values("date")
        merged = pd.merge_asof(
            price_grp, fund_grp, left_on="date", right_on="available_date", direction="backward"
        )
        merged_parts.append(merged.drop(columns="available_date"))

    return pd.concat(merged_parts, ignore_index=True)
