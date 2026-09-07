"""Turn a composite score into an actual portfolio: quintile sorting,
long-only and long-short construction, turnover, and the passive
benchmark. All buy-and-hold-between-rebalances — no interim trading.
See Main.ipynb sections 11 and 13 for how this is used."""

import pandas as pd


def quintile_top_holdings(panel: pd.DataFrame, score_col: str, date, eligible_col: str = "above_cap_floor") -> list[str]:
    """Tickers in the top quintile of `score_col`, among cap-floor-
    eligible stocks with a valid score, on a single rebalance date.
    Nothing here needs to know about weights -- equal weighting is applied
    later, in compute_portfolio_returns, as 1 / n_holdings."""
    cross_section = panel[(panel["date"] == date) & panel[eligible_col]].dropna(subset=[score_col])
    if len(cross_section) < 5:
        return []
    quintile = pd.qcut(cross_section[score_col], 5, labels=False, duplicates="drop")
    top_bin = quintile.max()  # dynamic, not hardcoded to 4 -- robust to duplicates collapsing bins
    return cross_section.loc[quintile == top_bin, "ticker"].tolist()


def quintile_bottom_holdings(panel: pd.DataFrame, score_col: str, date, eligible_col: str = "above_cap_floor") -> list[str]:
    """Mirror of quintile_top_holdings: the BOTTOM quintile by score --
    the short leg of the long-short portfolio."""
    cross_section = panel[(panel["date"] == date) & panel[eligible_col]].dropna(subset=[score_col])
    if len(cross_section) < 5:
        return []
    quintile = pd.qcut(cross_section[score_col], 5, labels=False, duplicates="drop")
    bottom_bin = quintile.min()
    return cross_section.loc[quintile == bottom_bin, "ticker"].tolist()


def compute_portfolio_returns(panel: pd.DataFrame, score_col: str, rebalance_dates: list, price_col: str = "price") -> pd.DataFrame:
    """Long-only, equal-weight-within-quintile portfolio: buy the top
    quintile by `score_col` at each rebalance date, hold with no interim
    trading until the next one. Returned rows are indexed by the date each
    holding period *ends* -- "the return earned by holding from the
    previous rebalance date through this one" -- same convention as
    fwd_return in weighting.py: attach a number to the date it becomes
    knowable."""
    records = []
    for i in range(len(rebalance_dates) - 1):
        start_date, end_date = rebalance_dates[i], rebalance_dates[i + 1]
        holdings = quintile_top_holdings(panel, score_col, start_date)
        if not holdings:
            continue

        start_prices = panel[(panel["date"] == start_date) & panel["ticker"].isin(holdings)].set_index("ticker")[price_col]
        end_prices = panel[(panel["date"] == end_date) & panel["ticker"].isin(holdings)].set_index("ticker")[price_col]
        common = start_prices.index.intersection(end_prices.index)
        stock_returns = end_prices[common] / start_prices[common] - 1

        records.append({
            "date": end_date,
            "n_holdings": len(holdings),
            "n_priced": len(common),
            "portfolio_return": stock_returns.mean(),
        })

    return pd.DataFrame(records).set_index("date")


def compute_long_short_returns(panel: pd.DataFrame, score_col: str, rebalance_dates: list, price_col: str = "price") -> pd.DataFrame:
    """Dollar-neutral long-short: long the top quintile, short the bottom
    quintile, equal-weighted within each leg. long_short_return is the
    long leg's holding-period return minus the short leg's -- same
    buy-and-hold-between-rebalances mechanics as compute_portfolio_returns,
    applied to both legs independently."""
    records = []
    for i in range(len(rebalance_dates) - 1):
        start_date, end_date = rebalance_dates[i], rebalance_dates[i + 1]
        long_holdings = quintile_top_holdings(panel, score_col, start_date)
        short_holdings = quintile_bottom_holdings(panel, score_col, start_date)
        if not long_holdings or not short_holdings:
            continue

        def leg_return(holdings):
            start_prices = panel[(panel["date"] == start_date) & panel["ticker"].isin(holdings)].set_index("ticker")[price_col]
            end_prices = panel[(panel["date"] == end_date) & panel["ticker"].isin(holdings)].set_index("ticker")[price_col]
            common = start_prices.index.intersection(end_prices.index)
            return (end_prices[common] / start_prices[common] - 1).mean(), len(common)

        long_return, n_long = leg_return(long_holdings)
        short_return, n_short = leg_return(short_holdings)

        records.append({
            "date": end_date,
            "n_long": n_long,
            "n_short": n_short,
            "long_return": long_return,
            "short_return": short_return,
            "long_short_return": long_return - short_return,
        })

    return pd.DataFrame(records).set_index("date")


def compute_benchmark_returns(panel: pd.DataFrame, rebalance_dates: list, price_col: str = "price", eligible_col: str = "above_cap_floor") -> pd.DataFrame:
    """Equal-weighted return of the FULL cap-floor-eligible universe --
    no factor tilt -- the actual benchmark the factor portfolios are being
    measured against. Same buy-and-hold-between-rebalances mechanics as
    compute_portfolio_returns, just with 'every eligible stock' as the
    holdings rule instead of a quintile sort."""
    records = []
    for i in range(len(rebalance_dates) - 1):
        start_date, end_date = rebalance_dates[i], rebalance_dates[i + 1]
        holdings = panel.loc[(panel["date"] == start_date) & panel[eligible_col], "ticker"].tolist()
        if not holdings:
            continue
        start_prices = panel[(panel["date"] == start_date) & panel["ticker"].isin(holdings)].set_index("ticker")[price_col]
        end_prices = panel[(panel["date"] == end_date) & panel["ticker"].isin(holdings)].set_index("ticker")[price_col]
        common = start_prices.index.intersection(end_prices.index)
        stock_returns = end_prices[common] / start_prices[common] - 1
        records.append({"date": end_date, "n_holdings": len(holdings), "portfolio_return": stock_returns.mean()})
    return pd.DataFrame(records).set_index("date")


def compute_turnover_by_period(panel: pd.DataFrame, score_col: str, rebalance_dates: list) -> pd.Series:
    """Turnover incurred entering each holding period, indexed by the
    date that period *ends* -- the same convention compute_portfolio_returns
    uses -- so it lines up directly against that period's gross return
    with no separate date-alignment step. The very first holding period
    has no prior portfolio to compare against, so it has no entry here
    (callers typically treat that as zero cost via .fillna(0) -- there's
    nothing to have traded out of yet)."""
    turnovers = {}
    prev_holdings = None
    for i in range(len(rebalance_dates) - 1):
        start_date, end_date = rebalance_dates[i], rebalance_dates[i + 1]
        holdings = set(quintile_top_holdings(panel, score_col, start_date))
        if prev_holdings is not None and len(prev_holdings) > 0:
            turnovers[end_date] = len(holdings - prev_holdings) / len(prev_holdings)
        prev_holdings = holdings
    return pd.Series(turnovers)


def compute_long_short_turnover(panel: pd.DataFrame, score_col: str, rebalance_dates: list) -> pd.Series:
    """Turnover summed across BOTH legs, same end-date indexing
    convention as compute_turnover_by_period -- both legs get rebalanced
    every quarter and both incur trading costs."""
    turnovers = {}
    prev_long, prev_short = None, None
    for i in range(len(rebalance_dates) - 1):
        start_date, end_date = rebalance_dates[i], rebalance_dates[i + 1]
        long_holdings = set(quintile_top_holdings(panel, score_col, start_date))
        short_holdings = set(quintile_bottom_holdings(panel, score_col, start_date))
        if prev_long is not None and len(prev_long) > 0 and len(prev_short) > 0:
            long_turnover = len(long_holdings - prev_long) / len(prev_long)
            short_turnover = len(short_holdings - prev_short) / len(prev_short)
            turnovers[end_date] = long_turnover + short_turnover
        prev_long, prev_short = long_holdings, short_holdings
    return pd.Series(turnovers)
