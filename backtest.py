"""Standard backtest performance metrics, and a paired block-bootstrap
for testing whether a Sharpe-ratio difference between two strategies is
distinguishable from noise. See Main.ipynb sections 12 and 14 for how
this is used."""

import numpy as np
import pandas as pd


def performance_metrics(returns: pd.Series, periods_per_year: int, risk_free_rate: float) -> dict:
    """Standard backtest performance metrics from a period return series.
    Annualization compounds geometrically -- total_growth ** (1 / years) - 1
    -- matching how a growth-of-$1 curve is built, rather than a naive
    (mean_return * periods_per_year) approximation that ignores
    compounding."""
    returns = returns.dropna()
    n = len(returns)
    growth = (1 + returns).cumprod()
    years = n / periods_per_year

    annualized_return = growth.iloc[-1] ** (1 / years) - 1
    annualized_vol = returns.std() * np.sqrt(periods_per_year)
    sharpe = (annualized_return - risk_free_rate) / annualized_vol if annualized_vol > 0 else np.nan

    running_max = growth.cummax()
    drawdown = growth / running_max - 1

    return {
        "annualized_return": annualized_return,
        "annualized_vol": annualized_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": drawdown.min(),
        "n_periods": n,
    }


def block_bootstrap_indices(n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    """One resampled path of length n, built from randomly chosen
    contiguous blocks of block_length consecutive positions (wrapping
    around the end of the series), then trimmed to exactly n. Blocks
    rather than single points preserve whatever short-run autocorrelation
    exists in the return series -- an ordinary iid bootstrap would
    implicitly assume returns are independent quarter to quarter, which a
    real return series generally isn't."""
    n_blocks_needed = int(np.ceil(n / block_length))
    starts = rng.integers(0, n, size=n_blocks_needed)
    idx = np.concatenate([np.arange(s, s + block_length) % n for s in starts])
    return idx[:n]


def bootstrap_metric_distribution(returns_dict: dict, metric_fn, n_bootstrap: int, block_length: int, seed: int = 0) -> pd.DataFrame:
    """Paired block-bootstrap distribution of `metric_fn` for every series
    in `returns_dict`. All series are aligned on a common date index
    first (inner join), and the SAME resampled date sequence is applied to
    every series on each iteration -- the strategies aren't independent of
    each other (they hold overlapping stocks and lived through the same
    historical quarters), so bootstrapping each series separately would
    overstate the uncertainty in any *difference* between them."""
    aligned = pd.DataFrame(returns_dict).dropna()
    n = len(aligned)
    rng = np.random.default_rng(seed)

    results = {name: [] for name in aligned.columns}
    for _ in range(n_bootstrap):
        idx = block_bootstrap_indices(n, block_length, rng)
        resampled = aligned.iloc[idx]
        for name in aligned.columns:
            results[name].append(metric_fn(resampled[name]))

    return pd.DataFrame(results)
