# Multi-Factor Equity Backtest: Does Weighting Scheme Sophistication Matter?

A sector-neutral, four-factor systematic equity model built on the S&P 500, testing whether more sophisticated factor-weighting schemes actually outperform a simple equal-weight baseline.

*Built with Claude as a pair-programming/research partner — used to accelerate implementation and stress-test methodology choices (e.g. flagging the look-ahead risk in naive IC-weighting, or that a long-short construction was needed to separate factor premium from market beta). All modeling decisions, interpretation, and the final writeup are my own.*

## Summary

This project builds a long-only, quintile-based factor portfolio using four academically-grounded factors (momentum, value, quality, low volatility) on the S&P 500, using point-in-time FactSet fundamentals to avoid lookahead bias. Three factor-weighting schemes were implemented and backtested: equal-weight, rolling IC-weighted, and Kalman-filtered dynamic weighting.

**Key finding**: none of the three weighting schemes significantly outperformed the passive equal-weight benchmark on a risk-adjusted basis — a 2,000-iteration block bootstrap put nearly every pairwise Sharpe comparison's 90% confidence interval across zero, with one exception (IC-weighted significantly underperformed Kalman-weighted). More importantly, isolating each scheme's own factor premium from market exposure via a long-short construction (long the top quintile, short the bottom) produced a *negative* net-of-cost return for all three — meaning the long-only outperformance shown below is mostly market beta, not genuine factor-picking skill. This result is broadly consistent with DeMiguel, Garlappi, and Uppal (2009), "Optimal Versus Naive Diversification," which found that naive 1/N weighting frequently outperforms optimized weighting out-of-sample due to estimation error in optimized weights. The takeaway: factor selection and construction discipline appear to matter more than combination sophistication, at least for this universe and factor set.

## Methodology

**Universe**: S&P 500 constituents, market-cap floor of $500M applied dynamically at each rebalance date.

**Factors** (cross-sectionally z-scored within sector, quarterly rebalance):
| Factor | Definition | Academic anchor |
|---|---|---|
| Momentum | 12-1 month return | Jegadeesh & Titman (1993) |
| Value | EV/EBITDA (inverted) | Fama-French |
| Quality | ROE | Novy-Marx; Asness, Frazzini & Pedersen ("Quality Minus Junk") |
| Low volatility | Trailing 12-month return volatility (inverted) | Low-volatility anomaly literature |

**Weighting schemes tested**:
1. Equal-weight (baseline)
2. IC-weighted, rolling window
3. Kalman-filtered dynamic weighting (smoothed factor return/covariance estimates fed into portfolio construction)

**Portfolio construction**: long-only, top quintile by composite score, equal-weighted within quintile, quarterly rebalance. A long-short variant (long the top quintile, short the bottom quintile, dollar-neutral) was also tested, to isolate each scheme's factor premium from general market exposure.

**Backtest mechanics**: point-in-time fundamentals (with a standard reporting lag applied where exact filing dates weren't available), flat transaction cost assumption (~10bps/trade), tracked turnover per rebalance.

## Results

Net of a 10bps/trade transaction cost assumption, over 39 quarterly rebalances (~10 years):

| Strategy | Ann. Return | Ann. Vol | Sharpe | Max Drawdown |
|---|---|---|---|---|
| Equal-weight | 15.45% | 16.81% | 0.92 | -23.3% |
| IC-weighted | 14.95% | 17.36% | 0.86 | -20.7% |
| Kalman-weighted | 17.35% | 17.96% | 0.97 | -23.3% |
| Benchmark (equal-weight universe, no factor tilt) | 15.77% | 17.39% | 0.91 | -23.0% |

A 2,000-iteration block bootstrap found only one pairwise Sharpe difference distinguishable from noise at 90% confidence: IC-weighted underperformed Kalman-weighted. Every other pairwise comparison — including each scheme vs. the passive benchmark — was not statistically distinguishable given this sample size.

**Long-short (isolating the factor premium from market beta)** — long the top quintile, short the bottom, dollar-neutral, net of costs:

| Strategy | Ann. Return | Sharpe |
|---|---|---|
| Equal-weight | -4.45% | -0.40 |
| IC-weighted | -3.32% | -0.27 |
| Kalman-weighted | -1.11% | -0.08 |

All three are net losers market-neutral — the long-only numbers above are mostly market beta, not stock-picking skill.

See `/results` for the full chart set:
- `equity_curve_comparison.png` — growth of $1, all four series
- `factor_performance_breakdown.png` — cumulative factor returns and IC t-stats vs. the significance threshold
- `weight_allocation_over_time.png` — how IC- and Kalman-weighted schemes split weight across factors over time
- `drawdown_underwater.png` — drawdown from running peak, all four series
- `long_short_vs_long_only.png` — the clearest single visual of the market-beta finding above

## Limitations

- **Survivorship bias**: the universe is built from *current* S&P 500 constituents applied retroactively across the full ~10-year window, not point-in-time historical index membership. This almost certainly inflates every strategy's absolute performance (including the passive benchmark) by construction, since it excludes any company that was delisted, acquired, or dropped from the index for underperforming. Treat the relative comparisons between strategies as more trustworthy than any single strategy's absolute return.
- **Fundamentals lag**: a standard 45-day reporting lag was assumed between a fundamental's period-end date and when it would actually have been public, rather than using real filing dates (not available on this license). A common approximation, but not verified against actual disclosure timing.
- **Sector classification**: GICS sector/sub-industry came directly from the FactSet export, applied as a single static snapshot across the whole backtest window — it does not account for any sector reclassifications a covered company may have gone through over the 10-year period.
- **Small sample size**: 39 quarterly rebalances is a small sample for comparing three strategies' Sharpe ratios, which is exactly why the bootstrap above matters more than the raw point estimates.
- **Covariance estimation (Kalman scheme)**: the mean-variance optimizer's covariance matrix is built by Kalman-filtering each entry independently, which isn't guaranteed to produce a valid (positive semi-definite) matrix every period. A fixed-intensity shrinkage toward a diagonal target was applied to stabilize it — a simplification, not an optimal (e.g. Ledoit-Wolf) estimator.

## Data

Fundamental and pricing data sourced via a FactSet academic license and not included in this repo due to redistribution restrictions. All methodology and code are documented above and in `/src`; the pipeline can be run against equivalent data from any point-in-time fundamentals provider.

## References

- Jegadeesh, N., & Titman, S. (1993). Returns to Buying Winners and Selling Losers.
- Fama, E. F., & French, K. R. (1992, 1993, 2015). Common factor models.
- Novy-Marx, R. (2013). The Other Side of Value: The Gross Profitability Premium.
- Asness, C., Frazzini, A., & Pedersen, L. H. Quality Minus Junk.
- DeMiguel, V., Garlappi, L., & Uppal, R. (2009). Optimal Versus Naive Diversification.