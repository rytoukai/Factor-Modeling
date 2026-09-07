# Multi-Factor Equity Backtest: Does Weighting Scheme Sophistication Matter?

A sector-neutral, four-factor systematic equity model built on the S&P 500, testing whether more sophisticated factor-weighting schemes actually outperform a simple equal-weight baseline.

## Summary

This project builds a long-only, quintile-based factor portfolio using four academically-grounded factors (momentum, value, quality, low volatility) on the S&P 500, using point-in-time FactSet fundamentals to avoid lookahead bias. Three factor-weighting schemes were implemented and backtested: equal-weight, rolling IC-weighted, and Kalman-filtered dynamic weighting.

**Key finding**: no statistically significant difference in risk-adjusted performance across the three weighting schemes. This result is consistent with DeMiguel, Garlappi, and Uppal (2009), "Optimal Versus Naive Diversification," which found that naive 1/N weighting frequently outperforms optimized weighting out-of-sample due to estimation error in optimized weights. The takeaway: factor selection and construction discipline appear to matter more than combination sophistication, at least for this universe and factor set.

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

**Portfolio construction**: long-only, top quintile by composite score, equal-weighted within quintile, quarterly rebalance.

**Backtest mechanics**: point-in-time fundamentals (with a standard reporting lag applied where exact filing dates weren't available), flat transaction cost assumption (~10bps/trade), tracked turnover per rebalance.

## Results

See `/results` for full output. Headline charts:
- Equity curve comparison across the three weighting schemes
- Factor-level performance breakdown

## Limitations

- **Survivorship bias**: [state here whether historical index membership was available or whether current constituents were applied retroactively]
- **Fundamentals lag**: [state whether actual filing dates were available, or whether a standard ~45-60 day lag assumption was used]
- **Sector classification**: [note if GICS came from FactSet directly or was sourced externally due to access limits]

## Repo Structure

```
factor-model-backtest/
├── README.md
├── notebooks/          # exploratory analysis
├── src/
│   ├── data_loader.py
│   ├── factors.py       # factor construction
│   ├── weighting.py     # equal-weight, IC-weighted, Kalman-filtered
│   ├── backtest.py
│   └── metrics.py
├── results/             # charts and output
└── requirements.txt
```

## Data

Fundamental and pricing data sourced via a FactSet academic license and not included in this repo due to redistribution restrictions. All methodology and code are documented above and in `/src`; the pipeline can be run against equivalent data from any point-in-time fundamentals provider.

## References

- Jegadeesh, N., & Titman, S. (1993). Returns to Buying Winners and Selling Losers.
- Fama, E. F., & French, K. R. (1992, 1993, 2015). Common factor models.
- Novy-Marx, R. (2013). The Other Side of Value: The Gross Profitability Premium.
- Asness, C., Frazzini, A., & Pedersen, L. H. Quality Minus Junk.
- DeMiguel, V., Garlappi, L., & Uppal, R. (2009). Optimal Versus Naive Diversification.