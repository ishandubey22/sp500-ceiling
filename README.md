# sp500_ceiling.py

A single-file pipeline for measuring survivorship bias in S&P 500 factor backtests.

Most free-data backtests on the S&P 500 are biased in ways that are hard to quantify: they use today's index composition for all historical dates, silently drop the last monthly return of every delisted stock, and rely on Yahoo Finance which no longer serves most pre-2015 delisted tickers. This project builds the infrastructure to measure each bias layer separately and report the exact CAGR and Sharpe impact.

**The strategy is a vehicle. The contribution is the bias measurement framework.**

---

## What it produces

```
  BIAS DELTAS (positive = naïve outperforms corrected)

  Membership bias  (Layer 0 − Layer 1):
    OOS CAGR   delta: -1.95%
    OOS Sharpe delta: -0.11

  Terminal-return bias  (Layer 1 − Layer 2):
    OOS CAGR   delta:  0.00%
    OOS Sharpe delta:  0.00

  Total measurable survivorship bias  (Layer 0 − Layer 2):
    OOS CAGR   delta: -1.95%
    OOS Sharpe delta: -0.11
```

**The negative sign is not a typo.** Using correct historical membership *improved* the anti-consensus strategy by 1.95 pp annually. Survivorship bias is directional — it inflates naive backtests of long-winners strategies; it deflates naive backtests of fade-the-winner strategies. Measuring the direction is the point.

The terminal-return delta is zero because 57% of delisted tickers still lack full price history. That gap is documented, not hidden.

---

## The three bias layers

| Layer | Membership | Terminal returns | Data |
|-------|-----------|-----------------|------|
| 0 — Naïve | Current S&P 500 for all dates | Drop NaN | Yahoo only |
| 1 — Membership fixed | Historical point-in-time | Drop NaN | Yahoo only |
| 2 — Full pipeline ★ | Historical + quarterly anchors | Recovered (FIX #1) | Yahoo + Tiingo |



### Wikipedia revision-history API
Instead of the Wayback Machine (which fails when archive.org hasn't crawled a target date), the pipeline queries Wikipedia's own Action API for the full revision history of the S&P 500 article. For every quarter-end since 2001, it finds the exact page revision live on that date and extracts the constituent list directly. This gives point-in-time snapshots with 3-month granularity instead of annual, and covers 71/104 target quarters successfully.

```
  2008-09-30  ✓  500 tickers  (revision 238868546, 13d before target)
  2009-03-31  ✓  496 tickers  (revision 278249766, 11d before target)
  ...
  2024-12-31  ✓  503 tickers  (revision 1265285344, 4d before target)
```

### Dynamic missing-ticker discovery
Rather than relying on a hardcoded list of 173 missing tickers, the pipeline cross-references the full reconstructed membership universe against the daily CSV at runtime. Any historical constituent absent from the price file — including names not on the hardcoded list — is added to the recovery queue.

### SEC EDGAR 8-K terminal price extraction (opt-in)
For acquisition targets still missing price history, the pipeline queries the EDGAR full-text search API for SC TO-T, DEFM14A, and 8-K filings. Per-share cash consideration is extracted by regex patterns that require explicit merger language (e.g., `"$102.43 per share in cash"`) and exclude EPS disclosures and dividend announcements. Enabled with `--edgar`.

### OpenFIGI rename resolution (opt-in)
Maps retired ticker symbols to their current equivalents for same-entity renames only. Acquisitions and mergers are handled via terminal price injection rather than ticker substitution to avoid mixing company histories. Enabled with `--openfigi`.

### Bias layer quantification
Layer 0 re-builds the factor panel with a naive `get_members()` that always returns today's 500 tickers, so factor values (momentum, volatility) themselves reflect the naïve analyst's universe — not just the membership filter applied after the fact. This correctly captures the full scope of static-membership bias.

---

## Backtest results (anti-consensus strategy)

| Period | Gross CAGR | Gross Sharpe | Max Drawdown |
|--------|:----------:|:------------:|:------------:|
| IS (2000–2014) | +1.14% | 0.17 | −51.38% |
| ★ OOS (2015–2026) | +3.12% | 0.26 | −34.34% |

| Cost tier | OOS net CAGR |
|-----------|:------------:|
| Tier A — Institutional (10 bps RT) | +1.48% |
| Tier B — Hedge fund (30 bps RT) | +0.41% |
| Tier C — Conservative (50 bps RT) | −0.64% |

The strategy is exploratory. It is not the contribution of this project.

---

## Data completeness (current run)

| Metric | Count |
|--------|------:|
| Historical tickers reconstructed | 850 |
| Tickers with price data | ~177 |
| Still absent after all recovery | 158 |
| Terminal prices injected (CURATED_TERMINAL) | 15 |
| Factor panel coverage | 43% |

The 57% absence rate is the honest ceiling of free-data approaches. The `missing_ticker_registry.csv` output documents every absent ticker with its last-known event type, so the gap is transparent and auditable rather than hidden.

---

## Curated terminal prices

For 36 tickers where Yahoo Finance no longer serves data, prices are injected from hand-verified SEC filings. A selection:

| Ticker | Price | Event | Source |
|--------|------:|-------|--------|
| CELG | $102.43 | BMS acquisition | 8-K 2019-11-20 |
| ATVI | $95.00 | MSFT acquisition | 8-K 2023-10-13 |
| RHT | $190.00 | IBM acquisition | 8-K 2019-07-09 |
| LEH | $0.21 | Bankruptcy | NYSE halt 2008-09-15 |
| EK | $0.36 | Bankruptcy | NYSE delist 2012-01-19 |
| FNM | $0.44 | Conservatorship | NYSE delist 2010-06-16 |

---

## Setup

```bash
pip install pandas numpy requests beautifulsoup4 yfinance scipy matplotlib
```

For delisted ticker recovery, a free Tiingo API key is required:

```bash
export TIINGO_API_KEY=your_key_here
```

Tiingo's free tier covers 1,000 API calls per day and supports historical data for delisted US equities. Sign up at [tiingo.com](https://www.tiingo.com).

---

## Usage

```bash
# Full pipeline (downloads everything fresh)
python sp500_ceiling.py

# Skip price download, use existing daily CSV
python sp500_ceiling.py --no-download

# Skip recovery (useful for fast iteration on factor/backtest changes)
python sp500_ceiling.py --no-download --no-recovery

# Run the bias layer study
python sp500_ceiling.py --no-download --bias-study

# See what Tiingo would be called before spending quota
python sp500_ceiling.py --no-download --tiingo-plan

# Cap Tiingo calls to 50 for this run
python sp500_ceiling.py --no-download --tiingo-max-calls 50

# Opt into EDGAR terminal price extraction
python sp500_ceiling.py --no-download --no-recovery --edgar

# Skip Fama-French download (offline mode)
python sp500_ceiling.py --no-ff

# Save all section CSVs (sensitivity grid, bias layers)
python sp500_ceiling.py --bias-study --csv

# No writes, analysis only
python sp500_ceiling.py --dry-run
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TIINGO_API_KEY` | — | Required for delisted ticker recovery |
| `TIINGO_TOKEN` | — | Accepted as an alias |
| `TIINGO_DAILY_LIMIT` | 1000 | Local quota guard |
| `TIINGO_RESERVE` | 25 | Calls to leave unused at end of run |
| `TIINGO_DELAY` | 0.25 | Seconds between Tiingo requests |
| `SP500_PIPELINE_DATA_DIR` | script directory | Output folder for CSV files |

---

## Output files

| File | Description |
|------|-------------|
| `sp500_daily_clean.csv` | Daily OHLCV for all historical S&P 500 tickers |
| `factors_clean.csv` | Monthly factor panel (momentum, vol, vol-change, fwd_ret) |
| `sp500_wiki_quarterly_anchors.csv` | Wikipedia quarterly membership snapshots |
| `missing_ticker_registry.csv` | Per-ticker transparency log |
| `tiingo_usage_ledger.json` | Local Tiingo quota tracker |
| `bias_layers.csv` | Bias layer results (with `--csv`) |
| `sensitivity_grid.csv` | IS sensitivity grid (with `--csv`) |

---

## Known limitations and honest caveats

**57% of historical tickers are absent from the factor panel.** The absent tickers are disproportionately bankruptcies, distressed delistings, and pre-2010 acquisitions — exactly the stocks with the most negative terminal returns. The measured bias delta is a lower bound on the true survivorship bias.

**Pre-2007 membership accuracy is lower.** The Wikipedia article's `id="constituents"` table format only reliably parses back to 2007. Before that, the fja05680 dataset provides monthly snapshots, but its pre-2005 coverage is also limited by the data it was compiled from.

**EDGAR terminal prices recover 0 for this run.** The EDGAR EFTS search is sensitive to query phrasing and document structure. The regex patterns require explicit cash-deal language; stock-for-stock mergers, CVR payouts, and earnings 8-Ks are intentionally excluded but may miss some valid acquisitions.

**Short interest proxy is static.** The cost model uses current Yahoo Finance `shortPercentOfFloat` as a uniform proxy for historical borrow costs. Historical per-ticker monthly short interest requires a paid subscription (CRSP, S3 Shortsight).

**The strategy is not the contribution.** The anti-consensus composite (fade high-momentum, high-vol, high-volume stocks) is presented as a test vehicle. Its OOS Sharpe of 0.26 on a long-short gross basis is modest. Do not use these results to trade.

---

## Project history

This pipeline consolidates and supersedes five earlier files:

- `diagnosis.py` — core download and factor construction
- `sp500_membership.py` — fja05680 + Wayback Machine anchors
- `missing_delisted.py` — Stooq / Nasdaq / BigCharts waterfall
- `recover_terminal_prices.py` — Alpha Vantage + manual injection
- `backtest_pro.py` — IS/OOS backtest with FF5 regression

The original files used Stooq as the primary recovery source. Stooq geo-blocks requests from parts of Asia; the waterfall returned 0/172 tickers in the development environment. Tiingo replaced it as the recovery source because it has an authenticated API that explicitly supports delisted history.

---

## Citation

If you use this work or the bias measurement framework, please cite:

```
sp500_ceiling.py — Survivorship Bias Measurement Pipeline for S&P 500 Factor Backtests
https://github.com/dubey.ishan22/sp500-ceiling
```

---

## License

GPL