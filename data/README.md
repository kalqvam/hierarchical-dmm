# Data

Snapshots of the raw inputs used to train and evaluate the model, downloaded in May 2026.
All files except `sp500_more_data.xlsx` can be re-downloaded with [`fetch_data.py`](../fetch_data.py)
(this needs a free FRED API key; see the main README).

**These files are not covered by the repository's MIT license.** They remain subject to their providers' terms
and are included only so the notebooks run without extra setup, for non-commercial research and educational use.
If you reuse the data, get it from the original source.

## Market data

| File | Contents | Coverage | Source |
|---|---|---|---|
| `spy_ohlcv.csv` | SPDR S&P 500 ETF (SPY) daily open, high, low, close, volume; split- and dividend-adjusted | 1993-01-29 to 2026-05-15 | Yahoo Finance, via the [`yfinance`](https://github.com/ranaroussi/yfinance) package (`auto_adjust=True`) |
| `sp500_more_data.xlsx` | S&P 500 quarterly earnings per share (EPS) and dividends per share (DPS) | 1988-Q1 to 2025-Q3 | S&P Dow Jones Indices, S&P 500 earnings data published on [spglobal.com](https://www.spglobal.com/spdji/en/indices/equity/sp-500/). The original download page is no longer available. |

## FRED series

All retrieved from FRED, Federal Reserve Bank of St. Louis, https://fred.stlouisfed.org/.

| File | Series ID | Description | Original source | Coverage |
|---|---|---|---|---|
| `fred_t10y3m.csv` | [T10Y3M](https://fred.stlouisfed.org/series/T10Y3M) | 10-Year Treasury Constant Maturity Minus 3-Month Treasury Constant Maturity | Federal Reserve Bank of St. Louis | 1982-01-04 to 2026-05-15 |
| `fred_baa10y.csv` | [BAA10Y](https://fred.stlouisfed.org/series/BAA10Y) | Moody's Seasoned Baa Corporate Bond Yield Relative to Yield on 10-Year Treasury Constant Maturity | Federal Reserve Bank of St. Louis; Moody's | 1986-01-02 to 2026-05-14 |
| `fred_dgs2.csv` | [DGS2](https://fred.stlouisfed.org/series/DGS2) | Market Yield on U.S. Treasury Securities at 2-Year Constant Maturity | Board of Governors of the Federal Reserve System (H.15) | 1976-06-01 to 2026-05-14 |
| `fred_dgs10.csv` | [DGS10](https://fred.stlouisfed.org/series/DGS10) | Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity | Board of Governors of the Federal Reserve System (H.15) | 1962-01-02 to 2026-05-21 |
| `fred_icsa.csv` | [ICSA](https://fred.stlouisfed.org/series/ICSA) | Initial Claims (weekly, seasonally adjusted) | U.S. Employment and Training Administration | 1967-01-07 to 2026-05-09 |
| `fred_cpi.csv` | [CPIAUCSL](https://fred.stlouisfed.org/series/CPIAUCSL) | Consumer Price Index for All Urban Consumers: All Items in U.S. City Average | U.S. Bureau of Labor Statistics | 1947-01-01 to 2026-04-01 |
| `fred_umcsent.csv` | [UMCSENT](https://fred.stlouisfed.org/series/UMCSENT) | University of Michigan: Consumer Sentiment | Surveys of Consumers, University of Michigan | 1952-11-01 to 2026-03-01 |
| `fred_vix.csv` | [VIXCLS](https://fred.stlouisfed.org/series/VIXCLS) | CBOE Volatility Index: VIX | Cboe Global Markets | 1990-01-02 to 2026-05-14 |

`fred_dgs10.csv` is downloaded by `fetch_data.py` but not used by the model.

Suggested citation format (per FRED):

> Board of Governors of the Federal Reserve System (US), *Market Yield on U.S. Treasury Securities at 2-Year Constant
> Maturity, Quoted on an Investment Basis* [DGS2], retrieved from FRED, Federal Reserve Bank of St. Louis;
> https://fred.stlouisfed.org/series/DGS2, May 2026.
