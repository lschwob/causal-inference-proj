# Do US macro releases move Bitcoin? A causal analysis

Final project for **Causal Inference & Experimentation (MSA-DAT09-04)**, Albert School × Mines Paris.
Authors: Alexandro Bizeul, Julius Hayter, Léonard Schwob.

We estimate the causal effect of scheduled US macro releases (CPI and NFP at 08:30 New York, FOMC decisions at 14:00)
on Bitcoin in a ±120-minute window, using 1-minute BTCUSDT prices (Binance, 2022-01 to 2026-10) and 151 releases.

- **Volatility:** realised volatility in the 30 minutes after a release is about ×2.4 that of comparable days without a release;
  the effect vanishes when the clock is shifted (falsification).
- **Direction:** the reaction follows the surprise (actual − consensus), about −65 bps per standard deviation at +5 minutes.
  The sign is robust; the magnitude is not stable over time.
- **Parallel trends are rejected** (BTC drifts up before the release), so the main estimator is a synthetic control
  (a "synthetic day" built from non-release days); the DiD is kept as a comparison and a falsification device.

## Deliverables

| File | Content |
|---|---|
| `notebooks/01_btc_us_macro_causal.ipynb` | Full analysis: DAG and backdoor criterion, naive comparison, DiD with parallel-trends test, synthetic control with donor pool and placebos, directional effect, falsification, identification critique, one-page recommendation |
| `report/report.pdf` | Written report (built by `report/build_report.py`) |
| `figures/`, `outputs/` | Figures and tables produced by the notebook |

## How to run

Locally:

```bash
pip install -r requirements.txt
python src/download_binance.py --start 2022-01 --end 2026-10   # ~2 min, writes data/btcusdt_1m.parquet
jupyter nbconvert --to notebook --execute --inplace notebooks/01_btc_us_macro_causal.ipynb
python report/build_report.py                                   # rebuilds report/report.pdf
```

On Google Colab: open the notebook and run all cells. The first cell clones this repository and downloads the Binance data.

## Repository layout

| Path | Role |
|---|---|
| `src/causal.py` | Event windows, DiD, synthetic day (ridge and classical synthetic control), placebo inference, pre-trend tests, directional slopes |
| `src/download_binance.py` | BTCUSDT 1-minute klines from data.binance.vision |
| `src/build_calendar.py` | CPI and NFP release dates from FRED (needs `FRED_API_KEY`) |
| `src/fetch_fomc.py` | FOMC decision dates (Federal Reserve) and Bauer-Swanson policy surprises (FRBSF) |
| `src/parse_investing.py` | Parses consensus tables copied from Investing.com into surprise files |
| `src/scrape_surprises.py`, `src/cpi_nowcast_surprise.py` | Exploratory surprise sources (not used in the main analysis) |
| `src/simulate.py` | Simulated data with a known effect, used to validate the estimators |
| `data/events.csv` | Treated events (CPI, NFP, FOMC) with the FOMC surprise |
| `data/surprises_manual.csv` | NFP and core CPI actual and consensus (Investing.com) |

## Data sources

Binance public data (prices), FRED and the Federal Reserve (release calendars), Investing.com (consensus forecasts, copied manually),
FRBSF Bauer-Swanson monetary policy surprises. No company data is used.
