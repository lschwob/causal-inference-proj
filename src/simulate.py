"""SIMULATED DATA - only used to test the code when the real data are not available.

Generates a synthetic 1-minute BTC series with intraday seasonality, volatility regimes and an
INJECTED CAUSAL effect around releases (true values in TRUTH). Used to check that the estimators
(DiD, synthetic day) recover the effect. Results on these data have no economic meaning.
"""
import numpy as np, pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

TRUTH = dict(
    beta_bps_per_sd=12.0,     # cumulative causal drift at +30 min, bps per std. dev. of surprise
    vol_mult_first10=3.0,     # volatility multiplier over the first 10 minutes
    vol_mult_10_30=1.5,       # then over minutes 10-30
)


def _bdays(start, end):
    d = pd.bdate_range(start, end)
    hol = USFederalHolidayCalendar().holidays(start, end)
    return d.difference(hol)


def simulate_calendar(start="2022-01-01", end="2026-09-30", seed=1):
    rng = np.random.default_rng(seed)
    days = _bdays(start, end)
    rows, other = [], set()
    for ym, g in pd.Series(days, index=days).groupby(days.to_period("M")):
        g = pd.DatetimeIndex(g.values)
        cpi = g[min(len(g) - 1, 7 + rng.integers(0, 3))]          # ~ 10th-12th of the month
        nfp = g[g.dayofweek == 4][0]                                # first Friday
        rows += [("CPI", cpi, "08:30"), ("NFP", nfp, "08:30")]
        other |= {g[min(len(g) - 1, 8)], g[min(len(g) - 1, 12)]}   # PPI, retail sales
    for y in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
        for m in (1, 3, 5, 6, 7, 9, 11, 12):
            g = days[(days.year == y) & (days.month == m) & (days.dayofweek == 2)]
            if len(g) >= 3:
                rows.append(("FOMC", g[2], "14:00"))
    ev = pd.DataFrame(rows, columns=["event_type", "date", "time_ny"])
    ev["date_ny"] = ev["date"].dt.strftime("%Y-%m-%d")
    ev = ev[(ev.date_ny >= start) & (ev.date_ny <= end)].drop(columns="date")
    ev = ev.drop_duplicates(["date_ny", "time_ny"]).sort_values("date_ny").reset_index(drop=True)
    ev["surprise_z"] = rng.standard_normal(len(ev))
    thursdays = {d.strftime("%Y-%m-%d") for d in days[days.dayofweek == 3]}
    o830 = {d.strftime("%Y-%m-%d") for d in other} | thursdays | set(ev.loc[ev.time_ny == "08:30", "date_ny"])
    return ev, sorted(o830)


def simulate_btc(events, start="2022-01-01", end="2026-10-01", seed=2):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq="1min", tz="UTC", inclusive="left")   # = close times
    n = len(idx)
    ndays = n // 1440 + 2
    lv = np.zeros(ndays)
    for i in range(1, ndays):                                     # daily AR(1) volatility regime
        lv[i] = 0.97 * lv[i - 1] + 0.12 * rng.standard_normal()
    daily_sig = 0.022 * np.exp(lv)
    sig_day = np.repeat(daily_sig, 1440)[:n]
    hour = idx.hour.to_numpy() + idx.minute.to_numpy() / 60
    season = 0.75 + 0.7 * np.exp(-0.5 * ((hour - 16.5) / 3.0) ** 2)    # peak during the US session (UTC)
    season = season / np.sqrt((season ** 2).mean())
    sigma = sig_day / np.sqrt(1440) * season
    eps = rng.standard_t(6, n) / np.sqrt(6 / 4)
    r = sigma * eps
    # --- causal effect injected at each release ---
    mult = np.ones(n)
    drift = np.zeros(n)
    t0s = [pd.Timestamp(f"{d} {h}", tz="America/New_York").tz_convert("UTC")
           for d, h in zip(events.date_ny, events.time_ny)]
    pos = np.round((pd.DatetimeIndex(t0s) - idx[0]).total_seconds().to_numpy() / 60).astype(int)
    for p, z in zip(pos, events.surprise_z.to_numpy()):
        if p < 0 or p + 130 >= n:
            continue
        mult[p + 1:p + 11] *= TRUTH["vol_mult_first10"]            # bar m closes at t0+m
        mult[p + 11:p + 31] *= TRUTH["vol_mult_10_30"]
        drift[p + 1:p + 31] += TRUTH["beta_bps_per_sd"] * 1e-4 * z / 30
    r = r * mult + drift
    lp = np.log(30000) + np.cumsum(r)
    vol = np.exp(rng.normal(3.0, 0.4, n)) * (sigma / sigma.mean()) * mult
    return pd.DataFrame({"close": np.exp(lp), "volume": vol, "n_trades": (vol * 20).astype(int)}, index=idx)
