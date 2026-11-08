"""Methodological core: event windows, DiD, "synthetic day" (ridge synthetic control),
in-space placebo inference, falsification tests.

Convention: price at time t = last price before t (1-min bar indexed by its CLOSE time).
Window path: Y(tau) = 1e4 * [ln P(t0+tau) - ln P(t0)]  (bps), tau in minutes, 5-minute steps.
Treatment happens at tau = 0; tau<0 = pre-period (24 points), tau>0 = post-period (24 points).
"""
import os
import numpy as np, pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

TAUS = np.arange(-120, 121, 5)
PRE, POST = TAUS < 0, TAUS > 0
I0 = int(np.where(TAUS == 0)[0][0])
NY = "America/New_York"


# ------------------------------------------------------------------ data
def load_prices(path_or_df):
    df = pd.read_parquet(path_or_df) if isinstance(path_or_df, (str, os.PathLike)) else path_or_df.copy()
    df.index = pd.DatetimeIndex(df.index)
    df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
    full = pd.date_range(df.index.min(), df.index.max(), freq="1min")
    df = df.reindex(full)
    df["close"] = df["close"].ffill(limit=3)
    df["volume"] = df["volume"].fillna(0.0)
    return df


def to_utc(date_ny, hhmm, shift_min=0):
    return pd.Timestamp(f"{date_ny} {hhmm}", tz=NY).tz_convert("UTC") + pd.Timedelta(minutes=shift_min)


class Prices:
    def __init__(self, df):
        self.start = df.index[0]
        self.lp = np.log(df["close"].to_numpy(float))
        r = np.diff(self.lp, prepend=np.nan)
        self.cs2 = np.cumsum(np.nan_to_num(r) ** 2)
        self.cv = np.cumsum(df["volume"].to_numpy(float))
        self.n = len(self.lp)

    def windows(self, t0s):
        t0s = pd.DatetimeIndex(t0s)
        pos = np.round((t0s - self.start).total_seconds().to_numpy() / 60).astype(int)
        ok = (pos - 1440 - 130 >= 0) & (pos + TAUS.max() < self.n)
        pc = np.where(ok, pos, 1440 + 130)
        P = self.lp[pc[:, None] + TAUS[None, :]]
        path = (P - P[:, [I0]]) * 1e4
        ok &= np.isfinite(path).all(1)
        p2 = pc - 120
        rv24 = np.sqrt(np.maximum(self.cs2[p2] - self.cs2[p2 - 1440], 0)) * 1e4
        mom = (self.lp[p2] - self.lp[p2 - 1440]) * 1e4
        lvol = np.log1p(np.maximum(self.cv[pc] - self.cv[pc - 120], 0))
        cov = np.c_[np.log(rv24 + 1), mom / (rv24 + 1), lvol]
        ok &= np.isfinite(cov).all(1)
        sig = np.diff(path[:, :I0 + 1], axis=1).std(axis=1)
        sig = np.maximum(np.nan_to_num(sig), 1e-6)
        return dict(path=np.nan_to_num(path), cov=np.nan_to_num(cov), sigma=sig, ok=ok)


def donor_dates(start, end, clock, excl_dates, no_thursday):
    d = pd.bdate_range(start, end)
    d = d.difference(USFederalHolidayCalendar().holidays(start, end))
    keep = ~d.strftime("%Y-%m-%d").isin(set(excl_dates))
    if no_thursday:
        keep &= d.dayofweek != 3          # weekly jobless claims = Thursday 08:30
    return d[keep]


# ------------------------------------------------------------------ SC ridge
def _rel_solve(G, b, lam):
    n = G.shape[0]
    return np.linalg.solve(G + lam * np.trace(G) / n * np.eye(n), b)


def _proj_simplex(v):
    """Euclidean projection onto the simplex {w >= 0, sum w = 1}."""
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1.0
    k = np.arange(1, len(v) + 1)
    rho = np.nonzero(u - css / k > 0)[0][-1]
    return np.maximum(v - css[rho] / (rho + 1), 0.0)


def _simplex_ls(A, b, iters=400, w0=None):
    """Classical synthetic control: min ||A w - b||^2 s.t. w >= 0, sum w = 1 (accelerated projected gradient)."""
    J = A.shape[1]
    G, c = A.T @ A, A.T @ b
    L = max(np.linalg.eigvalsh(G)[-1], 1e-12)
    w = np.full(J, 1.0 / J) if w0 is None else w0
    y, t = w.copy(), 1.0
    for _ in range(iters):
        w_new = _proj_simplex(y - (G @ y - c) / L)
        t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
        y = w_new + ((t - 1) / t_new) * (w_new - w)
        w, t = w_new, t_new
    return w


def log_rv(path, lo, hi):
    """Log realised volatility over the 5-minute increments whose index falls in [lo, hi) (grid positions)."""
    inc = np.diff(path, axis=1)
    return 0.5 * np.log((inc[:, lo:hi] ** 2).sum(1) + 1e-9)


class EventStudy:
    """One object = one set of events (same local clock time) + its pool of donor days."""

    def __init__(self, prices, events, excl, shift_min=0, donor_window_days=180,
                 kappa=0.5, lam=1.0, pre_start=None, pre_end=None):
        self.px, self.W, self.kappa, self.lam = prices, donor_window_days, kappa, lam
        ev = events.copy().reset_index(drop=True)
        ev["t0"] = [to_utc(d, h, shift_min) for d, h in zip(ev.date_ny, ev.time_ny)]
        self.ev, self.pools = ev, {}
        self.res = {}
        lo = (pd.Timestamp(ev.date_ny.min()) - pd.Timedelta(days=self.W)).strftime("%Y-%m-%d")
        hi = (pd.Timestamp(ev.date_ny.max()) + pd.Timedelta(days=self.W)).strftime("%Y-%m-%d")
        W_ev = {}
        for clock, g in ev.groupby("time_ny"):
            ed = set(g.date_ny)
            dd = donor_dates(lo, hi, clock, set(excl.get(clock, [])) | ed, no_thursday=(clock == "08:30"))
            t0d = [to_utc(d.strftime("%Y-%m-%d"), clock, shift_min) for d in dd]
            w = self.px.windows(t0d)
            keep = w["ok"]
            dd = dd[keep]
            path, cov, sig = w["path"][keep], w["cov"][keep], w["sigma"][keep]
            mu, sd = cov.mean(0), cov.std(0) + 1e-9
            self.pools[clock] = dict(dates=dd, path=path, sig=sig, pathn=path / sig[:, None],
                                     z=(cov - mu) / sd, mu=mu, sd=sd, logrv_ratio=log_rv(path, I0, I0 + 6) - log_rv(path, I0 - 6, I0))
        w = self.px.windows(ev["t0"])
        self.ev_w = w
        self.ev["ok"] = w["ok"]
        self.ev["sigma"] = w["sigma"]
        self.ev_logrv = log_rv(w["path"], I0, I0 + 6) - log_rv(w["path"], I0 - 6, I0)

    # ---- donor selection for one event
    def donors(self, i):
        e = self.ev.iloc[i]
        P = self.pools[e.time_ny]
        d0 = pd.Timestamp(e.date_ny)
        dist = np.abs((P["dates"] - d0).days)
        return np.where((dist <= self.W) & (dist > 0))[0], P

    # ---- DiD (equal weights)
    def did(self):
        E, T = len(self.ev), len(TAUS)
        gap = np.full((E, T), np.nan)
        dlr = np.full(E, np.nan)
        nj = np.zeros(E, int)
        for i in range(E):
            if not self.ev.ok.iloc[i]:
                continue
            dj, P = self.donors(i)
            if len(dj) < 10:
                continue
            gap[i] = self.ev_w["path"][i] - P["path"][dj].mean(0)
            dlr[i] = self.ev_logrv[i] - P["logrv_ratio"][dj].mean()
            nj[i] = len(dj)
        self.res["did"] = dict(gap=gap, dlogrv=dlr, nj=nj)
        return self.res["did"]

    # ---- lambda tuned by validation on placebo days (no leakage: treated days are never used)
    def tune_lambda(self, grid=(0.01, 0.03, 0.1, 0.3, 1, 3, 10), n_pseudo=60, seed=0):
        rng = np.random.default_rng(seed)
        fit, hold = np.where(TAUS <= -45)[0], np.where((TAUS >= -40) & (TAUS < 0))[0]
        scores = {g: [] for g in grid}
        for clock, P in self.pools.items():
            J = len(P["dates"])
            for j in rng.choice(J, min(n_pseudo, J), replace=False):
                dist = np.abs((P["dates"] - P["dates"][j]).days)
                dj = np.where((dist <= self.W) & (dist > 0))[0]
                if len(dj) < 15:
                    continue
                A = np.vstack([P["pathn"][dj][:, fit].T, self.kappa * P["z"][dj].T])
                b = np.r_[P["pathn"][j, fit], self.kappa * P["z"][j]]
                G = A @ A.T
                for g in grid:
                    w = A.T @ _rel_solve(G, b, g)
                    scores[g].append(np.sqrt(np.mean((P["pathn"][j, hold] - P["pathn"][dj][:, hold].T @ w) ** 2)))
        mean = {g: float(np.mean(v)) for g, v in scores.items()}
        self.lam = min(mean, key=mean.get)
        return mean

    # ---- synthetic day (ridge SC) + in-space placebos
    def sc(self, n_placebo=100, seed=0, method="ridge"):
        """method='ridge': synthetic day (unconstrained ridge weights); 'simplex': classical SC (weights >= 0, sum 1)."""
        classic = method == "simplex"
        rng = np.random.default_rng(seed)
        E, nP = len(self.ev), POST.sum()
        near30 = (TAUS[POST] <= 30)
        gap = np.full((E, nP), np.nan)
        out = dict(rm_pre=np.full(E, np.nan), rm_post=np.full(E, np.nan), ratio=np.full(E, np.nan),
                   s30=np.full(E, np.nan), p_ratio=np.full(E, np.nan), p_s30=np.full(E, np.nan),
                   gap_bps=gap, n_donors=np.zeros(E, int), pl_gap=[], eff_n=np.full(E, np.nan), pl_s30_mean=np.full(E, np.nan))
        for i in range(E):
            if not self.ev.ok.iloc[i]:
                out["pl_gap"].append(None); continue
            dj, P = self.donors(i)
            if len(dj) < 15:
                out["pl_gap"].append(None); continue
            Dp, Dq = P["pathn"][dj][:, PRE].T, P["pathn"][dj][:, POST].T      # (n_pre,J), (n_post,J)
            Zc = self.kappa * P["z"][dj].T
            A = np.vstack([Dp, Zc])
            zev = (self.px_cov(i) - P["mu"]) / P["sd"]
            yn = self.ev_w["path"][i] / self.ev["sigma"].iloc[i]
            b = np.r_[yn[PRE], self.kappa * zev]
            G = A @ A.T
            w = _simplex_ls(A, b) if classic else A.T @ _rel_solve(G, b, self.lam)
            res_pre = yn[PRE] - Dp @ w
            gn = yn[POST] - Dq @ w
            rm_pre, rm_post = np.sqrt((res_pre ** 2).mean()), np.sqrt((gn ** 2).mean())
            s30 = np.abs(gn[near30]).mean()
            out["gap_bps"][i] = gn * self.ev["sigma"].iloc[i]
            out["rm_pre"][i], out["rm_post"][i], out["ratio"][i], out["s30"][i] = rm_pre, rm_post, rm_post / (rm_pre + 1e-9), s30
            out["n_donors"][i] = len(dj)
            out["eff_n"][i] = (np.abs(w).sum() ** 2) / ((w ** 2).sum() + 1e-12)
            # placebos: each sampled donor day becomes a fake treated unit
            M = min(n_placebo, len(dj))
            pj = rng.choice(len(dj), M, replace=False)
            rat, ss, pg = [], [], []
            for j in pj:
                a = A[:, j]
                if classic:
                    wj = np.zeros(A.shape[1])
                    keep = np.arange(A.shape[1]) != j
                    wj[keep] = _simplex_ls(A[:, keep], a)
                else:
                    alpha = _rel_solve(G - np.outer(a, a), a, self.lam)
                    wj = A.T @ alpha
                    wj[j] = 0.0
                rp = Dp[:, j] - Dp @ wj
                gj = Dq[:, j] - Dq @ wj
                r_pre, r_post = np.sqrt((rp ** 2).mean()), np.sqrt((gj ** 2).mean())
                rat.append(r_post / (r_pre + 1e-9)); ss.append(np.abs(gj[near30]).mean()); pg.append(gj)
            rat, ss = np.array(rat), np.array(ss)
            out["pl_s30_mean"][i] = ss.mean()
            out["p_ratio"][i] = (1 + (rat >= out["ratio"][i]).sum()) / (1 + M)
            out["p_s30"][i] = (1 + (ss >= s30).sum()) / (1 + M)
            out["pl_gap"].append(np.array(pg))
        self.res["sc_classic" if classic else "sc"] = out
        return out

    def sc_example(self, i):
        """Normalised paths (by sigma_pre) of the treated unit and its synthetic day, for plots."""
        dj, P = self.donors(i)
        A = np.vstack([P["pathn"][dj][:, PRE].T, self.kappa * P["z"][dj].T])
        zev = (self.px_cov(i) - P["mu"]) / P["sd"]
        yn = self.ev_w["path"][i] / self.ev["sigma"].iloc[i]
        w = A.T @ _rel_solve(A @ A.T, np.r_[yn[PRE], self.kappa * zev], self.lam)
        synth = P["pathn"][dj].T @ w
        return dict(taus=TAUS, y=yn, synth=synth, weights=w, donor_dates=P["dates"][dj])

    def px_cov(self, i):
        return self.ev_w["cov"][i]


# ------------------------------------------------------------------ aggregation / inference
def att_table(gap, taus=TAUS, horizons=(5, 15, 30, 60, 120), signed_by=None):
    """Mean ATT (bps) and 95% CI by horizon (standard errors across events)."""
    rows = []
    for h in horizons:
        k = np.where(taus == h)[0][0] if gap.shape[1] == len(taus) else np.where(taus[taus > 0] == h)[0][0]
        x = gap[:, k]; x = x[np.isfinite(x)]
        if signed_by is not None:
            x = x * signed_by[:len(x)]
        m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
        rows.append(dict(horizon_min=h, n_events=len(x), att_bps=m, se=se, ci_low=m - 1.96 * se, ci_high=m + 1.96 * se))
    return pd.DataFrame(rows)


def pretrend_test(gap, n_flip=5000, seed=0):
    """Pre-trend test: (i) joint Wald (FRAGILE: 24 highly correlated points, fat tails -> often anti-conservative)
    and (ii) max-|t| test by sign flipping (Rademacher), which gives a more reliable joint p-value."""
    from scipy import stats
    g = gap[:, PRE]; g = g[np.isfinite(g).all(1)]
    E, k = g.shape
    beta = g.mean(0)
    S = np.cov(g.T) / E
    W = float(beta @ np.linalg.pinv(S) @ beta)
    t_obs = np.abs(beta / np.sqrt(np.diag(S)))
    rng = np.random.default_rng(seed)
    sg = rng.choice([-1.0, 1.0], size=(n_flip, E))
    mu = sg @ g / E
    sd = np.sqrt(np.maximum((g ** 2).mean(0) - mu ** 2, 1e-12) * E / (E - 1))
    tmax = np.abs(mu / (sd / np.sqrt(E))).max(1)
    return dict(wald=W, df=k, p_wald=float(1 - stats.chi2.cdf(W, k)), max_abs_t=float(t_obs.max()),
                p_maxt_signflip=float((1 + (tmax >= t_obs.max()).sum()) / (n_flip + 1)), n_events=E)


def directional(gap_h, z, n_perm=5000, seed=0, groups=None):
    """Slope (bps per std. dev. of surprise) + Fisher permutation p-value (within type if groups is given)."""
    import statsmodels.api as sm
    m = np.isfinite(gap_h) & np.isfinite(z)
    y, x = gap_h[m], z[m]
    fit = sm.OLS(y, sm.add_constant(x)).fit(cov_type="HC1")
    rng = np.random.default_rng(seed)
    g = None if groups is None else np.asarray(groups)[m]
    obs = np.cov(x, y)[0, 1] / np.var(x, ddof=1)
    cnt = 0
    for _ in range(n_perm):
        xp = x.copy()
        if g is None:
            rng.shuffle(xp)
        else:
            for u in np.unique(g):
                idx = np.where(g == u)[0]; xp[idx] = rng.permutation(x[idx])
        s = np.cov(xp, y)[0, 1] / np.var(xp, ddof=1)
        cnt += abs(s) >= abs(obs)
    return dict(slope_bps_per_sd=float(fit.params[1]), se_hc1=float(fit.bse[1]), p_hc1=float(fit.pvalues[1]),
                p_perm=(cnt + 1) / (n_perm + 1), n=int(m.sum()))


def placebo_did(pool, W, E, n_draw=500, seed=0):
    """DiD inference calibrated by in-space placebos: draw E donor days as fake events,
    compute their gap versus the other donors (same ±W-day window, hence the same donor sharing as the real
    events), and repeat n_draw times. Gives the NULL distribution of (i) the mean gap by tau, (ii) the pre-period max|t|.
    Corrects the naive standard error, which is too small because neighbouring events share donors."""
    dates, path = pool["dates"], pool["path"]
    J = len(dates)
    d = np.asarray(dates.values.astype("datetime64[D]").astype(int))
    A = (np.abs(d[:, None] - d[None, :]) <= W) & ~np.eye(J, dtype=bool)
    rng = np.random.default_rng(seed)
    E = min(E, J // 2)
    means, tmax = np.empty((n_draw, path.shape[1])), np.empty(n_draw)
    for b in range(n_draw):
        S = rng.choice(J, E, replace=False)
        s = np.zeros(J, bool); s[S] = True
        M = A[S][:, ~s].astype(float); cnt = M.sum(1); ok = cnt >= 10
        gap = path[S][ok] - (M[ok] @ path[~s]) / cnt[ok, None]
        g = gap[:, PRE]
        tt = g.mean(0) / (g.std(0, ddof=1) / np.sqrt(len(g)))
        means[b] = gap.mean(0); tmax[b] = np.abs(tt).max()
    return means, tmax


# ------------------------------------------------------------------ hour-by-hour profile
def hourly_paths(prices, t0s, offsets_min):
    """Paths 1e4*[ln P(t0+o) - ln P(t0)] (bps) for arbitrary offsets, e.g. -24 h to +3 h (outside the 5-minute grid)."""
    t0s = pd.DatetimeIndex(t0s)
    pos = np.round((t0s - prices.start).total_seconds().to_numpy() / 60).astype(int)
    o = np.asarray(offsets_min)
    ok = (pos + o.min() >= 0) & (pos + o.max() < prices.n)
    pc = np.where(ok, pos, -o.min())
    P = prices.lp[pc[:, None] + o[None, :]]
    path = (P - prices.lp[pc][:, None]) * 1e4
    ok &= np.isfinite(path).all(1)
    return np.nan_to_num(path), ok


def hourly_profile(es, clock, offsets_min=np.arange(-1440, 181, 60)):
    """Treated minus donors gap (same pool, same ±W-day windows as the DiD) at hourly offsets.
    Returns gap (E, K), the pool paths and the pool validity mask (for placebo calibration)."""
    P = es.pools[clock]
    pool_path, okp = hourly_paths(es.px, [to_utc(d.strftime("%Y-%m-%d"), clock) for d in P["dates"]], offsets_min)
    ev_path, oke = hourly_paths(es.px, es.ev["t0"], offsets_min)
    gap = np.full((len(es.ev), len(offsets_min)), np.nan)
    for i in range(len(es.ev)):
        if not (es.ev.ok.iloc[i] and oke[i]):
            continue
        dj, _ = es.donors(i)
        dj = dj[okp[dj]]
        if len(dj) >= 10:
            gap[i] = ev_path[i] - pool_path[dj].mean(0)
    return gap, pool_path, okp


def placebo_profile(dates, path, W, E, n_draw=500, seed=0):
    """Null distribution of the mean gap (fake events drawn among donors), for paths of any shape."""
    J = len(dates)
    d = np.asarray(pd.DatetimeIndex(dates).values.astype("datetime64[D]").astype(int))
    A = (np.abs(d[:, None] - d[None, :]) <= W) & ~np.eye(J, dtype=bool)
    rng = np.random.default_rng(seed)
    E = min(E, J // 2)
    out = np.empty((n_draw, path.shape[1]))
    for b in range(n_draw):
        S = rng.choice(J, E, replace=False)
        s = np.zeros(J, bool); s[S] = True
        M = A[S][:, ~s].astype(float); cnt = M.sum(1); ok = cnt >= 10
        out[b] = (path[S][ok] - (M[ok] @ path[~s]) / cnt[ok, None]).mean(0)
    return out
