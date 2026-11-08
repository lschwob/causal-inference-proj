"""Downloads 1-minute BTCUSDT (spot) klines from data.binance.vision
and writes data/btcusdt_1m.parquet (index = bar CLOSE time, UTC).

Usage (internet access required):
    python src/download_binance.py --start 2022-01 --end 2026-10

Note: since 2025, Binance spot files use microsecond timestamps
(milliseconds before). The script handles both.
"""
import argparse, io, zipfile, sys
from pathlib import Path
import datetime as dt
import urllib.request, urllib.error
import pandas as pd

BASE = "https://data.binance.vision/data/spot"
COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "n_trades", "taker_buy_base", "taker_buy_quote", "ignore"]


def fetch(url):
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def parse_zip(raw):
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        name = z.namelist()[0]
        with z.open(name) as f:
            df = pd.read_csv(f, header=None)
    # some recent files contain a header row
    if not str(df.iloc[0, 0]).isdigit():
        df = df.iloc[1:].reset_index(drop=True)
    df = df.iloc[:, :12]
    df.columns = COLS
    df["open_time"] = pd.to_numeric(df["open_time"])
    unit = "us" if df["open_time"].iloc[0] > 1e14 else "ms"
    df["t_open"] = pd.to_datetime(df["open_time"], unit=unit, utc=True)
    for c in ["close", "volume", "n_trades"]:
        df[c] = pd.to_numeric(df[c])
    return df[["t_open", "close", "volume", "n_trades"]]


def month_range(start, end):
    y, m = map(int, start.split("-"))
    y2, m2 = map(int, end.split("-"))
    while (y, m) <= (y2, m2):
        yield y, m
        m += 1
        if m == 13:
            y, m = y + 1, 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--start", default="2022-01")
    ap.add_argument("--end", default=dt.date.today().strftime("%Y-%m"))
    ap.add_argument("--out", default="data/btcusdt_1m.parquet")
    a = ap.parse_args()

    frames, today = [], dt.date.today()
    for y, m in month_range(a.start, a.end):
        ym = f"{y}-{m:02d}"
        complete = (y, m) < (today.year, today.month)
        if complete:
            url = f"{BASE}/monthly/klines/{a.symbol}/1m/{a.symbol}-1m-{ym}.zip"
            raw = fetch(url)
            if raw is not None:
                frames.append(parse_zip(raw)); print("OK  monthly", ym); continue
            print("monthly file missing, trying daily files:", ym)
        # current month (or monthly file missing): daily files
        d = dt.date(y, m, 1)
        while d.month == m and d < today:
            url = f"{BASE}/daily/klines/{a.symbol}/1m/{a.symbol}-1m-{d.isoformat()}.zip"
            raw = fetch(url)
            if raw is not None:
                frames.append(parse_zip(raw))
            d += dt.timedelta(days=1)
        print("OK  daily", ym)

    if not frames:
        sys.exit("No data downloaded: check the connection / the dates.")
    df = pd.concat(frames).drop_duplicates("t_open").sort_values("t_open")
    # index by the bar CLOSE time: the price "at t" is the last price before t
    df["t"] = df["t_open"] + pd.Timedelta(minutes=1)
    df = df.set_index("t")[["close", "volume", "n_trades"]]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out)
    print(f"{len(df):,} bars, {df.index.min()} -> {df.index.max()}  ->  {a.out}")
    gaps = df.index.to_series().diff().dt.total_seconds().div(60)
    print("gaps > 1 min:", int((gaps > 1).sum()), "| largest gap (min):", gaps.max())


if __name__ == "__main__":
    main()
