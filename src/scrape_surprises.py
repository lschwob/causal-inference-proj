"""Scrapes CPI and NFP actual values from public pages, without Bloomberg.

Source: public pages of the MQL5 economic calendar (history "actual / forecast / previous",
using the page's own pagination). One request per second.

WARNING: only the actual value is reliable. MQL5's "forecast" column is the site's own estimate
(e.g. NFP January 2022: -192K versus a market consensus of about +125K): it is NOT a consensus and is not used.
The consensus used in the analysis comes from Investing.com (src/parse_investing.py).
Outputs in data/: surprises_raw_<slug>.csv (New York release date, period, actual, forecast).

Usage: python src/scrape_surprises.py [--cpi consumer-price-index-mm]
"""
import argparse, html, re, time, urllib.request
from pathlib import Path
import pandas as pd

BASE = "https://www.mql5.com/en/economic-calendar/united-states"
UA = "Mozilla/5.0 (Albert School university project; reads public pages, 1 request/s)"
ITEM = re.compile(r'<div class="event-table-history__item">(.*?)</div>\s*</div>\s*(?=<div class="event-table-history__item">|</div>|$)', re.S)


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=30) as r:
        return r.read().decode("utf-8", "replace")


def num(s):
    """'165 K' -> 165.0 ; '-0.3%' -> -0.3 ; '' -> NaN (the unit is constant within a series)."""
    s = html.unescape(s).replace("\xa0", " ").strip()
    m = re.search(r"-?\d+(?:[.,]\d+)?", s)
    return float(m.group(0).replace(",", ".")) if m else float("nan")


def parse(page):
    rows = []
    for blk in re.split(r'<div class="event-table-history__item">', page)[1:]:
        ms = re.search(r'data-date="(\d+)"', blk)
        per = re.search(r'event-table-history__period">([^<]*)<', blk)
        act = re.search(r'actual__value[^>]*>\s*([^<]*?)\s*</span>', blk, re.S)
        fc = re.search(r'event-table-history__forecast">\s*([^<]*?)\s*</div>', blk, re.S)
        if not (ms and act):
            continue
        t = pd.Timestamp(int(ms.group(1)), unit="ms", tz="UTC").tz_convert("America/New_York")
        rows.append(dict(date_ny=t.strftime("%Y-%m-%d"), time_ny=t.strftime("%H:%M"), period=per.group(1).strip() if per else "",
                         actual=num(act.group(1)), forecast=num(fc.group(1)) if fc else float("nan"),
                         actual_txt=html.unescape(act.group(1)).strip(), forecast_txt=html.unescape(fc.group(1)).strip() if fc else ""))
    return rows


def scrape(slug, max_pages=12):
    out = []
    for p in range(1, max_pages + 1):
        rows = parse(get(f"{BASE}/{slug}/history?page={p}"))
        seen = {(r["date_ny"], r["period"]) for r in out}
        new = [r for r in rows if (r["date_ny"], r["period"]) not in seen]
        if not new:                                     # pagination repeats the last page beyond the end
            break
        rows = new
        out += rows
        print(f"{slug} page {p}: {len(rows)} rows, {rows[-1]['date_ny']} -> {rows[0]['date_ny']}")
        time.sleep(1.0)
    df = pd.DataFrame(out).drop_duplicates(["date_ny", "period"]).sort_values("date_ny").reset_index(drop=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nfp", default="nonfarm-payrolls")
    ap.add_argument("--cpi", default="consumer-price-index-mm")      # alternatives: consumer-price-index-ex-food-energy-mm, ...-yy
    ap.add_argument("--start", default="2022-01-01")
    a = ap.parse_args()
    D = Path("data"); D.mkdir(exist_ok=True)
    parts = []
    for lab, slug in (("NFP", a.nfp), ("CPI", a.cpi)):
        df = scrape(slug)
        df.to_csv(D / f"surprises_raw_{slug}.csv", index=False)
        d = df[df.date_ny >= a.start].copy()
        d["event_type"] = lab
        parts.append(d.rename(columns={"forecast": "consensus"})[["event_type", "date_ny", "actual", "consensus"]])
    print("Actual values saved in data/surprises_raw_*.csv. The MQL5 forecast column is NOT a market consensus: do not use it.")


if __name__ == "__main__":
    main()
