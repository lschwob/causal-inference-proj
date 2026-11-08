"""Builds the US macro event calendar.

Outputs in data/:
  events.csv              treated events (CPI, NFP, FOMC): event_type, date_ny, time_ny
  other_0830_dates.csv    other days with a release at 08:30 New York (excluded from donors)
  other_1400_dates.csv    other days with a release at 14:00 New York (excluded, FOMC)

Sources (all free):
  * FRED API  (free key: https://fred.stlouisfed.org/docs/api/api_key.html)
      export FRED_API_KEY=xxxx
    -> release dates (CPI, Employment Situation, PPI, Retail Sales, GDP,
       Personal Income and Outlays, Employment Cost Index, JOLTS, ...)
  * Bauer-Swanson surprise file (FRBSF), downloaded manually:
      https://www.frbsf.org/research-and-insights/data-and-indicators/monetary-policy-surprises/
    -> FOMC dates + monetary policy surprise (column MPS)
  * Recent FOMC dates (not covered by the BS file): add them to data/fomc_extra.csv
    (column date_ny) from https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm

WARNING: running this script overwrites data/events.csv without the FOMC rows added by src/fetch_fomc.py;
run src/fetch_fomc.py afterwards. Check the output: events per year (~12 CPI, ~12 NFP, ~8 FOMC).
"""
import os, re, sys, json, argparse
import urllib.request, urllib.parse
from pathlib import Path
import pandas as pd

FRED = "https://api.stlouisfed.org/fred"
TARGETS = {  # regex on the FRED release name -> label
    "CPI": r"^Consumer Price Index$",
    "NFP": r"^Employment Situation$",
}
OTHER_0830 = [r"Producer Price Index", r"Advance Monthly Sales for Retail", r"Gross Domestic Product",
              r"Personal Income and Outlays", r"Employment Cost Index", r"Import/Export Price",
              r"Productivity and Costs", r"Advance Report on Durable Goods", r"Housing Starts|New Residential Construction",
              r"Unemployment Insurance Weekly Claims", r"Trade|International Trade"]


def fred(endpoint, **params):
    key = os.environ.get("FRED_API_KEY") or sys.exit("Set FRED_API_KEY (free FRED key).")
    params.update(api_key=key, file_type="json")
    url = f"{FRED}/{endpoint}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def release_dates(release_id, start):
    out, offset = [], 0
    while True:
        j = fred("release/dates", release_id=release_id, realtime_start="2000-01-01",
                 include_release_dates_with_no_data="true", limit=1000, offset=offset,
                 sort_order="asc")
        out += [d["date"] for d in j["release_dates"]]
        offset += 1000
        if offset >= j["count"]:
            break
    return sorted({d for d in out if d >= start})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default="2026-10-15")
    ap.add_argument("--bs-file", default=None, help="downloaded Bauer-Swanson xlsx/csv")
    a = ap.parse_args()
    Path("data").mkdir(exist_ok=True)

    rel = fred("releases", limit=1000)["releases"]
    names = {r["id"]: r["name"] for r in rel}
    rows, other = [], set()
    for rid, name in names.items():
        for lab, pat in TARGETS.items():
            if re.search(pat, name):
                for d in release_dates(rid, a.start):
                    if d <= a.end:
                        rows.append((lab, d, "08:30"))
                print(lab, "->", name, f"(id {rid})")
        if any(re.search(p, name) for p in OTHER_0830):
            other |= {d for d in release_dates(rid, a.start) if d <= a.end}
    ev = pd.DataFrame(rows, columns=["event_type", "date_ny", "time_ny"])

    # FOMC: Bauer-Swanson file + manual additions
    fomc = []
    if a.bs_file:
        bs = pd.read_excel(a.bs_file) if a.bs_file.endswith("xlsx") else pd.read_csv(a.bs_file)
        print("BS columns:", list(bs.columns))
        dcol = next(c for c in bs.columns if re.search("date|announce", str(c), re.I))
        scol = next((c for c in bs.columns if re.fullmatch(r"MPS", str(c), re.I)), None) or \
               next(c for c in bs.columns if "MPS" in str(c).upper())
        bs = bs[[dcol, scol]].rename(columns={dcol: "date_ny", scol: "surprise"})
        bs["date_ny"] = pd.to_datetime(bs["date_ny"]).dt.strftime("%Y-%m-%d")
        bs = bs[(bs.date_ny >= a.start) & (bs.date_ny <= a.end)]
        fomc.append(bs)
    if Path("data/fomc_extra.csv").exists():
        fomc.append(pd.read_csv("data/fomc_extra.csv")[["date_ny"]])
    if fomc:
        f = pd.concat(fomc).drop_duplicates("date_ny", keep="first")
        f["event_type"], f["time_ny"] = "FOMC", "14:00"
        ev = pd.concat([ev, f], ignore_index=True)
    else:
        print("!! No FOMC: provide --bs-file and/or data/fomc_extra.csv")

    ev = ev.sort_values("date_ny").reset_index(drop=True)
    ev.to_csv("data/events.csv", index=False)
    ev_dates = set(ev.loc[ev.time_ny == "08:30", "date_ny"])
    pd.Series(sorted(other | ev_dates), name="date_ny").to_csv("data/other_0830_dates.csv", index=False)
    pd.Series([], name="date_ny", dtype=str).to_csv("data/other_1400_dates.csv", index=False)
    print(ev.groupby([ev.date_ny.str[:4], "event_type"]).size().unstack(fill_value=0))
    print("08:30 days excluded from donors:", len(other | ev_dates))


if __name__ == "__main__":
    main()
