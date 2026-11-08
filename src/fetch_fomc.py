"""Fetches FOMC decision dates (federalreserve.gov) and Bauer-Swanson monetary policy surprises (FRBSF).

Outputs in data/:
  monetary-policy-surprises-data.xlsx   FRBSF file (2023 update: ends in late 2023)
  fomc_extra.csv                        date_ny of FOMC decisions (statement day, 14:00 New York)
  events.csv                            completed with the FOMC rows + column 'surprise' (MPS; empty outside BS)
  other_1400_dates.csv                  days with another 14:00 release (FOMC minutes): excluded from donors
Usage: python src/fetch_fomc.py [--start 2022-01-01 --end 2026-10-15]
"""
import argparse, re, urllib.request
from pathlib import Path
import pandas as pd

UA = "Mozilla/5.0 (Albert School university project)"
MONTHS = {m: i for i, m in enumerate(["January", "February", "March", "April", "May", "June", "July", "August", "September",
                                      "October", "November", "December"], 1)}
CAL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
BS = "https://www.frbsf.org/wp-content/uploads/monetary-policy-surprises-data.xlsx"


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=60) as r:
        return r.read()


def parse_calendar(html):
    """Returns (decisions, minutes releases) as lists of ISO dates."""
    txt = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "|", html))
    dec, minutes = [], []
    for ym in re.finditer(r"(\d{4}) FOMC Meetings", txt):
        year = int(ym.group(1))
        nxt = re.search(r"\d{4} FOMC Meetings", txt[ym.end():])
        block = txt[ym.end(): ym.end() + (nxt.start() if nxt else 20000)]
        # a month (or "Jan/Feb" for a meeting spanning two months), then days "27-28" or "31-1"; notation votes are skipped
        for m in re.finditer(r"\|((?:January|February|March|April|May|June|July|August|September|October|November|December)|(?:[A-Z][a-z]{2}/[A-Z][a-z]{2}))\|[ |]*(\d{1,2})(?:-(\d{1,2}))?\*?([^|]{0,40})", block):
            if "notation" in m.group(4):
                continue
            lab = m.group(1).split("/")[-1]
            mo = next(v for k, v in MONTHS.items() if k.startswith(lab[:3]))
            d1, d2 = int(m.group(2)), m.group(3)
            day = int(d2) if d2 is not None else d1
            dec.append(f"{year}-{mo:02d}-{day:02d}")
        for m in re.finditer(r"Released (\w+) (\d{1,2}), (\d{4})", block):
            minutes.append(pd.Timestamp(f"{m.group(1)} {m.group(2)}, {m.group(3)}").strftime("%Y-%m-%d"))
    return sorted(set(dec)), sorted(set(minutes))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default="2026-10-15")
    a = ap.parse_args()
    D = Path("data"); D.mkdir(exist_ok=True)
    dec, minutes = parse_calendar(get(CAL).decode("utf-8", "replace"))
    dec = [d for d in dec if a.start <= d <= a.end]
    minutes = [d for d in minutes if a.start <= d <= a.end and d not in dec]
    pd.DataFrame({"date_ny": dec}).to_csv(D / "fomc_extra.csv", index=False)
    pd.Series(minutes, name="date_ny").to_csv(D / "other_1400_dates.csv", index=False)
    (D / "monetary-policy-surprises-data.xlsx").write_bytes(get(BS))
    bs = pd.read_excel(D / "monetary-policy-surprises-data.xlsx", sheet_name="FOMC (update 2023)")
    bs = bs[["Date", "MPS", "Unscheduled"]].rename(columns={"Date": "date_ny", "MPS": "surprise"})
    bs["date_ny"] = pd.to_datetime(bs["date_ny"]).dt.strftime("%Y-%m-%d")
    f = pd.DataFrame({"date_ny": dec}).merge(bs[["date_ny", "surprise"]], on="date_ny", how="left")
    f["event_type"], f["time_ny"] = "FOMC", "14:00"
    ev = pd.read_csv(D / "events.csv", dtype={"date_ny": str})
    ev = ev[ev.event_type != "FOMC"]
    if "surprise" not in ev:
        ev["surprise"] = float("nan")
    ev = pd.concat([ev, f[["event_type", "date_ny", "time_ny", "surprise"]]], ignore_index=True).sort_values(["date_ny", "event_type"])
    ev.to_csv(D / "events.csv", index=False)
    print(f"{len(dec)} FOMC decisions ({a.start} -> {a.end}), {int(f.surprise.notna().sum())} with a Bauer-Swanson surprise (last: {bs.date_ny.max()})")
    print("minutes releases (14:00) excluded from donors:", len(minutes))
    print(f.groupby(f.date_ny.str[:4]).agg(n=("date_ny", "size"), with_surprise=("surprise", "count")).T.to_string())


if __name__ == "__main__":
    main()
