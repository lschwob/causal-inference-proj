"""Approximate CPI surprise: actual minus the Cleveland Fed nowcast (last value before the release).

The Cleveland nowcast is public and archived day by day, but it is a MODEL, not the economists' consensus
(nor Bloomberg's). Present it as an exploratory proxy only. There is no public equivalent for NFP.
Not used in the main analysis (which uses the Investing.com consensus).

Output: data/surprises_cpi_nowcast.csv  (event_type, date_ny, actual, consensus)  [consensus = nowcast]
Usage: python src/cpi_nowcast_surprise.py [--series "CPI Inflation"|"Core CPI Inflation"]
"""
import argparse, json, re, urllib.request
from pathlib import Path
import pandas as pd

URL = "https://www.clevelandfed.org/-/media/files/webcharts/inflationnowcasting/nowcast_month.json"
UA = "Mozilla/5.0 (Albert School university project)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", default="CPI Inflation")
    ap.add_argument("--events", default="data/events.csv")
    a = ap.parse_args()
    with urllib.request.urlopen(urllib.request.Request(URL, headers={"User-Agent": UA}), timeout=120) as r:
        charts = json.load(r)
    rows = []
    for c in charts:
        y, m = map(int, c["chart"]["subcaption"].split("-"))          # target month
        labels = [x["label"] for x in c["categories"][0]["category"]]
        ds = {d["seriesname"]: d["data"] for d in c["dataset"]}
        if a.series not in ds:
            continue
        vals_all = [float(x["value"]) if x.get("value") not in (None, "") else float("nan") for x in ds[a.series]]
        # mm/dd dates: the nowcast starts before the target month; the year increments when going from 12 to 01
        pairs = [(l, v) for l, v in zip(labels, vals_all) if re.fullmatch(r"\d\d/\d\d", l)]
        if not pairs:
            continue
        yr, prev, dates, vals = (y - 1 if int(pairs[0][0][:2]) > m else y), None, [], []
        for lab, v in pairs:
            mm = int(lab[:2])
            if prev is not None and mm < prev:
                yr += 1
            prev = mm
            dates.append(pd.Timestamp(year=yr, month=mm, day=int(lab[3:5]))); vals.append(v)
        act = [x.get("value") for x in ds.get("Actual " + a.series, [])]
        act = next((float(v) for v in reversed(act) if v not in (None, "")), float("nan"))
        rows.append(dict(target=f"{y}-{m:02d}", dates=dates, vals=vals, actual=act))
    ev = pd.read_csv(a.events, dtype=str)
    ev = ev[ev.event_type == "CPI"]
    out = []
    for d in ev.date_ny:
        rel = pd.Timestamp(d)
        # target month = month before the release
        t = (rel.replace(day=1) - pd.Timedelta(days=1)).strftime("%Y-%m")
        r = next((x for x in rows if x["target"] == t), None)
        if r is None:
            continue
        prior = [(dt, v) for dt, v in zip(r["dates"], r["vals"]) if dt < rel and v == v]
        if prior and r["actual"] == r["actual"]:
            out.append(dict(event_type="CPI", date_ny=d, actual=round(r["actual"], 3), consensus=round(prior[-1][1], 3)))
    df = pd.DataFrame(out)
    Path("data").mkdir(exist_ok=True)
    df.to_csv("data/surprises_cpi_nowcast.csv", index=False)
    df["surprise"] = df.actual - df.consensus
    print(df.head(8).to_string(index=False)); print(len(df), "CPI events; surprise: mean", round(df.surprise.mean(), 3), "std. dev.", round(df.surprise.std(), 3))


if __name__ == "__main__":
    main()
