"""Turns a table copied from Investing.com (Release date / Time / Actual / Forecast / Previous)
into a surprise CSV: event_type, date_ny, actual, consensus.   Usage: python src/parse_investing.py NFP data/nfp_investing.txt
Dates are release dates; rows without a consensus (e.g. the October 2025 catch-up release) are dropped."""
import re, sys
from pathlib import Path
import pandas as pd


def val(s):
    s = s.strip().replace(",", "")
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)\s*([KMB%]?)", s)
    return float(m.group(1)) if m else float("nan")


def parse(path, label, start="2022-01-01"):
    rows = []
    for line in Path(path).read_text().splitlines()[1:]:
        p = [x for x in line.split("\t")]
        if len(p) < 4:
            continue
        m = re.match(r"([A-Za-z]{3} \d{2}, \d{4})", p[0])
        if not m:
            continue
        rows.append(dict(event_type=label, date_ny=pd.to_datetime(m.group(1), format="%b %d, %Y").strftime("%Y-%m-%d"),
                         actual=val(p[2]), consensus=val(p[3])))
    d = pd.DataFrame(rows)
    d = d[d.date_ny >= start].dropna()
    return d.drop_duplicates("date_ny").sort_values("date_ny")


if __name__ == "__main__":
    lab, path = sys.argv[1], sys.argv[2]
    out = parse(path, lab)
    out.to_csv(f"data/surprises_{lab.lower()}.csv", index=False)
    print(len(out), "rows ->", f"data/surprises_{lab.lower()}.csv")
