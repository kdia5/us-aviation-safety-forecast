"""DCA look-back: public ASRS reports at or near Reagan National involving helicopters,
before the 29 Jan 2025 mid-air collision.

Usage:
    python src/dca_lookback.py --asrs data/raw/asrs --out outputs
"""
import argparse
import glob
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asrs", default="data/raw/asrs")
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    parts = []
    for f in sorted(glob.glob(str(Path(a.asrs) / "*.csv"))):
        df = pd.read_csv(f, header=[0, 1], low_memory=False)
        df.columns = [f"{x.strip()}|{y}" for x, y in df.columns]
        parts.append(df)
    df = pd.concat(parts).drop_duplicates("|ACN")
    text = df["Report 1|Synopsis"].fillna("") + " " + df["Report 1|Narrative"].fillna("")
    at_dca = (df["Place|Locale Reference"].fillna("").str.contains(r"\bDCA\b")
              | text.str.contains(r"\bDCA\b|Reagan National|Washington National"))
    heli = text.str.contains(r"helicopter|rotorcraft|\bhelo\b", case=False)
    d = df[at_dca & heli].copy()
    d["date"] = d["Time|Date"].astype(str).str[:6]
    d = d[d.date < "202502"]  # before the collision (29 Jan 2025)
    out = d[["|ACN", "date", "Events|Anomaly", "Report 1|Synopsis"]].rename(
        columns={"|ACN": "ACN", "Events|Anomaly": "anomaly", "Report 1|Synopsis": "synopsis"}).sort_values("date")
    out.to_csv(Path(a.out) / "dca_helicopter_reports_pre_collision.csv", index=False)
    print(f"{len(out)} DCA helicopter-related ASRS reports before Feb 2025; "
          f"{out.anomaly.str.contains('NMAC').sum()} coded as NMAC")


if __name__ == "__main__":
    main()
