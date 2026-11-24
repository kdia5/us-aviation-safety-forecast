"""Extract the NTSB aviation database (avall.mdb) into compact CSV files.

Requires `mdbtools` (macOS: `brew install mdbtools`).

Usage:
    python src/extract.py --zip data/raw/avall.zip --out data/interim
"""
import argparse
import io
import subprocess
import tempfile
import zipfile
from pathlib import Path

import pandas as pd

# Only the columns the pipeline uses, to keep interim files small.
TABLES = {
    "events": ["ev_id", "ntsb_no", "ev_type", "ev_date", "ev_year", "ev_month",
               "ev_state", "ev_country", "light_cond", "wx_cond_basic",
               "ev_highest_injury", "apt_name", "ev_nr_apt_id"],
    "Events_Sequence": ["ev_id", "Aircraft_Key", "Occurrence_No", "Occurrence_Code",
                        "Occurrence_Description", "phase_no", "eventsoe_no", "Defining_ev"],
    "Findings": ["ev_id", "Aircraft_Key", "finding_no", "finding_code",
                 "finding_description", "category_no", "Cause_Factor"],
    "aircraft": ["ev_id", "Aircraft_Key", "far_part", "acft_category", "damage",
                 "acft_make", "acft_model"],
    "narratives": ["ev_id", "Aircraft_Key", "narr_accp"],
}


def export_table(mdb: Path, table: str) -> pd.DataFrame:
    raw = subprocess.run(["mdb-export", str(mdb), table], check=True,
                         capture_output=True).stdout
    df = pd.read_csv(io.BytesIO(raw), low_memory=False)
    keep = [c for c in TABLES[table] if c in df.columns]
    return df[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="data/raw/avall.zip")
    ap.add_argument("--out", default="data/interim")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(args.zip) as z:
            z.extract("avall.mdb", tmp)
        mdb = Path(tmp) / "avall.mdb"
        for table in TABLES:
            df = export_table(mdb, table)
            df.to_csv(out / f"{table}.csv.gz", index=False, compression="gzip")
            print(f"{table:16s} {len(df):>8,d} rows")


if __name__ == "__main__":
    main()
