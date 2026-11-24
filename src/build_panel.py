"""Build a quarterly panel: US Census division x ICAO high-risk category x quarter.

Each row holds the number of NTSB cases in that category (the forecast target)
and precursor features computed ONLY from information available two quarters
earlier. NTSB coding, findings and narratives are added after investigation,
so a two-quarter reporting lag keeps the evaluation free of look-ahead leakage.

Usage:
    python src/build_panel.py --interim data/interim --out data/processed
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

from nlp import keyword_flags, scrub_negations

# ICAO / CAST-ICAO Common Taxonomy high-risk categories, matched on NTSB event-sequence text.
CATEGORIES = {
    "MAC": r"Midair collision|Near midair",
    "RI": r"Runway incursion",
    "RE": r"Runway excursion",
    "LOC-I": r"Loss of control in flight",
    "CFIT": r"Controlled flight into terr",
}

# Non-defining occurrence types used as precursor signals.
PRECURSOR_OCC = {
    "occ_engine_power": r"Loss of engine power",
    "occ_fuel": r"Fuel (?:exhaustion|starvation|related|contamination)",
    "occ_vfr_into_imc": r"VFR encounter with IMC",
    "occ_abnormal_runway_contact": r"Abnormal runway contact|Hard landing",
    "occ_separation_tcas": r"Near midair|loss of separation|TCAS",
    "occ_wind_weather": r"Windshear|thunderstorm|Turbulence|Dropping|Wind",
    "occ_system_failure": r"Sys/Comp malf/fail|structural failure",
    "occ_ground_collision": r"Collision during takeoff/land|Ground collision",
}

DIVISION = {
    "New England": "CT ME MA NH RI VT",
    "Middle Atlantic": "NJ NY PA",
    "East North Central": "IL IN MI OH WI",
    "West North Central": "IA KS MN MO NE ND SD",
    "South Atlantic": "DE DC FL GA MD NC SC VA WV",
    "East South Central": "AL KY MS TN",
    "West South Central": "AR LA OK TX",
    "Mountain": "AZ CO ID MT NV NM UT WY",
    "Pacific": "AK CA HI OR WA",
}
STATE_TO_DIV = {s: d for d, states in DIVISION.items() for s in states.split()}

FEATURE_LAG = 2          # quarters between data availability and the forecast target
TOPIC_FIT_END = 2017     # NMF topics are fit on narratives up to this year only
N_TOPICS = 8


def quarter_index(year, month):
    return (year.astype(int) - 2008) * 4 + (month.astype(int) - 1) // 3


def load(interim: Path):
    ev = pd.read_csv(interim / "events.csv.gz", low_memory=False)
    ev = ev[(ev.ev_country == "USA") & ev.ev_state.isin(STATE_TO_DIV)].copy()
    d = pd.to_datetime(ev.ev_date, format="%m/%d/%y %H:%M:%S", errors="coerce")
    ev = ev[d.notna()].copy()
    ev["year"], ev["month"] = d[d.notna()].dt.year, d[d.notna()].dt.month
    ev["q"] = quarter_index(ev.year, ev.month)
    ev["division"] = ev.ev_state.map(STATE_TO_DIV)
    seq = pd.read_csv(interim / "Events_Sequence.csv.gz", low_memory=False)
    fnd = pd.read_csv(interim / "Findings.csv.gz", dtype=str)
    acft = pd.read_csv(interim / "aircraft.csv.gz", low_memory=False)
    nar = pd.read_csv(interim / "narratives.csv.gz", low_memory=False)
    return ev, seq, fnd, acft, nar


def event_features(ev, seq, fnd, acft, nar):
    """One row per event: category flags + precursor attributes."""
    desc = seq.Occurrence_Description.fillna("")
    for c, pat in CATEGORIES.items():
        seq[c] = desc.str.contains(pat, regex=True)
    for c, pat in PRECURSOR_OCC.items():
        seq[c] = desc.str.contains(pat, regex=True, case=False)
    occ = seq.groupby("ev_id")[list(CATEGORIES) + list(PRECURSOR_OCC)].any().astype(int)

    fnd["cat"] = fnd.category_no.map({"1": "fnd_aircraft", "2": "fnd_personnel",
                                      "3": "fnd_environment", "4": "fnd_organizational"})
    fsh = pd.crosstab(fnd.ev_id, fnd.cat).clip(upper=1)

    a1 = acft.sort_values("Aircraft_Key").groupby("ev_id").first()
    acf = pd.DataFrame({
        "ac_part121_135": a1.far_part.isin(["121", "135"]).astype(int),
        "ac_helicopter": (a1.acft_category == "HELI").astype(int),
    })

    n1 = nar.sort_values("Aircraft_Key").groupby("ev_id").narr_accp.first().fillna("")
    kw = pd.DataFrame(keyword_flags(n1.tolist()), index=n1.index)

    e = ev.set_index("ev_id")
    base = pd.DataFrame({
        "imc": (e.wx_cond_basic == "IMC").astype(int),
        "night": e.light_cond.isin(["NITE", "NDRK", "NBRT", "DUSK", "DAWN"]).astype(int),
        "fatal": (e.ev_highest_injury == "FATL").astype(int),
    }, index=e.index)
    feat = base.join([occ, fsh, acf, kw], how="left").fillna(0)
    feat[["q", "year", "division"]] = e[["q", "year", "division"]]
    return feat, n1


def topic_features(feat, narratives):
    """Negation-aware TF-IDF + NMF topics; fit on early years only (no leakage)."""
    text = narratives.reindex(feat.index).fillna("").map(scrub_negations)
    fit_mask = (feat.year <= TOPIC_FIT_END) & (text.str.len() > 50)
    vec = TfidfVectorizer(max_features=4000, stop_words="english", min_df=20, max_df=0.5,
                          ngram_range=(1, 2))
    X_fit = vec.fit_transform(text[fit_mask])
    nmf = NMF(n_components=N_TOPICS, random_state=0, init="nndsvda", max_iter=400).fit(X_fit)
    W = nmf.transform(vec.transform(text))
    terms = np.array(vec.get_feature_names_out())
    labels = {f"topic_{i}": ", ".join(terms[np.argsort(c)[-6:][::-1]])
              for i, c in enumerate(nmf.components_)}
    return pd.DataFrame(W, index=feat.index, columns=list(labels)), labels


def load_asrs(raw_dir: Path):
    """NASA ASRS near mid-air, airborne conflict and runway incursion reports.

    Returns one row per report with quarter index, division (None when the report
    has no single state) and 0/1 flags. ASRS exports have a two-row header."""
    files = sorted(raw_dir.glob("ASRS_conflict_incursion_*.csv"))
    if not files:
        return None
    parts = []
    for f in files:
        df = pd.read_csv(f, header=[0, 1], low_memory=False)
        df.columns = [f"{a.strip()}|{b}" for a, b in df.columns]
        parts.append(df)
    df = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["|ACN"])
    ym = df["Time|Date"].astype(str).str[:6]
    df = df[ym.str.fullmatch(r"\d{6}")].copy()
    ym = ym[df.index]
    year, month = ym.str[:4].astype(int), ym.str[4:6].astype(int)
    an = df["Events|Anomaly"].fillna("")
    text = (df["Report 1|Synopsis"].fillna("") + " " + df["Report 1|Narrative"].fillna("")).map(scrub_negations)
    out = pd.DataFrame({
        "q": quarter_index(year, month).values,
        "division": df["Place|State Reference"].map(STATE_TO_DIV).values,
        "asrs_nmac": an.str.contains("NMAC").astype(int).values,
        "asrs_airborne_conflict": an.str.contains("Airborne Conflict").astype(int).values,
        "asrs_runway_incursion": an.str.contains("Ground Incursion Runway").astype(int).values,
        "asrs_helicopter": text.str.contains(r"\bhelicopter|\brotorcraft", case=False).astype(int).values,
        "asrs_tcas_ra": text.str.contains(r"\bTCAS\b.{0,40}\bRA\b|resolution advisory", case=False).astype(int).values,
    })
    return out[out.q >= 0]


def asrs_features(asrs, divs, quarters):
    """Division- and national-level ASRS counts, lagged and smoothed like the other precursors."""
    cols = [c for c in asrs.columns if c.startswith("asrs_")]
    grid = pd.MultiIndex.from_product([divs, quarters], names=["division", "q"])
    div = asrs.dropna(subset=["division"]).groupby(["division", "q"])[cols].sum().reindex(grid, fill_value=0)
    div_l = div.groupby(level=0).transform(lambda s: s.shift(FEATURE_LAG).rolling(4, min_periods=1).mean())
    nat = asrs.groupby("q")[cols].sum().reindex(quarters, fill_value=0)
    nat_l = nat.shift(FEATURE_LAG).rolling(4, min_periods=1).mean().add_prefix("nat_")
    # Growth: latest available quarter relative to the prior two years (captures rising reports).
    growth = (nat.shift(FEATURE_LAG) / nat.shift(FEATURE_LAG + 1).rolling(8, min_periods=4).mean()).add_suffix("_growth")
    nat_all = pd.concat([nat_l, growth.add_prefix("nat_")], axis=1).fillna(0)
    out = div_l.reset_index().merge(nat_all, left_on="q", right_index=True, how="left")
    return out


def build_panel(feat, horizon=FEATURE_LAG, asrs=None):
    """Panel through the last complete quarter, plus `horizon` future quarters (y = NaN)."""
    cats = list(CATEGORIES)
    divs = sorted(DIVISION)
    last_full_q = int(feat.q.max()) - 1          # the newest quarter is still incomplete
    feat = feat[feat.q <= last_full_q]
    quarters = range(0, last_full_q + horizon + 1)

    # Division x quarter context (all events), then trailing 4-quarter means, lagged.
    ctx_cols = [c for c in feat.columns if c not in cats + ["q", "year", "division"]]
    g = feat.groupby(["division", "q"])
    ctx = g[ctx_cols].mean()
    ctx["n_events"] = g.size()
    grid = pd.MultiIndex.from_product([divs, quarters], names=["division", "q"])
    ctx = ctx.reindex(grid)
    ctx["n_events"] = ctx.n_events.fillna(0)
    ctx = ctx.groupby(level=0).transform(lambda s: s.ffill().fillna(0))
    lagged = (ctx.groupby(level=0).transform(lambda s: s.shift(FEATURE_LAG).rolling(4, min_periods=1).mean())
              .add_prefix("ctx_"))
    lagged["ctx_n_events_lag1"] = ctx.n_events.groupby(level=0).shift(1)  # event counts are known fast

    # Target counts on a full grid (zero-filled), then own-history features.
    cnt = feat.groupby(["division", "q"])[cats].sum().reindex(grid, fill_value=0)
    long = cnt.stack().rename("y").reset_index().rename(columns={"level_2": "category"})
    long = long.sort_values(["division", "category", "q"])
    gy = long.groupby(["division", "category"]).y
    for L in (2, 3, 4, 5):
        long[f"y_lag{L}"] = gy.shift(L)
    long["y_ma4"] = long[[f"y_lag{L}" for L in (2, 3, 4, 5)]].mean(axis=1)
    long["y_ewm"] = gy.transform(lambda s: s.shift(FEATURE_LAG).ewm(alpha=0.3).mean())
    long["y_ma8"] = gy.transform(lambda s: s.shift(FEATURE_LAG).rolling(8, min_periods=4).mean())
    nat = long.groupby(["category", "q"]).y.sum().rename("nat_y").reset_index()
    nat["nat_ma4_lag2"] = nat.groupby("category").nat_y.transform(lambda s: s.shift(2).rolling(4).mean())
    long = long.merge(nat[["category", "q", "nat_ma4_lag2"]], on=["category", "q"], how="left")
    long["qtr"] = long.q % 4 + 1
    long["sin_q"], long["cos_q"] = np.sin(2 * np.pi * long.qtr / 4), np.cos(2 * np.pi * long.qtr / 4)
    long["year"] = 2008 + long.q // 4
    long = long.merge(lagged.reset_index(), on=["division", "q"], how="left")
    if asrs is not None:
        long = long.merge(asrs_features(asrs, divs, quarters), on=["division", "q"], how="left")
    long = long[long.q >= 8].reset_index(drop=True)  # need two years of history
    long["is_future"] = long.q > last_full_q
    long.loc[long.is_future, "y"] = np.nan
    return long


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interim", default="data/interim")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--asrs", default="data/raw/asrs")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ev, seq, fnd, acft, nar = load(Path(args.interim))
    feat, narratives = event_features(ev, seq, fnd, acft, nar)
    topics, labels = topic_features(feat, narratives)
    feat = feat.join(topics)
    asrs = load_asrs(Path(args.asrs))
    if asrs is not None:
        print(f"ASRS reports loaded: {len(asrs):,}")
    panel = build_panel(feat, asrs=asrs)

    panel.to_csv(out / "panel.csv.gz", index=False, compression="gzip")
    pd.Series(labels, name="top_terms").to_csv(out / "topics.csv")
    hist = panel[~panel.is_future]
    print(f"panel rows {len(panel):,} (history {len(hist):,})  last complete quarter "
          f"{2008 + hist.q.max() // 4}Q{hist.q.max() % 4 + 1}, future quarters "
          f"{sorted(panel[panel.is_future].q.unique())}")
    print(hist.groupby("category").y.agg(["sum", "mean"]).round(2).to_string())
    for k, v in labels.items():
        print(k, "|", v)


if __name__ == "__main__":
    main()
