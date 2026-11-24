"""Walk-forward forecasting of high-risk aviation occurrence counts, benchmarked
against simple baselines, with 95% intervals, spike alerts and feature importance.

Each test year is predicted by models trained only on earlier years.

Usage:
    python src/model.py --panel data/processed/panel.csv.gz --out outputs
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_poisson_deviance
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TEST_YEARS = range(2018, 2027)
SEED = 0


def feature_matrix(df):
    num = [c for c in df.columns if c.startswith(("y_", "ctx_", "nat_", "asrs_"))] + ["sin_q", "cos_q"]
    X = pd.concat([df[num].fillna(0),
                   pd.get_dummies(df.category, prefix="cat", dtype=float),
                   pd.get_dummies(df.division, prefix="div", dtype=float)], axis=1)
    return X


def make_models():
    return {
        "poisson_glm": make_pipeline(StandardScaler(), PoissonRegressor(alpha=0.3, max_iter=1000)),
        "gbm_poisson": HistGradientBoostingRegressor(loss="poisson", max_depth=3, learning_rate=0.05,
                                                     max_iter=300, l2_regularization=1.0,
                                                     random_state=SEED),
        "random_forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=20, n_jobs=-1,
                                               random_state=SEED),
    }


def fit_predict(train, test, X_cols):
    Xtr, Xte = train[X_cols], test[X_cols]
    preds = {}
    for name, m in make_models().items():
        m.fit(Xtr, train.y)
        preds[name] = np.clip(m.predict(Xte), 1e-3, None)
    preds["ensemble"] = np.mean([preds[k] for k in ("poisson_glm", "gbm_poisson", "random_forest")], axis=0)
    return preds


def nb_interval(mu, alpha, level=0.95):
    """Central interval of a negative binomial with mean mu and dispersion alpha
    (alpha = 0 falls back to Poisson)."""
    mu, alpha = np.asarray(mu, float), np.asarray(alpha, float)
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    lo, hi = poisson.ppf(lo_q, mu), poisson.ppf(hi_q, mu)
    m = alpha > 1e-6
    n = 1.0 / alpha[m]
    p = n / (n + mu[m])
    lo[m], hi[m] = nbinom.ppf(lo_q, n, p), nbinom.ppf(hi_q, n, p)
    return lo, hi


def metrics(y, p):
    p = np.clip(np.asarray(p, float), 1e-3, None)
    return {"MAE": float(np.mean(np.abs(y - p))),
            "RMSE": float(np.sqrt(np.mean((y - p) ** 2))),
            "PoissonDev": float(mean_poisson_deviance(y, p))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/processed/panel.csv.gz")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(args.panel)
    X = feature_matrix(panel)
    X_cols = list(X.columns)
    panel = pd.concat([panel, X.drop(columns=[c for c in X.columns if c in panel.columns])], axis=1)
    hist, future = panel[~panel.is_future].copy(), panel[panel.is_future].copy()

    # ---- Walk-forward backtest -------------------------------------------
    # Ablation: the same models without any precursor (ctx_*) features, to test
    # whether precursors add value beyond each series' own history.
    is_asrs = lambda c: "asrs" in c
    variants = {
        "ensemble_history_only": [c for c in X_cols if not c.startswith("ctx_") and not is_asrs(c)],
        "ensemble_history_ntsb": [c for c in X_cols if not is_asrs(c)],
        "ensemble_history_asrs": [c for c in X_cols if not c.startswith("ctx_")],
    }
    rows = []
    for yr in TEST_YEARS:
        train, test = hist[hist.year < yr], hist[hist.year == yr].copy()
        if test.empty:
            continue
        for k, v in fit_predict(train, test, X_cols).items():
            test[f"pred_{k}"] = v
        for name, cols in variants.items():
            test[f"pred_{name}"] = fit_predict(train, test, cols)["ensemble"]
        rows.append(test)
    bt = pd.concat(rows)
    # Primary model = history + ASRS near-miss precursors (best in the v2 ablation);
    # the all-features ensemble is kept for comparison.
    bt["pred_ensemble_all_features"] = bt.pred_ensemble
    bt["pred_ensemble"] = bt.pred_ensemble_history_asrs
    bt = bt.drop(columns=["pred_ensemble_history_asrs"])
    bt["pred_baseline_ma4"] = bt.y_ma4.fillna(0)
    bt["pred_baseline_seasonal"] = bt.y_lag4.fillna(0)
    bt["pred_baseline_ma8"] = bt.y_ma8.fillna(bt.y_ma4).fillna(0)
    pred_cols = [c for c in bt.columns if c.startswith("pred_")]

    # National level = sum over the nine divisions.
    nat = bt.groupby(["category", "q", "year", "qtr"])[["y"] + pred_cols].sum().reset_index()
    # Overdispersion: for each test year, estimate a negative binomial alpha from
    # residuals of EARLIER backtest years only (Poisson when none are available yet).
    nat["nb_alpha"] = 0.0
    for cat, g in nat.groupby("category"):
        for yr in g.year.unique():
            past = g[g.year < yr]
            if len(past) >= 4:
                mu = past.pred_ensemble
                a = ((past.y - mu) ** 2 - mu).mean() / (mu ** 2).mean()
                nat.loc[(nat.category == cat) & (nat.year == yr), "nb_alpha"] = max(a, 0.0)
    lo, hi = nb_interval(nat.pred_ensemble.values, nat.nb_alpha.values)
    nat["lower95"], nat["upper95"] = lo, hi
    nat["in_interval"] = (nat.y >= lo) & (nat.y <= hi)
    nat["alert_spike"] = nat.y > hi

    met = []
    for level, df in (("division", bt), ("national", nat)):
        for cat, g in list(df.groupby("category")) + [("ALL", df)]:
            for c in pred_cols:
                met.append({"level": level, "category": cat, "model": c.replace("pred_", ""),
                            **metrics(g.y.values, g[c].values)})
    met = pd.DataFrame(met)
    base = met[met.model == "baseline_ma4"].set_index(["level", "category"]).RMSE
    met["RMSE_skill_vs_MA4_pct"] = [100 * (1 - r.RMSE / base[(r.level, r.category)]) for r in met.itertuples()]

    # ---- Feature importance (permutation, last training window) ----------
    train = hist[hist.year < max(TEST_YEARS)]
    test = hist[hist.year == max(TEST_YEARS)]
    primary_cols = variants["ensemble_history_asrs"]
    gbm = make_models()["gbm_poisson"].fit(train[primary_cols], train.y)
    pi = permutation_importance(gbm, test[primary_cols], test.y, n_repeats=10, random_state=SEED,
                                scoring="neg_mean_poisson_deviance")
    imp = pd.DataFrame({"feature": primary_cols, "importance": pi.importances_mean}).sort_values(
        "importance", ascending=False)

    # ---- Forward forecast for the next two quarters -----------------------
    fut_pred = fit_predict(hist, future, primary_cols)
    future["forecast"] = fut_pred["ensemble"]
    fnat = future.groupby(["category", "q", "year", "qtr"]).forecast.sum().reset_index()
    alpha_now = {}
    for cat, g in nat.groupby("category"):
        mu = g.pred_ensemble
        alpha_now[cat] = max(((g.y - mu) ** 2 - mu).mean() / (mu ** 2).mean(), 0.0)
    fnat["lower95"], fnat["upper95"] = nb_interval(fnat.forecast.values,
                                                   fnat.category.map(alpha_now).values)

    # ---- Save --------------------------------------------------------------
    keep = ["division", "category", "q", "year", "qtr", "y"] + pred_cols
    bt[keep].to_csv(out / "backtest_division.csv", index=False)
    nat.to_csv(out / "backtest_national.csv", index=False)
    met.round(4).to_csv(out / "metrics.csv", index=False)
    imp.to_csv(out / "feature_importance.csv", index=False)
    fnat.round(2).to_csv(out / "forecast_next_quarters.csv", index=False)
    nat[nat.alert_spike].to_csv(out / "spike_alerts.csv", index=False)

    summary = {
        "backtest_years": [min(TEST_YEARS), max(TEST_YEARS)],
        "interval_coverage_national": float(nat.in_interval.mean()),
        "n_spike_alerts_national": int(nat.alert_spike.sum()),
        "rmse_skill_vs_ma4_pct": met[(met.model == "ensemble")].set_index(["level", "category"])
                                   .RMSE_skill_vs_MA4_pct.round(1).to_dict(),
    }
    summary["rmse_skill_vs_ma4_pct"] = {f"{a}|{b}": v for (a, b), v in summary["rmse_skill_vs_ma4_pct"].items()}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    show = met[met.model.isin(["ensemble", "ensemble_history_only", "ensemble_history_ntsb",
                               "ensemble_all_features", "baseline_ma4"])]
    print(show.pivot_table(index=["level", "category"], columns="model", values="RMSE").round(3).to_string())
    print(json.dumps(summary, indent=2))
    print(imp.head(15).to_string(index=False))
    print(fnat.to_string(index=False))


if __name__ == "__main__":
    main()
