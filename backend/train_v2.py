"""Train the v2 F1 race-winner model.

    python train_v2.py [path/to/f1_multiyear.csv] [output_dir]

1. Honest evaluation: for each of 2022-2025, train on EARLIER seasons only and score
   the race-level predictions (winner picked, log loss, Brier, calibration).
2. Final model: train on every season in the CSV.
3. Writes model_v2.json, f1_feature_importance.json, f1_shap_values.json, model_metrics.json.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import StandardScaler

from features import FEATURES, LABELS, build_frame, load_history

warnings.filterwarnings("ignore")
CSV = sys.argv[1] if len(sys.argv) > 1 else "f1_multiyear.csv"
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else ".")
OUT.mkdir(parents=True, exist_ok=True)
TEST_YEARS = [2022, 2023, 2024, 2025]
C = 0.5

df = load_history(CSV)
frame, _ = build_frame(df)
print(f"Loaded {len(frame)} driver-rows from {frame['race_id'].nunique()} races "
      f"({int(frame['Year'].min())}-{int(frame['Year'].max())})")


def fit(train):
    """Returns (scaler, model, team_median) fitted on `train`."""
    med = float(train["team_form_points"].median())
    X = train[FEATURES].copy()
    X["team_form_points"] = X["team_form_points"].fillna(med)
    sc = StandardScaler().fit(X)
    m = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(X), train["Winner"])
    return sc, m, med


def predict(sc, m, med, test):
    X = test[FEATURES].copy()
    X["team_form_points"] = X["team_form_points"].fillna(med)
    return m.predict_proba(sc.transform(X))[:, 1]


def normalise(frame_, p):
    """Independent driver probabilities -> one distribution per race (sums to 1)."""
    s = pd.Series(np.clip(p, 1e-4, None), index=frame_.index)
    return s / s.groupby(frame_["race_id"]).transform("sum")


# ── 1. Honest, season-held-out evaluation ─────────────────────
held = []
for yr in TEST_YEARS:
    train, test = frame[frame["Year"] < yr], frame[frame["Year"] == yr]
    sc, m, med = fit(train)
    held.append(test.assign(p=normalise(test, predict(sc, m, med, test))))
held = pd.concat(held)

races = held.groupby("race_id")
n_races = races.ngroups
top1 = np.mean([r.loc[r["p"].idxmax(), "Winner"] for _, r in races])
pole = np.mean([r.loc[r["GridPosition"].idxmin(), "Winner"] for _, r in races])
logloss = -np.mean([np.log(r.loc[r["Winner"] == 1, "p"].iloc[0]) for _, r in races])
brier = brier_score_loss(held["Winner"], held["p"])

bins = pd.cut(held["p"], [0, .02, .05, .1, .2, .4, .7, 1.0])
calib = (held.groupby(bins, observed=True)
         .agg(n=("p", "size"), predicted=("p", "mean"), actual=("Winner", "mean")).round(3))
print(f"\nHeld-out seasons {TEST_YEARS} ({n_races} races):")
print(f"  winner picked correctly : {100 * top1:.1f}%   (pole-sitter baseline: {100 * pole:.1f}%)")
print(f"  race log loss           : {logloss:.3f}   (random guess ~ {np.log(20):.2f}; lower is better)")
print(f"  brier score             : {brier:.3f}")
print("\nCalibration (predicted vs actual win rate):")
print(calib.to_string())

# ── 2. Final model on all seasons ─────────────────────────────
sc, model, med = fit(frame)
spec = {
    "features": FEATURES,
    "coef": model.coef_[0].tolist(),
    "intercept": float(model.intercept_[0]),
    "mean": sc.mean_.tolist(),
    "scale": sc.scale_.tolist(),
    "team_median": med,
    "meta": {"trained_through": int(frame["Year"].max()), "C": C},
}
(OUT / "model_v2.json").write_text(json.dumps(spec, indent=2))

# ── 3. Explanation + metrics files used by the API ────────────
coef = np.abs(model.coef_[0])
importance = {LABELS[f]: float(c / coef.sum()) for f, c in zip(FEATURES, coef)}
Z = sc.transform(frame[FEATURES].fillna({"team_form_points": med}))
mean_abs = np.abs(Z * model.coef_[0]).mean(axis=0)                 # exact SHAP for a linear model
shap_importance = {LABELS[f]: float(v) for f, v in zip(FEATURES, mean_abs)}
(OUT / "f1_feature_importance.json").write_text(json.dumps(importance))
(OUT / "f1_shap_values.json").write_text(json.dumps(shap_importance))

metrics = {
    "model": "Logistic regression on grid, qualifying, recent form and team form",
    "n_samples": int(len(frame)),
    "n_races": int(frame["race_id"].nunique()),
    "seasons": f"{int(frame['Year'].min())}-{int(frame['Year'].max())}",
    "test_seasons": f"{TEST_YEARS[0]}-{TEST_YEARS[-1]} (each trained only on earlier seasons)",
    "held_out_winner_accuracy": round(100 * top1, 1),
    "pole_baseline_accuracy": round(100 * pole, 1),
    "held_out_race_logloss": round(float(logloss), 3),
    "held_out_brier": round(float(brier), 3),
    "held_out_races": int(n_races),
    "note": "Evaluated per race on seasons the model had not seen. Small sample (about 90 races).",
}
(OUT / "model_metrics.json").write_text(json.dumps(metrics, indent=2))

print("\nMost influential features (mean |SHAP|, log-odds):")
for k, v in sorted(shap_importance.items(), key=lambda kv: -kv[1]):
    print(f"  {k:28s} {v:.3f}")
print(f"\nSaved to {OUT.resolve()}: model_v2.json, f1_feature_importance.json, "
      f"f1_shap_values.json, model_metrics.json")
