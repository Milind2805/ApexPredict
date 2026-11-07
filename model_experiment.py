"""Season-held-out evaluation for the F1 race winner model.

Run from the folder that contains f1_multiyear.csv:
    python model_experiment.py [path/to/f1_multiyear.csv]

For each test season (2022-2025) every model is trained ONLY on earlier seasons,
so there is no leakage from the race being predicted. Metrics are computed per
race, because the real question is "who wins this race?", not "is this row a win?".
"""
import sys
import warnings

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
CSV = sys.argv[1] if len(sys.argv) > 1 else "f1_multiyear.csv"
TEST_YEARS = [2022, 2023, 2024, 2025]

# ── Load + clean ──────────────────────────────────────────────
df = pd.read_csv(CSV)
df["race_idx"] = df.groupby("Year", sort=False)["Race"].transform(lambda s: pd.factorize(s)[0])
df = df.sort_values(["Year", "race_idx"], kind="stable").reset_index(drop=True)
df["race_id"] = df["Year"].astype(str) + "|" + df["race_idx"].astype(str)
df = df.dropna(subset=["Position"]).copy()
df["Winner"] = (df["Position"] == 1).astype(int)

# keep only races with exactly one recorded winner
ok = df.groupby("race_id")["Winner"].sum().eq(1)
df = df[df["race_id"].isin(ok[ok].index)].copy()

df["GridPosition"] = df["GridPosition"].fillna(20).replace(0, 20)  # 0 = pit-lane start
df["QualiPosition"] = df["QualiPosition"].fillna(df["GridPosition"])

# ── Feature engineering (every rolling feature uses PRIOR races only) ─────
df = df.reset_index(drop=True)
df["race_order"] = df["race_id"].map({r: i for i, r in enumerate(df["race_id"].drop_duplicates())})
df = df.sort_values(["race_order"], kind="stable").reset_index(drop=True)


def prior_rolling(frame, by, col, window, fn="mean"):
    """Rolling stat over the previous `window` races of each group, excluding the current one."""
    g = frame.groupby(by)[col]
    shifted = g.shift(1)
    return shifted.groupby(frame[by]).transform(
        lambda s: s.rolling(window, min_periods=1).agg(fn)
    )


df["driver_form_finish"] = prior_rolling(df, "Abbreviation", "Position", 5)
df["driver_form_points"] = prior_rolling(df, "Abbreviation", "Points", 5)
df["driver_win_rate10"] = prior_rolling(df, "Abbreviation", "Winner", 10)

team_race = df.groupby(["TeamName", "race_order"], as_index=False)["Points"].mean().rename(
    columns={"Points": "team_pts"})
team_race["team_form_points"] = prior_rolling(team_race, "TeamName", "team_pts", 5)
df = df.merge(team_race[["TeamName", "race_order", "team_form_points"]], on=["TeamName", "race_order"], how="left")

df["driver_form_finish"] = df["driver_form_finish"].fillna(10)
df["driver_form_points"] = df["driver_form_points"].fillna(0)
df["driver_win_rate10"] = df["driver_win_rate10"].fillna(0)
df["team_form_points"] = df["team_form_points"].fillna(df["team_form_points"].median())

g = df.groupby("race_id")
df["grid_rank"] = g["GridPosition"].rank(method="min")
df["pole"] = (df["GridPosition"] == 1).astype(int)
df["front_row"] = (df["GridPosition"] <= 2).astype(int)
df["quali_vs_team"] = df["QualiPosition"] - df.groupby(["race_id", "TeamName"])["QualiPosition"].transform("min")
max_pts = g["StandingPoints"].transform("max")
df["points_share"] = np.where(max_pts > 0, df["StandingPoints"] / max_pts.replace(0, 1), 0.0)
df["standing_rank"] = g["StandingPoints"].rank(ascending=False, method="min")
df["form_rank"] = g["driver_form_points"].rank(ascending=False, method="min")
df["team_form_rank"] = g["team_form_points"].rank(ascending=False, method="min")

BASE_FEATURES = ["GridPosition", "QualiPosition", "Rainfall", "AirTemp", "TrackTemp",
                 "StandingPoints", "HomeRace"]
NEW_FEATURES = ["GridPosition", "QualiPosition", "grid_rank", "pole", "front_row", "quali_vs_team",
                "points_share", "standing_rank", "driver_form_finish", "driver_form_points",
                "driver_win_rate10", "team_form_points", "form_rank", "team_form_rank", "Rainfall"]


# ── Models ────────────────────────────────────────────────────
def current_pipeline(train, test):
    """Replicates f1_multiyear_modeltraining.py (label-encoded IDs + SMOTE + scaling)."""
    enc = {k: LabelEncoder().fit(df[c]) for k, c in
           [("d", "Abbreviation"), ("t", "TeamName"), ("c", "Race")]}

    def X(frame):
        return np.column_stack([
            enc["d"].transform(frame["Abbreviation"]), enc["t"].transform(frame["TeamName"]),
            enc["c"].transform(frame["Race"]), frame[BASE_FEATURES[:2]].values,
            frame[["Rainfall", "AirTemp", "TrackTemp", "StandingPoints", "HomeRace"]].values])

    Xtr, ytr = SMOTE(random_state=42).fit_resample(X(train), train["Winner"])
    sc = StandardScaler().fit(Xtr)
    m = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
    m.fit(sc.transform(Xtr), ytr)
    return m.predict_proba(sc.transform(X(test)))[:, 1]


def logreg(train, test):
    sc = StandardScaler().fit(train[NEW_FEATURES])
    m = LogisticRegression(C=0.5, max_iter=2000)
    m.fit(sc.transform(train[NEW_FEATURES]), train["Winner"])
    return m.predict_proba(sc.transform(test[NEW_FEATURES]))[:, 1]


def xgb_new(train, test):
    m = XGBClassifier(n_estimators=250, max_depth=3, learning_rate=0.04, subsample=0.8,
                      colsample_bytree=0.8, min_child_weight=5, reg_lambda=3,
                      random_state=42, eval_metric="logloss")
    m.fit(train[NEW_FEATURES], train["Winner"])
    return m.predict_proba(test[NEW_FEATURES])[:, 1]


def pole_baseline(train, test):
    return (test["GridPosition"] == 1).astype(float).values * 0.9 + 0.1 / 20


MODELS = {
    "Pole-sitter baseline": pole_baseline,
    "Current model (SMOTE + IDs)": current_pipeline,
    "Logistic regression (new features)": logreg,
    "XGBoost (new features, no SMOTE)": xgb_new,
}


# ── Race-level evaluation ─────────────────────────────────────
def normalise(frame, p):
    """Turn independent driver probabilities into one distribution per race."""
    s = pd.Series(np.clip(p, 1e-4, None), index=frame.index)
    return s / s.groupby(frame["race_id"]).transform("sum")


def evaluate(frame, p):
    p = normalise(frame, p)
    f = frame.assign(p=p)
    races = f.groupby("race_id")
    top1 = np.mean([r.loc[r["p"].idxmax(), "Winner"] for _, r in races])
    logloss = -np.mean([np.log(r.loc[r["Winner"] == 1, "p"].iloc[0]) for _, r in races])
    brier = brier_score_loss(f["Winner"], f["p"])
    return top1, logloss, brier, races.ngroups


rows = []
for yr in TEST_YEARS:
    train, test = df[df["Year"] < yr], df[df["Year"] == yr]
    for name, fn in MODELS.items():
        top1, ll, br, n = evaluate(test, fn(train, test))
        rows.append({"year": yr, "model": name, "top1": top1, "logloss": ll, "brier": br, "races": n})

res = pd.DataFrame(rows)
summary = res.groupby("model", sort=False).apply(
    lambda g: pd.Series({
        "winner_accuracy_%": 100 * np.average(g["top1"], weights=g["races"]),
        "race_logloss": np.average(g["logloss"], weights=g["races"]),
        "brier": np.average(g["brier"], weights=g["races"]),
    }))
print("\nPer-season winner accuracy (% of races where the top-ranked driver won):")
print((res.pivot(index="model", columns="year", values="top1") * 100).round(1).reindex(MODELS).to_string())
print("\nOverall, seasons", TEST_YEARS, "(lower log loss / brier is better):")
print(summary.round(3).to_string())
print("\nReference: random guess among ~20 drivers gives a race log loss of about", round(np.log(20), 2))
res.to_csv("experiment_results.csv", index=False)
