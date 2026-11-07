"""Shared feature engineering + tiny linear model for the F1 Race Winner Predictor.

Used by BOTH train_v2.py (training) and core.py (serving), so a feature can never
be computed one way in training and another way in production.

Only numpy/pandas are needed at serving time (no scikit-learn).
"""
import json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

# Order matters: this is the column order the model was trained on.
FEATURES = [
    "GridPosition", "QualiPosition", "pole", "front_row", "StandingPoints",
    "driver_form_finish", "driver_form_points", "driver_win_rate10",
    "team_form_points", "Rainfall",
]
# Friendly names shown in the "What drove this prediction" chart and the Model tab.
LABELS = {
    "GridPosition": "Grid position",
    "QualiPosition": "Qualifying position",
    "pole": "Starting on pole",
    "front_row": "Starting on front row",
    "StandingPoints": "Championship points",
    "driver_form_finish": "Driver's recent finishes",
    "driver_form_points": "Driver's recent points",
    "driver_win_rate10": "Driver's recent win rate",
    "team_form_points": "Team's recent points",
    "Rainfall": "Wet race",
}

# Defaults for a driver with no history (rookie) - same values used in training.
DEFAULT_FINISH = 10.0
DEFAULT_POINTS = 0.0
DEFAULT_WIN_RATE = 0.0


def canon_team(name: str) -> str:
    """Collapse renamed teams so form carries over (Racing Point -> Aston Martin, etc.)."""
    n = str(name).lower().strip()
    if "red bull" in n:
        return "Red Bull"
    for key, canon in [("ferrari", "Ferrari"), ("mercedes", "Mercedes"), ("mclaren", "McLaren"),
                       ("aston martin", "Aston Martin"), ("racing point", "Aston Martin"),
                       ("alpine", "Alpine"), ("renault", "Alpine"), ("williams", "Williams"),
                       ("haas", "Haas"), ("cadillac", "Cadillac")]:
        if key in n:
            return canon
    if any(k in n for k in ("alphatauri", "toro rosso", "racing bulls")) or n == "rb" or n.startswith("rb "):
        return "Racing Bulls"
    if any(k in n for k in ("sauber", "alfa romeo", "audi")):
        return "Sauber"
    return str(name)


# ── Form tracking (identical code path for training and serving) ──────────────
class FormTracker:
    """Rolling recent-form state. Call update_race() AFTER a race has finished."""

    def __init__(self):
        self._driver = defaultdict(lambda: deque(maxlen=10))  # (position, points)
        self._team = defaultdict(lambda: deque(maxlen=5))     # mean points per team-race

    def driver_features(self, code: str):
        h = self._driver.get(code)
        if not h:
            return DEFAULT_FINISH, DEFAULT_POINTS, DEFAULT_WIN_RATE
        last5 = list(h)[-5:]
        return (float(np.mean([p for p, _ in last5])),
                float(np.mean([pts for _, pts in last5])),
                float(np.mean([1.0 if p == 1 else 0.0 for p, _ in h])))

    def team_form(self, team: str):
        """Mean recent points per car for the team, or None if the team has no history."""
        h = self._team.get(canon_team(team))
        return float(np.mean(h)) if h else None

    def update_race(self, rows):
        """rows: iterable of dicts with driver, team, position, points for ONE finished race."""
        by_team = defaultdict(list)
        for r in rows:
            pos, pts = r["position"], r["points"]
            if pos is None or pd.isna(pos):
                continue
            pts = 0.0 if pts is None or pd.isna(pts) else float(pts)
            self._driver[r["driver"]].append((float(pos), pts))
            by_team[canon_team(r["team"])].append(pts)
        for team, pts in by_team.items():
            self._team[team].append(float(np.mean(pts)))


# ── Training data ─────────────────────────────────────────────
def load_history(csv_path) -> pd.DataFrame:
    """Read f1_multiyear.csv, keep only usable races, add race_order / Winner."""
    df = pd.read_csv(csv_path)
    df["race_idx"] = df.groupby("Year", sort=False)["Race"].transform(lambda s: pd.factorize(s)[0])
    df = df.sort_values(["Year", "race_idx"], kind="stable").reset_index(drop=True)
    df["race_id"] = df["Year"].astype(str) + "|" + df["race_idx"].astype(str)
    df = df.dropna(subset=["Position"]).copy()
    df["Winner"] = (df["Position"] == 1).astype(int)
    ok = df.groupby("race_id")["Winner"].sum().eq(1)          # races with exactly one winner
    df = df[df["race_id"].isin(ok[ok].index)].copy()
    df["GridPosition"] = df["GridPosition"].fillna(20).replace(0, 20)   # 0 = pit-lane start
    df["QualiPosition"] = df["QualiPosition"].fillna(df["GridPosition"])
    df["race_order"] = df["race_id"].map({r: i for i, r in enumerate(df["race_id"].drop_duplicates())})
    return df.sort_values("race_order", kind="stable").reset_index(drop=True)


def history_rows(df: pd.DataFrame):
    """Rows -> dicts for FormTracker.update_race."""
    return [{"driver": a, "team": t, "position": p, "points": pts}
            for a, t, p, pts in zip(df["Abbreviation"], df["TeamName"], df["Position"], df["Points"])]


def build_frame(df: pd.DataFrame):
    """Compute model features for every row by walking races in order (prior races only).

    Returns (frame, tracker) - the tracker holds the form state after the last race,
    which is exactly what the live app needs to predict the next one.
    """
    tracker = FormTracker()
    parts = []
    for _, race in df.groupby("race_order", sort=True):
        f = race.copy()
        feats = [tracker.driver_features(a) for a in f["Abbreviation"]]
        f["driver_form_finish"] = [x[0] for x in feats]
        f["driver_form_points"] = [x[1] for x in feats]
        f["driver_win_rate10"] = [x[2] for x in feats]
        f["team_form_points"] = pd.Series([tracker.team_form(t) for t in f["TeamName"]],
                                  index=f.index, dtype="float64")
        parts.append(f)
        tracker.update_race(history_rows(race))
    out = pd.concat(parts)
    out["pole"] = (out["GridPosition"] == 1).astype(int)
    out["front_row"] = (out["GridPosition"] <= 2).astype(int)
    out["Rainfall"] = out["Rainfall"].fillna(0).astype(int)
    return out.reset_index(drop=True), tracker


# ── Tiny logistic model (numpy only) ──────────────────────────
class LinearModel:
    def __init__(self, spec: dict):
        self.features = spec["features"]
        self.coef = np.array(spec["coef"], dtype=float)
        self.intercept = float(spec["intercept"])
        self.mean = np.array(spec["mean"], dtype=float)
        self.scale = np.array(spec["scale"], dtype=float)
        self.team_median = float(spec["team_median"])
        self.meta = spec.get("meta", {})

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))

    def _z(self, x):
        return (np.asarray(x, dtype=float) - self.mean) / self.scale

    def predict_proba(self, X) -> np.ndarray:
        """X: (n, n_features) in FEATURES order. Returns P(win) per row."""
        z = self._z(np.atleast_2d(X))
        return 1.0 / (1.0 + np.exp(-(z @ self.coef + self.intercept)))

    def contributions(self, x) -> dict:
        """Exact per-feature contribution to the log-odds, relative to an average row."""
        z = self._z(np.asarray(x, dtype=float))
        return {LABELS[f]: float(c * v) for f, c, v in zip(self.features, self.coef, z)}


def feature_row(grid, quali, standing_points, rainfall, driver_feats, team_form, team_median):
    """One driver's feature vector in FEATURES order (single source of truth for serving)."""
    finish, points, win_rate = driver_feats
    tf = team_median if team_form is None else team_form
    return [float(grid), float(quali), 1.0 if grid == 1 else 0.0, 1.0 if grid <= 2 else 0.0,
            float(standing_points), finish, points, win_rate, tf, float(bool(rainfall))]
