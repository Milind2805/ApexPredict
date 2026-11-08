"""Model + data layer for the F1 Race Winner Predictor (no Streamlit)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import time
import copy
import requests
from features import FEATURES, LinearModel, build_frame, feature_row, load_history

STANDINGS_URL = "https://api.jolpi.ca/ergast/f1/current/driverStandings.json"
RESULTS_URL = "https://api.jolpi.ca/ergast/f1/2026/results.json?limit=1000"
_cache = {"time": 0, "data": None}

def fetch_live_standings(ttl=3600):
    """Returns {driver_code: (position, points)}, or None only if it has never succeeded."""
    now = time.time()
    if _cache["data"] and now - _cache["time"] < ttl:
        return _cache["data"]
    try:
        r = requests.get(STANDINGS_URL, timeout=10)
        r.raise_for_status()
        rows = r.json()["MRData"]["StandingsTable"]["StandingsLists"][0]["DriverStandings"]
        data = {row["Driver"]["code"]: (int(row["position"]), float(row["points"])) for row in rows}
    except Exception as err:
        print("standings fetch failed:", repr(err))      # shows up in the Render logs
        if _cache["data"]:                               # keep the last good copy
            _cache["time"] = now - ttl + 300             # try again in 5 minutes
            return _cache["data"]
        return None
    _cache.update(time=now, data=data)
    return data

DATA_DIR = Path(__file__).parent / "data"

# Display name -> what the user sees
TEAM_DISPLAY_ALIAS = {
    "Racing Point": "Aston Martin",
    "AlphaTauri": "Racing Bulls",
    "RB": "Racing Bulls",
    "Alfa Romeo": "Audi",
    "Kick Sauber": "Audi",
    "Alfa Romeo Racing": "Cadillac",
}


def _records(df: pd.DataFrame) -> list:
    """DataFrame -> JSON-safe list of dicts (avoids numpy types)."""
    return json.loads(df.to_json(orient="records"))


class F1Engine:
    def get_tracker(self, ttl=6 * 3600):
        """Form tracker = end-of-2025 state + every 2026 race finished so far."""
        cache = self._live_tracker
        if cache["tracker"] is not None and time.time() < cache["expires"]:
            return cache["tracker"]
        tracker, ok = self.base_tracker, False
        try:
            r = requests.get(RESULTS_URL, timeout=10)
            r.raise_for_status()
            races = r.json()["MRData"]["RaceTable"]["Races"]
            t = copy.deepcopy(self.base_tracker)      # fresh copy, so races are never counted twice
            for race in races:
                t.update_race([
                    {"driver": x["Driver"]["code"], "team": x["Constructor"]["name"],
                     "position": int(x["position"]), "points": float(x["points"])}
                    for x in race["Results"]
                ])
            tracker, ok = t, True
        except Exception:
            pass                                      # API down: use end-of-2025 form
        cache.update(tracker=tracker, expires=time.time() + (ttl if ok else 300))
        return tracker

    def __init__(self, data_dir: Path = DATA_DIR):
        self.model = LinearModel.load(data_dir / "model_v2.json")
        _, self.base_tracker = build_frame(load_history(data_dir / "f1_multiyear.csv"))
        self._live_tracker = {"tracker": None, "expires": 0.0}

        self.df = pd.read_csv(data_dir / "f1_multiyear.csv")
        with open(data_dir / "f1_feature_importance.json") as f:
            self.importance = json.load(f)
        with open(data_dir / "f1_shap_values.json") as f:
            self.shap = json.load(f)
        with open(data_dir / "model_metrics.json") as f:
            self.metrics = json.load(f)
        with open(data_dir / "default_grids.json") as f:
            self.default_grids = json.load(f)

        self.drivers = sorted(self.df["Abbreviation"].unique())
        self.circuits = sorted(self.df["Race"].unique())
        raw_teams = self.df["TeamName"].unique()
        self.teams = sorted({TEAM_DISPLAY_ALIAS.get(t, t) for t in raw_teams})
        self.years = sorted(int(y) for y in self.df["Year"].unique())

    # ── Prediction ────────────────────────────────────────────

    def predict(self, driver, team, circuit, grid_position, quali_position,
                rainfall, air_temp, track_temp, standing_points, home_race):
        """Returns (predicted_win: bool, win_probability: float in 0..1)."""
        row = feature_row(grid_position, quali_position, standing_points, rainfall,
        self.get_tracker().driver_features(driver),
        self.get_tracker().team_form(team), self.model.team_median)
        prob = float(self.model.predict_proba(row)[0])
        return prob >= 0.5, prob

    # ── Season analytics ──────────────────────────────────────
    def _season(self, year: int) -> pd.DataFrame:
        s = self.df[self.df["Year"] == year]
        if s.empty:
            raise ValueError(f"No data for season {year}")
        return s

    def driver_progression(self, year: int, top_n: int = 5) -> dict:
        s = self._season(year)
        top = s.groupby("Abbreviation")["Points"].sum().nlargest(top_n).index
        # Assumes the CSV rows are in chronological order
        race_order = s["Race"].drop_duplicates().tolist()
        rows = s[s["Abbreviation"].isin(top)][["Race", "Abbreviation", "Points"]]
        return {"race_order": race_order, "rows": _records(rows)}

    def driver_wins(self, year: int) -> list:
        s = self._season(year)
        w = s[s["Position"] == 1]["Abbreviation"].value_counts().reset_index()
        w.columns = ["driver", "wins"]
        return _records(w)

    def constructor_points(self, year: int) -> list:
        s = self._season(year)
        c = s.groupby("TeamName")["Points"].sum().reset_index()
        c.columns = ["team", "points"]
        return _records(c.sort_values("points", ascending=False))

    def constructor_wins(self, year: int) -> list:
        s = self._season(year)
        w = s[s["Position"] == 1]["TeamName"].value_counts().reset_index()
        w.columns = ["team", "wins"]
        return _records(w)

    # ── Explainability ────────────────────────────────────────
    def explain(self, driver, team, circuit, grid_position, quali_position,rainfall, air_temp, track_temp, standing_points, home_race):
        row = feature_row(grid_position, quali_position, standing_points, rainfall,
                        self.get_tracker().driver_features(driver),
                        self.get_tracker().team_form(team), self.model.team_median)
        prob = float(self.model.predict_proba(row)[0])
        contribs = self.model.contributions(row)
        return prob >= 0.5, prob, contribs
        
    # ── Grid simulator ────────────────────────────────────────
    def current_grid(self, season: str = "2026") -> list:
        """The JSON decides WHO is on the grid; the live API (2026 only)
        decides their points and order. Falls back to the JSON if offline."""
        if season not in self.default_grids:
            raise ValueError(f"No default grid for {season}")
        rows = [dict(r) for r in self.default_grids[season]]
        live = fetch_live_standings() if season == "2026" else None
        if not live:
            return rows
        for r in rows:
            if r["driver"] in live:
                r["_live_pos"], r["points"] = live[r["driver"]]
        rows.sort(key=lambda r: r.get("_live_pos", 999))
        for i, r in enumerate(rows, start=1):
            r["position"] = i
            r.pop("_live_pos", None)
        return rows
    
    def default_grid(self, season: int) -> list:
        grid = self.current_grid(str(season))
        return [
            {
                **row,
                "team": TEAM_DISPLAY_ALIAS.get(row["team"], row["team"]),
                "supported": row["driver"] in self.drivers,
            }
            for row in grid
        ]

    def simulate_grid(self, entries, circuit, rainfall, air_temp, track_temp):
        """Win probability for every car on the grid. Drivers the model has
        never seen are returned with supported=False and no probability."""
        out = []
        for e in entries:
            row = {"driver": e["driver"], "team": e["team"],
                   "quali_position": e["quali_position"]}
            try:
                _, prob = self.predict(
                    e["driver"], e["team"], circuit,
                    e["grid_position"], e["quali_position"],
                    rainfall, air_temp, track_temp,
                    e["standing_points"], e["home_race"],
                )
                out.append({**row, "supported": True, "win_probability": prob})
            except ValueError:
                out.append({**row, "supported": False, "win_probability": None})
        total = sum(r["win_probability"] for r in out if r["supported"])
        
        for r in out:
            if r["supported"]:
                r["win_probability"] = round(r["win_probability"] / max(total, 1e-9), 4)
        out.sort(key=lambda r: (-1 if r["win_probability"] is None else r["win_probability"]),
                 reverse=True)
        return out

    # ── Season projection (Monte Carlo) ───────────────────────
    POINTS = np.array([25, 18, 15, 12, 10, 8, 6, 4, 2, 1], dtype=float)

    def project_season(self, remaining_races, n_sims=500, seed=None):
        """Monte Carlo projection of the 2026 standings.

        Same logic as the Streamlit tab, vectorised: for each race, ONE batched
        model call covers every simulation and driver. Finishing orders are drawn
        from Plackett-Luce via the Gumbel-max trick (identical distribution to
        sequential weighted sampling without replacement).
        """
        grid = self.current_grid("2026")
        names = [r["driver"] for r in grid]
        D, S = len(grid), int(n_sims)
        rng = np.random.default_rng(seed)

        current = np.array([r["points"] for r in grid], dtype=float)

        # Recent form (2025 averages); neutral midfield guess for rookies
        form_grid = np.full(D, 12.0)
        form_quali = np.full(D, 12.0)
        tracker = self.get_tracker()
        for i, drv in enumerate(names):
            rows = self.df[(self.df["Abbreviation"] == drv) & (self.df["Year"] == 2025)]
            if len(rows):
                form_grid[i] = rows["GridPosition"].mean()
                form_quali[i] = rows["QualiPosition"].mean()
        
        skipped = []
        rows_idx = np.arange(S)
        drv_feats = np.array([tracker.driver_features(n) for n in names])   # shape (D, 3)
        team_f = np.array([tracker.team_form(r["team"]) for r in grid], dtype=float)     # shape (D,)    
        team_f = np.where(np.isnan(team_f), self.model.team_median, team_f)
        season_pts = np.tile(current, (S, 1))  

        for race in remaining_races:
            gp = np.clip(rng.normal(form_grid, 2, (S, D)).round(), 1, 22)
            qp = np.clip(rng.normal(form_quali, 2, (S, D)).round(), 1, 22)
            X = np.empty((S, D, 10))
            X[:, :, FEATURES.index("GridPosition")] = gp
            X[:, :, FEATURES.index("QualiPosition")] = qp
            X[:, :, FEATURES.index("pole")] = (gp == 1)
            X[:, :, FEATURES.index("Rainfall")] = 0
            X[:, :, FEATURES.index("front_row")] = (gp <= 2)
            X[:, :, FEATURES.index("driver_form_finish")] = drv_feats[:, 0]
            X[:, :, FEATURES.index("driver_form_points")] = drv_feats[:, 1]
            X[:, :, FEATURES.index("driver_win_rate10")] = drv_feats[:, 2]
            X[:, :, FEATURES.index("team_form_points")] = team_f
            X[:, :, FEATURES.index("StandingPoints")] = season_pts[rows_idx]
    
            p = self.model.predict_proba(X.reshape(-1, 10))
            win = np.maximum(p.reshape(S, D), 0.001)
            keys = np.log(win) + rng.gumbel(size=(S, D))
            order = np.argsort(-keys, axis=1)[:, : min(10, D)]
            for rank in range(order.shape[1]):
                season_pts[rows_idx, order[:, rank]] += self.POINTS[rank]

        champs = np.argmax(season_pts, axis=1)
        title_pct = np.bincount(champs, minlength=D) * 100.0 / S

        summary = []
        for i, r in enumerate(grid):
            v = season_pts[:, i]
            summary.append({
                "driver": r["driver"],
                "team": TEAM_DISPLAY_ALIAS.get(r["team"], r["team"]),
                "current_points": float(current[i]),
                "projected_mean": round(float(v.mean()), 1),
                "p10": round(float(np.percentile(v, 10)), 1),
                "p90": round(float(np.percentile(v, 90)), 1),
                "title_win_pct": round(float(title_pct[i]), 1),
                "modelled": True,
            })
        summary.sort(key=lambda x: x["projected_mean"], reverse=True)
        return {"n_sims": S, "skipped_races": skipped, "standings": summary}
