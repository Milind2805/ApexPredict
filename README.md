# ApexPredict

A web app that estimates each driver's chance of winning the next Formula 1 Grand Prix, explains the estimate, and lets you test "what if" scenarios. The model is a logistic regression trained on 2019–2025 races and kept current with live 2026 results.

**Live app:** https://f1-race-winner-predictor-app-fronte.vercel.app
**API:** https://f1-race-winner-predictor-app.onrender.com/docs  *(free tier, the first request can take about 30 seconds to wake the server)*

---

## What it does

| Tab / feature | What you get |
|---|---|
| **Next race** | One click predicts the upcoming Grand Prix for the whole grid. It pulls the calendar, qualifying results, standings and a weather forecast automatically. Before qualifying it uses championship order as a provisional grid and says so. |
| **Custom scenario** | Pick any race, driver and conditions, then see the win probability and a chart of which factors pushed it up or down. |
| **Head-to-head** | Compare two drivers on the same grid and conditions. |
| **Race simulator** | Win probability for every car on an editable grid. Probabilities add up to 100% across the field. |
| **Season projection** | Monte Carlo simulation of the remaining races, giving projected points, a likely range and title odds. |
| **Backtest** | The model trained on 2019–2024, then asked to pick every 2025 winner, shown race by race next to the "pick the pole sitter" baseline. |

## How it works

1. **Data.** Race results, grid and qualifying positions, rain flag and championship points for 2019–2025 (152 races, 3,035 driver entries), collected with [FastF1](https://docs.fastf1.dev/).
2. **Features (10).** Grid position, qualifying position, on pole, on front row, championship points before the race, the driver's average finish and points over the last 5 races, the driver's win rate over the last 10, the team's average points over the last 5, and a wet-race flag. Every "recent form" feature uses **only earlier races**.
3. **Model.** Logistic regression on standardised features. Predictions are normalised within each race so the field adds up to 100%.
4. **Explanations.** For a linear model, each feature's contribution to the log-odds is exact (coefficient × standardised value), so the "What drove this prediction" chart adds up to the prediction.
5. **Live form.** The backend rebuilds the recent-form state from the 2025 data and then applies every 2026 race result from the [Jolpica F1 API](https://github.com/jolpica/jolpica-f1), cached for six hours. If the API is unreachable it falls back to end-of-2025 form.
6. **Shared feature code.** `features.py` is used by both the training script and the live API, so a feature can never be calculated one way in training and another way in production.

## Results

Evaluated honestly: for each test season the model is trained **only on earlier seasons**, then scored per race (did it rank the winner first, and how much probability did it give the real winner?).

**Held-out seasons 2022–2025 (92 races)**

| | Winner picked | Race log loss (lower is better) | Brier score |
|---|---|---|---|
| Always pick the pole sitter | 55.4% | n/a | n/a |
| Version 1 (XGBoost + SMOTE, driver/team/circuit IDs) | 48.9% | 1.637 | 0.035 |
| **Version 2 (this repo)** | **58.7%** | **1.129** | **0.027** |

A random guess among 20 drivers scores a race log loss of about 3.0.

**Calibration (held-out):** when the model says about 53%, the driver wins about 57% of the time; when it says about 77%, about 90%. It slightly over-predicts in the 2–5% range.

**2025 backtest (24 races):** the model picked 14 winners (58.3%), the pole sitter won 16 (66.7%), and the winner was inside the model's top three in all 24.

**How to read this.** Pole position is a very strong predictor, and with 24 races one race moves a hit rate by about 4 points. As a single "pick the winner" the model roughly matches the pole sitter. What it adds is a calibrated probability for every driver, which is where the improvement in log loss comes from.

### Version history

Version 1 reported **96.4% test accuracy**, which was misleading: only 5% of entries are winners, so predicting "nobody wins" scores 95%. The random row-level split also put rows from the same race in both train and test. When rebuilt with season-held-out evaluation, version 1 picked the winner less often than the pole sitter. Version 2 replaces it with recent-form features, drops the label-encoded IDs, removes SMOTE, and uses a simpler model that is better calibrated.

## Limitations

- About 90 test races is a small sample. Differences of a few points in winner accuracy are within noise.
- The model does not use the circuit, temperatures, tyre strategy, penalties or safety cars. Only the wet-race flag comes from conditions.
- New drivers get neutral "rookie" form until they have results, and a new team uses the median team form.
- Before qualifying, the Next race card uses championship order as a stand-in for the grid, so its numbers are provisional.
- Predictions are statistical estimates, not guarantees.

## API

| Endpoint | Purpose |
|---|---|
| `GET /health`, `GET /meta` | Health check, lists of drivers, teams, circuits and seasons |
| `POST /predict` | Win probability for one driver |
| `POST /explain` | Probability plus per-feature contributions |
| `POST /compare` | Two drivers on the same grid |
| `GET /grid/{season}` | Default grid with live standings |
| `POST /simulate` | Normalised win probabilities for a full grid |
| `POST /season-projection` | Monte Carlo season projection |
| `GET /season/{year}/drivers`, `/constructors` | Season analytics |
| `GET /model/importance`, `/model/shap`, `/model/metrics` | Model details |

Interactive docs are at `/docs` on the API.

## Run it locally

```bash
git clone https://github.com/Milind2805/ApexPredict.git
cd ApexPredict

# Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload          # http://127.0.0.1:8000

# Frontend (second terminal): open frontend/index.html with Live Server,
# and set the API constant near the top of the script to http://127.0.0.1:8000
```

**Retrain the model** (needs scikit-learn; not required to run the app):

```bash
cd backend
pip install scikit-learn
python train_v2.py data/f1_multiyear.csv data
```

This writes `model_v2.json`, the importance and SHAP files, and `model_metrics.json`, and prints the held-out results.

**Regenerate the Backtest tab** (run from the project root):

```bash
python training/backtest_data.py
```

## Project structure

```
backend/
  main.py          FastAPI routes
  core.py          prediction, simulation, season projection, live form
  features.py      shared feature engineering and the small linear model
  train_v2.py      held-out evaluation and final training
  data/            model_v2.json, f1_multiyear.csv, default_grids.json, metrics and importance files
frontend/
  index.html       the whole UI (vanilla JavaScript)
  backtest.json    2025 backtest results
training/
  backtest_data.py regenerates backtest.json
  ...              earlier experiments
```

## Tech stack

Python, FastAPI, pandas, NumPy, scikit-learn (training only), vanilla JavaScript, deployed on Render (API) and Vercel (frontend).

## Data sources and credits

- [FastF1](https://docs.fastf1.dev/) for historical race and qualifying data
- [Jolpica F1 API](https://github.com/jolpica/jolpica-f1) (Ergast-compatible) for the live calendar, results and standings
- [Open-Meteo](https://open-meteo.com/) for race-day weather forecasts

Built by [Milind Kumar Singh](https://github.com/Milind2805).
