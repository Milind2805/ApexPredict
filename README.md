# F1 Race Winner Predictor

Predict who wins a Formula 1 race from qualifying, weather and championship standings, then see *why* the model decided that.

**Live demo:** https://f1-race-winner-predictor-app-fronte.vercel.app
**API docs:** https://f1-race-winner-predictor-app.onrender.com/docs

> The API runs on a free Render instance, so the first request after a quiet period can take up to a minute while it wakes up.

[App screenshot](Screenshot 2026-10-06 115513.png)

## What it does

- **Prediction:** win probability for one driver, shown on a team-coloured driver card with an animated count-up, plus a SHAP chart showing which inputs pushed the prediction up or down.
- **Head-to-head:** two driver cards in their team colours and a tug-of-war bar showing each driver's share of their combined win chance.
- **Race simulator:** a team-coloured timing tower with win probabilities for the full 2026 grid (or the 2023 grid), with an editable lineup.
- **Season projection:** Monte Carlo simulation of the remaining races, shown as a Chart.js chart of projected final points with a 10th-90th percentile band, plus a standings table with title chances.
- A lights-out loader is shown while the free backend wakes up, so the cold start reads as part of the experience rather than a hang.

Championship standings are fetched live from the [Jolpica-F1 API](https://api.jolpi.ca/ergast/f1/) and cached for 6 hours. If the API is unreachable, the app falls back to the standings saved in `backend/data/default_grids.json`.

## Architecture

```
Browser (Vercel)  ──HTTPS/JSON──▶  FastAPI (Render)  ──▶  XGBoost model, scaler, encoders
 index.html                          backend/main.py       backend/data/*.pkl
                                     backend/core.py   ──▶  Jolpica-F1 API (live standings)
```

- `backend/core.py` holds all model and data logic, with no web code.
- `backend/main.py` exposes it as a REST API (FastAPI, Pydantic validation, CORS).
- `frontend/index.html` is a single-page UI written in plain HTML, CSS and JavaScript. Chart.js and canvas-confetti are loaded from a CDN, so there is no build step.

## How the model works

- **Features (10):** driver, team, circuit, grid position, qualifying position, rain, air temperature, track temperature, championship points, home race.
- **Model:** XGBoost classifier trained on 2019-2025 race weekends. Metrics are in `backend/data/model_metrics.json`.
- **Explanations:** SHAP values, computed per prediction with XGBoost's built-in contributions.
- **Season projection:** each remaining race samples a full top-10 finishing order from the model's win probabilities (Plackett-Luce, drawn with the Gumbel-max trick) and repeats this thousands of times. It is vectorised, so one batched model call covers every simulation.

## Known limitations

- The model leans heavily on qualifying and grid position (clearly visible in the SHAP chart), so a pole sitter often gets a very high win probability.
- It is trained only on 2019-2025 data. 2026 form is approximated from current championship order, not learned.
- Teams with no history (Cadillac) use a midfield team as a proxy.
- Predictions are statistical estimates, not guarantees.
- Win probabilities are estimated per driver, so they do not sum to 100% across the grid. The head-to-head bar normalises the two drivers' values for display only.
- In the season projection, the title chance comes from the simulated finishing orders, which depend heavily on the same qualifying-driven model.

## Run locally

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

Then open `frontend/index.html` (for example with VS Code Live Server). To point it at the local API, set the `API` constant near the top of the script to `http://127.0.0.1:8000`.

## Project structure

```
backend/    FastAPI app, model files and data (data/)
frontend/   single-page UI
training/   model training script
```

## Tech stack

Python, FastAPI, XGBoost, scikit-learn, pandas, NumPy, SHAP, plain JavaScript, Chart.js, canvas-confetti, Render, Vercel.

## Data

Race and standings data come from the community-maintained Jolpica-F1 API (the successor to the Ergast API).

## Author

Milind Kumar Singh · [GitHub](https://github.com/Milind2805) · [LinkedIn](https://www.linkedin.com/in/milind-kumar-singh-a84173382)
