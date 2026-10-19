# F1 Race Winner Predictor

Predict who wins a Formula 1 race from qualifying, weather and championship standings, then see *why* the model decided that, and how it would have done on a season it never saw.

**Live demo:** https://f1-race-winner-predictor-app-fronte.vercel.app
**API docs:** https://f1-race-winner-predictor-app.onrender.com/docs

> The API runs on a free Render instance, so the first request after a quiet period can take up to a minute while it wakes up. The app shows a lights-out loader while it does.

## What it does

- **Prediction:** win probability for one driver on a team-coloured card with an animated gauge, plus a SHAP chart of which inputs pushed the prediction up or down. A clickable 2026 starting grid loads any driver into the form, and a what-if explorer lets you drag grid position and weather sliders and watch the probability change.
- **Head-to-head:** two driver cards in their team colours and a bar showing each driver's share of their combined win chance, with one-click classic rivalries.
- **Race simulator:** a team-coloured timing tower with win probabilities for the full 2026 grid (or the 2023 grid) and an editable lineup, with reverse, shuffle and reset shortcuts.
- **Season projection:** Monte Carlo simulation of the remaining races, shown as a chart of projected final points with a 10th-90th percentile band, alongside current standings and title chances.
- **Backtest:** the model trained on 2019-2024 and judged race by race on 2025, compared with a simple "pick the pole sitter" baseline.

Championship standings are fetched live from the [Jolpica-F1 API](https://api.jolpi.ca/ergast/f1/) and cached for 6 hours. If the API is unreachable, the app falls back to the standings saved in `backend/data/default_grids.json`.

## Architecture

```
Browser (Vercel)  ──HTTPS/JSON──▶  FastAPI (Render)  ──▶  XGBoost model, scaler, encoders
 index.html                          backend/main.py       backend/data/*.pkl
 backtest.json (static)              backend/core.py   ──▶  Jolpica-F1 API (live standings)
```

- `backend/core.py` holds all model and data logic, with no web code.
- `backend/main.py` exposes it as a REST API (FastAPI, Pydantic validation, CORS).
- `frontend/index.html` is a single-page UI in plain HTML, CSS and JavaScript. Chart.js and canvas-confetti load from a CDN, so there is no build step.
- `frontend/backtest.json` is generated offline by `training/backtest_data.py` and served as a static file, so the backtest costs the API nothing.
- A lights-out loader covers the free instance's cold start.

## How the model works

- **Features (10):** driver, team, circuit, grid position, qualifying position, rain, air temperature, track temperature, championship points, home race.
- **Model:** XGBoost classifier trained on 2019-2025 race weekends (3,031 driver-race entries). The training set is balanced with SMOTE because only about 1 in 20 entries is a
