import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from core import F1Engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.engine = F1Engine()  # load model + data once at startup
    yield


app = FastAPI(title="F1 Race Winner Predictor API", version="1.0.0", lifespan=lifespan)

# Set ALLOWED_ORIGINS on Render to your frontend URL(s), comma-separated.
origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


def engine(request: Request) -> F1Engine:
    return request.app.state.engine


# ── Schemas ───────────────────────────────────────────────────
class DriverEntry(BaseModel):
    driver: str
    team: str
    grid_position: int = Field(1, ge=1, le=22)
    quali_position: int = Field(1, ge=1, le=22)
    standing_points: float = Field(0, ge=0, le=1000)
    home_race: bool = False


class RaceConditions(BaseModel):
    circuit: str
    rainfall: bool = False
    air_temp: float = Field(25, ge=0, le=60)
    track_temp: float = Field(35, ge=0, le=80)


class PredictRequest(DriverEntry, RaceConditions):
    pass


class CompareRequest(RaceConditions):
    a: DriverEntry
    b: DriverEntry


def _run(e: F1Engine, entry: DriverEntry, cond: RaceConditions):
    try:
        return e.predict(
            entry.driver, entry.team, cond.circuit,
            entry.grid_position, entry.quali_position,
            cond.rainfall, cond.air_temp, cond.track_temp,
            entry.standing_points, entry.home_race,
        )
    except ValueError as err:
        raise HTTPException(status_code=422, detail=str(err))


# ── Routes ────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/meta")
def meta(request: Request):
    e = engine(request)
    return {
        "drivers": e.drivers,
        "teams": e.teams,
        "circuits": e.circuits,
        "years": e.years,
    }


@app.post("/predict")
def predict(req: PredictRequest, request: Request):
    e = engine(request)
    pred, prob = _run(e, req, req)
    return {
        "driver": req.driver,
        "predicted_win": pred,
        "win_probability": round(prob, 4),
        "note": (
            "Cadillac has no historical F1 data; a midfield team is used as a rough proxy."
            if req.team == "Cadillac" else None
        ),
    }

@app.post("/compare")
def compare(req: CompareRequest, request: Request):
    e = engine(request)
    _, pa = _run(e, req.a, req)
    _, pb = _run(e, req.b, req)
    return {
        "a": {"driver": req.a.driver, "win_probability": round(pa, 4)},
        "b": {"driver": req.b.driver, "win_probability": round(pb, 4)},
        "favourite": req.a.driver if pa > pb else req.b.driver,
    }


@app.get("/model/importance")
def importance(request: Request):
    return engine(request).importance


@app.get("/model/shap")
def shap_values(request: Request):
    return engine(request).shap


@app.get("/model/metrics")
def metrics(request: Request):
    return engine(request).metrics


def _season_call(fn, *args):
    try:
        return fn(*args)
    except ValueError as err:
        raise HTTPException(status_code=404, detail=str(err))


@app.get("/season/{year}/drivers")
def season_drivers(year: int, request: Request):
    e = engine(request)
    return {
        "progression": _season_call(e.driver_progression, year),
        "wins": _season_call(e.driver_wins, year),
    }


@app.get("/season/{year}/constructors")
def season_constructors(year: int, request: Request):
    e = engine(request)
    return {
        "points": _season_call(e.constructor_points, year),
        "wins": _season_call(e.constructor_wins, year),
    }


class SimulateRequest(RaceConditions):
    entries: list[DriverEntry] = Field(..., min_length=2, max_length=24)


@app.post("/explain")
def explain(req: PredictRequest, request: Request):
    e = engine(request)
    try:
        pred, prob, contribs = e.explain(
            req.driver, req.team, req.circuit,
            req.grid_position, req.quali_position,
            req.rainfall, req.air_temp, req.track_temp,
            req.standing_points, req.home_race,
        )
    except ValueError as err:
        raise HTTPException(status_code=422, detail=str(err))
    return {
        "driver": req.driver,
        "predicted_win": pred,
        "win_probability": round(prob, 4),
        "contributions": contribs,
    }


@app.get("/grid/{season}")
def default_grid(season: int, request: Request):
    return _season_call(engine(request).default_grid, season)


@app.post("/simulate")
def simulate(req: SimulateRequest, request: Request):
    e = engine(request)
    results = e.simulate_grid(
        [x.model_dump() for x in req.entries],
        req.circuit, req.rainfall, req.air_temp, req.track_temp,
    )
    ranked = [r for r in results if r["supported"]]
    return {
        "circuit": req.circuit,
        "predicted_winner": ranked[0]["driver"] if ranked else None,
        "results": results,
    }


class ProjectionRequest(BaseModel):
    remaining_races: list[str] = Field(..., min_length=1, max_length=24)
    n_sims: int = Field(500, ge=100, le=3000)
    seed: int | None = None


@app.post("/season-projection")
def season_projection(req: ProjectionRequest, request: Request):
    return engine(request).project_season(req.remaining_races, req.n_sims, req.seed)
