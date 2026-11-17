"""FastAPI service: create experiment configs, ingest events, fetch the analysis readout.

State is held in memory (one process); swap ``Store`` for a database to persist. Run with
``uvicorn expplatform.api:app``.
"""

from __future__ import annotations

import io
import threading
import uuid

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from . import design
from .assignment import assign
from .config import ExperimentConfig
from .report import build_readout


class Store:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.configs: dict[str, ExperimentConfig] = {}
        self.events: dict[str, pd.DataFrame] = {}


class SampleSizeRequest(BaseModel):
    kind: str = Field(pattern="^(means|proportions)$")
    delta: float | None = None  # means: absolute difference
    sd: float | None = None
    p_control: float | None = None  # proportions
    p_treatment: float | None = None
    alpha: float = 0.05
    power: float = 0.8
    ratio: float = 1.0
    r2: float = Field(default=0.0, ge=0.0, lt=1.0)


def _read_events(body: bytes, fmt: str) -> pd.DataFrame:
    try:
        if fmt == "csv":
            return pd.read_csv(io.BytesIO(body))
        if fmt == "parquet":
            return pd.read_parquet(io.BytesIO(body))
    except Exception as exc:
        raise HTTPException(400, f"could not parse {fmt} payload: {exc}") from exc
    raise HTTPException(400, "format must be 'csv' or 'parquet'")


def create_app() -> FastAPI:
    app = FastAPI(title="Experimentation Platform", version="0.1.0")
    store = Store()
    app.state.store = store

    def get_cfg(exp_id: str) -> ExperimentConfig:
        if exp_id not in store.configs:
            raise HTTPException(404, "unknown experiment")
        return store.configs[exp_id]

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "experiments": len(store.configs)}

    @app.post("/design/sample-size")
    def sample_size(req: SampleSizeRequest) -> dict:
        try:
            if req.kind == "means":
                if req.delta is None or req.sd is None:
                    raise HTTPException(422, "means requires delta and sd")
                ss = design.sample_size_means(req.delta, req.sd, req.alpha, req.power,
                                              req.ratio, True, req.r2)
            else:
                if req.p_control is None or req.p_treatment is None:
                    raise HTTPException(422, "proportions requires p_control and p_treatment")
                ss = design.sample_size_proportions(req.p_control, req.p_treatment, req.alpha,
                                                    req.power, req.ratio, True, req.r2)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return {"n_control": ss.n_control, "n_treatment": ss.n_treatment, "total": ss.total}

    @app.post("/experiments", status_code=201)
    def create_experiment(cfg: ExperimentConfig) -> dict:
        exp_id = uuid.uuid4().hex[:12]
        with store.lock:
            store.configs[exp_id] = cfg
        return {"id": exp_id, "config": cfg.model_dump()}

    @app.get("/experiments")
    def list_experiments() -> list[dict]:
        return [{"id": k, "name": v.name, "n_events": len(store.events.get(k, []))}
                for k, v in store.configs.items()]

    @app.get("/experiments/{exp_id}")
    def get_experiment(exp_id: str) -> dict:
        return {"id": exp_id, "config": get_cfg(exp_id).model_dump(),
                "n_events": len(store.events.get(exp_id, []))}

    @app.get("/experiments/{exp_id}/assign/{unit_id}")
    def assign_unit(exp_id: str, unit_id: str) -> dict:
        cfg = get_cfg(exp_id)
        return {"unit_id": unit_id, "variant": assign(unit_id, cfg.salt, cfg.variants)}

    @app.post("/experiments/{exp_id}/events")
    async def ingest(exp_id: str, request: Request, format: str = "csv") -> dict:
        """Append unit-level rows (one per unit) as a raw CSV or parquet request body."""
        cfg = get_cfg(exp_id)
        df = _read_events(await request.body(), format)
        need = {cfg.unit_col, cfg.variant_col}
        if not need <= set(df.columns):
            raise HTTPException(422, f"events must contain columns {sorted(need)}")
        unknown = set(df[cfg.variant_col].unique()) - set(cfg.variants)
        if unknown:
            raise HTTPException(422, f"unknown variants in events: {sorted(map(str, unknown))}")
        with store.lock:
            prev = store.events.get(exp_id)
            merged = df if prev is None else pd.concat([prev, df], ignore_index=True)
            merged = merged.drop_duplicates(subset=cfg.unit_col, keep="last")
            store.events[exp_id] = merged
            total = len(merged)
        return {"ingested": len(df), "total_units": total}

    @app.get("/experiments/{exp_id}/analysis")
    def analysis(exp_id: str) -> dict:
        cfg = get_cfg(exp_id)
        df = store.events.get(exp_id)
        if df is None or df.empty:
            raise HTTPException(409, "no events ingested yet")
        try:
            return build_readout(df, cfg)
        except KeyError as exc:
            raise HTTPException(422, str(exc)) from exc

    return app


app = create_app()
