"""FastAPI service exposing the credit risk model.

Run locally:
    uvicorn credit_risk.api:app --reload
Then open http://127.0.0.1:8000/docs for interactive documentation.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from credit_risk.config import BILL_COLS, MODEL_PATH, PAY_AMT_COLS, PAY_STATUS_COLS
from credit_risk.predict import Scorer

logger = logging.getLogger("credit_risk.api")


class Applicant(BaseModel):
    """Account data for one client. Lists hold six months, most recent month first."""

    limit_bal: float = Field(gt=0, description="Credit limit", examples=[50000])
    education: int = Field(ge=1, le=4, description="1 grad school, 2 university, 3 high school, 4 other",
                           examples=[2])
    age: int = Field(ge=18, le=100, examples=[35])
    repayment_status: list[int] = Field(
        min_length=6, max_length=6,
        description="-2 no consumption, -1 paid in full, 0 revolving, 1..8 months of delay",
        examples=[[2, 2, 0, 0, 0, 0]],
    )
    bill_amounts: list[float] = Field(min_length=6, max_length=6,
                                      examples=[[48000, 47000, 45000, 40000, 38000, 30000]])
    payment_amounts: list[float] = Field(min_length=6, max_length=6,
                                         examples=[[0, 1500, 2000, 1500, 1500, 2000]])

    def to_row(self) -> dict:
        row = {"LIMIT_BAL": self.limit_bal, "EDUCATION": self.education, "AGE": self.age}
        row.update(zip(PAY_STATUS_COLS, self.repayment_status, strict=True))
        row.update(zip(BILL_COLS, self.bill_amounts, strict=True))
        row.update(zip(PAY_AMT_COLS, self.payment_amounts, strict=True))
        return row


class RiskDriver(BaseModel):
    feature: str
    description: str
    impact: float


class ScoreResponse(BaseModel):
    probability_of_default: float
    risk_band: str
    decision: str
    top_risk_drivers: list[RiskDriver]
    model_version: str


class BatchRequest(BaseModel):
    applicants: list[Applicant] = Field(min_length=1, max_length=1000)


state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = os.getenv("MODEL_PATH", str(MODEL_PATH))
    try:
        state["scorer"] = Scorer(path)
        logger.info("Loaded model %s from %s", state["scorer"].version, path)
    except FileNotFoundError:
        logger.error("No model at %s. Run `python -m credit_risk.train` first.", path)
    yield
    state.clear()


app = FastAPI(
    title="Credit Risk Scoring API",
    description="Probability of default with risk band, decision and explanation.",
    version="1.0.0",
    lifespan=lifespan,
)


def get_scorer() -> Scorer:
    if "scorer" not in state:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return state["scorer"]


def _score(applicants: list[Applicant]) -> list[ScoreResponse]:
    scorer = get_scorer()
    df = pd.DataFrame([a.to_row() for a in applicants])
    return [ScoreResponse(**vars(r), model_version=scorer.version) for r in scorer.score(df)]


@app.get("/health")
def health() -> dict:
    scorer = state.get("scorer")
    return {"status": "ok" if scorer else "model_missing",
            "model_version": scorer.version if scorer else None}


@app.post("/score", response_model=ScoreResponse)
def score(applicant: Applicant) -> ScoreResponse:
    return _score([applicant])[0]


@app.post("/score/batch", response_model=list[ScoreResponse])
def score_batch(request: BatchRequest) -> list[ScoreResponse]:
    return _score(request.applicants)
