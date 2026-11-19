"""FastAPI service: play against the solved strategies, and solve push/fold on demand.

Run locally:
    uv run --extra api uvicorn api.main:app --reload

Endpoints (interactive docs at /docs):
    GET  /health                  wake-up / liveness check
    GET  /meta                    hand classes, stack limits, precomputed stacks
    POST /hands                   deal a new hand (Leduc or push/fold)
    GET  /hands/{id}              current view of a hand
    POST /hands/{id}/actions      take an action; the bot replies
    POST /solve/pushfold          Nash push/call charts for any stack depth

Strategies are loaded from ``api/strategies/*.json`` at startup (built by
``scripts/export_strategies.py``); nothing is solved at startup.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from api.hands import Hand, HandStore, InvalidAction, LeducHand, PushFoldHand
from api.solver import MAX_STACK, MIN_STACK, PushFoldSolver, charts
from pokerbot.poker.hand_classes import CLASS_NAMES

STRATEGY_DIR = Path(__file__).parent / "strategies"


def _load(name: str) -> dict:
    with (STRATEGY_DIR / name).open() as f:
        return json.load(f)


LEDUC = _load("leduc.json")
SOLVER = PushFoldSolver(_load("pushfold.json"))
HANDS = HandStore()
RNG = random.SystemRandom()

app = FastAPI(title="pokerbot", description="Play against CFR-solved poker strategies.")
app.add_middleware(
    CORSMiddleware,
    # Comma-separated front-end origins, e.g. "https://pokerbot.vercel.app".
    allow_origins=[
        o.strip()
        for o in os.environ.get("ALLOWED_ORIGINS", "http://localhost:5173").split(",")
        if o.strip()
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# --- Schemas ------------------------------------------------------------------------------


class NewHand(BaseModel):
    game: Literal["leduc", "pushfold"]
    seat: Literal[0, 1] | None = Field(
        None, description="0 = acts first (small blind in push/fold); omitted = random"
    )
    stack: float = Field(10, description="push/fold only: stack depth in big blinds")


class Action(BaseModel):
    action: str = Field(..., max_length=1, description="an action code from legal_actions")


class SolveRequest(BaseModel):
    stack: float = Field(..., ge=MIN_STACK, le=MAX_STACK, description="stack depth in big blinds")


# --- Routes ---------------------------------------------------------------------------------


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/meta")
def meta() -> dict:
    return {
        "hand_classes": CLASS_NAMES,
        "stack_range": [MIN_STACK, MAX_STACK],
        "precomputed_stacks": sorted(SOLVER.precomputed),
        "leduc": {
            "iterations": LEDUC["iterations"],
            "exploitability_mbb": LEDUC["exploitability_mbb"],
        },
    }


@app.post("/hands")
def new_hand(request: NewHand) -> dict:
    seat = request.seat if request.seat is not None else RNG.randrange(2)
    rng = random.Random(RNG.getrandbits(64))
    if request.game == "leduc":
        hand: Hand = LeducHand(LEDUC["strategy"], seat, rng)
    else:
        try:
            strategy = SOLVER.strategy(request.stack)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        hand = PushFoldHand(strategy, SOLVER.normalize(request.stack), seat, rng)
    return HANDS.add(hand).view()


def _get(hand_id: str) -> Hand:
    try:
        return HANDS.get(hand_id)
    except KeyError:
        raise HTTPException(404, "hand not found (it may have expired); deal a new one") from None


@app.get("/hands/{hand_id}")
def get_hand(hand_id: str) -> dict:
    return _get(hand_id).view()


@app.post("/hands/{hand_id}/actions")
def act(hand_id: str, request: Action) -> dict:
    hand = _get(hand_id)
    try:
        hand.act(request.action)
    except InvalidAction as exc:
        raise HTTPException(409, str(exc)) from exc
    return hand.view()


@app.post("/solve/pushfold")
def solve_pushfold(request: SolveRequest) -> dict:
    # A plain `def` endpoint: FastAPI runs it in a worker thread, so a solve
    # doesn't block other requests.
    result = SOLVER.solve(request.stack)
    return {
        **{k: v for k, v in result.items() if k != "strategy"},
        **charts(result["strategy"]),
        "hand_classes": CLASS_NAMES,
    }
