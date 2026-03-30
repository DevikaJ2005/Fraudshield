"""FastAPI server exposing the FraudShield environment."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from fraudshield_env import FraudShieldEnvironment
from models import FraudCheckAction, TaskDifficulty

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).resolve().parents[1] / "data"
env = FraudShieldEnvironment(data_path=str(DATA_PATH), seed=42)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load the bundled task set when the API process starts."""

    if not env.load_data():
        logger.error("FraudShield failed to load its bundled data from %s", DATA_PATH)
    yield


app = FastAPI(
    title="FraudShield",
    description="OpenEnv-compatible e-commerce fraud review environment.",
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Container health probe."""

    if not env.data_loaded:
        env.load_data()

    return {
        "status": "healthy" if env.data_loaded else "degraded",
        "service": "fraudshield",
        "data_loaded": env.data_loaded,
    }


@app.post("/reset")
async def reset(task: TaskDifficulty = TaskDifficulty.EASY) -> Dict[str, Any]:
    """Start a new episode for the requested task."""

    try:
        result = env.reset(task.value)
        return {
            "observation": result.observation.model_dump(mode="json"),
            "info": result.info,
            "episode_id": env.episode_id,
        }
    except Exception as exc:
        logger.exception("Reset error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/step")
async def step(action: FraudCheckAction) -> Dict[str, Any]:
    """Submit one action to the environment."""

    try:
        result = env.step(action)
        return {
            "observation": result.observation.model_dump(mode="json"),
            "reward": result.reward.model_dump(mode="json"),
            "done": result.done,
            "info": result.info,
        }
    except Exception as exc:
        logger.exception("Step error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/state")
async def get_state() -> Dict[str, Any]:
    """Return the current episode state."""

    try:
        return env.state().model_dump(mode="json")
    except Exception as exc:
        logger.exception("State error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/info")
async def get_info() -> Dict[str, Any]:
    """Return static environment metadata."""

    return {
        "name": "fraudshield",
        "version": "0.2.0",
        "description": "E-commerce fraud review environment built from a frozen public-data snapshot.",
        "tasks": {
            task.value: {"max_steps": max_steps}
            for task, max_steps in env.max_steps.items()
        },
        "data_path": str(DATA_PATH),
        "data_snapshot": env.data_loader.get_bundle_summary(),
    }


@app.get("/tasks")
async def get_tasks() -> Dict[str, Any]:
    """Describe the available task variants."""

    return {
        "easy": {
            "difficulty": "easy",
            "num_transactions": env.max_steps[TaskDifficulty.EASY],
            "description": "Clear-cut fraud indicators and low-noise legitimate cases.",
        },
        "medium": {
            "difficulty": "medium",
            "num_transactions": env.max_steps[TaskDifficulty.MEDIUM],
            "description": "Mixed-signal reviews where confidence calibration matters.",
        },
        "hard": {
            "difficulty": "hard",
            "num_transactions": env.max_steps[TaskDifficulty.HARD],
            "description": "Coordinated abuse and legitimate flash-sale edge cases.",
        },
    }


@app.exception_handler(Exception)
async def global_exception_handler(_, exc: Exception) -> JSONResponse:
    """Catch any unhandled exception with a JSON error body."""

    logger.exception("Unhandled exception")
    return JSONResponse(status_code=500, content={"detail": str(exc)})


@app.get("/")
async def root() -> Dict[str, Any]:
    """Service landing page."""

    return {
        "service": "FraudShield OpenEnv",
        "version": "0.2.0",
        "description": "E-commerce fraud review environment for agent training and evaluation.",
        "endpoints": {
            "health": "GET /health",
            "reset": "POST /reset?task=easy|medium|hard",
            "step": "POST /step",
            "state": "GET /state",
            "info": "GET /info",
            "tasks": "GET /tasks",
        },
        "docs": "/docs",
    }


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")), workers=1)
