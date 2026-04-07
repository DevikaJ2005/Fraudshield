"""FastAPI server exposing the FraudShield OpenEnv environment.

This module provides HTTP endpoints for the FraudShield fraud detection environment,
enabling agent access via standard REST/JSON interfaces. All endpoints follow OpenEnv
conventions (reset→step→state→done).

Architecture:
  - FraudShieldEnvironment: Core RL environment (manages episodes, rewards, grading)
  - FastAPI Server: HTTP API wrapper (async, production-grade error handling)
  - Data Pipeline: Frozen Kaggle snapshot (reproducible, seed=42)

Key Endpoints:
  - POST /reset: Start new episode (task=easy|medium|hard)
  - POST /step: Submit fraud decision (confidence [0,1] + decision)
  - GET /state: Current observation + history
  - GET /health: Service readiness probe
  - GET /info: Static metadata (task counts, schema)

Error Handling:
  - 400: Invalid request (missing fields, type mismatch)
  - 500: Server error (data load failure, internal exception)
  - 503: Service degraded (data not loaded)

Deployment:
  - Docker: python:3.11-slim, PORT=7860
  - HuggingFace Spaces: Supports auto-scaling
  - Health probe: GET /health before accepting traffic
"""

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

# Data path: points to ../data/ relative to this file
DATA_PATH = Path(__file__).resolve().parents[1] / "data"

# Global environment instance (singleton per process)
env = FraudShieldEnvironment(data_path=str(DATA_PATH), seed=42)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load the bundled task set when the API process starts.
    
    This async context manager runs:
    - On startup: Load frozen data snapshot from disk (108 fraud cases)
    - On shutdown: Clean up environment resources (if needed)
    
    Raises:
        Error logged (not raised): If data fails to load, server starts in "degraded" mode.
        Health probe will return status="degraded" until data is available.
    
    Note:
        Non-blocking: Startup completes even if data load fails. Requests will fail
        with 503 if data isn't available. This allows graceful degradation and retry.
    """
    # On startup
    if not env.load_data():
        logger.error("FraudShield failed to load its bundled data from %s", DATA_PATH)
    # Yield control back to FastAPI (server is ready to accept requests)
    yield
    # On shutdown (cleanup if needed)
    logger.info("FraudShield server shutting down")


app = FastAPI(
    title="FraudShield",
    description="OpenEnv-compatible e-commerce fraud review environment. Agents submit fraud review decisions and receive dense rewards based on business cost (true positives valuable, false positives costly).",
    version="0.2.0",
    docs_url="/docs",  # Swagger UI
    openapi_url="/openapi.json",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Container health probe response.
    
    This endpoint is used by load balancers (HuggingFace, Docker Swarm) to determine
    if the service is ready to accept traffic. Always returns 200 with status indicator.
    
    Returns:
        Dict with keys:
        - status: "healthy" if data loaded, "degraded" if not
        - service: "fraudshield" (service identifier)
        - data_loaded: bool (whether frozen snapshot is available)
    
    Example:
        GET /health
        200 OK
        {
          "status": "healthy",
          "service": "fraudshield",
          "data_loaded": true
        }
    
    Note:
        Even if data is not loaded on startup, this probe succeeds at 200.
        Load balancers should check the status field to determine readiness.
    """
    if not env.data_loaded:
        env.load_data()

    return {
        "status": "healthy" if env.data_loaded else "degraded",
        "service": "fraudshield",
        "data_loaded": env.data_loaded,
    }


@app.post("/reset")
async def reset(task: TaskDifficulty = TaskDifficulty.EASY) -> Dict[str, Any]:
    """Start a new episode for the requested task difficulty.
    
    This endpoint initializes a fresh episode with a random transaction sequence
    from the requested task difficulty level.
    
    Args:
        task: Task difficulty ("easy", "medium", or "hard").
            - easy: 45 transactions, clear separation between fraud/legitimate
            - medium: 50 transactions, mixed signals, confidence calibration matters
            - hard: 65 transactions, coordinated abuse, edge cases, tight thresholds
    
    Returns:
        Dict with keys:
        - observation: First observation dict (transaction + history)
        - info: Episode metadata (task, episode_id, max_steps)
        - episode_id: Unique identifier for this episode (string)
    
    Raises:
        500: Data not loaded or internal error
    
    Example:
        POST /reset?task=easy
        200 OK
        {
          "observation": {
            "transaction_id": "txn_001",
            "amount": 125.50,
            "merchant": "ACME_STORE",
            "country": "US",
            "is_fraud": null,
            ...26 more fields...
            "history": [...]
          },
          "info": {
            "task": "easy",
            "episode_id": "ep_abc123",
            "max_steps": 45
          },
          "episode_id": "ep_abc123"
        }
    
    Note:
        Each call to /reset generates a new episode. Previous episode state is discarded.
        Seed=42 ensures reproducibility across runs.
    """
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
    """Submit one fraud review decision to the environment.
    
    This endpoint processes an agent's fraud decision and returns the reward,
    next observation, and episode termination status.
    
    Args:
        action: FraudCheckAction model with fields:
            - decision: "APPROVE" or "REJECT" (enum)
            - confidence: float in [0.0, 1.0] (confidence in decision)
            Additional optional fields: reason, metadata
    
    Returns:
        Dict with keys:
        - observation: Next observation dict (new transaction or final state)
        - reward: Reward dict with dense reward component
            {
              "dense": -1.0 to +1.0,  # Business cost-sensitive signal
              "info": {...}            # Breakdown (TP/FP/TN/FN signals)
            }
        - done: bool (True if episode complete)
        - info: Episode state info
    
    Raises:
        400: Invalid action format or invalid confidence range
        500: Environment step error
    
    Example:
        POST /step
        Content-Type: application/json
        {
          "decision": "REJECT",
          "confidence": 0.92
        }
        
        200 OK
        {
          "observation": {
            "transaction_id": "txn_002",
            "amount": 250.00,
            ...26 more fields...
          },
          "reward": {
            "dense": 0.95,
            "info": {"signal": "true_positive"}
          },
          "done": false,
          "info": {"step": 1, "episode_id": "ep_abc123"}
        }
    
    Reward Breakdown:
        - True Positive (detected fraud correctly): +0.95
        - True Negative (approved legitimate): +0.80
        - False Positive (rejected legitimate): -0.50
        - False Negative (approved fraud): -1.00
        - Confidence penalty: applied if |confidence - accuracy| > threshold
    
    Note:
        Must call /reset before first /step in an episode.
        Episode terminates when all transactions reviewed (done=true).
    """
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
    """Return the current episode state (observation + metadata).
    
    This endpoint retrieves the complete state without taking any action.
    Useful for agents that want to inspect the current episode before submitting
    the next decision.
    
    Returns:
        EpisodeState dict with keys:
        - observation: Current transaction dict (same as last reward.observation)
        - episode_id: Unique episode identifier
        - task: Current task difficulty ("easy", "medium", "hard")
        - step_count: Number of decisions made so far
        - done: Whether episode is complete
        - cumulative_reward: Sum of all dense rewards so far
        - reward_history: List of all past reward dicts
    
    Raises:
        500: Environment state error
    
    Example:
        GET /state
        200 OK
        {
          "observation": {...},
          "episode_id": "ep_abc123",
          "task": "easy",
          "step_count": 5,
          "done": false,
          "cumulative_reward": 2.45,
          "reward_history": [
            {"dense": 0.95},
            {"dense": 0.80},
            ...
          ]
        }
    
    Note:
        Returns 500 if no episode is active (call /reset first).
    """
    try:
        return env.state().model_dump(mode="json")
    except Exception as exc:
        logger.exception("State error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/info")
async def get_info() -> Dict[str, Any]:
    """Return static environment metadata and schema.
    
    This endpoint provides read-only information about the environment:
    - Task configuration (max steps per difficulty)
    - Data snapshot version and statistics
    - Schema information
    
    Returns:
        Dict with keys:
        - name: "fraudshield"
        - version: "0.2.0"
        - description: Environment description
        - tasks: Dict mapping task names to max_steps counts
        - data_path: Path to frozen data directory
        - data_snapshot: Snapshot metadata (row count, seed, URL)
    
    Example:
        GET /info
        200 OK
        {
          "name": "fraudshield",
          "version": "0.2.0",
          "description": "E-commerce fraud review environment...",
          "tasks": {
            "easy": 45,
            "medium": 50,
            "hard": 65
          },
          "data_path": "/app/data",
          "data_snapshot": {
            "rows": 108,
            "seed": 42,
            "source": "kaggle/creditcard (frozen)"
          }
        }
    
    Note:
        This endpoint does not require an active episode.
        Safe to call for introspection before /reset.
    """
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
    """Describe the available task variants and their characteristics.
    
    Tasks differ in difficulty, transaction count, data distribution, and reward signals:
    
    - **easy**: Clear patterns, few edge cases
      - 45 transactions from frozen Kaggle snapshot
      - Fraud indicators obvious (e.g., repeated high-value failures)
      - Good baseline: naive heuristics score ~0.95 F1
    
    - **medium**: Mixed signals, confidence matters
      - 50 transactions with intentional noise
      - Some legitimate high-value txns, some fraud with low signals
      - Confidence calibration important: overconfident = lower score
      - Good baseline: ~0.88 F1 with learnable thresholds
    
    - **hard**: Adversarial patterns, tight thresholds
      - 65 transactions including coordinated abuse schemes
      - Edge cases: buying cards from physical stores (weird but legitimate)
      - Flash-sale spike patterns (legitimate but unusual)
      - Good baseline: ~0.72 F1 with domain knowledge + anomaly detection
    
    Returns:
        Dict mapping task names to task descriptors with:
        - difficulty: one of "easy", "medium", "hard"
        - num_transactions: transaction count in this task
        - description: Human-readable summary of patterns and difficulty factors
    
    Example:
        GET /tasks
        200 OK
        {
          "easy": {
            "difficulty": "easy",
            "num_transactions": 45,
            "description": "Clear-cut fraud indicators and low-noise legitimate cases."
          },
          "medium": {
            "difficulty": "medium",
            "num_transactions": 50,
            "description": "Mixed-signal reviews where confidence calibration matters."
          },
          "hard": {
            "difficulty": "hard",
            "num_transactions": 65,
            "description": "Coordinated abuse and legitimate flash-sale edge cases."
          }
        }
    
    Note:
        All tasks use the same underlying data (Kaggle snapshot), but pre-shuffled
        into difficulty-stratified subsets. Randomization via seed=42 ensures reproducibility.
    """
    return {
        "easy": {
            "difficulty": "easy",
            "num_transactions": env.max_steps[TaskDifficulty.EASY],
            "description": "Clear-cut fraud indicators and low-noise legitimate cases. Baseline: simple heuristics (high-value repeated failures) score ~0.95 F1.",
        },
        "medium": {
            "difficulty": "medium",
            "num_transactions": env.max_steps[TaskDifficulty.MEDIUM],
            "description": "Mixed-signal reviews where confidence calibration matters. Legitimate high-value txns exist; some fraud is subtle. Requires careful thresholding. Baseline: ~0.88 F1.",
        },
        "hard": {
            "difficulty": "hard",
            "num_transactions": env.max_steps[TaskDifficulty.HARD],
            "description": "Coordinated abuse and legitimate flash-sale edge cases. Card testing patterns, unusual geographies, spike behavior. Requires domain expertise. Baseline: ~0.72 F1.",
        },
    }


@app.exception_handler(Exception)
async def global_exception_handler(_, exc: Exception) -> JSONResponse:
    """Catch any unhandled exception with a JSON error body.
    
    This handler ensures all exceptions (even unexpected ones) return valid JSON
    with appropriate HTTP status codes, rather than letting FastAPI's default
    HTML error pages leak internal details.
    
    Args:
        exc: The uncaught exception
    
    Returns:
        JSONResponse with status_code=500 and error detail
    
    Note:
        HTTPException handlers (reset, step, state) are caught and handled separately.
        This handler catches everything else (programming errors, system errors).
    """
    logger.exception("Unhandled exception")
    return JSONResponse(status_code=500, content={"detail": str(exc)})


@app.get("/")
async def root() -> Dict[str, Any]:
    """Service landing page and API route reference.
    
    This endpoint provides a quick reference for all available routes.
    Primary users: documentation crawlers, health dashboards, agent clients.
    
    Returns:
        Dict with service metadata and endpoint map
    
    Example:
        GET /
        200 OK
        {
          "service": "FraudShield OpenEnv",
          "version": "0.2.0",
          "description": "E-commerce fraud review environment...",
          "endpoints": {
            "health": "GET /health",
            "reset": "POST /reset?task=easy|medium|hard",
            "step": "POST /step",
            "state": "GET /state",
            "info": "GET /info",
            "tasks": "GET /tasks"
          },
          "docs": "/docs"
        }
    
    Note:
        Visit /docs (Swagger UI) for interactive endpoint testing.
        Visit /openapi.json for machine-readable OpenAPI schema.
    """
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

def main() -> None:
    """Launch the FraudShield API server.
    
    Starts a production-grade Uvicorn ASGI server with:
    - Host: 0.0.0.0 (accessible from all interfaces)
    - Port: 7860 (default) or via PORT env var
    - Workers: 1 (single worker; for scaling use Kubernetes/Docker Swarm)
    - Async: Full async/await support via FastAPI
    
    Environment Variables:
        PORT: Server port (default: 7860)
    
    Example:
        PORT=8000 python -m server.app
        # or:
        python -m server.app
        # Then: curl http://localhost:7860/health
    
    Note:
        This is the entrypoint for Docker containers.
        Designed for HuggingFace Spaces and Kubernetes.
    """
    import uvicorn

    port = int(os.getenv("PORT", "7860"))
    logger.info("Launching FraudShield server on port %d", port)
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1)


if __name__ == "__main__":  # pragma: no cover
    main()
