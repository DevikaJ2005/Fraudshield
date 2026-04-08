"""FastAPI server exposing the FraudShield OpenEnv environment."""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from fraudshield_env import FraudShieldEnvironment
from models import (
    FraudCheckAction,
    FraudCheckObservation,
    InvestigationTargetEnum,
    ResetResult,
    Reward,
    StepResult,
    TaskDifficulty,
    EpisodeState,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).resolve().parents[1] / "data"
APP_VERSION = "0.3.0"

TASK_DESCRIPTIONS = {
    TaskDifficulty.EASY: {
        "difficulty": "easy",
        "description": (
            "Low-noise triage cases with strong single-transaction signals. "
            "A good agent should decide quickly and only investigate when a high-value signal conflicts."
        ),
    },
    TaskDifficulty.MEDIUM: {
        "difficulty": "medium",
        "description": (
            "Mixed-signal fraud reviews where partial observability matters. "
            "Investigations help separate risky-but-legitimate traffic from genuine abuse."
        ),
    },
    TaskDifficulty.HARD: {
        "difficulty": "hard",
        "description": (
            "Coordinated abuse and flash-sale edge cases. "
            "Agents must balance investigation budget, ambiguity, and business cost."
        ),
    },
}

env = FraudShieldEnvironment(data_path=str(DATA_PATH), seed=42)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load the bundled task set when the API process starts."""

    if not env.load_data():
        logger.error("FraudShield failed to load its bundled data from %s", DATA_PATH)
    yield
    logger.info("FraudShield server shutting down")


app = FastAPI(
    title="FraudShield",
    description=(
        "OpenEnv-compatible marketplace fraud investigation environment with typed "
        "actions, dense rewards, and optional multi-step evidence gathering."
    ),
    version=APP_VERSION,
    docs_url="/docs",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)


def _ensure_data_loaded() -> None:
    if not env.data_loaded and not env.load_data():
        raise RuntimeError(f"FraudShield failed to load data from {DATA_PATH}")


def _task_payload() -> Dict[str, Any]:
    return {
        task.value: {
            **TASK_DESCRIPTIONS[task],
            "num_transactions": env.case_counts[task],
            "max_steps": env.max_steps[task],
            "investigation_budget_per_case": env.investigation_budget_per_case[task],
        }
        for task in TaskDifficulty
    }


def _metadata_payload() -> Dict[str, Any]:
    _ensure_data_loaded()
    return {
        "name": "fraudshield",
        "title": "FraudShield",
        "version": APP_VERSION,
        "description": (
            "Agentic marketplace trust-and-safety environment for transaction review. "
            "Agents may either make a final decision or spend limited budget to reveal "
            "extra evidence before deciding."
        ),
        "transport": {
            "rest": {
                "health": "/health",
                "reset": "/reset",
                "step": "/step",
                "state": "/state",
                "info": "/info",
                "tasks": "/tasks",
                "metadata": "/metadata",
                "schema": "/schema",
            },
            "mcp": "/mcp",
            "openapi": "/openapi.json",
        },
        "action_families": ["decide", "investigate"],
        "investigation_targets": [target.value for target in InvestigationTargetEnum],
        "tasks": _task_payload(),
        "data_snapshot": env.data_loader.get_bundle_summary(),
    }


def _schema_payload() -> Dict[str, Any]:
    action_schema = FraudCheckAction.model_json_schema()
    observation_schema = FraudCheckObservation.model_json_schema()
    reward_schema = Reward.model_json_schema()
    state_schema = EpisodeState.model_json_schema()
    reset_schema = ResetResult.model_json_schema()
    step_schema = StepResult.model_json_schema()
    return {
        "name": "fraudshield",
        "version": APP_VERSION,
        "action": action_schema,
        "observation": observation_schema,
        "reward": reward_schema,
        "state": state_schema,
        "reset_result": reset_schema,
        "step_result": step_schema,
        "models": {
            "action": action_schema,
            "observation": observation_schema,
            "reward": reward_schema,
            "state": state_schema,
            "reset_result": reset_schema,
            "step_result": step_schema,
        },
        "tasks": _task_payload(),
    }


def _mcp_success(request_id: Any, result: Dict[str, Any]) -> JSONResponse:
    return JSONResponse({"jsonrpc": "2.0", "id": request_id, "result": result})


def _mcp_error(request_id: Any, code: int, message: str) -> JSONResponse:
    return JSONResponse({"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}})


def _mcp_tool_result(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=True)}],
        "structuredContent": payload,
        "isError": False,
    }


def _mcp_tool_descriptors() -> list[Dict[str, Any]]:
    task_enum_values = [task.value for task in TaskDifficulty]
    return [
        {
            "name": "environment.reset",
            "description": "Start a new episode for easy, medium, or hard review.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "enum": task_enum_values, "default": TaskDifficulty.EASY.value}
                },
            },
        },
        {
            "name": "environment.step",
            "description": "Submit either a final decision or an investigation request.",
            "inputSchema": FraudCheckAction.model_json_schema(),
        },
        {
            "name": "environment.state",
            "description": "Get the current episode state without changing it.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "environment.info",
            "description": "Read static environment information and snapshot details.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "environment.tasks",
            "description": "List the graded task variants and budgets.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "environment.metadata",
            "description": "Read runtime metadata for OpenEnv and MCP clients.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "environment.schema",
            "description": "Read JSON schemas for the typed environment models.",
            "inputSchema": {"type": "object", "properties": {}},
        },
    ]


def _run_mcp_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    _ensure_data_loaded()

    if name == "environment.reset":
        task = arguments.get("task", TaskDifficulty.EASY.value)
        result = env.reset(str(task))
        return {
            "observation": result.observation.model_dump(mode="json"),
            "info": result.info,
            "episode_id": env.episode_id,
        }
    if name == "environment.step":
        action = FraudCheckAction.model_validate(arguments)
        result = env.step(action)
        return {
            "observation": result.observation.model_dump(mode="json"),
            "reward": result.reward.model_dump(mode="json"),
            "done": result.done,
            "info": result.info,
        }
    if name == "environment.state":
        return env.state().model_dump(mode="json")
    if name == "environment.info":
        return {
            "name": "fraudshield",
            "version": APP_VERSION,
            "description": "Marketplace fraud investigation environment built from a frozen public-data snapshot.",
            "tasks": {
                task.value: {
                    "max_steps": env.max_steps[task],
                    "num_transactions": env.case_counts[task],
                }
                for task in TaskDifficulty
            },
            "data_path": str(DATA_PATH),
            "data_snapshot": env.data_loader.get_bundle_summary(),
        }
    if name == "environment.tasks":
        return _task_payload()
    if name == "environment.metadata":
        return _metadata_payload()
    if name == "environment.schema":
        return _schema_payload()
    raise ValueError(f"Unknown MCP tool: {name}")


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    if not env.data_loaded:
        env.load_data()

    return {
        "status": "healthy" if env.data_loaded else "degraded",
        "service": "fraudshield",
        "data_loaded": env.data_loaded,
    }


@app.post("/reset")
async def reset(task: TaskDifficulty = TaskDifficulty.EASY) -> Dict[str, Any]:
    try:
        _ensure_data_loaded()
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
    try:
        _ensure_data_loaded()
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
    try:
        _ensure_data_loaded()
        return env.state().model_dump(mode="json")
    except Exception as exc:
        logger.exception("State error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/info")
async def get_info() -> Dict[str, Any]:
    _ensure_data_loaded()
    return {
        "name": "fraudshield",
        "version": APP_VERSION,
        "description": "Marketplace fraud investigation environment built from a frozen public-data snapshot.",
        "tasks": {
            task.value: {
                "max_steps": env.max_steps[task],
                "num_transactions": env.case_counts[task],
            }
            for task in TaskDifficulty
        },
        "data_path": str(DATA_PATH),
        "data_snapshot": env.data_loader.get_bundle_summary(),
    }


@app.get("/tasks")
async def get_tasks() -> Dict[str, Any]:
    _ensure_data_loaded()
    return _task_payload()


@app.get("/metadata")
async def get_metadata() -> Dict[str, Any]:
    return _metadata_payload()


@app.get("/schema")
async def get_schema() -> Dict[str, Any]:
    _ensure_data_loaded()
    return _schema_payload()


@app.post("/mcp")
async def mcp_endpoint(request: Dict[str, Any]) -> JSONResponse:
    request_id = request.get("id")
    method = request.get("method")
    params = request.get("params", {}) or {}

    try:
        if method == "initialize":
            return _mcp_success(
                request_id,
                {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {"tools": {}, "prompts": {}, "resources": {}},
                    "serverInfo": {"name": "fraudshield", "version": APP_VERSION},
                },
            )
        if method in {"notifications/initialized", "initialized"}:
            return _mcp_success(request_id, {})
        if method == "ping":
            return _mcp_success(request_id, {})
        if method == "tools/list":
            return _mcp_success(request_id, {"tools": _mcp_tool_descriptors()})
        if method == "tools/call":
            tool_name = params.get("name")
            if not tool_name:
                return _mcp_error(request_id, -32602, "tools/call requires a tool name")
            arguments = params.get("arguments", {}) or {}
            return _mcp_success(request_id, _mcp_tool_result(_run_mcp_tool(tool_name, arguments)))
        if method == "resources/list":
            return _mcp_success(request_id, {"resources": []})
        if method == "prompts/list":
            return _mcp_success(request_id, {"prompts": []})
        return _mcp_error(request_id, -32601, f"Method not found: {method}")
    except Exception as exc:
        logger.exception("MCP error")
        return _mcp_error(request_id, -32000, str(exc))


@app.exception_handler(Exception)
async def global_exception_handler(_, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception")
    return JSONResponse(status_code=500, content={"detail": str(exc)})


@app.get("/")
async def root() -> Dict[str, Any]:
    return {
        "service": "FraudShield OpenEnv",
        "version": APP_VERSION,
        "description": (
            "Marketplace fraud investigation environment with final-decision actions "
            "and optional budgeted evidence gathering."
        ),
        "endpoints": {
            "health": "GET /health",
            "reset": "POST /reset?task=easy|medium|hard",
            "step": "POST /step",
            "state": "GET /state",
            "info": "GET /info",
            "tasks": "GET /tasks",
            "metadata": "GET /metadata",
            "schema": "GET /schema",
            "mcp": "POST /mcp",
        },
        "docs": "/docs",
    }


def main() -> None:
    import uvicorn

    port = int(os.getenv("PORT", "7860"))
    logger.info("Launching FraudShield server on port %d", port)
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1)


if __name__ == "__main__":  # pragma: no cover
    main()
