"""FastAPI server exposing the FraudShield OpenEnv API."""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

from fraudshield_env import FraudShieldEnvironment, TASK_CONFIG
from llm_agent import SnapshotCalibratedFraudDetectionAgent
from models import (
    ActionTypeEnum,
    CaseScreenEnum,
    EpisodeState,
    FraudCheckAction,
    FraudCheckObservation,
    ResetResult,
    Reward,
    StepResult,
    TaskDifficulty,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data"
APP_VERSION = "0.6.0"

env = FraudShieldEnvironment(data_path=str(DATA_PATH), seed=42)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load the frozen snapshot on startup."""

    if not env.load_data():
        logger.error("FraudShield failed to load bundled data from %s", DATA_PATH)
    yield


app = FastAPI(
    title="FraudShield",
    description=(
        "Simulated fraud-investigation environment for OpenEnv. Agents operate under partial "
        "observability, reveal evidence with investigation tools, and route cases under limited budgets."
    ),
    version=APP_VERSION,
    docs_url="/docs",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)


def _explorer_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>FraudShield Explorer</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f7fb;
      --panel: #ffffff;
      --line: #d7e0ea;
      --ink: #102033;
      --muted: #536579;
      --accent: #0b6dff;
      --accent-soft: #dbe9ff;
      --success: #1f7a4d;
      --warning: #b4690e;
      --danger: #b42318;
      --shadow: 0 10px 24px rgba(16, 32, 51, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", system-ui, sans-serif;
      background: linear-gradient(180deg, #eef4ff 0%, var(--bg) 280px);
      color: var(--ink);
    }
    .wrap {
      max-width: 1180px;
      margin: 0 auto;
      padding: 28px 18px 44px;
    }
    .hero {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 24px;
      box-shadow: var(--shadow);
      margin-bottom: 20px;
    }
    .hero h1 { margin: 0 0 8px; font-size: 2rem; }
    .hero p { margin: 0; color: var(--muted); line-height: 1.5; }
    .hero-links {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin-top: 16px;
    }
    .hero-links a {
      color: var(--accent);
      text-decoration: none;
      font-weight: 600;
    }
    .grid {
      display: grid;
      grid-template-columns: 340px minmax(0, 1fr);
      gap: 20px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 18px;
      box-shadow: var(--shadow);
    }
    .panel h2, .panel h3 {
      margin: 0 0 12px;
      font-size: 1rem;
    }
    .field { margin-bottom: 12px; }
    .field label {
      display: block;
      font-size: 0.88rem;
      color: var(--muted);
      margin-bottom: 6px;
      font-weight: 600;
    }
    select, textarea, input {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px 12px;
      font: inherit;
      background: #fff;
      color: var(--ink);
    }
    textarea { min-height: 92px; resize: vertical; }
    .actions {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-top: 10px;
    }
    button {
      border: 0;
      border-radius: 10px;
      background: var(--accent);
      color: #fff;
      padding: 10px 12px;
      font: inherit;
      font-weight: 600;
      cursor: pointer;
    }
    button.secondary {
      background: #edf2fa;
      color: var(--ink);
      border: 1px solid var(--line);
    }
    button:disabled {
      opacity: 0.45;
      cursor: not-allowed;
    }
    .status {
      margin-top: 12px;
      padding: 10px 12px;
      border-radius: 10px;
      background: #f7f9fc;
      color: var(--muted);
      font-size: 0.94rem;
      border: 1px solid var(--line);
    }
    .cards {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }
    .card {
      border: 1px solid var(--line);
      background: #fbfdff;
      border-radius: 14px;
      padding: 12px;
    }
    .card .label {
      font-size: 0.78rem;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    .card .value {
      margin-top: 6px;
      font-size: 1rem;
      font-weight: 700;
      word-break: break-word;
    }
    .chips {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 8px 0 0;
    }
    .chip {
      padding: 6px 10px;
      border-radius: 999px;
      background: var(--accent-soft);
      color: #154284;
      font-size: 0.84rem;
      font-weight: 600;
    }
    .trace-list {
      display: grid;
      gap: 12px;
      margin-top: 16px;
    }
    .trace-item {
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px;
      background: #fff;
    }
    .trace-item h4 {
      margin: 0 0 8px;
      font-size: 0.95rem;
    }
    .trace-item p {
      margin: 6px 0;
      color: var(--muted);
      line-height: 1.45;
    }
    pre {
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 0.82rem;
      line-height: 1.45;
      color: #163047;
      background: #f7f9fc;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px;
      overflow: auto;
    }
    .muted { color: var(--muted); }
    .inline {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
    }
    @media (max-width: 920px) {
      .grid { grid-template-columns: 1fr; }
      .actions { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>FraudShield Explorer</h1>
      <p>
        This is a lightweight browser UI for the OpenEnv environment. It lets you reset a task,
        reveal hidden evidence step by step, and see how the heuristic baseline investigates before
        making a final routing decision.
      </p>
      <div class="hero-links">
        <a href="/docs" target="_blank" rel="noreferrer">Open API docs</a>
        <a href="/metadata" target="_blank" rel="noreferrer">View metadata JSON</a>
        <a href="/schema" target="_blank" rel="noreferrer">View schema JSON</a>
      </div>
    </section>

    <div class="grid">
      <aside class="panel">
        <h2>Controls</h2>

        <div class="field">
          <label for="task">Task</label>
          <select id="task">
            <option value="easy">easy</option>
            <option value="medium" selected>medium</option>
            <option value="hard">hard</option>
          </select>
        </div>

        <div class="actions">
          <button id="resetBtn">Reset Episode</button>
          <button id="stateBtn" class="secondary">Refresh State</button>
          <button id="traceBtn" class="secondary">Run Heuristic Walkthrough</button>
        </div>

        <div class="field" style="margin-top:18px;">
          <label for="reasoning">Reasoning</label>
          <textarea id="reasoning">Review the visible evidence before taking the next action.</textarea>
        </div>

        <div class="field">
          <label for="noteText">Case Note</label>
          <textarea id="noteText">Reviewed the currently visible evidence before selecting the next workflow step.</textarea>
        </div>

        <div class="field">
          <label for="resolution">Resolution</label>
          <select id="resolution">
            <option value="approve">approve</option>
            <option value="block" selected>block</option>
            <option value="hold">hold</option>
            <option value="request_docs">request_docs</option>
            <option value="escalate">escalate</option>
          </select>
        </div>

        <h3 style="margin-top:18px;">Action Buttons</h3>
        <div class="actions">
          <button data-action="review_transaction">review_transaction</button>
          <button data-action="fetch_customer_profile">fetch_customer_profile</button>
          <button data-action="fetch_merchant_profile">fetch_merchant_profile</button>
          <button data-action="fetch_network_graph">fetch_network_graph</button>
          <button data-action="check_policy">check_policy</button>
          <button data-action="add_case_note">add_case_note</button>
          <button data-action="resolve_case">resolve_case</button>
        </div>

        <div id="status" class="status">Reset a task to begin exploring the environment.</div>
      </aside>

      <main class="panel">
        <h2>Current Observation</h2>
        <div id="overview" class="cards"></div>

        <div class="panel" style="box-shadow:none; padding:0; border:0; background:transparent;">
          <h3>Visible Workflow Hints</h3>
          <div id="hints" class="chips"></div>
        </div>

        <div class="panel" style="box-shadow:none; padding:0; border:0; background:transparent;">
          <h3>Revealed Evidence</h3>
          <pre id="evidence">{}</pre>
        </div>

        <div class="panel" style="box-shadow:none; padding:0; border:0; background:transparent;">
          <h3>Current State Snapshot</h3>
          <pre id="state">No active episode yet.</pre>
        </div>

        <div class="panel" style="box-shadow:none; padding:0; border:0; background:transparent;">
          <div class="inline">
            <h3 style="margin:0;">Heuristic Walkthrough</h3>
            <span class="muted">Useful before RL training so you can see the current baseline behavior.</span>
          </div>
          <div id="trace" class="trace-list"></div>
        </div>
      </main>
    </div>
  </div>

  <script>
    let currentObservation = null;

    const statusEl = document.getElementById("status");
    const overviewEl = document.getElementById("overview");
    const hintsEl = document.getElementById("hints");
    const evidenceEl = document.getElementById("evidence");
    const stateEl = document.getElementById("state");
    const traceEl = document.getElementById("trace");

    function setStatus(message, kind = "neutral") {
      statusEl.textContent = message;
      const palette = {
        neutral: ["#f7f9fc", "#536579"],
        success: ["#ecfdf3", "#1f7a4d"],
        warning: ["#fff7ed", "#b4690e"],
        error: ["#fef3f2", "#b42318"],
      };
      const [bg, color] = palette[kind] || palette.neutral;
      statusEl.style.background = bg;
      statusEl.style.color = color;
    }

    function pretty(value) {
      return JSON.stringify(value, null, 2);
    }

    function renderObservation(observation) {
      currentObservation = observation;
      const budget = observation.app_context?.investigation_budget_remaining ?? "n/a";
      const timestamp = observation.app_context?.timestamp ?? "n/a";
      const category = observation.app_context?.item_category ?? "n/a";
      const linked = observation.linked_case_ids?.length ? observation.linked_case_ids.join(", ") : "hidden / none";

      const cards = [
        ["Case ID", observation.case_id],
        ["Task", observation.task_name],
        ["Workflow View", observation.current_screen],
        ["Episode Step", observation.episode_step],
        ["Amount", "$" + observation.case_summary.amount_usd],
        ["Category", category],
        ["Timestamp", timestamp],
        ["Budget Left", budget],
        ["Remaining Steps", observation.remaining_steps],
        ["Remaining SLA", observation.remaining_sla],
        ["Note Required", observation.note_required ? "yes" : "no"],
        ["Linked Cases", linked],
      ];

      overviewEl.innerHTML = cards.map(([label, value]) => `
        <div class="card">
          <div class="label">${label}</div>
          <div class="value">${value}</div>
        </div>
      `).join("");

      const hints = [
        "queue_reason: " + observation.case_summary.queue_reason,
        ...observation.visible_panels.map((item) => "panel: " + item),
        ...observation.allowed_actions.map((item) => "allowed: " + item),
      ];
      hintsEl.innerHTML = hints.map((hint) => `<span class="chip">${hint}</span>`).join("");
      evidenceEl.textContent = pretty(observation.revealed_evidence || {});
      updateActionButtons();
    }

    async function fetchState() {
      const response = await fetch("/state");
      const data = await response.json();
      stateEl.textContent = pretty(data);
      return data;
    }

    async function resetEpisode() {
      const task = document.getElementById("task").value;
      setStatus("Resetting " + task + " episode...", "neutral");
      const response = await fetch("/reset?task=" + encodeURIComponent(task), { method: "POST" });
      const data = await response.json();
      if (!response.ok) {
        setStatus(data.detail || "Reset failed.", "error");
        return;
      }
      renderObservation(data.observation);
      await fetchState();
      traceEl.innerHTML = "";
      setStatus("Episode ready. Start with review_transaction to reveal the transaction trace.", "success");
    }

    async function step(actionType) {
      if (!currentObservation) {
        setStatus("Reset an episode first.", "warning");
        return;
      }

      const payload = {
        case_id: currentObservation.case_id,
        action_type: actionType,
        reasoning: document.getElementById("reasoning").value.trim(),
      };

      if (actionType === "add_case_note") {
        payload.note_text = document.getElementById("noteText").value.trim();
      }
      if (actionType === "resolve_case") {
        payload.resolution = document.getElementById("resolution").value;
      }

      setStatus("Submitting " + actionType + "...", "neutral");
      const response = await fetch("/step", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await response.json();
      if (!response.ok) {
        setStatus(data.detail || "Step failed.", "error");
        return;
      }
      renderObservation(data.observation);
      await fetchState();
      const reward = data.reward;
      const doneSuffix = data.done ? " Episode finished." : "";
      setStatus(
        `${actionType} -> reward ${reward.value}. ${reward.reason}${doneSuffix}`,
        data.done ? "success" : "neutral"
      );
    }

    function updateActionButtons() {
      const allowed = new Set((currentObservation?.allowed_actions || []).map(String));
      document.querySelectorAll("button[data-action]").forEach((button) => {
        button.disabled = !allowed.has(button.dataset.action);
      });
    }

    async function runTrace() {
      const task = document.getElementById("task").value;
      setStatus("Running heuristic walkthrough for " + task + "...", "neutral");
      const response = await fetch("/demo/trace?task=" + encodeURIComponent(task));
      const data = await response.json();
      if (!response.ok) {
        setStatus(data.detail || "Could not run heuristic walkthrough.", "error");
        return;
      }

      const cards = data.action_trace.map((item) => `
        <div class="trace-item">
          <h4>Step ${item.step}: ${item.action.action_type} on ${item.action.case_id}</h4>
          <p><strong>Reasoning:</strong> ${item.action.reasoning || "(no reasoning text)"}</p>
          <p><strong>Reward:</strong> ${item.reward.value} | <strong>Why:</strong> ${item.reward.reason}</p>
          ${item.action.resolution ? `<p><strong>Resolution:</strong> ${item.action.resolution}</p>` : ""}
          ${item.action.note_text ? `<p><strong>Note:</strong> ${item.action.note_text}</p>` : ""}
        </div>
      `).join("");
      traceEl.innerHTML = cards || '<div class="trace-item"><p>No trace steps returned.</p></div>';
      setStatus("Heuristic walkthrough finished. You can compare this flow with your later RL-trained policy.", "success");
    }

    document.getElementById("resetBtn").addEventListener("click", resetEpisode);
    document.getElementById("stateBtn").addEventListener("click", fetchState);
    document.getElementById("traceBtn").addEventListener("click", runTrace);
    document.querySelectorAll("button[data-action]").forEach((button) => {
      button.addEventListener("click", () => step(button.dataset.action));
    });
  </script>
</body>
</html>"""


def _ensure_data_loaded() -> None:
    if not env.data_loaded and not env.load_data():
        raise RuntimeError(f"FraudShield failed to load data from {DATA_PATH}")


def _task_payload() -> Dict[str, Any]:
    return {
        task.value: {
            "difficulty": task.value,
            "description": TASK_CONFIG[task]["description"],
            "num_cases": TASK_CONFIG[task]["num_cases"],
            "max_steps": TASK_CONFIG[task]["max_steps"],
            "sla_limit": TASK_CONFIG[task]["sla_limit"],
            "investigation_budget": TASK_CONFIG[task]["investigation_budget"],
        }
        for task in TaskDifficulty
    }


def _workflow_views() -> list[str]:
    return [screen.value for screen in CaseScreenEnum]


def _metadata_payload() -> Dict[str, Any]:
    _ensure_data_loaded()
    return {
        "name": "fraudshield",
        "title": "FraudShield",
        "version": APP_VERSION,
        "description": app.description,
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
        "action_families": [action.value for action in ActionTypeEnum],
        "workflow_views": _workflow_views(),
        "tasks": _task_payload(),
        "data_snapshot": env.data_loader.get_bundle_summary(),
    }


def _schema_payload() -> Dict[str, Any]:
    return {
        "name": "fraudshield",
        "version": APP_VERSION,
        "action": FraudCheckAction.model_json_schema(),
        "observation": FraudCheckObservation.model_json_schema(),
        "reward": Reward.model_json_schema(),
        "state": EpisodeState.model_json_schema(),
        "reset_result": ResetResult.model_json_schema(),
        "step_result": StepResult.model_json_schema(),
        "tasks": _task_payload(),
    }


def _demo_trace_payload(task: TaskDifficulty) -> Dict[str, Any]:
    demo_env = FraudShieldEnvironment(data_path=str(DATA_PATH), seed=42)
    demo_env.load_data()
    reset_result = demo_env.reset(task.value)
    agent = SnapshotCalibratedFraudDetectionAgent()

    observation = reset_result.observation
    action_trace: list[Dict[str, Any]] = []
    max_steps = TASK_CONFIG[task]["max_steps"]

    while not demo_env.is_done and demo_env.step_count < max_steps:
        action = agent.decide(observation)
        result = demo_env.step(action)
        action_trace.append(
            {
                "step": demo_env.step_count,
                "action": action.model_dump(mode="json"),
                "reward": result.reward.model_dump(mode="json"),
                "done": result.done,
            }
        )
        observation = result.observation

    return {
        "task": task.value,
        "agent_name": agent.name,
        "initial_observation": reset_result.observation.model_dump(mode="json"),
        "action_trace": action_trace,
        "episode_report": demo_env.get_episode_report(),
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
    task_values = [task.value for task in TaskDifficulty]
    return [
        {
            "name": "environment.reset",
            "description": "Start a new easy, medium, or hard FraudShield episode.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "enum": task_values, "default": TaskDifficulty.EASY.value}
                },
            },
        },
        {
            "name": "environment.step",
            "description": "Submit one investigation or resolution action for the active case.",
            "inputSchema": FraudCheckAction.model_json_schema(),
        },
        {
            "name": "environment.state",
            "description": "Read the full current episode state.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "environment.info",
            "description": "Read static environment information and dataset metadata.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "environment.tasks",
            "description": "List the available graded tasks.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "environment.metadata",
            "description": "Read runtime metadata for OpenEnv clients.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "environment.schema",
            "description": "Read the JSON schema for the typed models.",
            "inputSchema": {"type": "object", "properties": {}},
        },
    ]


def _run_mcp_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    _ensure_data_loaded()

    if name == "environment.reset":
        task = arguments.get("task", TaskDifficulty.EASY.value)
        result = env.reset(str(task))
        return {"observation": result.observation.model_dump(mode="json"), "info": result.info}
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
            "tasks": _task_payload(),
            "workflow_views": _workflow_views(),
            "data_snapshot": env.data_loader.get_bundle_summary(),
        }
    if name == "environment.tasks":
        return _task_payload()
    if name == "environment.metadata":
        return _metadata_payload()
    if name == "environment.schema":
        return _schema_payload()
    raise ValueError(f"Unknown MCP tool: {name}")


@app.get("/", response_class=HTMLResponse)
async def explorer() -> HTMLResponse:
    return HTMLResponse(_explorer_html())


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    if not env.data_loaded:
        env.load_data()
    return {
        "status": "healthy" if env.data_loaded else "degraded",
        "service": "fraudshield",
        "data_loaded": env.data_loaded,
        "workflow_views": _workflow_views(),
    }


@app.post("/reset")
async def reset(task: TaskDifficulty = TaskDifficulty.EASY) -> Dict[str, Any]:
    try:
        _ensure_data_loaded()
        result = env.reset(task.value)
        return {"observation": result.observation.model_dump(mode="json"), "info": result.info}
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
        "description": app.description,
        "tasks": _task_payload(),
        "workflow_views": _workflow_views(),
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


@app.get("/demo/trace")
async def demo_trace(task: TaskDifficulty = TaskDifficulty.MEDIUM) -> Dict[str, Any]:
    try:
        return _demo_trace_payload(task)
    except Exception as exc:
        logger.exception("Demo trace error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


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
        if method in {"notifications/initialized", "initialized", "ping"}:
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
async def global_exception_handler(_: Any, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception")
    return JSONResponse(status_code=500, content={"detail": str(exc)})


def main() -> None:
    import uvicorn

    port = int(os.getenv("PORT", "7860"))
    logger.info("Launching FraudShield server on port %d", port)
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1)


if __name__ == "__main__":  # pragma: no cover
    main()
