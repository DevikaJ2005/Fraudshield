# FraudShield Submission Checklist

Updated: 2026-04-08

## Core submission requirements

- [x] `openenv.yaml` present and aligned with the current environment
- [x] Root `inference.py` runs successfully
- [x] `inference.py` emits `[START]`, `[STEP]`, and `[END]` blocks to stdout
- [x] Inference uses injected `API_BASE_URL` and `API_KEY` when provided
- [x] Task scores stay strictly inside `(0, 1)`
- [x] Frozen snapshot committed in `data/fraudshield_cases.json`
- [x] Dockerfile included
- [x] FastAPI runtime listens on port `7860`

## Environment quality

- [x] Typed Pydantic action, observation, reward, state, reset, and step models
- [x] Three graded tasks: easy, medium, hard
- [x] Deterministic graders and fixed seed
- [x] Business-cost-sensitive reward shaping
- [x] Optional multi-step investigation workflow
- [x] Partial observability via budgeted evidence reveals

## Runtime surface

- [x] `GET /health`
- [x] `POST /reset`
- [x] `POST /step`
- [x] `GET /state`
- [x] `GET /info`
- [x] `GET /tasks`
- [x] `GET /metadata`
- [x] `GET /schema`
- [x] `POST /mcp`

## Local validations completed

- [x] `python inference.py`
- [x] `python -X utf8 validate_enhancements.py`
- [x] `python -X utf8 validate_api.py`
- [x] `python -m openenv.cli validate .`
- [x] `python -m openenv.cli validate --url http://127.0.0.1:7860`

## Current baseline

- [x] Easy: `0.9999`
- [x] Medium: `0.8773`
- [x] Hard: `0.7206`
- [x] Overall: `0.8659`

## Submission URLs

- GitHub: `https://github.com/DevikaJ2005/Fraudshield`
- Hugging Face Space: `https://huggingface.co/spaces/DevikaJ2005/fraudshield-1`

## Presentation strengths

- [x] Real-world marketplace fraud use case
- [x] Clear agentic loop instead of pure one-shot classification
- [x] Strong deterministic submission path for validator reliability
- [x] Richer demo path through `agentic_demo.py`
- [x] REST and MCP access patterns for tooling and judge exploration
