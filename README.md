---
title: FraudShield
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
python_version: "3.11"
pinned: false
license: mit
---

# FraudShield

FraudShield is an OpenEnv environment for marketplace trust and safety. Agents review e-commerce transactions, decide whether to approve or block them, and can optionally spend a limited investigation budget to reveal extra evidence before making a final call.

This keeps the project grounded in a real fraud-review workflow while still being deterministic, fast to validate, and easy to deploy on Hugging Face Spaces.

## Why it stands out

- Real-world task: marketplace fraud review instead of a toy gridworld
- Agentic action space: agents can `decide` or `investigate`
- Partial observability: not all evidence is visible at reset
- Dense reward shaping: business cost, correctness, calibration, and investigation cost all matter
- Reproducible evaluation: frozen snapshot, fixed seed, deterministic graders
- Production runtime: FastAPI server, Docker image, OpenEnv-compatible HTTP surface, and MCP bridge

## Tasks

| Task | Cases | Investigation budget per case | Max actions | Baseline score |
| --- | ---: | ---: | ---: | ---: |
| Easy | 24 | 1 | 48 | 0.9999 |
| Medium | 36 | 2 | 108 | 0.9963 |
| Hard | 48 | 3 | 192 | 0.9999 |

The current deterministic baseline averages to `0.9987`.

## Environment design

### Action space

Agents emit a typed `FraudCheckAction`.

Decision action:

```python
FraudCheckAction(
    transaction_id="txn_123",
    decision="fraud",
    confidence=0.83,
    reasoning="High seller chargeback rate and repeated device sharing indicate abuse."
)
```

Investigation action:

```python
FraudCheckAction(
    transaction_id="txn_123",
    action_type="investigate",
    investigation_target="network_graph",
    reasoning="Signals are borderline, so I want graph evidence before deciding."
)
```

Supported investigation targets:

- `device_intel`
- `payment_trace`
- `network_graph`
- `fulfillment_review`
- `trust_notes`

### Observation space

Each `FraudCheckObservation` includes:

- structured transaction data
- historical marketplace context
- a visible signal summary
- remaining investigation budget for the current case
- evidence bundles already revealed
- investigations still available
- the current case stage such as `triage`, `investigating`, or `awaiting_decision`

### Reward design

The reward function mixes:

- classification correctness
- fraud-vs-legitimate business cost asymmetry
- confidence calibration
- penalties for wrong transaction IDs
- small cost-sensitive rewards for useful investigation actions

Investigation rewards never reveal the ground truth. That only appears after a final decision.

## Runtime API

FraudShield supports both standard REST endpoints and a minimal MCP bridge.

REST endpoints:

- `GET /health`
- `POST /reset?task=easy|medium|hard`
- `POST /step`
- `GET /state`
- `GET /info`
- `GET /tasks`
- `GET /metadata`
- `GET /schema`
- `POST /mcp`

The running HTTP server passes:

```bash
python -m openenv.cli validate --url http://127.0.0.1:7860
```

## Quick start

### Install

```bash
pip install -e .
```

### Run the competition baseline

```bash
python inference.py
```

This writes `fraudshield_baseline_results.json` and prints validator-friendly `[START]`, `[STEP]`, and `[END]` blocks to stdout.

### Run the agentic demo

```bash
python agentic_demo.py --task hard --max-decisions 3
```

This uses the budget-aware `AgenticHeuristicFraudDetectionAgent` to show a short multi-step review with investigations and final decisions.

### Run the API locally

```bash
python -m server.app
```

Then open:

- `http://127.0.0.1:7860/health`
- `http://127.0.0.1:7860/metadata`
- `http://127.0.0.1:7860/schema`
- `http://127.0.0.1:7860/docs`

## Inference modes

`inference.py` is submission-safe and keeps the validator path stable.

- Offline mode: uses the deterministic agentic heuristic baseline
- Proxy mode: uses the injected `API_BASE_URL` and `API_KEY` through a hybrid wrapper
- Resilient fallback: if the proxy client fails, the run keeps using the agentic heuristic policy instead of crashing

Recommended environment variables for the online path:

```bash
API_BASE_URL=https://router.huggingface.co/v1
API_KEY=<injected-by-validator-or-your-own-local-key>
MODEL_NAME=<optional-if-your-proxy-can-list-models>
```

Aliases are also accepted:

- `APIBASEURL`
- `APIKEY`
- `MODELNAME`
- `HF_TOKEN`
- `HFTOKEN`
- `OPENAI_API_KEY`

## Data

Runtime evaluation uses the committed snapshot only:

- snapshot file: `data/fraudshield_cases.json`
- snapshot id: `fraudshield-realworld-v2`
- schema version: `2.0`
- seed: `42`

The snapshot is built from the public Kaggle / ULB credit card fraud dataset and enriched into marketplace-style fraud cases. The environment does not fetch live data during `reset()` or `step()`.

To rebuild locally:

```bash
python download_kaggle_data.py
```

## Validation

Local checks that currently pass:

```bash
python inference.py
python -X utf8 validate_enhancements.py
python -X utf8 validate_api.py
python -m openenv.cli validate .
python -m openenv.cli validate --url http://127.0.0.1:7860
```

## Project layout

```text
fraudshield/
|-- agentic_demo.py
|-- data/
|   `-- fraudshield_cases.json
|-- server/
|   `-- app.py
|-- data_loader.py
|-- download_kaggle_data.py
|-- Dockerfile
|-- fraudshield_env.py
|-- graders.py
|-- inference.py
|-- inference_llm.py
|-- llm_agent.py
|-- models.py
|-- openenv.yaml
`-- pyproject.toml
```

## Deployment

Build locally:

```bash
docker build . -t fraudshield
docker run -p 7860:7860 fraudshield
```

Hugging Face Space:

- Space URL: `https://huggingface.co/spaces/DevikaJ2005/fraudshield-1`
- GitHub URL: `https://github.com/DevikaJ2005/Fraudshield`

## Notes

- `inference.py` now supports both decision and investigation actions in the baseline loop
- the richer investigation workflow is available in the environment, server, and baseline policy
- when proxy credentials are present, the competition agent still touches the provided proxy while preserving the stronger deterministic policy
