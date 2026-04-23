---
title: FraudShield
emoji: "🛡️"
colorFrom: blue
colorTo: indigo
sdk: docker
python_version: "3.12"
pinned: false
license: mit
---

# FraudShield

FraudShield is an OpenEnv environment for **enterprise fraud operations**. Instead of a one-shot fraud classifier, the agent behaves like a trust-and-safety analyst working across multiple internal tools: it opens a queue case, reviews the transaction, fetches customer and merchant evidence, checks policy rules, writes case notes, and resolves or escalates the case under SLA pressure.

This is designed for Theme `#3.1 Professional Tasks`: a partially observable, multi-step enterprise workflow where world state matters and shortcutting is punished.

## Why this fits the hackathon

- Real professional workflow instead of a toy benchmark
- Explicit tool actions rather than hidden investigation internals
- Partial observability with staged evidence gathering
- Deterministic offline runtime from a frozen public-data snapshot
- Reward shaping that values routing quality, policy compliance, notes, and efficiency
- Stronger training story because the baseline leaves room for improvement

## Simulated enterprise apps

FraudShield exposes one workflow across five internal apps:

1. `Queue`
2. `Case Console`
3. `Customer Profile`
4. `Merchant Profile`
5. `Policy & Escalation`

## Tasks

| Task | Episode shape | What the agent must do |
| --- | --- | --- |
| Easy | 1 low-noise case | Review, document, and route a clear-cut case |
| Medium | 1 ambiguous case | Use profile evidence plus policy rules before routing |
| Hard | 2 linked fraud cases | Connect shared evidence, note both cases, and choose the correct routing for each |

## Action space

Agents emit a typed `FraudCheckAction` with one of these explicit workflow actions:

- `review_transaction`
- `fetch_customer_profile`
- `fetch_merchant_profile`
- `fetch_network_graph`
- `check_policy`
- `add_case_note`
- `resolve_case`

Final routing is limited to:

- `approve`
- `block`
- `hold`
- `request_docs`
- `escalate`

Example resolution action:

```python
FraudCheckAction(
    case_id="medium_case_01",
    action_type="resolve_case",
    resolution="request_docs",
    reasoning="Customer and policy evidence show mixed signals that require document verification."
)
```

## Observation space

Each `FraudCheckObservation` includes:

- `case_id`
- `task_name`
- `current_screen`
- `visible_panels`
- `revealed_evidence`
- `linked_case_ids`
- `remaining_steps`
- `remaining_sla`
- `note_required`
- `allowed_actions`
- `queue_items`
- `case_summary`
- `app_context`

## Reward design

FraudShield uses layered verifier-first rewards:

- positive reward for first-time useful evidence retrieval
- positive reward for checking policy when it matters
- positive reward for adding a case note before closure
- large terminal reward for correct routing
- penalties for redundant fetches, bad action order, note spam, SLA burn, and policy misses

The final reward depends on **correct routing plus workflow quality**, not only on the hidden fraud label.

## Runtime API

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

## Quick start

### Install

```bash
pip install -e .
```

### Run the baseline

```bash
python inference.py
```

### Run the workflow demo

```bash
python agentic_demo.py --task hard --max-actions 10
```

### Run the API locally

```bash
python -m server.app
```

Then open:

- `http://127.0.0.1:7860/health`
- `http://127.0.0.1:7860/metadata`
- `http://127.0.0.1:7860/schema`
- `http://127.0.0.1:7860/docs`

## Baseline modes

`inference.py` supports two paths:

- Offline deterministic workflow heuristic
- Optional remote LLM path through OpenAI-compatible env vars

Latest verified offline baseline:

- Easy: `0.9900`
- Medium: `0.3500`
- Hard: `0.7063`
- Final: `0.6821`

Supported env vars:

```bash
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=<model>
HF_TOKEN=<token>
```

Aliases are also accepted:

- `APIBASEURL`
- `MODELNAME`
- `HFTOKEN`
- `API_KEY`
- `OPENAI_API_KEY`

## Data

Runtime evaluation is fully offline and deterministic.

- snapshot file: `data/fraudshield_cases.json`
- seed: `42`
- source: public Kaggle / ULB credit-card fraud data enriched into enterprise fraud-ops cases

The environment does **not** fetch live records during `reset()` or `step()`.

## Validation

Recommended local checks:

```bash
python inference.py
python -X utf8 validate_enhancements.py
python -X utf8 validate_api.py
python -m openenv.cli validate .
python -m openenv.cli validate --url http://127.0.0.1:7860
docker build -t fraudshield .
docker run -p 7860:7860 fraudshield
```

## Training artifact

The repo includes a minimal TRL Colab notebook for the required hackathon training artifact:

- `notebooks/fraudshield_trl_colab.ipynb`

Default training model:

- `Qwen/Qwen2.5-0.5B-Instruct`

## Public artifact

The repo also includes a Hugging Face blog draft outline:

- `HF_BLOG_DRAFT.md`

## Project layout

```text
fraudshield/
|-- agentic_demo.py
|-- data/
|   `-- fraudshield_cases.json
|-- notebooks/
|   `-- fraudshield_trl_colab.ipynb
|-- server/
|   `-- app.py
|-- data_loader.py
|-- Dockerfile
|-- fraudshield_env.py
|-- graders.py
|-- inference.py
|-- llm_agent.py
|-- models.py
|-- openenv.yaml
|-- HF_BLOG_DRAFT.md
`-- pyproject.toml
```
