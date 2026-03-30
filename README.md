# FraudShield

FraudShield is an OpenEnv environment for marketplace fraud review. An agent inspects one transaction at a time, predicts whether it is `fraud` or `legitimate`, and receives dense reward shaped by business impact, confidence calibration, and correctness.

The environment is grounded in real public fraud data, but it does not fetch live records during `reset()` or `step()`. Instead, it uses a frozen, versioned snapshot stored in `data/fraudshield_cases.json`. That gives you real-world grounding with deterministic grading, fast Docker startup, and reproducible evaluation on Hugging Face Spaces.

## Why this design

For an OpenEnv submission, the safest pattern is:

- Fetch or refresh public source data offline
- Build a deterministic FraudShield snapshot
- Commit the snapshot used for evaluation
- Keep the environment runtime fully offline

That avoids runtime API failures, privacy issues, and non-reproducible scores.

## Real-world data strategy

FraudShield currently builds its snapshot from the public Kaggle / ULB credit card fraud dataset:

- Source ID: `kaggle_creditcardfraud`
- Dataset: `mlg-ulb/creditcardfraud`
- URL: `https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud`

The loader is now source-agnostic in code:

- `data_loader.py` exposes a public-source snapshot pipeline
- `download_kaggle_data.py` refreshes the local source CSV and rebuilds the frozen snapshot
- `fraudshield_env.py` reads the snapshot only at runtime

The checked-in snapshot currently reports:

- Snapshot ID: `fraudshield-realworld-v2`
- Schema version: `2.0`
- Seed: `42`
- Task sizes: easy `24`, medium `36`, hard `48`

## Tasks

| Task | Cases | Goal | What makes it hard |
| --- | ---: | --- | --- |
| Easy | 24 | Catch obvious fraud while avoiding basic false positives | Single-transaction red flags are strong and low-noise |
| Medium | 36 | Balance fraud capture with calibration | No single signal is decisive; tradeoffs matter |
| Hard | 48 | Handle coordinated abuse and edge-case legitimate traffic | Fraud rings and flash-sale behavior intentionally overlap |

## Action space

Agents emit a single `FraudCheckAction`:

```python
FraudCheckAction(
    transaction_id: str,
    decision: Literal["fraud", "legitimate"],
    confidence: float,  # 0.0 to 1.0
    reasoning: str,
)
```

## Observation space

Each step returns a `FraudCheckObservation` with:

- Structured transaction facts such as amount, seller age, buyer age, geo mismatch, rating, prior flags, chargeback rate, shared-device counts, and address velocity
- Historical context such as seller velocity, linked cards, refund counts, cluster alert score, and source snapshot metadata
- Task metadata including difficulty and episode step

## Reward design

Rewards in `fraudshield_env.py` are dense and cost-sensitive:

- Correct fraud catches receive the strongest positive reward
- Correct legitimate approvals still earn positive reward, but less than catching fraud
- False negatives are punished more than false positives
- Confidence is rewarded when it matches hidden case difficulty and punished when it is overconfident
- Submitting the wrong `transaction_id` adds an extra penalty

## Graders

The three task graders in `graders.py` are deterministic and return scores from `0.0` to `1.0`.

- Easy: accuracy, F1, recall, and specificity
- Medium: F1, ROC-AUC, precision, and confidence calibration
- Hard: recall, precision, F1, ROC-AUC, and calibration

## Baseline inference

The required root script is `inference.py`.

- Competition mode: if `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN` are set, it uses the OpenAI client against that endpoint
- Local smoke-test mode: if those variables are missing, it falls back to a deterministic heuristic agent

Required environment variables for the competition path:

```bash
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=<your-model-id>
HF_TOKEN=<your-token>
```

Run it with:

```bash
python inference.py
```

The script writes `fraudshield_baseline_results.json` to the project root.

### Tested local baseline

I reran the baseline after the snapshot-loader changes. With the deterministic heuristic fallback and seed `42`, the tested local scores are:

| Task | Score |
| --- | ---: |
| Easy | 1.0000 |
| Medium | 0.8773 |
| Hard | 0.7206 |
| Final | 0.8660 |

## Project layout

```text
fraudshield/
|-- data/
|   |-- fraudshield_cases.json
|-- server/
|   |-- __init__.py
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

## Setup

Install the project:

```bash
python -m pip install -e .
```

If you want to rebuild the frozen snapshot from the public source CSV:

```bash
python -m pip install -e ".[data]"
python download_kaggle_data.py
```

If `data/creditcard.csv` already exists locally, the script rebuilds the snapshot without needing to download again.

## Running locally

### Python API

```python
from fraudshield_env import FraudShieldEnvironment
from models import DecisionEnum, FraudCheckAction

env = FraudShieldEnvironment(data_path="data", seed=42)
env.load_data()
reset_result = env.reset("medium")

action = FraudCheckAction(
    transaction_id=reset_result.observation.transaction_id,
    decision=DecisionEnum.LEGITIMATE,
    confidence=0.62,
    reasoning="Signals are mixed but seller history is reasonably stable.",
)

step_result = env.step(action)
print(step_result.reward.value, step_result.done)
```

### FastAPI server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

Endpoints:

- `GET /health`
- `POST /reset?task=easy|medium|hard`
- `POST /step`
- `GET /state`
- `GET /info`
- `GET /tasks`

## Docker

Build and run:

```bash
docker build -t fraudshield .
docker run -p 7860:7860 fraudshield
```

The container listens on port `7860`, which matches Hugging Face Docker Spaces expectations.

## Validation checklist

Before submission:

```bash
python inference.py
openenv validate openenv.yaml
docker build -t fraudshield .
docker run -p 7860:7860 fraudshield
```

Then verify:

- `http://localhost:7860/health`
- `POST http://localhost:7860/reset?task=easy`

## Notes

- Runtime uses the committed snapshot only
- Public source refresh is optional and intended for offline rebuilds
- `inference_llm.py` remains as a thin wrapper to `inference.py`
