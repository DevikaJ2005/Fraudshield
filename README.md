# FraudShield

FraudShield is an OpenEnv environment for marketplace fraud review. An agent receives one e-commerce transaction at a time, decides whether it is `fraud` or `legitimate`, and gets dense reward shaped by business impact, confidence calibration, and correctness.

The environment is built from a compact task bundle derived from the Kaggle credit card fraud dataset. The bundle is committed as `data/fraudshield_cases.json`, so the repo stays self-contained for Docker and Hugging Face Spaces while still grounding the tasks in real fraud data.

## Why this environment

Real commerce teams review risky orders all day: new sellers, chargeback-heavy merchants, reused devices, flash-sale spikes, and account-takeover style behavior. FraudShield turns that workflow into an agent training environment with:

- A real-world domain instead of a toy game
- Typed `Action`, `Observation`, and `Reward` models
- `reset()`, `step()`, and `state()` APIs
- Three graded tasks with deterministic scoring from `0.0` to `1.0`
- Dense step rewards with partial progress signals
- A root `inference.py` baseline compatible with the required OpenAI client flow

## Tasks

| Task | Cases | Goal | What makes it hard |
| --- | ---: | --- | --- |
| Easy | 24 | Catch obvious fraud while avoiding basic false positives | Single-transaction red flags are strong and low-noise |
| Medium | 36 | Balance fraud capture with calibration | No single signal is decisive; tradeoffs matter |
| Hard | 48 | Handle coordinated abuse and edge-case legitimate traffic | Fraud rings and flash-sale behavior intentionally overlap |

Each task uses a deterministic grader in [graders.py](/c:/Users/Jayashanker/Downloads/fraudshield_kaggle_ready/fraudshield_kaggle/graders.py).

## Action space

Agents emit a single [FraudCheckAction](/c:/Users/Jayashanker/Downloads/fraudshield_kaggle_ready/fraudshield_kaggle/models.py):

```python
FraudCheckAction(
    transaction_id: str,
    decision: Literal["fraud", "legitimate"],
    confidence: float,  # 0.0 to 1.0
    reasoning: str,
)
```

## Observation space

Each step returns a [FraudCheckObservation](/c:/Users/Jayashanker/Downloads/fraudshield_kaggle_ready/fraudshield_kaggle/models.py) with structured transaction facts and rolling context:

- Transaction facts: amount, seller age, buyer age, payment method, geo mismatch, rating, prior flags, chargeback rate, shared-device counts, same-address velocity, and more
- Historical context: seller velocity, linked cards, refund counts, cluster alert score, and task-specific notes
- Task metadata: difficulty and episode step

## Reward design

Rewards are dense and cost-sensitive in [fraudshield_env.py](/c:/Users/Jayashanker/Downloads/fraudshield_kaggle_ready/fraudshield_kaggle/fraudshield_env.py):

- Correct fraud catches receive the strongest positive reward
- Correct legitimate approvals still earn positive reward, but less than catching fraud
- False negatives are punished more than false positives
- Confidence is rewarded when it matches hidden case difficulty and punished when it is overconfident
- Submitting the wrong `transaction_id` adds an extra penalty

This gives the agent signal across the full trajectory instead of only at episode end.

## Graders

The three task graders are deterministic and return `0.0` to `1.0`.

- Easy: accuracy, F1, recall, and specificity
- Medium: F1, ROC-AUC, precision, and confidence calibration
- Hard: recall, precision, F1, ROC-AUC, and calibration

## Baseline inference

The required root script is [inference.py](/c:/Users/Jayashanker/Downloads/fraudshield_kaggle_ready/fraudshield_kaggle/inference.py).

- Competition mode: if `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN` are set, it uses the OpenAI client against that endpoint
- Local smoke-test mode: if those variables are missing, it falls back to a deterministic heuristic agent so the repo can still be verified offline

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

The script writes `fraudshield_baseline_results.json` in the project root.

### Local offline baseline

With the deterministic heuristic fallback and seed `42`, the current local smoke-test scores are:

| Task | Score |
| --- | ---: |
| Easy | 1.0000 |
| Medium | 0.8773 |
| Hard | 0.7206 |
| Final | 0.8660 |

## Project layout

```text
fraudshield_kaggle/
├── data/
│   └── fraudshield_cases.json
├── server/
│   ├── __init__.py
│   └── app.py
├── data_loader.py
├── download_kaggle_data.py
├── Dockerfile
├── fraudshield_env.py
├── graders.py
├── inference.py
├── inference_llm.py
├── llm_agent.py
├── models.py
├── openenv.yaml
└── pyproject.toml
```

## Setup

Install the project:

```bash
python -m pip install -e .
```

Optional: if you want to regenerate the compact bundle from the original Kaggle CSV instead of using the committed task file:

```bash
python -m pip install -e ".[data]"
python download_kaggle_data.py
```

## Running the environment locally

### Python API

```python
from fraudshield_env import FraudShieldEnvironment
from models import DecisionEnum, FraudCheckAction

env = FraudShieldEnvironment(data_path="data", seed=42)
env.load_kaggle_data()
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

The container listens on port `7860`, which is the expected default for Hugging Face Docker Spaces.

## Hugging Face Spaces

This repo is ready for a Docker Space:

- Include `openenv` in the Space tags
- Use the provided `Dockerfile`
- Expose the app on port `7860`
- Set `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN` in the Space secrets if you want the LLM baseline to run there

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

- The committed task bundle is small on purpose so the repo stays deployable without external downloads
- The source CSV is optional and only needed if you want to regenerate the bundle
- `inference_llm.py` is kept as a backward-compatible wrapper to the main baseline entrypoint
