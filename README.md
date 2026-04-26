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

FraudShield is a partial-observability OpenEnv environment for simulated fraud investigation and workflow-aware routing.

## Training-First Architecture

FraudShield now includes a modular LLM + RL training stack alongside the OpenEnv runtime:

- `environment.py`: text-first wrapper for multi-step rollouts
- `reward.py`: decomposed numeric reward with measurable subscores
- `train.py`: Colab-friendly QLoRA training pipeline
- `evaluate.py`: fixed-task evaluation and comparison plots
- `config.py`: experiment, model, environment, and reward configuration
- `utils.py`: seeding, JSON handling, logging helpers, and moving averages
- `configs/colab_qlora_grpo.json`: default Colab experiment config

This layer is designed so you can generate rollouts, score model behavior with decomposed rewards, save checkpoints, resume runs, and compare before/after performance in a repeatable way.

Experimental tracking is enabled by default through TensorBoard logs under `artifacts/rl_runs/.../tb_logs`, and the training pipeline also writes plot artifacts such as `loss_vs_steps.png` and `reward_vs_steps.png`. If you want hosted tracking, set `report_to=["wandb"]` or `["tensorboard","wandb"]` in the experiment config before the run.

## What This Is

FraudShield is an RL-ready simulation, not a live fraud platform. An agent receives a limited triage view of a case, chooses investigation actions to reveal hidden evidence, and then routes the case with one of the supported final resolutions.

The environment is built for OpenEnv evaluation and training. It keeps the runtime fully offline by using the frozen snapshot in `data/fraudshield_cases.json`.

## Why It Matters For Theme 3.1

Theme 3.1 is about professional tasks, tool use, and world modeling under partial observability. FraudShield fits that directly:

- the agent starts with incomplete information
- useful evidence appears only after the right action is taken
- the environment rewards workflow quality, not just final correctness
- harder tasks require multi-step investigation and linked-case reasoning

This makes it a better fit for training decision-making agents than a one-shot fraud classifier.

## Lightweight Explorer UI

FraudShield now includes a small browser explorer at `/` so you can inspect the environment without sending raw API requests by hand. The explorer lets you:

- reset an easy, medium, or hard episode
- click investigation and resolution actions one step at a time
- inspect the live observation and full environment state
- run the current heuristic baseline as a walkthrough before RL training

This UI is intentionally lightweight. It is there to make the environment easier to understand, not to turn FraudShield into a fake production product.

## Environment Design

### Action Space

FraudShield keeps a fixed typed action space:

- `review_transaction`: open the operational transaction trace for the active case
- `fetch_customer_profile`: reveal buyer age, dispute history, and repeat-buyer status
- `fetch_merchant_profile`: reveal seller age, rating, reviews, and chargeback rate
- `fetch_network_graph`: reveal shared-device activity, prior flags, cluster risk, linked cards, and linked case IDs when present
- `check_policy`: reveal routing policy guidance
- `add_case_note`: write the required audit note before final closure
- `resolve_case`: submit one final resolution

Supported final resolutions:

- `approve`
- `block`
- `hold`
- `request_docs`
- `escalate`

### Observation Space

The public observation model stays the same, but the reset-time contents are intentionally sparse.

At reset, the agent only sees:

- `case_id`
- `task_name`
- `remaining_steps`
- `episode_step`
- `case_summary.amount_usd`
- a short triage summary in `case_summary.queue_reason`
- coarse context in `app_context`:
  - `item_category`
  - `timestamp`
  - `investigation_budget_remaining`
  - `available_investigations`
- the currently valid public actions in `allowed_actions`

Hidden details do **not** appear until the matching action is taken. In particular, seller profile, buyer profile, network risk, payment method, shipping behavior, and linked-case structure are progressively revealed through `revealed_evidence`.

### Reward Design

FraudShield keeps the existing correctness-driven terminal structure and adds workflow-shaped rewards:

- `+0.05` for a first-time useful fetch
- `+0.08` for `review_transaction` on cases with hidden high-risk payment or fulfillment facts
- `+0.08` for `fetch_network_graph` on cases with high hidden cluster risk
- `-0.05` for redundant repeated fetches
- `-0.03` for fetches after the case fetch budget is exhausted
- `-0.10` for resolving a medium or hard case with no fetch-based evidence
- `+0.15` terminal bonus for correct medium or hard routing when at least one investigation was used

The grader in `graders.py` is unchanged. Final task scores still depend on resolution accuracy, evidence coverage, policy compliance, workflow completion, efficiency, and linked-case consistency.

### Task Difficulty

FraudShield has three graded tasks:

| Task | Design goal | What makes it hard |
| --- | --- | --- |
| Easy | obvious routing with minimal investigation | strong visible cues, 1 fetch budget |
| Medium | mixed-signal routing | at least 1 investigation needed, 2 evidence points typically matter |
| Hard | linked-case reasoning | misleading triage, hidden linkage, 3 fetch budget, graph evidence usually required |

## How To Run Locally

Install the package:

```bash
pip install -e .
```

Run the heuristic or configured agent:

```bash
python inference.py
```

FraudShield supports three agent modes:

- `heuristic` by default when no model credentials are set
- `llm_local` when `LOCAL_MODEL_PATH` points to a trained Hugging Face / PEFT checkpoint
- `llm_remote` when an API-compatible model is configured

For a no-paid-model open-source setup, the recommended options are:

### Option 1: Use your locally trained model

```bash
LOCAL_MODEL_PATH=trained_policy python inference.py
```

### Option 2: Use a Hugging Face hosted open-source model

```bash
HF_TOKEN=your_token_here \
MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct \
API_BASE_URL=https://router.huggingface.co/v1 \
python inference.py
```

If `HF_TOKEN` is present and `API_BASE_URL` is not set, FraudShield defaults to the Hugging Face router automatically.

Run the OpenEnv API server:

```bash
python -m server.app
```

Then open the lightweight explorer:

- `http://127.0.0.1:7860/`

Important endpoints:

- `GET /health`
- `POST /reset?task=easy|medium|hard`
- `POST /step`
- `GET /state`
- `GET /info`
- `GET /tasks`
- `GET /metadata`
- `GET /schema`
- `POST /mcp`
- `GET /docs`

Validation:

```bash
python validate_api.py
python -m openenv.cli validate .
docker build -t fraudshield .
docker run -p 7860:7860 fraudshield
```

## How To Run The Training Notebook

The Colab notebook lives at:

- `notebooks/fraudshield_trl_colab.ipynb`

It is designed to:

1. install `openenv-core`, `trl`, `unsloth`, `transformers`, `datasets`, and `peft`
2. clone the repo and install FraudShield
3. load a public fraud curriculum dataset from Hugging Face
4. build a second-stage training set from real FraudShield rollouts
5. run two-stage fine-tuning with Unsloth LoRA and TRL `SFTTrainer`
   - stage 1: public fraud-data adaptation
   - stage 2: FraudShield policy adaptation
5. save a reusable local policy checkpoint
6. save:
   - `reward_curve.png`
   - `loss_curve.png`
   - `training_summary.json`
7. evaluate:
   - heuristic via `python inference.py`
   - trained model via `LOCAL_MODEL_PATH=... python inference.py`

The notebook is designed for Colab + GPU execution and does not require a paid proprietary LLM. The current public curriculum source is `Phoenix21/mock_fraud-detection-dataset`, which gives the model broader fraud-signal exposure before it is adapted to FraudShield actions.

## Results

Current heuristic baseline, measured with `python inference.py`:

- Easy: `0.9900`
- Medium: `0.3500`
- Hard: `0.7425`
- Final: `0.6942`

This baseline is intentionally rule-based and not trained. It is strong on easy, weaker on medium, and still imperfect on hard, which leaves headroom for a trained policy that can learn broader fraud patterns from public data and then adapt them to FraudShield.

Once training is completed, this section should include:

- reward curve image
- loss curve image
- trained-vs-heuristic comparison table
- one short qualitative trace comparison

The preferred final story is:

- heuristic baseline
- base open-source LLM or hosted HF model
- fine-tuned local policy checkpoint

## Live Links

- Hugging Face Space: `https://huggingface.co/spaces/DevikaJ2005/fraudshield-1`
- Code repository: `https://github.com/DevikaJ2005/Fraudshield`
- Colab notebook: `https://colab.research.google.com/github/DevikaJ2005/Fraudshield/blob/main/notebooks/fraudshield_trl_colab.ipynb`
- Blog draft: `HF_BLOG_DRAFT.md`

The Space root can double as a quick explorer UI for judges before they open the API docs.

For final submission, make sure the README links:

- the public HF Space
- the public GitHub repo
- the public Colab notebook
- the final Hugging Face blog post or video/slides link
- the committed reward/loss plot images

## Simulation vs Production

FraudShield is a simulation for training and evaluation.

What it does:

- models partial observability
- enforces investigation budgets
- exposes hidden evidence only through actions
- grades routing behavior in a reproducible way

What it does **not** do:

- connect to live financial systems
- process real customer data
- move money or block real payments
- provide production security, auth, or compliance guarantees

A production fraud platform would still need real data pipelines, authentication, authorization, monitoring, compliance controls, and human-review operations beyond this environment.
