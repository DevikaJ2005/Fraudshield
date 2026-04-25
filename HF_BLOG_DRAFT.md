# FraudShield: Training Agents to Investigate Fraud, Not Just Classify It

## Working title

FraudShield: A partial-observability OpenEnv environment for fraud investigation and workflow-aware routing

## Opening

Most fraud benchmarks flatten the problem into a single prediction: fraud or legitimate.

Real fraud review work is more sequential than that. An analyst starts with incomplete case information, opens the transaction trace, pulls a few targeted evidence bundles, checks policy when the case is ambiguous, writes a note, and only then chooses a final route such as approve, hold, request documents, block, or escalate.

FraudShield is built around that gap. It is an OpenEnv environment where the agent learns what to investigate, when to stop, and how to route the case under a limited budget.

## Why this fits Theme 3.1

Theme 3.1 is about world modeling and professional tasks. FraudShield fits because it is:

- partially observable at reset
- tool based rather than one-shot
- sequential, with consequences for bad workflow
- easy to verify with deterministic grading
- realistic enough to feel like professional case handling without pretending to be a live financial system

The environment is meant to train behavior, not just labels.

## What the agent sees and does

At reset, the agent only gets a limited triage view:

- case ID
- task name
- amount
- short queue reason
- coarse context such as category, timestamp, and remaining investigation budget

Most of the useful evidence is hidden at first.

The agent can choose from seven public actions:

- `review_transaction`
- `fetch_customer_profile`
- `fetch_merchant_profile`
- `fetch_network_graph`
- `check_policy`
- `add_case_note`
- `resolve_case`

Each investigation action reveals a different evidence bundle. That means the model has to decide which tool to use first instead of seeing the whole case upfront.

## Task design

FraudShield has three graded tasks.

### Easy

One obvious case with strong visible cues and one fetch budget.

### Medium

One mixed-signal case where at least one investigation is needed before the final route is reliable.

### Hard

Two linked cases with misleading triage and hidden graph structure. The model usually needs graph evidence and consistent routing across both cases.

## Reward design

FraudShield keeps deterministic case grading and adds workflow-shaped step rewards.

Positive signals:

- useful first-time evidence fetches
- reviewing transaction details on hidden high-risk payment or fulfillment cases
- graph review on hidden high-risk cluster cases
- correct final routing on medium and hard after real investigation

Negative signals:

- repeated fetches
- fetches after budget is exhausted
- resolving medium or hard without any revealed evidence
- wrong final routing

So the reward is not just about the final answer. It also teaches whether the workflow was disciplined.

## Baseline before RL training

Current heuristic baseline:

- Easy: `0.9900`
- Medium: `0.3500`
- Hard: `0.7425`
- Final: `0.6942`

That shape is useful for RL:

- easy is already stable
- medium clearly has room to improve
- hard is decent but still misses important routing behavior

## Why the RL setup matters

The main goal is not to build a better rule engine. The goal is to train a policy that learns:

- which evidence to fetch
- when enough evidence has been gathered
- when a case should be approved, blocked, held, documented, or escalated

That is a much better fit for OpenEnv than direct fraud prediction.

## Training plan

The repo includes a Colab notebook at `notebooks/fraudshield_trl_colab.ipynb`.

The intended training flow is:

1. install `openenv-core`, `trl`, `unsloth`, `transformers`, `datasets`, and `peft`
2. load FraudShield in Colab
3. train a small instruction-tuned model with GRPO-style updates
4. evaluate heuristic vs trained policy
5. save:
   - `reward_curve.png`
   - `loss_curve.png`
   - `training_summary.json`

Once the real run is complete, this post should be updated with the final curves and the before/after comparison table.

## Lightweight explorer UI

To make the environment easier to understand before training, the Hugging Face Space root includes a lightweight browser explorer.

It lets a judge:

- reset an easy, medium, or hard episode
- click investigation actions manually
- inspect the live observation and hidden evidence as it appears
- run the current heuristic baseline as a walkthrough

This is intentionally a small environment explorer, not a fake enterprise product.

## What FraudShield is and is not

FraudShield is:

- a simulated RL environment
- deterministic and reproducible
- built on a frozen offline snapshot
- suitable for OpenEnv evaluation and Colab training

FraudShield is not:

- a live fraud system
- connected to real bank rails
- processing real customer records at runtime
- making production fraud decisions

## Assets to add before publishing

- reward curve image
- loss curve image
- trained-vs-heuristic comparison table
- one trace comparison screenshot
- Hugging Face Space link
- Colab notebook link
- GitHub repo link
