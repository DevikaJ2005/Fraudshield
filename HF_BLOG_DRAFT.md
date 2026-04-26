# Training Agents to Investigate Fraud, Not Just Classify It

Most fraud detection benchmarks reduce the job to one prediction: look at a transaction and say whether it is fraud. Real fraud review does not work like that.

A human analyst usually starts with a thin triage summary, not a perfect spreadsheet. They open a case, inspect the transaction, check customer and merchant history, look for linked activity, consult policy, write a note, and only then decide whether to approve, hold, request documents, block, or escalate.

FraudShield was built around that gap.

Instead of treating fraud review as a one-shot classifier, FraudShield turns it into a **partial-observability investigation environment**. The question is no longer just “can a model predict fraud?” The question becomes:

**Can an agent learn how to investigate before it decides?**

---

## The Problem We Wanted To Model

The usual fraud benchmark setup is simple:

- show all features up front
- ask for one label
- score the final answer

That is useful for classification research, but it misses most of the workflow that matters in real operations.

Professional fraud review is sequential. The reviewer has to:

1. decide what to inspect first
2. reveal evidence step by step
3. avoid wasting time on redundant checks
4. document the case properly
5. choose the final route under uncertainty

That makes it a much better fit for **OpenEnv**, **tool use**, and eventually **reinforcement learning**, because the model is being trained on behavior, not just labels.

---

## What FraudShield Is

FraudShield is a simulated OpenEnv environment for fraud investigation and workflow-aware routing.

It is:

- partially observable
- multi-step
- deterministic and reproducible
- trainable in Colab with open-source LLM tooling
- designed for professional-task evaluation, not just classification

It is **not**:

- a live financial product
- connected to real bank rails
- making production fraud decisions
- pulling live customer data at runtime

The runtime stays offline and uses a frozen snapshot in `data/fraudshield_cases.json`.

---

## What The Agent Experiences

At reset, the agent does **not** get the whole case.

It sees only a triage view:

- case ID
- task name
- amount
- short queue reason
- remaining steps
- limited workflow context such as category, timestamp, and investigation budget

Most of the important evidence is hidden at first.

To reveal more, the agent has to use workflow actions:

- `review_transaction`
- `fetch_customer_profile`
- `fetch_merchant_profile`
- `fetch_network_graph`
- `check_policy`
- `add_case_note`
- `resolve_case`

Each action reveals a different bundle of evidence, so the model has to decide **what to investigate first** rather than seeing the answer up front.

That makes the task feel much closer to a real analyst workflow than a static fraud-classification benchmark.

---

## Three Tasks, Three Kinds Of Difficulty

FraudShield has three graded tasks.

### Easy

One obvious case with strong visible cues and a single fetch budget.

This is the “sanity check” task. If an agent cannot handle easy, it probably does not understand the workflow at all.

### Medium

One mixed-signal case where at least one investigation is needed before the final route is reliable.

This is the bottleneck task.

The model must decide whether the evidence supports:

- `approve`
- `hold`
- `request_docs`

without the answer being obvious from the initial view.

### Hard

Two linked cases with misleading triage and hidden graph structure.

Hard is about consistency and linked-case reasoning. The model usually needs graph evidence and has to keep both cases aligned.

---

## Reward Design: More Than Just The Final Label

FraudShield keeps deterministic grading, but it also uses workflow-shaped rewards.

Positive signals include:

- useful first-time evidence fetches
- reviewing transaction details on hidden high-risk payment or fulfillment cases
- graph review on hidden high-risk cluster cases
- correct final routing after real investigation

Negative signals include:

- repeated fetches
- fetches after budget is exhausted
- resolving medium or hard with no evidence
- incorrect routing

So the environment is not just asking “was the answer correct?”

It is also asking:

- did the agent investigate the right things?
- did it stop at the right time?
- did it follow the workflow cleanly?

That is exactly the kind of structure we wanted for Theme 3.1.

---

## The Baseline Taught Us Where The Real Difficulty Is

Before training, the environment had a rule-based baseline:

- Easy: `0.9900`
- Medium: `0.3500`
- Hard: `0.7425`
- Final: `0.6942`

That shape is revealing.

Easy is already strong. Hard is imperfect but usable. Medium is where the real headroom is.

That makes sense: medium cases are where the model has to decide whether the visible evidence is enough, whether more investigation is needed, and whether the safest route is `request_docs` or `hold`.

The baseline is useful as a reference, but it is not the end goal. FraudShield is supposed to be a **trainable** environment, so the real challenge is whether a learned policy can improve where the baseline struggles.

---

## Building The Training Stack Was Its Own Investigation

One of the most useful lessons from this project is that a working environment is not the same thing as a working learned policy.

We started with a training-first architecture:

- `environment.py`: text wrapper for multi-step rollouts
- `reward.py`: decomposed reward components
- `train.py`: Colab-friendly training entrypoint
- `evaluate.py`: before/after evaluation
- `config.py`: experiment and reward configuration
- `utils.py`: seeding, JSON handling, metrics helpers

The training stack supports:

- Colab-friendly QLoRA
- low-VRAM execution
- TensorBoard logging
- checkpointable runs
- evaluation plots

But the first training attempts taught us something important:

**a pipeline can run successfully and still fail to learn the right behavior.**

---

## What Went Wrong In Early Training Attempts

The first failure mode was straightforward: the model could learn formatting faster than it could learn policy.

A few things went wrong along the way:

- early training was too close to supervised imitation
- the model learned output shape better than sequential decision quality
- hosted LLM evaluation paths were noisy and sometimes fell back to the heuristic
- stale artifacts in notebook sessions made it easy to misread what had actually run
- one important bug created a mismatch between the prompt format used in training and the prompt format used in local-model inference

That last bug mattered a lot.

If a model is trained on one prompt structure and then evaluated on a different one, even a seemingly reasonable checkpoint can perform badly for the wrong reasons.

Fixing that prompt mismatch was one of the highest-confidence improvements to the training pipeline.

---

## What The Training Stack Looks Like Now

The current stack is more disciplined:

- stale Colab artifacts are cleared before a fresh run
- evaluation compares the heuristic baseline against the local trained checkpoint
- prompt format is aligned between training and local inference
- training targets use canonical investigation aliases
- the training curriculum is more FraudShield-heavy

We also moved away from relying only on weak heuristic imitation.

The current training direction uses:

1. a small public fraud curriculum for broad signal exposure
2. stronger FraudShield-specific trajectories from an expert teacher policy
3. a Colab-friendly local checkpoint evaluation path

That still is not the same as full reward-optimizing PPO/GRPO training. It is better described as a **curriculum SFT pipeline with a stronger task-specific teacher**.

That distinction matters, and we want to be honest about it.

---

## Why Reinforcement Learning Still Matters Here

FraudShield was always designed with RL in mind, because the task is fundamentally sequential:

- the agent chooses what to reveal
- the agent decides when to stop
- the agent trades off investigation cost against confidence
- the agent has to route ambiguous cases correctly

That is why the long-term goal is not just a better imitation model, but a policy that can improve against actual environment reward.

Even where the current training setup is not yet full GRPO/PPO optimization end to end, the environment and reward structure were built to support that future direction.

The strongest evidence that this matters is still the medium task. A one-shot classifier can look fine on easy cases. A true investigation policy is tested when the right action is **not** obvious.

---

## The Explorer UI Matters Too

FraudShield also includes a lightweight browser explorer on the Hugging Face Space root.

That explorer is intentionally simple and professional. It lets a non-technical reviewer:

- start an easy, medium, or hard episode
- click through investigation actions manually
- watch hidden evidence appear as the workflow progresses
- compare their own flow with the baseline walkthrough

The point is not to pretend FraudShield is a production case-management system.

The point is to make the environment legible to a human reviewer quickly.

---

## What We Want The Final Evidence To Show

The ideal final evidence bundle for FraudShield is:

- a working OpenEnv environment
- a reproducible Colab training script
- real training artifacts:
  - loss plot
  - reward plot
  - before/after evaluation
- a clear story for where learning should improve over the baseline

The strongest outcome would be:

- Easy stays stable
- Medium improves significantly
- Hard remains competitive or improves modestly

That would support the central claim of the project:

**fraud review is better modeled as a trainable investigation policy than as a one-shot classifier.**

---

## What FraudShield Shows Beyond Fraud

The larger point of the project goes beyond fraud.

FraudShield is a template for professional-task environments where:

- information is incomplete
- tools reveal hidden state
- actions have workflow cost
- the model must decide when to stop

The same structure could be adapted to:

- trust and safety review
- claims triage
- AML investigations
- compliance workflows
- customer-support escalations

That is why the environment matters even before the perfect learned policy exists.

---

## Closing Thought

FraudShield started from a simple frustration: too many fraud benchmarks ask a model to classify, when the real job is to investigate.

The environment now tests something much closer to the real cognitive problem:

- decide what to inspect
- gather evidence under budget
- reason over partial information
- document the case
- route it safely

That is the behavior we actually want to train.

Whether the final winning policy is heuristic, curriculum-trained, or fully reinforcement-optimized, the important shift is already clear:

**the right question is not only whether an agent can spot fraud. It is whether an agent can learn how to investigate before it decides.**

---

## Project Links

- Hugging Face Space: `https://huggingface.co/spaces/DevikaJ2005/fraudshield-1`
- GitHub repository: `https://github.com/DevikaJ2005/Fraudshield`
- Colab notebook: `https://colab.research.google.com/github/DevikaJ2005/Fraudshield/blob/main/notebooks/fraudshield_trl_colab.ipynb`

## Add Before Publishing

- final loss plot
- final reward plot
- final before/after comparison table
- one short trace comparison screenshot
- final blog URL or video URL linked from `README.md`
