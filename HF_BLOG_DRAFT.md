# FraudShield: Training an LLM for Enterprise FraudOps with OpenEnv

## Draft title

FraudShield: A Multi-App Fraud Operations Environment for OpenEnv

## One-line summary

FraudShield turns fraud review into a realistic enterprise workflow where an agent investigates cases across internal tools, documents evidence, and resolves or escalates under policy and SLA pressure.

## Story outline

### 1. Problem

- Most fraud benchmarks are static classifiers.
- Real fraud analysts do multi-step work across queue, case, profile, and policy tools.
- We wanted an environment that rewards investigation quality, policy compliance, and workflow discipline.

### 2. Environment design

- Five simulated enterprise apps:
  - Queue
  - Case Console
  - Customer Profile
  - Merchant Profile
  - Policy & Escalation
- Three tasks:
  - easy: one obvious case
  - medium: one ambiguous case
  - hard: two linked fraud cases
- Offline deterministic data built from a frozen public snapshot

### 3. Action and observation design

- Explicit enterprise actions:
  - review transaction
  - fetch customer profile
  - fetch merchant profile
  - fetch network graph
  - check policy
  - add case note
  - resolve case
- Observations expose:
  - current screen
  - visible panels
  - revealed evidence
  - linked cases
  - remaining steps and SLA
  - allowed actions

### 4. Reward design

- Positive reward for first-time useful evidence
- Positive reward for policy checks when required
- Positive reward for writing notes
- Large terminal reward for correct routing
- Penalties for:
  - redundant fetches
  - invalid order
  - note spam
  - missing policy
  - SLA burn

### 5. Anti-hacking safeguards

- repeated note spam is penalized
- repeated evidence fetches are penalized
- premature resolution is penalized
- hard mode requires linked-case evidence for full credit

### 6. Training setup

- TRL in Colab
- default base model: `Qwen/Qwen2.5-0.5B-Instruct`
- before/after evaluation on held-out easy and medium workflow states

### 7. Results section placeholders

- Insert reward curve screenshot
- Insert before/after score table
- Insert one failed baseline trace
- Insert one improved trained trace

### 8. Demo framing

- baseline model skips policy and under-documents cases
- trained model checks policy more consistently and routes medium cases better
- hard mode remains challenging because linked reasoning and workflow discipline both matter

## Assets to collect before publishing

- screenshot of `/docs`
- screenshot of one hard-mode queue state
- screenshot of a policy check observation
- screenshot of reward curve from Colab
- screenshot of before/after evaluation table
