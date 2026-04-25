# FraudShield Hugging Face Deployment Guide

FraudShield is packaged as a Docker Space and is ready to deploy to:

- GitHub: `https://github.com/DevikaJ2005/Fraudshield`
- Hugging Face Space: `https://huggingface.co/spaces/DevikaJ2005/fraudshield-1`

## What gets deployed

The Space runs:

- `python -m server.app`
- FastAPI on port `7860`
- frozen snapshot data from `data/fraudshield_cases.json`
- a lightweight explorer UI at `/`

The HTTP runtime exposes:

- `GET /health`
- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /info`
- `GET /tasks`
- `GET /metadata`
- `GET /schema`
- `POST /mcp`

## Deployment options

### Option 1: GitHub sync

Recommended for hackathon iteration speed.

1. Create a new Hugging Face Space.
2. Choose `Docker`.
3. Enable GitHub sync.
4. Connect `DevikaJ2005/Fraudshield`.
5. Let Hugging Face rebuild on every push.

### Option 2: Direct git push

If you prefer manual deployment:

```bash
git remote add space https://huggingface.co/spaces/DevikaJ2005/fraudshield-1
git push space main
```

## Local preflight

Run these before pushing:

```bash
python inference.py
python -X utf8 validate_api.py
python -m openenv.cli validate .
python -m server.app
```

Then verify:

```bash
curl http://127.0.0.1:7860/health
curl http://127.0.0.1:7860/metadata
curl http://127.0.0.1:7860/schema
```

You can also open:

```bash
http://127.0.0.1:7860/
```

to inspect the environment in the browser before training.

The live server should also pass:

```bash
python -m openenv.cli validate --url http://127.0.0.1:7860
```

## Optional environment variables

The server itself does not require LLM credentials to start.

`inference.py` supports proxy-based LLM evaluation when these are available:

```bash
API_BASE_URL=https://router.huggingface.co/v1
API_KEY=<optional-local-key>
MODEL_NAME=<optional-model>
```

Aliases accepted by the code:

- `APIBASEURL`
- `APIKEY`
- `MODELNAME`
- `HF_TOKEN`
- `HFTOKEN`
- `OPENAI_API_KEY`

## Deployment notes

- Runtime evaluation uses only the committed snapshot.
- The richer investigation workflow is available through both REST and MCP.
- The root page is a lightweight manual explorer, not a separate product layer.
- The submission-safe inference path remains deterministic when no API credentials are injected.
