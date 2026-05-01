# Vera Bot - magicpin AI Challenge Submission

## Approach

This bot implements the required stateful FastAPI server for the magicpin Vera AI challenge:

- `GET /v1/healthz`
- `GET /v1/metadata`
- `POST /v1/context`
- `POST /v1/tick`
- `POST /v1/reply`

## Core Architecture

1. **Versioned context store** - in-memory context keyed by `(scope, context_id)`. Same-version reposts are accepted no-ops; higher versions replace atomically.
2. **Deterministic trigger routing** - `/tick` composes from trigger kind, merchant metrics, category voice, offer catalog, and customer facts. It caps output at 20 actions per tick.
3. **Grounded message composer** - default path is pure Python, no network call, so replay behavior is stable and fast. Optional Claude composition is available with `USE_LLM_COMPOSER=true`.
4. **Conversation state machine** - `/reply` handles opt-out, hostile replies, auto-replies, and clear intent transitions before using any fallback response.
5. **Suppression tracking** - fired triggers and suppression keys prevent duplicate sends in the same run.

## Model Choice

Default submission mode is deterministic rule-based composition. This is intentional: the judge injects fresh facts and runs under a timeout, so stable grounded decisions are safer than 20 sequential LLM calls.

Optional experimentation:

```bash
USE_LLM_COMPOSER=true
ANTHROPIC_API_KEY=sk-ant-your-real-key
```

## Quick Start

```bash
pip install -r requirements.txt
python bot.py
```

Bot URL:

```text
http://localhost:8080
```

Health check:

```bash
curl http://localhost:8080/v1/healthz
curl http://localhost:8080/v1/metadata
```

## Docker

```bash
docker build -t vera-bot .
docker run -p 8080:8080 vera-bot
```

## Deployment

Deploy the Docker image or Python app to Railway, Render, Fly.io, or any HTTPS host. Submit the public base URL, for example:

```text
https://your-bot.example.com
```

The judge will call:

```text
POST /v1/context
POST /v1/tick
POST /v1/reply
GET  /v1/healthz
GET  /v1/metadata
```
