# Security

Forge is a local-first tool. Here is exactly what leaves your machine, per engine
(`forge config` shows which one you're using):

| Engine | What leaves the machine |
|---|---|
| `stub` | Nothing. Deterministic offline responses (`forge/llm.py`, class `Stub`). |
| `ollama` | Nothing beyond localhost: prompts go to your local Ollama server at `http://localhost:11434` (override: `FORGE_OLLAMA`) (`forge/llm.py`, `OLLAMA_URL`). |
| `anthropic` | Your prompts — including lesson text, your answers, and any `--from` source excerpts — are sent to the configured Anthropic endpoint (`https://api.anthropic.com`, override: `FORGE_ANTHROPIC_URL`), with the API key in the `x-api-key` request header (`forge/llm.py`, `AnthropicEngine._open`). |

## Where the API key lives

- Preferred: the `ANTHROPIC_API_KEY` environment variable (`forge/config.py`, `api_key()`).
- Optional: `~/.forge/config.json`, only when you explicitly opt in via
  `forge init --store-key`. The config file is written atomically with `0600`
  permissions (`forge/config.py`, `save()`). `forge doctor --full` warns if the
  perms have drifted.
- The key is never logged and never included in error messages
  (`forge/llm.py` error paths raise fixed, key-free hints).
- `forge bundle-debug` never packages `config.json`, the database, or
  environment variables.

## Dashboard

`forge web` binds to `127.0.0.1` only. Every request is checked against a
Host-header allowlist (DNS-rebinding guard) and an Origin allowlist (CSRF
guard) — `forge/web.py`, `Handler._local_host`.

## Telemetry

None. The only network calls in the codebase are the engine requests above.

## Reporting issues

Open an issue: https://github.com/kgorle1111/forge/issues
