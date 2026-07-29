# Security

Forge is a local-first tool. Here is exactly what leaves your machine, per engine
(`forge config` shows which one you're using):

| Engine | What leaves the machine |
|---|---|
| `stub` | Nothing. Deterministic offline responses (`forge/llm.py`, class `Stub`). |
| `ollama` | Nothing beyond localhost: prompts go to your local Ollama server at `http://localhost:11434` (override: `FORGE_OLLAMA`) (`forge/llm.py`, `OLLAMA_URL`). |
| `anthropic` | Your prompts — including lesson text, your answers, and any `--from` source excerpts — are sent to the configured Anthropic endpoint (`https://api.anthropic.com`, override: `FORGE_ANTHROPIC_URL`), with the API key in the `x-api-key` request header (`forge/llm.py`, `AnthropicEngine._open`). |

## Where the API key lives

- Sources, in the order the code actually reads them (`forge/config.py`, `api_key()`):
  a key stored in `config.json` (explicit `forge init --store-key` opt-in) **wins over**
  the environment variable named by `api_key_env` (default `ANTHROPIC_API_KEY`).
  If you rotate a key in your environment, also remove any stale stored key
  (`forge init` rewrites it). `config.json` is fully trusted local state.
- Optional: `~/.forge/config.json`, only when you explicitly opt in via
  `forge init --store-key`. The config file is written atomically with `0600`
  permissions (`forge/config.py`, `save()`). `forge doctor --full` warns if the
  perms have drifted.
- The key is never logged and never included in error messages
  (`forge/llm.py` error paths raise fixed, key-free hints).
- `forge bundle-debug` never packages `config.json`, the database, or any
  secret values. Non-secret paths/URLs from your environment (the `FORGE_DB`
  path, your Ollama URL) do appear in the bundled doctor output — review
  before sharing, as the tool itself reminds you.

## Docker networking

The container binds `0.0.0.0` by design (Docker's own network namespace);
the browser-facing security gate is the Host-header allowlist in
`forge/web.py` — non-`localhost` Host headers are rejected. Always access
the container as `http://localhost:8765`, not via the LAN IP; do not
publish port 8765 to public interfaces. The container also runs as an
unprivileged user (uid 10001, `USER forge`), so any process escape stays
non-root inside the namespace.

## Dashboard

`forge web` binds to `127.0.0.1` only. Every request is checked against a
Host-header allowlist (DNS-rebinding guard) and an Origin allowlist (CSRF
guard) — `forge/web.py`, `Handler._local_host`.

## Telemetry

None. The only network calls in the codebase are the engine requests above.

## Reporting issues

Open an issue: https://github.com/kgorle1111/forge/issues
