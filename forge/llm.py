"""LLM engines: local Ollama, Anthropic API, and a deterministic offline Stub.

Engine selection lives in forge.config.resolve_engine(); the default remains
fully offline (Ollama or Stub). Anthropic is opt-in via key/config.
"""
import json
import os
import re
import urllib.error
import urllib.request

OLLAMA_URL = os.environ.get("FORGE_OLLAMA", "http://localhost:11434")


def _post(path: str, body: dict, timeout: int = 600) -> dict:
    req = urllib.request.Request(
        f"{OLLAMA_URL}{path}",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"


class Ollama:
    name = "ollama"

    def __init__(self, model: str):
        self.model = model

    def healthy(self) -> tuple[bool, str]:
        try:
            models = _available_models()
        except (OSError, ValueError, KeyError):
            return (False, (f"Ollama not reachable at {OLLAMA_URL} — start it "
                            "with `ollama serve`, or switch engines with "
                            "FORGE_ENGINE=anthropic or FORGE_STUB=1"))
        if self.model not in models:
            return (False, (f"model '{self.model}' not installed — run "
                            f"`ollama pull {self.model}` or unset FORGE_MODEL"))
        return (True, f"ollama reachable, model '{self.model}' installed")

    def ask(self, prompt: str, as_json: bool = False) -> str:
        body = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.4},
        }
        if as_json:
            body["format"] = "json"
        text = _post("/api/generate", body)["response"]
        # reasoning models (deepseek-r1) wrap chain-of-thought in <think> tags
        return THINK_RE.sub("", text).strip()


class AnthropicEngine:
    """Anthropic Messages API over stdlib urllib. Key comes from env or
    ~/.forge/config.json; it is never logged and never appears in errors."""

    name = "anthropic"

    def __init__(self, model: str | None = None):
        self.model = model or DEFAULT_ANTHROPIC_MODEL

    def _key(self) -> str | None:
        from . import config
        return config.api_key()

    def healthy(self) -> tuple[bool, str]:
        if self._key():
            return (True, f"API key found, model '{self.model}'")
        return (False, ("no Anthropic API key — set ANTHROPIC_API_KEY or run "
                        "`forge init`"))

    def ask(self, prompt: str, as_json: bool = False) -> str:
        key = self._key()
        if not key:
            raise RuntimeError("no Anthropic API key — set ANTHROPIC_API_KEY "
                               "or run `forge init`")
        if as_json:
            prompt += "\n\nRespond with ONLY valid JSON, no prose."
        base = os.environ.get("FORGE_ANTHROPIC_URL", "https://api.anthropic.com")
        req = urllib.request.Request(
            f"{base}/v1/messages",
            json.dumps({
                "model": self.model,
                "max_tokens": 4096,
                "temperature": 0.4,
                "messages": [{"role": "user", "content": prompt}],
            }).encode(),
            {"Content-Type": "application/json", "x-api-key": key,
             "anthropic-version": ANTHROPIC_VERSION},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data = json.loads(r.read())
        except urllib.error.HTTPError as e:
            # never include the key in any message
            hints = {
                401: "API key rejected — check ANTHROPIC_API_KEY or rerun forge init",
                403: "API key lacks access — check your Anthropic account/plan",
                404: f"model '{self.model}' not found — check FORGE_MODEL",
                429: "rate limited — wait a minute and retry",
                529: "Anthropic API overloaded — retry shortly",
            }
            hint = hints.get(e.code, "see https://docs.anthropic.com/en/api/errors")
            raise RuntimeError(f"Anthropic API error HTTP {e.code}: {hint}") from None
        except urllib.error.URLError:
            raise RuntimeError("could not reach api.anthropic.com — check your "
                               "network, or use FORGE_ENGINE=ollama") from None
        text = "".join(b.get("text", "") for b in data.get("content", []))
        text = THINK_RE.sub("", text).strip()
        if as_json:
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
        return text


class Stub:
    """Deterministic offline fallback: the full agent loop runs with no model.

    Recognizes each agent's prompt by its JSON marker and answers with fixed,
    parseable content so behavior is testable byte-for-byte.
    """

    name = "stub"
    model = "stub"

    def healthy(self) -> tuple[bool, str]:
        return (True, "stub is always available (offline demo mode)")

    def ask(self, prompt: str, as_json: bool = False) -> str:
        if '"concepts"' in prompt:
            topic = _quoted(prompt, "topic")
            return json.dumps({"concepts": [
                {"name": f"{topic}: foundations", "summary": f"core definitions of {topic}"},
                {"name": f"{topic}: mechanics", "summary": f"how {topic} works step by step"},
                {"name": f"{topic}: applications", "summary": f"where {topic} is used and why"},
            ]})
        if '"questions"' in prompt:
            return json.dumps({"questions": [
                {"prompt": "Reconstruct the lesson from memory.",
                 "key_points": ["alpha", "beta", "gamma"]},
                {"prompt": "Explain the core mechanism without looking back.",
                 "key_points": ["alpha", "gamma"]},
            ]})
        # lesson prompt
        name = _quoted(prompt, "concept")
        angle = re.search(r"angle: (\w+)", prompt)
        return f"LESSON[{name}|{angle.group(1) if angle else '?'}]: remember alpha beta gamma."


def _quoted(prompt: str, after: str) -> str:
    m = re.search(rf'{after} "([^"]+)"', prompt)
    return m.group(1) if m else "topic"


def _available_models() -> list[str]:
    req = urllib.request.Request(f"{OLLAMA_URL}/api/tags")
    with urllib.request.urlopen(req, timeout=2) as r:
        return [m["name"] for m in json.loads(r.read())["models"]]


def get_llm():
    from . import config
    return config.resolve_engine()


def get_grader():
    """Model used only for grading answers. Defaults to the teaching model;
    set FORGE_GRADER (e.g. to a reasoning model like deepseek-r1) for stricter
    evaluation at the cost of latency. The model id applies to whichever
    engine type get_llm() resolved (Ollama or Anthropic)."""
    llm = get_llm()
    pick = os.environ.get("FORGE_GRADER")
    if pick and llm.name != "stub":
        return type(llm)(pick)
    return llm
