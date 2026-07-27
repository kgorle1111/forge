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

    def ask_stream(self, prompt: str, as_json: bool = False):
        """Yield raw text chunks; join + post-process happens in stream_or_ask."""
        body = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": 0.4},
        }
        if as_json:
            body["format"] = "json"
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/generate",
            json.dumps(body).encode(),
            {"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=600) as r:
            for line in r:  # JSON-lines: one object per line
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                # in-band error frame: a stream that "just ends" here would
                # store a truncated lesson as canon — fail loudly instead
                if obj.get("error"):
                    raise RuntimeError(
                        f"Ollama failed mid-generation: {obj['error']} — "
                        "retry, or try a smaller model")
                chunk = obj.get("response", "")
                if chunk:  # the final {"done": true} line carries no text
                    yield chunk


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

    def _open(self, prompt: str, stream: bool = False):
        """Build + open the Messages request; shared prescriptive error mapping."""
        key = self._key()
        if not key:
            raise RuntimeError("no Anthropic API key — set ANTHROPIC_API_KEY "
                               "or run `forge init`")
        base = os.environ.get("FORGE_ANTHROPIC_URL", "https://api.anthropic.com")
        payload = {
            "model": self.model,
            "max_tokens": 4096,
            "temperature": 0.4,
            "messages": [{"role": "user", "content": prompt}],
        }
        if stream:
            payload["stream"] = True
        req = urllib.request.Request(
            f"{base}/v1/messages",
            json.dumps(payload).encode(),
            {"Content-Type": "application/json", "x-api-key": key,
             "anthropic-version": ANTHROPIC_VERSION},
        )
        try:
            return urllib.request.urlopen(req, timeout=120)
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

    def ask(self, prompt: str, as_json: bool = False) -> str:
        if as_json:
            prompt += "\n\nRespond with ONLY valid JSON, no prose."
        try:
            with self._open(prompt) as r:
                data = json.loads(r.read())
        except json.JSONDecodeError:
            raise RuntimeError(
                "Anthropic API returned a non-JSON response — likely a proxy "
                "or captive portal in the way; check your network") from None
        text = "".join(b.get("text", "") for b in data.get("content", []))
        text = THINK_RE.sub("", text).strip()
        if as_json:
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
        return text

    def ask_stream(self, prompt: str, as_json: bool = False):
        """Yield text chunks from SSE content_block_delta events."""
        if as_json:
            prompt += "\n\nRespond with ONLY valid JSON, no prose."
        with self._open(prompt, stream=True) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                try:
                    ev = json.loads(line[5:])
                except json.JSONDecodeError:
                    continue  # e.g. the final "data: [DONE]" sentinel
                if ev.get("type") == "error":
                    # in-band SSE error (e.g. overloaded mid-stream): never
                    # return a truncated lesson as success
                    etype = ev.get("error", {}).get("type", "unknown")
                    raise RuntimeError(
                        f"Anthropic stream failed mid-generation ({etype}) — "
                        "retry shortly")
                if (ev.get("type") == "content_block_delta"
                        and ev.get("delta", {}).get("type") == "text_delta"):
                    yield ev["delta"]["text"]


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

    def ask_stream(self, prompt: str, as_json: bool = False):
        """Deterministic word-by-word chunks; joined == ask() exactly."""
        for chunk in re.split(r"(\s+)", self.ask(prompt, as_json)):
            if chunk:
                yield chunk


def _quoted(prompt: str, after: str) -> str:
    m = re.search(rf'{after} "([^"]+)"', prompt)
    return m.group(1) if m else "topic"


def _available_models() -> list[str]:
    req = urllib.request.Request(f"{OLLAMA_URL}/api/tags")
    with urllib.request.urlopen(req, timeout=2) as r:
        return [m["name"] for m in json.loads(r.read())["models"]]


def stream_or_ask(engine, prompt: str, on_chunk=None, as_json: bool = False) -> str:
    """Stream via engine.ask_stream when available and on_chunk is given,
    calling on_chunk per raw chunk; otherwise fall back to engine.ask().
    Returns the same post-processed text ask() would."""
    stream = getattr(engine, "ask_stream", None)
    if not (stream and on_chunk):
        return engine.ask(prompt, as_json=as_json)
    chunks = []
    for c in stream(prompt, as_json=as_json):
        chunks.append(c)
        on_chunk(c)
    text = THINK_RE.sub("", "".join(chunks)).strip()
    if as_json:
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
    return text


def get_llm():
    from . import config
    return config.resolve_engine()


def get_grader():
    """Model used only for grading answers. Defaults to the teaching model;
    set FORGE_GRADER (e.g. to a reasoning model like deepseek-r1) for stricter
    evaluation at the cost of latency. The model id applies to whichever
    engine type get_llm() resolved (Ollama or Anthropic)."""
    from . import config
    llm = get_llm()
    pick = os.environ.get("FORGE_GRADER") or config.model_for("grade")
    if pick and llm.name != "stub":
        return type(llm)(pick)
    return llm
