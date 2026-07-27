"""LLM access: local Ollama first, deterministic stub when no model is reachable.

Everything runs offline. No API keys, no cloud calls, no telemetry.
"""
import json
import os
import re
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


class Ollama:
    def __init__(self, model: str):
        self.model = model

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
        return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


class Stub:
    """Deterministic offline fallback: the full agent loop runs with no model.

    Recognizes each agent's prompt by its JSON marker and answers with fixed,
    parseable content so behavior is testable byte-for-byte.
    """

    model = "stub"

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
    if os.environ.get("FORGE_STUB"):
        return Stub()
    try:
        models = _available_models()
        if not models:
            return Stub()
        # prefer instruct/coder models over reasoning models for structured JSON
        pick = os.environ.get("FORGE_MODEL") or next(
            (m for m in models if any(k in m for k in ("coder", "llama", "qwen", "mistral"))),
            models[0],
        )
        return Ollama(pick)
    except Exception:
        return Stub()


def get_grader():
    """Model used only for grading answers. Defaults to the teaching model;
    set FORGE_GRADER (e.g. to a reasoning model like deepseek-r1) for stricter
    evaluation at the cost of latency."""
    pick = os.environ.get("FORGE_GRADER")
    if pick and not os.environ.get("FORGE_STUB"):
        return Ollama(pick)
    return get_llm()
