"""Engine configuration: ~/.forge/config.json (path override via FORGE_CONFIG).

Schema: {"engine": "ollama"|"anthropic"|"stub", "model": str|null,
         "api_key_env": "ANTHROPIC_API_KEY"} and optionally "api_key"
(only ever written by the init wizard with explicit opt-in; tolerated here).
Key values are never logged and never appear in error messages.
"""
import json
import os

DEFAULT_PATH = os.path.expanduser("~/.forge/config.json")


def config_path() -> str:
    return os.environ.get("FORGE_CONFIG") or DEFAULT_PATH


def load() -> dict:
    try:
        with open(config_path(), encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save(cfg: dict) -> str:
    path = config_path()
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    os.chmod(path, 0o600)
    return path


def api_key(cfg: dict | None = None) -> str | None:
    cfg = load() if cfg is None else cfg
    return cfg.get("api_key") or os.environ.get(
        cfg.get("api_key_env", "ANTHROPIC_API_KEY"))


def resolve_engine():
    """Single engine-resolution path used by get_llm().

    Order: FORGE_STUB > FORGE_ENGINE (hard fail if unhealthy) > config
    "engine" (same hard behavior) > auto Ollama > auto Anthropic > Stub.
    """
    from . import llm

    if os.environ.get("FORGE_STUB"):
        return llm.Stub()
    cfg = load()
    name = os.environ.get("FORGE_ENGINE") or cfg.get("engine")
    if name:
        engine = _build(name, cfg)
        ok, msg = engine.healthy()
        if not ok:
            raise RuntimeError(f"engine '{name}' is unavailable: {msg}")
        return engine
    # auto-detect: Ollama first, then Anthropic key, else demo stub
    try:
        models = llm._available_models()
    except (OSError, ValueError, KeyError):
        models = []
    if models:
        return llm.Ollama(_pick_ollama(models, cfg))
    if api_key(cfg):
        return llm.AnthropicEngine(
            model=os.environ.get("FORGE_MODEL") or cfg.get("model"))
    return llm.Stub()


def _pick_ollama(models: list[str], cfg: dict) -> str:
    return os.environ.get("FORGE_MODEL") or cfg.get("model") or next(
        (m for m in models if any(k in m for k in ("coder", "llama", "qwen", "mistral"))),
        models[0],
    )


def _build(name: str, cfg: dict):
    from . import llm

    if name == "stub":
        return llm.Stub()
    if name == "anthropic":
        return llm.AnthropicEngine(
            model=os.environ.get("FORGE_MODEL") or cfg.get("model"))
    if name == "ollama":
        model = os.environ.get("FORGE_MODEL") or cfg.get("model")
        if not model:
            try:
                models = llm._available_models()
            except (OSError, ValueError, KeyError):
                models = []
            if not models:
                raise RuntimeError(
                    f"Ollama not reachable at {llm.OLLAMA_URL} — start it with "
                    "`ollama serve`, or set FORGE_ENGINE=anthropic or FORGE_STUB=1")
            model = _pick_ollama(models, cfg)
        return llm.Ollama(model)
    raise RuntimeError(
        f"unknown engine '{name}' — set FORGE_ENGINE (or config \"engine\") "
        "to ollama, anthropic, or stub")
