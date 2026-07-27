"""forge init: engine setup wizard.

Interactive on a TTY; scriptable via --engine/--model. Writes the 0600 config
through forge.config.save(). API keys are read via getpass only with explicit
--store-key opt-in and are never echoed, logged, or sent over HTTP.
"""
import getpass
import os
import sys

from . import config, llm

_OPTIONS = """[Forge] stdin is not a terminal, so the wizard cannot ask questions.
Pick one non-interactively:
  forge init --engine ollama [--model NAME]   local models via Ollama
  forge init --engine anthropic               cloud API (export ANTHROPIC_API_KEY first)
  forge init --engine stub                    offline demo mode"""


def cmd_init(args):
    if args.engine:
        _write(args.engine, args.model)
        _finish(args.engine)
        return
    if not sys.stdin.isatty():
        print(_OPTIONS)
        sys.exit(1)
    models = _probe()
    if models:
        print(f"[Forge] Ollama detected at {llm.OLLAMA_URL} with {len(models)} model(s):")
        for i, m in enumerate(models, 1):
            print(f"  {i}. {m}")
        pick = input("Pick a model number, Enter for auto, or 'n' to skip Ollama: ").strip()
        if pick.lower() != "n":
            model = None
            if pick.isdigit() and 1 <= int(pick) <= len(models):
                model = models[int(pick) - 1]
            _write("ollama", model)
            _finish("ollama")
            return
    else:
        print(f"[Forge] Ollama not reachable at {llm.OLLAMA_URL} "
              "(install from https://ollama.com, then `ollama serve`).")
    if input("Use the Anthropic API instead? [y/N]: ").strip().lower() == "y":
        _init_anthropic(args)
        return
    print("[Forge] Demo mode: a deterministic offline stub teaches canned "
          "lessons so you can explore the full loop with no model installed.")
    _write("stub", None)
    _finish("stub")


def _init_anthropic(args):
    key = os.environ.get("ANTHROPIC_API_KEY")
    entered = None
    if not key:
        print("[Forge] No ANTHROPIC_API_KEY in your environment. Default path: "
              "run `export ANTHROPIC_API_KEY=sk-ant-...` and rerun `forge init`.")
    if args.store_key:
        confirm = input("Store the key itself inside the 0600 config file "
                        "instead of the environment? [y/N]: ").strip().lower()
        if confirm == "y":
            entered = getpass.getpass("ANTHROPIC_API_KEY (input hidden): ").strip()
    if not (key or entered):
        print("[Forge] No API key available — export ANTHROPIC_API_KEY=... "
              "then rerun `forge init`.")
        sys.exit(1)
    cfg = {"engine": "anthropic", "model": None,
           "api_key_env": "ANTHROPIC_API_KEY"}
    if entered:
        cfg["api_key"] = entered
    path = config.save(cfg)
    print(f"[Forge] wrote {path} (anthropic)")
    _finish("anthropic")


def _probe() -> list[str]:
    try:
        return llm._available_models()
    except (OSError, ValueError, KeyError):
        return []


def _write(engine: str, model: str | None) -> None:
    cfg = {"engine": engine, "model": model}
    if engine == "anthropic":
        cfg["api_key_env"] = "ANTHROPIC_API_KEY"
    path = config.save(cfg)
    extra = f", model {model}" if model else ""
    print(f"[Forge] wrote {path} ({engine}{extra})")


def _finish(engine_name: str) -> None:
    """Doctor-style health line for the chosen engine, then the next step."""
    try:
        engine = config._build(engine_name, config.load())
        ok, msg = engine.healthy()
    except RuntimeError as e:
        ok, msg = False, str(e)
    print(f"[Forge] engine health: {'ok' if ok else 'FAIL'} — {msg}")
    print('[Forge] next: `forge web` for the dashboard, '
          'or `forge learn "<topic>"` to start.')


def setup_state(memory=None) -> dict:
    """Shared by GET/POST /api/setup. Never includes key material: the only
    strings that can carry secrets (config values) are never copied here."""
    cfg = config.load()
    zero_concepts = memory is None or len(memory.stats()) == 0
    first_run = not os.path.exists(config.config_path()) and zero_concepts
    try:
        engine = config.resolve_engine()
        ok, msg = engine.healthy()
        name, model = engine.name, engine.model
    except RuntimeError as e:
        ok, msg = False, str(e)
        name = os.environ.get("FORGE_ENGINE") or cfg.get("engine") or "unknown"
        model = cfg.get("model")
    return {"engine": name, "model": model, "healthy": ok,
            "message": msg, "first_run": first_run}
