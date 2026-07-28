"""The Forge dashboard: stdlib-only local web server. Zero new dependencies.

    forge web          ->  http://127.0.0.1:8765

The agent loop is conversational (it blocks waiting for your answer), so each
learn/review session runs in a background thread bridged to the browser by two
queues: the browser polls /api/events for agent output and POSTs answers to
/api/answer. Binds to 127.0.0.1 only — this is a personal tool, not a service.
"""
import json
import os
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .graph import run_review, run_topic
from .graphview import layout
from .llm import get_llm
from .memory import DAY, Memory, retrievability, risk_reasons, risk_score
from .profiles import select_profile
from .sources import (SourceError, clamp_source, read_uploaded_source,
                      source_manifest as build_source_manifest)


class Session:
    """One live learn/review conversation between the agents and the browser."""

    def __init__(self):
        self.inbox: queue.Queue = queue.Queue()    # answers from the browser
        self.events: queue.Queue = queue.Queue()   # agent output to the browser
        self.active = False

    def emit(self, text=""):
        self.events.put({"type": "say", "text": str(text)})

    def emit_chunk(self, text: str):
        # additive event type: a live fragment of the lesson being generated;
        # the full lesson still arrives afterwards as a normal "say" event
        self.events.put({"type": "chunk", "text": str(text)})

    def ask(self, prompt: str) -> str:
        self.events.put({"type": "ask", "text": str(prompt)})
        return self.inbox.get()

    def start(self, target) -> bool:
        if self.active:
            return False
        while not self.events.empty():   # drop leftovers from a prior session
            self.events.get_nowait()
        while not self.inbox.empty():
            self.inbox.get_nowait()
        self.active = True

        def run():
            try:
                target()
            except Exception as e:
                self.events.put({"type": "say", "text": f"[error] {e}"})
            finally:
                self.active = False
                self.events.put({"type": "end"})

        threading.Thread(target=run, daemon=True).start()
        return True


SESSION = Session()
_MODEL = None


def _model() -> str:
    global _MODEL
    if _MODEL is None:
        _MODEL = get_llm().model
    return _MODEL


def _source_from_body(body: dict) -> tuple[str, dict]:
    if body.get("source_data"):
        name = str(body.get("source_name") or "upload.txt")
        source = read_uploaded_source(name, str(body.get("source_data") or ""))
        return source, build_source_manifest([(name, source)])
    source = clamp_source(str(body.get("source") or body.get("source_text") or ""))
    return source, build_source_manifest([("typed source", source)]) if source else {}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _local_host(self) -> bool:
        """Reject non-local Host headers (DNS-rebinding guard for a local API)."""
        host = (self.headers.get("Host") or "").split(":")[0].strip("[]")
        if host not in ("127.0.0.1", "localhost", "::1"):
            return False
        # CSRF guard: browsers attach Origin on cross-site requests; the Host
        # check alone can't stop an evil page blind-POSTing to 127.0.0.1
        origin = self.headers.get("Origin")
        if origin:
            try:
                ohost = origin.split("//", 1)[1].split(":")[0].strip("[]")
            except IndexError:
                return False
            return ohost in ("127.0.0.1", "localhost", "::1")
        return True

    def _send(self, obj, ctype="application/json", code=200):
        body = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if not self._local_host():
            return self._send({"error": "forbidden"}, code=403)
        if self.path == "/":
            page = open(os.path.join(os.path.dirname(__file__), "dashboard.html"),
                        "rb").read()
            self._send(page, "text/html; charset=utf-8")
        elif self.path == "/api/state":
            m = Memory()
            self._send({"active": SESSION.active, "model": _model(),
                        "learner": os.environ.get("FORGE_LEARNER", "default"),
                        "due": len(m.due_cards()),
                        "concepts": len(m.stats()),
                        "streak": m.streak()})
        elif self.path == "/api/stats":
            now = time.time()
            rows = []
            for r in Memory().stats():
                d = dict(r)
                d["recall"] = retrievability(r["interval_days"], r["due"], now)
                d["due_in_days"] = (r["due"] - now) / DAY
                d["risk"] = risk_score(r, now)
                d["reasons"] = risk_reasons(r, now)
                rows.append(d)
            self._send(rows)
        elif self.path == "/api/forecast":
            self._send(Memory().forecast())
        elif self.path == "/api/due":
            self._send(Memory().due_queue(limit=25))
        elif self.path == "/api/weak":
            self._send(Memory().weak_cards(25))
        elif self.path == "/api/topics":
            self._send(Memory().topic_summaries())
        elif self.path.startswith("/api/search?"):
            q = _query_param(self.path, "q")[:200]
            self._send(Memory().search_lessons(q, 10))
        elif self.path.startswith("/api/card?"):
            topic = _query_param(self.path, "topic")[:200]
            concept = _query_param(self.path, "concept")[:200]
            detail = Memory().card_detail(topic, concept)
            self._send(detail or {"error": "not found"}, code=200 if detail else 404)
        elif self.path == "/api/setup":
            from .wizard import setup_state
            self._send(setup_state(Memory()))
        elif self.path == "/api/debt":
            self._send(Memory().review_debt())
        elif self.path == "/api/progress":
            self._send(Memory().progress_rows())
        elif self.path.startswith("/api/artifact?"):
            path = os.path.realpath(_query_param(self.path, "path"))
            root = os.path.realpath(os.environ.get(
                "FORGE_ARTIFACT_DIR", os.path.expanduser("~/.forge/visuals")))
            if (not path.endswith(".svg") or not path.startswith(root + os.sep)
                    or not os.path.exists(path)):
                return self._send({"error": "not found"}, code=404)
            with open(path, "rb") as f:
                self._send(f.read(), "image/svg+xml; charset=utf-8")
        elif self.path.startswith("/api/transcript"):
            limit = int(_query_param(self.path, "limit") or 100)
            topic = _query_param(self.path, "topic")[:200]
            concept = _query_param(self.path, "concept")[:200]
            self._send({"text": Memory().transcript(limit, topic, concept)})
        elif self.path == "/api/graphdata":
            self._send(layout(Memory()))
        elif self.path == "/api/events":
            evs = []
            try:
                while len(evs) < 100:
                    evs.append(SESSION.events.get_nowait())
            except queue.Empty:
                pass
            self._send(evs)
        else:
            self._send({"error": "not found"}, code=404)

    def do_POST(self):
        if not self._local_host():
            return self._send({"error": "forbidden"}, code=403)
        n = int(self.headers.get("Content-Length") or 0)
        if n > 16 * 1024 * 1024:  # bound attacker-controlled reads (source uploads fit well under this)
            return self._send({"error": "body too large"}, code=413)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return self._send({"error": "bad json"}, code=400)
        if self.path == "/api/learn":
            topic = str(body.get("topic", "")).strip()[:200]
            if not topic:
                return self._send({"error": "topic required"}, code=400)
            try:
                source, manifest = _source_from_body(body)
            except SourceError as e:
                return self._send({"error": str(e)}, code=400)
            ok = SESSION.start(lambda: run_topic(
                topic, ask=SESSION.ask, emit=SESSION.emit,
                memory=Memory(), source=source, source_manifest=manifest,
                emit_chunk=SESSION.emit_chunk))
            self._send({"ok": ok} if ok else {"error": "session already running"})
        elif self.path == "/api/review":
            ok = SESSION.start(lambda: run_review(
                ask=SESSION.ask, emit=SESSION.emit, memory=Memory()))
            self._send({"ok": ok} if ok else {"error": "session already running"})
        elif self.path == "/api/answer":
            SESSION.inbox.put(str(body.get("text", "")))
            self._send({"ok": True})
        elif self.path == "/api/due-now":
            topic = str(body.get("topic", "")).strip()[:200]
            concept = str(body.get("concept", "")).strip()[:200]
            try:
                self._send(Memory().due_now(topic, concept))
            except ValueError as e:
                self._send({"error": str(e)}, code=404)
        elif self.path == "/api/suspend":
            topic = str(body.get("topic", "")).strip()[:200]
            concept = str(body.get("concept", "")).strip()[:200]
            try:
                self._send(Memory().suspend_card(topic, concept))
            except ValueError as e:
                self._send({"error": str(e)}, code=404)
        elif self.path == "/api/setup":
            from .config import load, save
            from .wizard import setup_state
            engine = str(body.get("engine", "")).strip()[:20]
            model = str(body.get("model") or "").strip()[:200] or None
            if engine == "anthropic":
                # keys never travel over HTTP, so this engine can't be set here
                return self._send({"error": "set ANTHROPIC_API_KEY and rerun "
                                   "forge init — keys are never sent over HTTP"},
                                  code=400)
            if engine not in ("ollama", "stub"):
                return self._send(
                    {"error": "engine must be ollama or stub — for the "
                     "Anthropic API run `forge init` in a terminal"}, code=400)
            # merge, never overwrite: a wizard-stored api_key must survive
            # an engine switch from the dashboard
            cfg = load()
            cfg.update({"engine": engine, "model": model})
            save(cfg)
            self._send(setup_state(Memory()))
        elif self.path == "/api/learner":
            name = str(body.get("learner", "")).strip()
            try:
                select_profile(name)
            except ValueError as e:
                return self._send({"error": str(e)}, code=400)
            self._send({"ok": True, "learner": os.environ.get("FORGE_LEARNER", "default")})
        else:
            self._send({"error": "not found"}, code=404)


def serve(port: int = 8765, learner: str | None = None):
    """FORGE_BIND overrides the bind address (containers); non-default values
    expose the server beyond localhost — the Host allowlist still applies."""
    select_profile(learner)
    host = os.environ.get("FORGE_BIND", "127.0.0.1")
    try:
        server = ThreadingHTTPServer((host, port), Handler)
    except OSError as e:
        raise SystemExit(
            f"[Forge] port {port} is already in use — is another Forge running? "
            f"Try `forge web --port {port + 1}` or stop the other process. ({e.strerror})"
        ) from None
    who = os.environ.get("FORGE_LEARNER")
    profile = f", learner: {who}" if who else ""
    print(f"[Forge] dashboard at http://127.0.0.1:{port}  (model: {_model()}{profile})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Forge] dashboard stopped.")


def _query_param(path: str, key: str) -> str:
    from urllib.parse import parse_qs, urlparse
    return parse_qs(urlparse(path).query).get(key, [""])[0]
