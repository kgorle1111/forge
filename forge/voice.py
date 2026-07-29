"""Voice loop: push-to-talk STT via whisper.cpp, TTS via `say`/`espeak-ng`.

100% local — audio never leaves the machine, nothing voice-related enters
bundle-debug. Everything here (whisper binary, model, recorder) is DETECTED,
never required: missing components produce a prescriptive error naming the
exact install command.

An unconfirmed dictation must never reach the mastery gate — every
transcript is shown to the learner for a one-key confirm/edit/type-instead.
"""
import os
import shutil
import subprocess
import sys
import tempfile

WHISPER_BINARIES = ("whisper-cli", "whisper-cpp", "whisper")
RECORDER_BINARIES = ("rec", "ffmpeg")


def _which_whisper() -> str | None:
    """FORGE_WHISPER overrides; else the first known binary on PATH.

    Explicit override that doesn't resolve fails closed — silently using a
    different binary than the user named would be a surprise, not a fallback.
    """
    override = os.environ.get("FORGE_WHISPER")
    if override:
        found = shutil.which(override)
        if found:
            return found
        if os.path.isabs(override) and os.path.exists(override):
            return override
        return None  # explicit override missed → don't fall through
    for name in WHISPER_BINARIES:
        found = shutil.which(name)
        if found:
            return found
    return None


def _which_recorder() -> str | None:
    for name in RECORDER_BINARIES:
        found = shutil.which(name)
        if found:
            return found
    return None


def _whisper_model() -> str | None:
    override = os.environ.get("FORGE_WHISPER_MODEL")
    if override and os.path.exists(override):
        return override
    models_dir = os.path.expanduser("~/.forge/whisper")
    if os.path.isdir(models_dir):
        bins = sorted((os.path.join(models_dir, f) for f in os.listdir(models_dir)
                       if f.endswith(".bin")),
                      key=os.path.getmtime, reverse=True)
        if bins:
            return bins[0]
    return None


def voice_status() -> list[tuple[str, bool, str]]:
    """(component, ok, prescriptive message) — surfaced by `forge doctor`."""
    whisper = _which_whisper()
    model = _whisper_model()
    recorder = _which_recorder()
    return [
        ("whisper", whisper is not None,
         (f"found at {whisper}" if whisper else
          "not found — brew install whisper-cpp (or set FORGE_WHISPER=/path)")),
        ("whisper model", model is not None,
         (f"using {os.path.basename(model)}" if model else
          "no .bin model in ~/.forge/whisper — download one (e.g. ggml-base.en.bin) "
          "or set FORGE_WHISPER_MODEL=/path/to/model.bin")),
        ("recorder", recorder is not None,
         (f"found at {recorder}" if recorder else
          "no recorder — brew install sox (for `rec`) or ffmpeg")),
    ]


def voice_ready() -> tuple[bool, str]:
    """Concise health check for --voice preflight."""
    status = voice_status()
    missing = [name for name, ok, _ in status if not ok]
    if missing:
        details = "; ".join(f"{n}: {m}" for n, ok, m in status if not ok)
        return False, f"voice unavailable ({', '.join(missing)}) — {details}"
    return True, "voice ready"


def _record(dest_wav: str) -> None:
    """Push-to-talk: record until Enter (SIGINT). Kills the recorder on Enter."""
    recorder = _which_recorder()
    if not recorder:
        raise RuntimeError("no recorder found — brew install sox or ffmpeg")
    if recorder.endswith("rec"):
        cmd = [recorder, "-q", "-c", "1", "-r", "16000", dest_wav]
    else:
        # ffmpeg: macOS avfoundation, else default
        if sys.platform == "darwin":
            cmd = [recorder, "-loglevel", "quiet", "-f", "avfoundation",
                   "-i", ":default", "-ar", "16000", "-ac", "1", "-y", dest_wav]
        else:
            cmd = [recorder, "-loglevel", "quiet", "-f", "alsa",
                   "-i", "default", "-ar", "16000", "-ac", "1", "-y", dest_wav]
    proc = subprocess.Popen(cmd)
    try:
        input()  # blocks until the user hits Enter again
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()


def _transcribe(wav: str) -> str:
    whisper = _which_whisper()
    model = _whisper_model()
    if not whisper:
        raise RuntimeError("whisper binary missing — brew install whisper-cpp")
    if not model:
        raise RuntimeError("whisper model missing — download a .bin to ~/.forge/whisper")
    proc = subprocess.run(
        [whisper, "-m", model, "-f", wav, "-nt", "-otxt"],
        capture_output=True, text=True, timeout=180,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"whisper failed: {proc.stderr.strip()[:200]}")
    txt_path = wav + ".txt"
    if not os.path.exists(txt_path):
        # stdout is progress/log lines, NOT the transcript — refuse to guess
        raise RuntimeError(
            "whisper wrote no transcript file — check the binary supports -otxt")
    with open(txt_path, encoding="utf-8") as f:
        text = f.read().strip()
    os.unlink(txt_path)
    return text


def record_and_transcribe() -> str:
    """One push-to-talk turn. Tempfile always cleaned up."""
    fd, wav = tempfile.mkstemp(prefix="forge-voice-", suffix=".wav")
    os.close(fd)
    try:
        _record(wav)
        return _transcribe(wav)
    finally:
        for p in (wav, wav + ".txt"):
            if os.path.exists(p):
                os.unlink(p)


def make_voice_ask(base_ask, base_emit):
    """Wrap the loop's ask() with confirm-before-submit dictation.

    The typed path is always available as a fallback so voice hiccups can
    never trap a learner — an unconfirmed transcript is NEVER submitted.
    """
    def _read(prompt: str) -> str:
        try:
            return input(prompt)
        except EOFError:  # piped/closed stdin → treat as blank, never crash
            return ""

    def ask(prompt: str) -> str:
        base_emit(prompt)  # existing emit prints + speaks if --speak on
        while True:
            hint = ("[voice] Enter to talk (Enter again to stop), type an "
                    "answer, or 'q' to abort:")
            user = _read(hint + "\n> ")
            if user.strip().lower() == "q":
                return ""  # caller handles empty answer per grading heuristic
            if user.strip():
                return user
            try:
                transcript = record_and_transcribe()
            except RuntimeError as e:
                base_emit(f"[voice] {e} — type your answer instead.")
                continue
            base_emit(f"[voice] heard: {transcript}")
            choice = _read("[voice] y=submit, e=edit, r=redo, t=type: ").strip().lower()
            if choice == "y" and transcript:
                return transcript
            if choice == "e":
                edited = _read("[voice] edit> ").strip()
                if edited:
                    return edited
            if choice == "t":
                typed = _read("> ")
                if typed.strip():
                    return typed
            # anything else → redo (fresh recording); show it so nothing looks silent
            base_emit("[voice] redo")

    return ask
