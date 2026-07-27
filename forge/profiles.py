"""Learner profile helpers.

Each learner is just a separate local SQLite file. This keeps the single-user
architecture intact while making the common "second learner" case explicit.
"""
import os
import re
import shutil
import time


_PROFILE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def profile_db_path(name: str) -> str:
    name = (name or "").strip()
    if not _PROFILE.fullmatch(name):
        raise ValueError("learner must be 1-64 letters, numbers, dots, dashes, or underscores")
    return os.path.expanduser(f"~/.forge/{name}.db")


def _stored_profile_path(name: str) -> str:
    return os.path.expanduser("~/.forge/forge.db") if name == "default" else profile_db_path(name)


def list_profiles() -> list[dict]:
    root = os.path.expanduser("~/.forge")
    current = os.environ.get("FORGE_LEARNER", "default")
    rows = []
    if os.path.isdir(root):
        for fn in sorted(os.listdir(root)):
            if not fn.endswith(".db"):
                continue
            path = os.path.join(root, fn)
            name = "default" if fn == "forge.db" else fn[:-3]
            st = os.stat(path)
            rows.append({"name": name, "path": path, "bytes": st.st_size,
                         "modified": time.strftime("%Y-%m-%d %H:%M",
                                                   time.localtime(st.st_mtime)),
                         "current": name == current})
    default_path = os.path.expanduser("~/.forge/forge.db")
    if os.path.exists(default_path) and not any(r["name"] == "default" for r in rows):
        st = os.stat(default_path)
        rows.insert(0, {"name": "default", "path": default_path, "bytes": st.st_size,
                        "modified": time.strftime("%Y-%m-%d %H:%M",
                                                  time.localtime(st.st_mtime)),
                        "current": current == "default"})
    return rows


def copy_profile(source: str, target: str, overwrite: bool = False) -> str:
    src = _stored_profile_path(source)
    dst = _stored_profile_path(target)
    if not os.path.exists(src):
        raise ValueError(f"profile not found: {source}")
    if os.path.exists(dst) and not overwrite:
        raise ValueError(f"profile already exists: {target}")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def select_profile(name: str | None) -> str | None:
    """Set FORGE_DB for this process and return the selected DB path."""
    if not name:
        return None
    path = profile_db_path(name)
    os.environ["FORGE_DB"] = path
    os.environ["FORGE_LEARNER"] = name
    return path
