#!/usr/bin/env python3
"""Doc DoD check: run safe README commands and verify relative links.

- Extracts every fenced ```bash block from README.md and UP-AND-RUNNING/README.md.
- SKIP: network/interactive/install commands and anything with a placeholder.
- RUN: safe offline commands (forge doctor, --help, python -c ...) against
  .venv/bin with FORGE_STUB=1 and temp FORGE_CONFIG/FORGE_DB; assert exit 0.
- Verifies every relative markdown link resolves to an existing path.

Exit non-zero listing failures. Stdlib only.
"""
import os
import re
import shlex
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = ["README.md", os.path.join("UP-AND-RUNNING", "README.md")]
VENV_BIN = os.path.join(ROOT, ".venv", "bin")

SKIP_TOKENS = (
    "uvx", "uv tool", "ollama", "git clone", "curl", "brew", "apt",
    "setup.sh", "pip install", "export ", "unset ", "alias", "cd ",
    "forge init", "forge web", "forge learn", "forge review",
    "forge serve", ">> ~/", "echo ",
)
PLACEHOLDER = re.compile(r"<[^>]+>|\.\.\.|\bNAME\b|\bmy_notes\b|\bnotes\.md\b|\bpaper\.pdf\b")
RUN_PREFIXES = ("forge doctor", "forge stats", "forge --help",
                "python tests/", "python -c")


def bash_blocks(path):
    text = open(path, encoding="utf-8").read()
    return re.findall(r"```bash\n(.*?)```", text, re.DOTALL)


def classify(cmd):
    if PLACEHOLDER.search(cmd):
        return "SKIP"
    if any(tok in cmd for tok in SKIP_TOKENS):
        return "SKIP"
    norm = cmd.replace(".venv/bin/", "")
    if any(norm.startswith(p) for p in RUN_PREFIXES):
        return "RUN"
    return "SKIP"  # ponytail: whitelist, never run unknown commands


def check_links(path):
    text = open(path, encoding="utf-8").read()
    base = os.path.dirname(path)
    bad = []
    for target in re.findall(r"\]\(([^)#]+)(?:#[^)]*)?\)", text):
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        if not os.path.exists(os.path.join(base, target)):
            bad.append(f"{path}: dead link -> {target}")
    return bad


def main():
    failures = []
    tmp = tempfile.mkdtemp(prefix="forge_docs_")
    env = dict(os.environ,
               FORGE_STUB="1",
               FORGE_CONFIG=os.path.join(tmp, "config.json"),
               FORGE_DB=os.path.join(tmp, "docs.db"),
               PATH=VENV_BIN + os.pathsep + os.environ.get("PATH", ""))
    ran = skipped = 0
    for doc in DOCS:
        path = os.path.join(ROOT, doc)
        for block in bash_blocks(path):
            for line in block.splitlines():
                cmd = line.split("#", 1)[0].strip()
                if not cmd:
                    continue
                if classify(cmd) == "SKIP":
                    skipped += 1
                    continue
                ran += 1
                argv = shlex.split(cmd.replace(".venv/bin/", VENV_BIN + "/"))
                if argv[0] in ("python", "python3"):
                    argv[0] = os.path.join(VENV_BIN, "python")
                elif not argv[0].startswith("/"):
                    argv[0] = os.path.join(VENV_BIN, argv[0])
                r = subprocess.run(argv, capture_output=True, text=True,
                                   cwd=ROOT, env=env, timeout=120)
                status = "ok" if r.returncode == 0 else f"exit {r.returncode}"
                print(f"RUN  {cmd} -> {status}")
                if r.returncode != 0:
                    failures.append(f"{doc}: `{cmd}` failed ({r.returncode}): "
                                    f"{(r.stderr or r.stdout).strip()[:300]}")
        failures.extend(check_links(path))
    print(f"commands: {ran} run, {skipped} skipped")
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)
    print("docs: OK")


if __name__ == "__main__":
    main()
