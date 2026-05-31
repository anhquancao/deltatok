#!/usr/bin/env python3
"""PreToolUse hook: enforces the "Do not run code locally" and "Never sync the
Karolina checkout yourself" rules from CLAUDE.md by rejecting Bash commands
that violate them.

Reads a Claude Code hook JSON payload from stdin. Exits 0 to allow, 2 to deny
(with the message on stderr surfaced back to the model).
"""
from __future__ import annotations

import json
import re
import sys


KAROLINA_FORBIDDEN = [
    (r"\bgit\s+clean\s+-[a-zA-Z]*f", "git clean -f"),
    (r"\bgit\s+reset\s+--hard\b", "git reset --hard"),
    (r"\bgit\s+checkout\s+\.(\s|$)", "git checkout ."),
    (r"\bgit\s+restore\s+\.(\s|$)", "git restore ."),
    (r"\bgit\s+stash\s+(drop|clear)\b", "git stash drop/clear"),
    (r"\bgit\s+pull\b", "git pull"),
    (r"\bgit\s+fetch\b[^;&|]*&&\s*git\s+merge\b", "git fetch && git merge"),
    (r"\bgit\s+rebase\b", "git rebase"),
]


def block(message: str) -> None:
    print(message, file=sys.stderr)
    sys.exit(2)


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        sys.exit(0)

    if payload.get("tool_name") != "Bash":
        sys.exit(0)

    cmd = payload.get("tool_input", {}).get("command", "")
    if not cmd:
        sys.exit(0)

    # Rule 1: destructive ops targeting the Karolina checkout.
    if re.search(r"\bssh\s+karolina\b", cmd):
        for pat, label in KAROLINA_FORBIDDEN:
            if re.search(pat, cmd):
                block(
                    f"BLOCKED by CLAUDE.md: `{label}` on Karolina is forbidden.\n"
                    "The Karolina checkout at /home/it4i-anhquan/OccAny is the user's.\n"
                    "Never reset, clean, pull, rebase, or drop stashes there.\n"
                    "If this is genuinely needed, stop and ask the user to run it."
                )

    # Rule 2: pushing source files local → karolina via rsync/scp.
    tokens = cmd.split()
    if tokens and tokens[0] in {"rsync", "scp"}:
        positional = [t for t in tokens[1:] if not t.startswith("-")]
        if positional and "karolina:" in positional[-1]:
            block(
                "BLOCKED by CLAUDE.md: rsync/scp pushing files local → karolina is forbidden.\n"
                "The user syncs the Karolina checkout manually. Pulling from karolina is fine."
            )

    # Rule 3: GPU-bound work must run on Karolina. CPU-only Python is fine
    # locally — only block this repo's heavy entrypoints (train/eval/inference/
    # extraction) and their shell wrappers.
    local = re.sub(r"ssh\s+karolina\s+\"[^\"]*\"", "", cmd)
    local = re.sub(r"ssh\s+karolina\s+'[^']*'", "", local)
    local = re.sub(r"ssh\s+karolina\s+[^\n;&|]+", "", local)

    heavy_patterns = [
        r"\bpython3?\b[^;&|]*\b(inference|test_occ_rae|train_\S+|extract_\S+|infer_\S+|eval_\S+)\.py\b",
        r"\btorchrun\b",
        r"\bpython3?\s+-m\s+torch\.distributed\.(run|launch)\b",
        r"\bbash\s+sh/(train|eval|extract|infer)_\S+\.sh\b",
    ]
    for pat in heavy_patterns:
        if re.search(pat, local):
            block(
                "BLOCKED by CLAUDE.md: GPU-bound work must run on Karolina.\n"
                "Wrap the command in: ssh karolina \"conda activate occany && <cmd>\"\n"
                "(CPU-only Python like `python3 -m json.tool` is fine locally.)"
            )

    sys.exit(0)


if __name__ == "__main__":
    main()
