"""On-disk case store for hermes-regress.

Layout under ~/.hermes/regress (or $HERMES_REGRESS_HOME):

    config.json
    cases/<case_id>.json
    runs/<run_id>/results.json
    runs/<run_id>/report.md
    runs/<run_id>/junit.xml
    runs/<run_id>/manifest.json

Everything is plain JSON on purpose: a regression suite you cannot read,
diff, or commit to git is not a regression suite.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable

SCHEMA = "hermes-regress/v1"
MAX_CASES = 5000

DEFAULT_CONFIG: dict[str, Any] = {
    "capture": True,
    "redact": True,
    "exclude_tools": ["approval", "human_input"],
    "max_result_chars": 20000,
    "min_prompt_chars": 12,
    "threshold": 0.55,
    "plan_threshold": 0.5,
    "timeout_s": 240,
}


def regress_home() -> Path:
    """Where cases live: the active Hermes home, under plugin-data.

    get_hermes_home() is the same resolver the rest of Hermes uses, so a
    gateway running several profiles writes each profile's cases into that
    profile's home instead of one shared ~/.hermes/regress.
    """
    override = os.environ.get("HERMES_REGRESS_HOME")
    if override:
        return Path(override).expanduser()
    try:
        from hermes_constants import get_hermes_home

        base = Path(get_hermes_home())
    except Exception:
        base = Path(os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes"))
    return base.expanduser() / "plugin-data" / "hermes-regress"


def ensure_dirs() -> Path:
    home = regress_home()
    (home / "cases").mkdir(parents=True, exist_ok=True)
    (home / "runs").mkdir(parents=True, exist_ok=True)
    return home


def load_config() -> dict[str, Any]:
    home = ensure_dirs()
    path = home / "config.json"
    cfg = dict(DEFAULT_CONFIG)
    if path.exists():
        try:
            cfg.update(json.loads(path.read_text("utf-8")))
        except (OSError, ValueError):
            pass
    return cfg


def save_config(patch: dict[str, Any]) -> dict[str, Any]:
    cfg = load_config()
    cfg.update(patch)
    home = ensure_dirs()
    (home / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n", "utf-8")
    return cfg


def case_id_for(session_id: str, turn_id: str, prompt: str) -> str:
    raw = f"{session_id}\x00{turn_id or ''}\x00{prompt[:400]}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def write_private(path: Path, text: str) -> Path:
    """Write a file only the owner can read.

    Cases hold prompts, answers and tool arguments, so they are 0600: a shared
    box or a synced home should not leak them to another account.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)
    return path


def save_case(case: dict[str, Any]) -> Path:
    home = ensure_dirs()
    return write_private(home / "cases" / f"{case['case_id']}.json",
                         json.dumps(case, indent=2, sort_keys=True) + "\n")


def load_case(case_id: str) -> dict[str, Any] | None:
    path = regress_home() / "cases" / f"{case_id}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        return None


def list_cases() -> list[dict[str, Any]]:
    home = regress_home()
    cases: list[dict[str, Any]] = []
    directory = home / "cases"
    if not directory.exists():
        return cases
    for path in directory.glob("*.json"):
        try:
            cases.append(json.loads(path.read_text("utf-8")))
        except (OSError, ValueError):
            continue
    cases.sort(key=lambda c: c.get("created_at", ""), reverse=True)
    return cases


def prune(keep: int) -> int:
    cases = list_cases()
    removed = 0
    for case in cases[keep:]:
        path = regress_home() / "cases" / f"{case['case_id']}.json"
        try:
            path.unlink()
            removed += 1
        except OSError:
            pass
    return removed


def new_run_id() -> str:
    return time.strftime("%Y%m%dT%H%M%S", time.localtime())


def run_dir(run_id: str, create: bool = False) -> Path:
    path = regress_home() / "runs" / run_id
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def save_run(run_id: str, payload: dict[str, Any]) -> Path:
    path = run_dir(run_id, create=True) / "results.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", "utf-8")
    return path


def load_run(run_id: str) -> dict[str, Any] | None:
    path = run_dir(run_id) / "results.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        return None


def latest_run_id() -> str | None:
    directory = regress_home() / "runs"
    if not directory.exists():
        return None
    runs = sorted(p.name for p in directory.iterdir() if p.is_dir())
    return runs[-1] if runs else None


def hash_chain(items: Iterable[dict[str, Any]]) -> tuple[str, list[str]]:
    """Tamper-evident chain over case files: each link covers the previous hash."""
    prev = "0" * 64
    links: list[str] = []
    for item in items:
        blob = json.dumps(item, sort_keys=True, separators=(",", ":")).encode("utf-8")
        prev = hashlib.sha256(prev.encode("ascii") + blob).hexdigest()
        links.append(prev)
    return prev, links
