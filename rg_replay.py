"""Replay a captured case against the current model/config with no side effects.

A replay is a real Hermes turn: the same prompt goes through the same agent
loop, with the same tools on the same system prompt. The one difference is
that every tool call is vetoed at ``pre_tool_call`` and answered with the
output recorded during capture, so replay can never delete a file, push a
branch, or send a message.

The rewrite tells you whether the turn still reaches the same place after a
model swap, a prompt edit, a plugin change, or a Hermes upgrade.
"""

from __future__ import annotations

import difflib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import rg_store as store

TRACE_ENV = "REGRESS_TRACE"
DEBUG_ENV = "REGRESS_DEBUG_LOG"

def _env(name: str) -> str:
    """Read a debugging variable from this process's environment."""
    return os.environ.get(name, "")


# --------------------------------------------------------------------------
# fixture serving (child process side)
# --------------------------------------------------------------------------

_STATE: dict[str, Any] = {"case": None, "consumed": set(), "path": None, "checked": 0.0}


def arm_path() -> Path:
    return store.regress_home() / "replay" / "armed.json"


def arm(case: dict[str, Any], *, ttl_s: int = 900, trace_path: str = "") -> Path:
    """Publish a case as the active replay fixture set.

    Node: the replayed turn runs in the Hermes gateway process, which is not
    our child process, so a module global or an env var cannot carry this
    state across. A short-lived file the hook re-reads is the one channel both
    processes share.
    """
    path = arm_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "case_id": case.get("case_id"),
        "prompt": (case.get("prompt") or "")[:200],
        "tool_calls": case.get("tool_calls", []),
        "trace_path": trace_path or "",
        "expires_at": time.time() + ttl_s,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    _STATE.update({"case": None, "consumed": set(), "path": None, "checked": 0.0})
    return path


def disarm() -> None:
    try:
        arm_path().unlink()
    except OSError:
        pass
    _STATE.update({"case": None, "consumed": set(), "path": None, "checked": 0.0})


def armed_case() -> dict[str, Any] | None:
    path = arm_path()
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    if _STATE["case"] is not None and _STATE["checked"] == mtime:
        case = _STATE["case"]
    else:
        try:
            case = json.loads(path.read_text("utf-8"))
        except (OSError, ValueError):
            return None
        _STATE.update({"case": case, "consumed": set(), "checked": mtime})
    if float(case.get("expires_at") or 0) < time.time():
        return None
    return case


def _load_fixtures() -> dict[str, Any] | None:
    return armed_case()


def args_key(args: Any) -> str:
    if not isinstance(args, dict):
        return str(args)
    focused = {k: v for k, v in args.items() if k in (
        "command", "path", "file_path", "url", "query", "cmd", "pattern", "name")}
    pick = focused or args
    return json.dumps(pick, sort_keys=True, default=str)


def _score(recorded: dict[str, Any], tool: str, args: Any) -> int:
    """3 = same tool and identical args, 2 = same tool and no args to disambiguate.

    Anything weaker must not serve a recorded result: replaying the output of a
    different command would silently answer the wrong question.
    """
    if (recorded.get("tool") or "").strip().lower() != (tool or "").strip().lower():
        return 0
    recorded_key = args_key(recorded.get("args"))
    incoming_key = args_key(args)
    if recorded_key == incoming_key:
        return 3
    if recorded_key in {"", "{}", "None"} or incoming_key in {"", "{}", "None"}:
        return 2
    return 0


def serve(*, tool_name: str = "", args: Any = None, **_: Any) -> dict[str, Any]:
    _debug(f"serve called tool={tool_name}")
    case = _load_fixtures()
    if case is None:
        return {}
    consumed = _STATE["consumed"]
    best: tuple[int, int] | None = None
    for idx, recorded in enumerate(case.get("tool_calls", [])):
        if idx in consumed:
            continue
        score = _score(recorded, tool_name, args)
        if score > 0 and (best is None or score > best[0]):
            best = (score, idx)
            if score == 3:
                break
    if best is None:
        payload = f"[hermes-regress replay] no recorded output for {tool_name}; this tool was not called during capture."
        matched = None
    else:
        matched = best[1]
        consumed.add(matched)
        payload = case["tool_calls"][matched].get("result") or "[hermes-regress replay] recorded output was empty."
    _trace({
        "case_id": case.get("case_id"),
        "tool": tool_name,
        "args": args if isinstance(args, dict) else {"_raw": str(args)},
        "fixture_index": matched,
        "exact": bool(best and best[0] == 3),
    }, case)
    return {"action": "block", "message": payload}


def _debug(message: str) -> None:
    path = _env(DEBUG_ENV)
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(message + "\n")
    except OSError:
        pass


def _trace(entry: dict[str, Any], case: dict[str, Any] | None = None) -> None:
    path = _env(TRACE_ENV) or (case or {}).get("trace_path") or ""
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, default=str) + "\n")
    except OSError:
        pass


# --------------------------------------------------------------------------
# comparing a replayed turn to the recording
# --------------------------------------------------------------------------

def _norm(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[`*_#>\[\]()]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def answer_similarity(recorded: str, replayed: str) -> float:
    a, b = _norm(recorded), _norm(replayed)
    if not a or not b:
        return 0.0
    ratio = difflib.SequenceMatcher(None, a[:6000], b[:6000]).ratio()
    ta, tb = set(a.split()), set(b.split())
    jaccard = len(ta & tb) / max(1, len(ta | tb))
    return round(max(ratio, jaccard), 4)


def plan_similarity(recorded: list[dict[str, Any]], replayed: list[dict[str, Any]]) -> float:
    a = [c.get("tool", "") for c in recorded]
    b = [c.get("tool", "") for c in replayed]
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return round(difflib.SequenceMatcher(None, a, b).ratio(), 4)


def check_expectations(expect: list[str], answer: str) -> list[dict[str, Any]]:
    out = []
    low = (answer or "").lower()
    for rule in expect or []:
        ok = rule.lower() in low
        out.append({"rule": rule, "ok": ok})
    return out


def verdict_for(result: dict[str, Any], case: dict[str, Any], cfg: dict[str, Any]) -> str:
    if result.get("error"):
        return "error"
    checks = result.get("expectations") or []
    failed = [c for c in checks if not c["ok"]]
    if failed:
        return "regressed"
    sim = result.get("answer_similarity")
    if sim is None:
        sim = answer_similarity(case.get("answer", ""), result.get("answer", ""))
    plan = result["plan_similarity"]
    threshold = float(cfg.get("threshold", 0.55))
    plan_threshold = float(cfg.get("plan_threshold", 0.5))
    if checks:
        return "pass" if plan >= plan_threshold else "drift"
    if sim >= threshold and plan >= plan_threshold:
        return "pass"
    if sim >= 0.35:
        return "drift"
    return "regressed"


# --------------------------------------------------------------------------
# preflight: never replay against a runtime that cannot intercept
# --------------------------------------------------------------------------

_PROBE = {
    "case_id": "_probe",
    "prompt": "Use the terminal tool to run exactly: echo regress-probe",
    "tool_calls": [{"tool": "terminal", "args": {"command": "echo regress-probe"},
                    "result": "regress-probe-ok"}],
}

_PREFLIGHT: dict[str, Any] = {"ok": None, "detail": "", "checked": 0.0}
_PREFLIGHT_TTL = 600


def preflight(*, timeout: int = 150, force: bool = False) -> dict[str, Any]:
    """Prove this Hermes runtime serves recorded fixtures before we replay anything.

    Plugins load when the runtime starts, so a freshly edited or freshly
    installed plugin is invisible until the runtime restarts. Replaying against
    a runtime that does not intercept would execute real tools -- the exact
    thing replay exists to prevent -- so every run is gated on this probe.
    """
    if not force and _PREFLIGHT["ok"] is not None and time.time() < _PREFLIGHT["checked"]:
        return {"ok": _PREFLIGHT["ok"], "detail": _PREFLIGHT["detail"]}

    workdir = Path(tempfile.mkdtemp(prefix="regress-probe-"))
    trace_path = workdir / "trace.jsonl"
    trace_path.write_text("", encoding="utf-8")
    arm(_PROBE, ttl_s=max(120, timeout), trace_path=str(trace_path))
    output = ""
    try:
        proc = subprocess.run(
            [hermes_bin(), "chat", "-Q", "--oneshot", "-q", _PROBE["prompt"], "--max-turns", "4"],
            capture_output=True, text=True, timeout=timeout)
        output = (proc.stdout or "").strip()
    except (subprocess.TimeoutExpired, OSError) as exc:
        output = f"probe failed: {exc}"
    finally:
        disarm()
        try:
            hits = [l for l in trace_path.read_text("utf-8").splitlines() if l.strip()]
        except OSError:
            hits = []
        shutil.rmtree(workdir, ignore_errors=True)

    ok = bool(hits)
    detail = ("runtime intercepted the probe and served the recorded output"
              if ok else
              "runtime did not intercept a tool call, so replay would execute real tools. "
              "Restart Hermes (plugins load at startup) and run /regress run again")
    _PREFLIGHT.update({"ok": ok, "detail": detail, "checked": time.time() + _PREFLIGHT_TTL})
    return {"ok": ok, "detail": detail, "output": output}


# --------------------------------------------------------------------------
# running a case
# --------------------------------------------------------------------------

def hermes_bin() -> str:
    found = shutil.which("hermes")
    if found:
        return found
    return str(Path(sys.executable).with_name("hermes"))


def run_case(case: dict[str, Any], *, model: str | None = None, timeout: int | None = None,
             cfg: dict[str, Any] | None = None, extra_args: list[str] | None = None) -> dict[str, Any]:
    cfg = cfg or store.load_config()
    timeout = timeout or int(cfg.get("timeout_s", 240))
    probe = preflight()
    if not probe["ok"]:
        return {
            "case_id": case.get("case_id"),
            "prompt": case.get("prompt", "")[:300],
            "model": model or "(current)",
            "recorded_model": case.get("model", ""),
            "answer": "",
            "error": probe["detail"],
            "duration_ms": 0,
            "recorded_duration_ms": case.get("recorded_duration_ms") or 0,
            "replayed_calls": [],
            "plan_similarity": 0.0,
            "answer_similarity": 0.0,
            "fixture_hits": 0,
            "fixture_exact": 0,
            "fixture_total": 0,
            "expectations": [],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
            "verdict": "error",
        }
    workdir = tempfile.mkdtemp(prefix="regress-replay-")
    case_path = Path(workdir) / "case.json"
    prompt_path = Path(workdir) / "prompt.txt"
    trace_path = Path(workdir) / "trace.jsonl"
    case_path.write_text(json.dumps(case), "utf-8")
    prompt_path.write_text(case.get("prompt", ""), "utf-8")
    trace_path.write_text("", "utf-8")

    cmd = [hermes_bin(), "chat", "-Q", "--oneshot", "--query-file", str(prompt_path),
           "--max-turns", "40"]
    if model:
        cmd += ["-m", model]
    if extra_args:
        cmd += extra_args

    env = dict(os.environ)
    env[TRACE_ENV] = str(trace_path)
    arm(case, ttl_s=max(120, timeout + 60), trace_path=str(trace_path))

    started = time.time()
    error = ""
    output = ""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env,
                              cwd=case.get("cwd") if os.path.isdir(case.get("cwd") or "") else None)
        output = (proc.stdout or "").strip()
        if proc.returncode != 0:
            error = (proc.stderr or output or f"exit {proc.returncode}").strip()[-2000:]
    except subprocess.TimeoutExpired:
        error = f"replay timed out after {timeout}s"
    except OSError as exc:
        error = f"could not start hermes: {exc}"
    duration_ms = int((time.time() - started) * 1000)
    disarm()

    replayed: list[dict[str, Any]] = []
    try:
        for line in trace_path.read_text("utf-8").splitlines():
            if line.strip():
                replayed.append(json.loads(line))
    except OSError:
        pass

    exact = sum(1 for call in replayed if call.get("exact"))
    served = sum(1 for call in replayed if call.get("fixture_index") is not None)
    result = {
        "case_id": case["case_id"],
        "prompt": case.get("prompt", "")[:300],
        "model": model or "(current)",
        "recorded_model": case.get("model", ""),
        "answer": output,
        "error": error,
        "duration_ms": duration_ms,
        "recorded_duration_ms": case.get("recorded_duration_ms") or 0,
        "replayed_calls": replayed,
        "plan_similarity": plan_similarity(case.get("tool_calls", []), replayed),
        "answer_similarity": answer_similarity(case.get("answer", ""), output),
        "fixture_hits": served,
        "fixture_exact": exact,
        "fixture_total": len(replayed),
        "expectations": check_expectations(case.get("expect", []), output),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
    }
    result["verdict"] = verdict_for(result, case, cfg)
    shutil.rmtree(workdir, ignore_errors=True)
    return result


def run_suite(*, model: str | None = None, case_id: str | None = None, limit: int | None = None,
              timeout: int | None = None, extra_args: list[str] | None = None,
              progress=None) -> dict[str, Any]:
    cfg = store.load_config()
    cases = store.list_cases()
    if case_id:
        cases = [c for c in cases if c["case_id"].startswith(case_id)]
    if limit:
        cases = cases[:limit]
    run_id = store.new_run_id()
    results = []
    for index, case in enumerate(cases, 1):
        if progress:
            progress(index, len(cases), case)
        results.append(run_case(case, model=model, timeout=timeout, cfg=cfg,
                               extra_args=extra_args))
    summary = summarize(results)
    payload = {
        "schema": store.SCHEMA,
        "run_id": run_id,
        "model": model or "(current)",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        "summary": summary,
        "results": results,
    }
    store.save_run(run_id, payload)
    payload["run_id"] = run_id
    return payload


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {"pass": 0, "drift": 0, "regressed": 0, "error": 0}
    for result in results:
        counts[result.get("verdict", "error")] = counts.get(result.get("verdict", "error"), 0) + 1
    sims = [r["answer_similarity"] for r in results if not r.get("error")]
    return {
        "cases": len(results),
        **counts,
        "mean_answer_similarity": round(sum(sims) / len(sims), 4) if sims else 0.0,
        "mean_plan_similarity": round(
            sum(r["plan_similarity"] for r in results) / len(results), 4) if results else 0.0,
        "fixture_coverage": round(
            sum(r["fixture_hits"] for r in results) / max(1, sum(r["fixture_total"] for r in results)), 3),
        "duration_ms": sum(r["duration_ms"] for r in results),
    }