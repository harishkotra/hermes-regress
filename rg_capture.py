"""Capture real turns into replayable cases.

Capture is deliberately passive: it never changes a tool call or an answer, it
only records what already happened. The only writes are JSON files under
``~/.hermes/regress/cases``.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

import rg_store as store

# Keys and credentials never belong in a case file.
SECRET_PATTERNS = [
    (re.compile(r"(?i)\b(password|passwd|pwd|secret|token|api[_-]?key|access[_-]?key)\b\s*[=:]\s*\S+"),
     r"\1=[REDACTED]"),
    (re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{4,}\b"),
     "[REDACTED_JWT]"),
    re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9_\-]{16,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"(?i)\b(bearer|token|api[_-]?key|secret|password)\b\s*[:=]\s*\S{8,}"),
    re.compile(r"\b[A-Fa-f0-9]{40,}\b"),
]


def redact(text: str) -> tuple[str, int]:
    hits = 0
    for pattern in SECRET_PATTERNS:
        if isinstance(pattern, tuple):
            regex, replacement = pattern
            text, n = regex.subn(replacement, text)
        else:
            text, n = pattern.subn("[REDACTED]", text)
        hits += n
    return text, hits


def _text_of(value: Any, limit: int) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        out = value
    elif isinstance(value, dict):
        for key in ("output", "stdout", "content", "text", "result", "message"):
            if isinstance(value.get(key), str):
                out = value[key]
                break
        else:
            out = json.dumps(value, ensure_ascii=False, default=str)
    elif isinstance(value, (list, tuple)):
        parts = [_text_of(v, limit) for v in value]
        out = "\n".join(p for p in parts if p)
    else:
        out = str(value)
    return out[:limit]


def _attr(msg: Any, name: str, default: Any = None) -> Any:
    if isinstance(msg, dict):
        return msg.get(name, default)
    return getattr(msg, name, default)


def _tool_calls_from_history(history: list[Any], limit: int) -> list[dict[str, Any]]:
    """Pair assistant tool calls with their tool results, in call order."""
    calls: list[dict[str, Any]] = []
    pending: dict[str, int] = {}
    for msg in history or []:
        role = _attr(msg, "role")
        if role == "assistant":
            for call in _attr(msg, "tool_calls") or []:
                fn = _attr(call, "function") or {}
                name = _attr(fn, "name") or ""
                raw_args = _attr(fn, "arguments") or "{}"
                if isinstance(raw_args, str):
                    try:
                        args = json.loads(raw_args)
                    except ValueError:
                        args = {"_raw": raw_args}
                else:
                    args = raw_args if isinstance(raw_args, dict) else {"_raw": str(raw_args)}
                calls.append({
                    "tool": name,
                    "args": args,
                    "result": "",
                    "source": "history",
                })
                call_id = _attr(call, "id") or ""
                if call_id:
                    pending[call_id] = len(calls) - 1
        elif role == "tool":
            call_id = _attr(msg, "tool_call_id") or ""
            idx = pending.pop(call_id, None)
            if idx is not None:
                calls[idx]["result"] = _text_of(_attr(msg, "content"), limit)
    return calls


class ToolBuffer:
    """Exact tool outputs observed live, used to fill gaps in the transcript."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.session_id: str | None = None

    def reset(self) -> None:
        self.events = []

    def record(self, tool_name: str, args: Any, result: Any, ok: Any = None,
               duration_ms: Any = None) -> None:
        event = {
            "tool": tool_name or "",
            "args": args if isinstance(args, dict) else {"_raw": str(args)},
            "result": result,
            "ok": ok,
            "ms": duration_ms,
        }
        self.events.append(event)


_BUFFER = ToolBuffer()


def buffer() -> ToolBuffer:
    return _BUFFER


def build_case(*, session_id: str, turn_id: str, prompt: str, answer: str,
               history: list[Any], model: str, platform: str) -> dict[str, Any] | None:
    cfg = store.load_config()
    prompt = (prompt or "").strip()
    if len(prompt) < int(cfg.get("min_prompt_chars", 12)):
        return None
    limit = int(cfg.get("max_result_chars", 20000))
    calls = _tool_calls_from_history(history, limit)

    # Live results are fuller than the transcript copy; use them in order when
    # the transcript did not carry a result for that call.
    live = list(_BUFFER.events)
    cursor = 0
    for call in calls:
        if call["result"]:
            check = call["tool"]
            while cursor < len(live) and live[cursor]["tool"] != check:
                cursor += 1
            if cursor < len(live):
                cursor += 1
            continue
        if cursor < len(live) and live[cursor]["tool"] == call["tool"]:
            event = live[cursor]
            call["result"] = _text_of(event["result"], limit)
            call["ms"] = event.get("ms")
            call["source"] = "live"
            cursor += 1

    excluded = {str(t) for t in cfg.get("exclude_tools", [])}
    calls = [c for c in calls if c["tool"] and c["tool"].split(":")[0] not in excluded]

    case_id = store.case_id_for(session_id, turn_id, prompt)
    redactions = 0
    if cfg.get("redact", True):
        for call in calls:
            call["result"], n = redact(call["result"] or "")
            redactions += n
            # Arguments carry secrets too: a curl with a bearer header, a
            # psql with a password, an api_key in a JSON body.
            raw_args = json.dumps(call.get("args", {}), default=str)
            clean_args, n = redact(raw_args)
            redactions += n
            if n:
                try:
                    call["args"] = json.loads(clean_args)
                except ValueError:
                    call["args"] = {"_redacted": clean_args}
        prompt, n = redact(prompt or "")
        redactions += n
        answer, n = redact(answer or "")
        redactions += n

    return {
        "schema": store.SCHEMA,
        "case_id": case_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        "session_id": session_id or "",
        "turn_id": turn_id or "",
        "platform": platform or "",
        "model": model or "",
        "cwd": os.getcwd(),
        "prompt": prompt,
        "answer": answer or "",
        "tool_calls": calls,
        "tool_count": len(calls),
        "redactions": redactions,
        "expect": [],
        "capture_version": 1,
    }


def _in_replay() -> bool:
    """True while a case is armed: a replay's own turns must not become cases.

    The env var covers a child process started by an older runner; armed_case()
    is the real signal, because the replay turn runs in a separate process that
    never shares our memory.
    """
    if os.environ.get("HERMES_REGRESS_REPLAY"):
        return True
    try:
        import rg_replay

        return rg_replay.armed_case() is not None
    except Exception:
        return False


def on_post_tool_call(*, tool_name: str = "", args: Any = None, result: Any = None,
                      ok: Any = None, duration_ms: Any = None, **_: Any) -> None:
    if _in_replay():
        return
    if not store.load_config().get("capture", True):
        return
    _BUFFER.record(tool_name, args, result, ok, duration_ms)


def on_post_llm_call(*, session_id: str = "", turn_id: str = "", user_message: Any = None,
                     assistant_response: Any = None, conversation_history: Any = None,
                     model: str = "", platform: str = "", **_: Any) -> None:
    if _in_replay():
        return
    if not store.load_config().get("capture", True):
        return
    case = build_case(
        session_id=session_id,
        turn_id=turn_id,
        prompt=_text_of(user_message, 8000),
        answer=_text_of(assistant_response, 40000),
        history=list(conversation_history or []),
        model=model,
        platform=str(platform or ""),
    )
    _BUFFER.reset()
    if case is None:
        return
    store.save_case(case)


def on_session_end(**_: Any) -> None:
    _BUFFER.reset()
