"""`/regress` — the operator surface for captured cases and replay runs."""

from __future__ import annotations

import json
import sys
import time
from typing import Any

import rg_replay
import rg_report
import rg_store as store

HELP = """hermes-regress

  /regress status              capture state, case count, last run
  /regress on | off            toggle capture (takes effect next turn)
  /regress list [n]            recent cases
  /regress show <id>           one case: prompt, tools, answer
  /regress expect <id> <text>  add a must-appear assertion to a case
  /regress run [flags]         replay every case; flags: --model X, --case ID,
                               --limit N, --timeout S
  /regress report [run_id]     markdown report for a run
  /regress export [run_id]     zip + hash-chained manifest, prints path
  /regress prune --keep N      keep the newest N cases
  /regress demo                create a demo case and replay it
  /regress selftest            prove this runtime intercepts tools before you
                               trust a replay (run it after any restart)
"""


def _fmt_ts(value: str) -> str:
    return (value or "").replace("T", " ")[:16]


def cmd_status(_argv: list[str]) -> str:
    cfg = store.load_config()
    cases = store.list_cases()
    last = store.latest_run_id()
    lines = [
        f"capture: {'on' if cfg.get('capture', True) else 'off'}   cases: {len(cases)}   "
        f"home: {store.regress_home()}",
    ]
    if last:
        run = store.load_run(last) or {}
        summary = run.get("summary", {})
        lines.append(
            f"last run {last} on {run.get('model', '?')}: {summary.get('cases', 0)} cases, "
            f"pass {summary.get('pass', 0)}, drift {summary.get('drift', 0)}, "
            f"regressed {summary.get('regressed', 0)}, error {summary.get('error', 0)}"
        )
    else:
        lines.append("last run: none yet. Work normally, then /regress run --model <other-model>.")
    return "\n".join(lines)


def cmd_on(_argv: list[str]) -> str:
    store.save_config({"capture": True})
    return "Capture on. Real turns from here on are recorded as cases."


def cmd_off(_argv: list[str]) -> str:
    store.save_config({"capture": False})
    return "Capture off. Existing cases are untouched."


def cmd_list(argv: list[str]) -> str:
    limit = int(argv[0]) if argv and argv[0].isdigit() else 15
    cases = store.list_cases()[:limit]
    if not cases:
        return "No cases yet. Talk to Hermes normally, then run /regress status."
    rows = []
    for case in cases:
        rows.append(
            f"{case['case_id']}  {_fmt_ts(case.get('created_at', ''))}  "
            f"{case.get('tool_count', 0)} tools  {case.get('model', '?')[:24]}\n"
            f"    {case.get('prompt', '')[:110]}"
        )
    return "\n".join(rows)


def cmd_show(argv: list[str]) -> str:
    if not argv:
        return "Usage: /regress show <case_id>"
    case = store.load_case(argv[0])
    if not case:
        matches = [c for c in store.list_cases() if c["case_id"].startswith(argv[0])]
        case = matches[0] if matches else None
    if not case:
        return f"No case matching {argv[0]}."
    lines = [
        f"case {case['case_id']}  captured {_fmt_ts(case.get('created_at', ''))}  "
        f"model {case.get('model', '?')}  platform {case.get('platform') or '-'}",
        f"prompt: {case.get('prompt', '')}",
        f"answer: {(case.get('answer') or '')[:400]}",
        f"tools: {', '.join(c['tool'] for c in case.get('tool_calls', [])) or '(none)'}",
    ]
    for index, call in enumerate(case.get("tool_calls", [])[:5]):
        preview = (call.get("result") or "")[:160].replace("\n", " ")
        lines.append(f"  [{index}] {call['tool']} {json.dumps(call.get('args', {}))[:120]}")
        lines.append(f"      -> {preview}")
    if case.get("expect"):
        lines.append(f"expects: {case['expect']}")
    return "\n".join(lines)


def cmd_expect(argv: list[str]) -> str:
    if len(argv) < 2:
        return "Usage: /regress expect <case_id> <text that must appear>"
    case = store.load_case(argv[0])
    if not case:
        return f"No case {argv[0]}."
    rule = " ".join(argv[1:])
    case.setdefault("expect", []).append(rule)
    store.save_case(case)
    holds = rule.lower() in (case.get("answer") or "").lower()
    note = "" if holds else " (heads up: the recorded answer does not contain this)"
    return f"Expectation added to {case['case_id']}: {rule!r}{note}"


def cmd_run(argv: list[str]) -> str:
    model = None
    case_id = None
    limit = None
    timeout = None
    index = 0
    while index < len(argv):
        token = argv[index]
        if token in {"--model", "-m"} and index + 1 < len(argv):
            model = argv[index + 1]
            index += 2
        elif token == "--case" and index + 1 < len(argv):
            case_id = argv[index + 1]
            index += 2
        elif token == "--limit" and index + 1 < len(argv):
            limit = int(argv[index + 1])
            index += 2
        elif token == "--timeout" and index + 1 < len(argv):
            timeout = int(argv[index + 1])
            index += 2
        else:
            index += 1
    if not store.list_cases():
        return "No cases to replay yet. Capture some real turns first."
    payload = rg_replay.run_suite(model=model, case_id=case_id, limit=limit, timeout=timeout)
    summary = payload["summary"]
    rg_report.export_bundle(payload)
    lines = [
        f"run {payload['run_id']} against {payload['model']}: {summary['cases']} cases, "
        f"pass {summary['pass']}, drift {summary['drift']}, "
        f"regressed {summary['regressed']}, error {summary['error']}",
        f"mean answer similarity {summary['mean_answer_similarity']:.2f}, "
        f"plan similarity {summary['mean_plan_similarity']:.2f}, "
        f"fixture coverage {summary['fixture_coverage']:.0%}",
    ]
    broken = [r for r in payload["results"] if r["verdict"] in {"regressed", "error"}]
    for result in broken[:6]:
        why = result.get("error") or f"answer sim {result['answer_similarity']:.2f}"
        lines.append(f"  {result['case_id']}  {result['verdict'].upper()}  {why}")
    lines.append(f"report: /regress report {payload['run_id']}")
    return "\n".join(lines)


def cmd_report(argv: list[str]) -> str:
    run_id = argv[0] if argv else store.latest_run_id()
    if not run_id:
        return "No runs yet."
    payload = store.load_run(run_id)
    if not payload:
        return f"No run {run_id}."
    md = rg_report.report_markdown(payload)
    path = store.run_dir(run_id, create=True) / "report.md"
    path.write_text(md, "utf-8")
    return md + f"\n\nSaved to {path}"


def cmd_export(argv: list[str]) -> str:
    run_id = argv[0] if argv else store.latest_run_id()
    if not run_id:
        return "No runs yet."
    payload = store.load_run(run_id)
    if not payload:
        return f"No run {run_id}."
    bundle = rg_report.export_bundle(payload)
    head = json.loads((store.run_dir(run_id) / "manifest.json").read_text("utf-8"))["chain_head"]
    return f"bundle: {bundle}\nchain head: {head[:32]}"


def cmd_prune(argv: list[str]) -> str:
    keep = 500
    if "--keep" in argv:
        keep = int(argv[argv.index("--keep") + 1])
    removed = store.prune(keep)
    return f"Removed {removed} case{'s' if removed != 1 else ''}; keeping the newest {keep}."


def cmd_demo(_argv: list[str]) -> str:
    """A self-contained case so a fresh install can see replay work immediately."""
    case = {
        "schema": store.SCHEMA,
        "case_id": "demo0000demo0000",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        "session_id": "demo",
        "turn_id": "demo",
        "platform": "demo",
        "model": "(demo)",
        "cwd": "",
        "prompt": "Use the terminal tool to run: echo regress-demo. Then reply with the exact output.",
        "answer": "regress-demo",
        "tool_calls": [{
            "tool": "terminal",
            "args": {"command": "echo regress-demo"},
            "result": "regress-demo",
            "source": "demo",
        }],
        "tool_count": 1,
        "redactions": 0,
        "expect": ["regress-demo"],
        "capture_version": 1,
    }
    store.save_case(case)
    result = rg_replay.run_case(case)
    return (
        f"demo case {case['case_id']} replayed without executing the terminal call: "
        f"{result['verdict']} (answer sim {result['answer_similarity']:.2f}, "
        f"fixture {result['fixture_exact']}/{result['fixture_total']} exact)."
    )


def cmd_selftest(_argv: list[str]) -> str:
    probe = rg_replay.preflight(force=True)
    if probe["ok"]:
        return ("selftest: pass — a tool call was intercepted and answered from a recorded "
                "fixture. Replay is safe to run.")
    return (
        "selftest: fail — " + probe["detail"] + "\n"
        "Plugins are loaded when the Hermes runtime starts, so a plugin installed or "
        "edited since then is invisible until you restart."
    )


COMMANDS = {
    "status": cmd_status,
    "on": cmd_on,
    "off": cmd_off,
    "list": cmd_list,
    "ls": cmd_list,
    "show": cmd_show,
    "expect": cmd_expect,
    "run": cmd_run,
    "report": cmd_report,
    "export": cmd_export,
    "prune": cmd_prune,
    "demo": cmd_demo,
    "selftest": cmd_selftest,
    "doctor": cmd_selftest,
}


def handle(raw_args: str) -> str:
    argv = (raw_args or "").strip().split()
    if not argv or argv[0] in {"help", "-h", "--help"}:
        return HELP
    handler = COMMANDS.get(argv[0])
    if handler is None:
        return f"Unknown subcommand {argv[0]!r}.\n\n{HELP}"
    try:
        return handler(argv[1:])
    except Exception as exc:  # a slash command should never take the session down
        return f"regress {argv[0]} failed: {type(exc).__name__}: {exc}"


def main(argv: list[str] | None = None) -> int:
    """Headless entry point, for CI: ``python3 rg_cli.py run --model X``.

    Exit codes: 0 clean, 1 usage/error, 2 a case regressed. Pass
    ``--fail-on-regressed`` on ``run`` to make a regression fail the build.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    fail_on_regressed = "--fail-on-regressed" in argv
    argv = [a for a in argv if a != "--fail-on-regressed"]
    if not argv:
        print(HELP)
        return 0
    out = handle(" ".join(argv))
    print(out)
    if out.startswith("regress ") and " failed:" in out:
        return 1
    if fail_on_regressed and argv[0] == "run":
        payload = store.load_run(store.latest_run_id()) if store.latest_run_id() else {}
        summary = (payload or {}).get("summary") or {}
        if int(summary.get("regressed", 0)) or int(summary.get("error", 0)):
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
