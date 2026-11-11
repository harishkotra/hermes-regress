"""Render a run as something you can act on: markdown, JUnit, a hash-chained bundle."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import rg_store as store

VERDICT_ICON = {"pass": "PASS", "drift": "DRIFT", "regressed": "REGRESSED", "error": "ERROR"}


def report_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    model = payload.get("model", "")
    lines = [
        f"# hermes-regress run {payload.get('run_id', '')}",
        "",
        f"Model under test: `{model}`  |  Cases: {summary.get('cases', 0)}  |  "
        f"pass {summary.get('pass', 0)} / drift {summary.get('drift', 0)} / "
        f"regressed {summary.get('regressed', 0)} / error {summary.get('error', 0)}",
        "",
        f"Mean answer similarity {summary.get('mean_answer_similarity', 0):.2f}, "
        f"mean tool-plan similarity {summary.get('mean_plan_similarity', 0):.2f}, "
        f"fixture coverage {summary.get('fixture_coverage', 0):.0%}.",
        "",
        "| verdict | case | tools (recorded → replay) | answer sim | plan sim | fixtures |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for result in payload.get("results", []):
        recorded = len(_case_calls(result["case_id"]))
        replayed = len(result.get("replayed_calls", []))
        lines.append(
            f"| {VERDICT_ICON.get(result.get('verdict'), '?')} | `{result['case_id']}` "
            f"| {recorded} → {replayed} | {result.get('answer_similarity', 0):.2f} "
            f"| {result.get('plan_similarity', 0):.2f} "
            f"| {result.get('fixture_exact', 0)}/{result.get('fixture_total', 0)} exact |"
        )
    failures = [r for r in payload.get("results", []) if r.get("verdict") in {"regressed", "error"}]
    if failures:
        lines += ["", "## What broke", ""]
        for result in failures:
            case = store.load_case(result["case_id"]) or {}
            lines.append(f"### `{result['case_id']}` — {VERDICT_ICON[result.get('verdict')]}")
            lines.append("")
            lines.append(f"Prompt: {result.get('prompt', '')[:280]}")
            if result.get("error"):
                lines.append(f"Error: `{result['error'][:400]}`")
            for check in result.get("expectations", []):
                if not check["ok"]:
                    lines.append(f"- missing expected content: `{check['rule']}`")
            recorded_answer = (case.get("answer") or "")[:600]
            answers = (result.get("answer") or "")[:600]
            if recorded_answer and answers:
                lines += ["", "Recorded answer:", "", f"> {recorded_answer}", "",
                          "Replayed answer:", "", f"> {answers}"]
            plan_a = ", ".join(c.get("tool", "") for c in case.get("tool_calls", [])) or "(none)"
            plan_b = ", ".join(c.get("tool", "") for c in result.get("replayed_calls", [])) or "(none)"
            lines += ["", f"Tools recorded: {plan_a}", "", f"Tools replayed: {plan_b}", ""]
    lines += ["", "Every tool call above was answered from the capture. Replay executed nothing.", ""]
    return "\n".join(lines)


def _case_calls(case_id: str) -> list[Any]:
    case = store.load_case(case_id) or {}
    return case.get("tool_calls", [])


def junit_xml(payload: dict[str, Any]) -> str:
    results = payload.get("results", [])
    failures = sum(1 for r in results if r.get("verdict") in {"regressed", "error"})
    cases = []
    for result in results:
        name = escape(f"regress.{result['case_id']}")
        time_s = round(result.get("duration_ms", 0) / 1000, 3)
        verdict = result.get("verdict")
        if verdict in {"pass", "drift"}:
            cases.append(f'    <testcase classname="hermes-regress" name="{name}" time="{time_s}"/>')
            continue
        message = escape(result.get("error") or f"{verdict}: answer similarity "
                        f"{result.get('answer_similarity', 0):.2f}, plan "
                        f"{result.get('plan_similarity', 0):.2f}")
        cases.append(
            f'    <testcase classname="hermes-regress" name="{name}" time="{time_s}">\n'
            f'      <failure message="{message}"/>\n'
            f'    </testcase>'
        )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<testsuite name="hermes-regress {escape(str(payload.get("run_id", "")))}" '
        f'tests="{len(results)}" failures="{failures}">\n'
        + "\n".join(cases) + "\n</testsuite>\n"
    )


def export_bundle(payload: dict[str, Any]) -> Path:
    """Zip the run plus every case it touched, chained so tampering is visible."""
    run_id = payload.get("run_id", "run")
    directory = store.run_dir(run_id, create=True)
    case_ids = [r["case_id"] for r in payload.get("results", [])]
    cases = [store.load_case(cid) for cid in case_ids]
    cases = [c for c in cases if c]
    chain, links = store.hash_chain(cases)
    manifest = {
        "schema": store.SCHEMA,
        "run_id": run_id,
        "model": payload.get("model"),
        "created_at": payload.get("created_at"),
        "cases": [{"case_id": c["case_id"], "sha256": links[i]} for i, c in enumerate(cases)],
        "chain_head": chain,
        "note": "sha256 chain over case files, in listed order. Recompute to verify nothing changed.",
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", "utf-8")
    (directory / "report.md").write_text(report_markdown(payload), "utf-8")
    (directory / "junit.xml").write_text(junit_xml(payload), "utf-8")
    bundle = directory / f"hermes-regress-{run_id}.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(directory / "results.json", "results.json")
        zf.write(directory / "report.md", "report.md")
        zf.write(directory / "junit.xml", "junit.xml")
        zf.write(directory / "manifest.json", "manifest.json")
        for case in cases:
            zf.writestr(f"cases/{case['case_id']}.json", json.dumps(case, indent=2, sort_keys=True))
    return bundle
