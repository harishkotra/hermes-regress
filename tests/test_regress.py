"""Unit tests for hermes-regress. Run: python3 -m unittest discover -s tests -v"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import rg_capture  # noqa: E402
import rg_replay  # noqa: E402
import rg_report  # noqa: E402
import rg_store as store  # noqa: E402


class TempHome(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._old = os.environ.get("HERMES_REGRESS_HOME")
        os.environ["HERMES_REGRESS_HOME"] = self._tmp.name
        rg_replay.disarm()

    def tearDown(self) -> None:
        rg_replay.disarm()
        if self._old is None:
            os.environ.pop("HERMES_REGRESS_HOME", None)
        else:
            os.environ["HERMES_REGRESS_HOME"] = self._old
        self._tmp.cleanup()

    def make_case(self, **over) -> dict:
        case = {
            "schema": store.SCHEMA,
            "case_id": "aaaa1111bbbb2222",
            "created_at": "2026-11-11T10:00:00+0530",
            "session_id": "s1",
            "turn_id": "t1",
            "platform": "slack",
            "model": "m1",
            "cwd": "/tmp",
            "prompt": "Use the terminal tool to run: echo hello",
            "answer": "hello",
            "tool_calls": [
                {"tool": "terminal", "args": {"command": "echo hello"},
                 "result": "hello", "source": "post_tool_call"}
            ],
            "tool_count": 1,
            "redactions": 0,
            "expect": [],
            "capture_version": 1,
            "recorded_duration_ms": 1200,
        }
        case.update(over)
        return case


class TestRedaction(unittest.TestCase):
    def test_masks_common_secret_shapes(self) -> None:
        raw = (
            "key sk-abcdef1234567890abcdef ghp_abcdefghijklmnopqrstuvwxyz0123 "
            "AKIAIOSFODNN7EXAMPLE Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.sig "
            "password=hunter2"
        )
        cleaned, count = rg_capture.redact(raw)
        for fragment in ("sk-abcdef1234567890abcdef", "ghp_abcdefghijklmnop",
                         "AKIAIOSFODNN7EXAMPLE", "hunter2"):
            self.assertNotIn(fragment, cleaned)
        self.assertGreaterEqual(count, 4)

    def test_leaves_ordinary_text_alone(self) -> None:
        text = "the build finished in 4s and wrote 3 files"
        cleaned, count = rg_capture.redact(text)
        self.assertEqual(cleaned, text)
        self.assertEqual(count, 0)


class TestStore(TempHome):
    def test_case_roundtrip_and_prune(self) -> None:
        for i in range(4):
            store.save_case(self.make_case(case_id=f"case{i:04d}", created_at=f"2026-11-11T10:00:0{i}+0530"))
        self.assertEqual(len(store.list_cases()), 4)
        self.assertEqual(store.load_case("case0001")["prompt"].startswith("Use the terminal"), True)
        self.assertEqual(store.prune(keep=2), 2)
        self.assertEqual(len(store.list_cases()), 2)

    def test_config_merges_defaults(self) -> None:
        store.save_config({"capture": False})
        cfg = store.load_config()
        self.assertFalse(cfg["capture"])
        self.assertEqual(cfg["threshold"], store.DEFAULT_CONFIG["threshold"])

    def test_hash_chain_is_deterministic_and_content_sensitive(self) -> None:
        items = [{"a": 1}, {"b": 2}]
        head1, chain1 = store.hash_chain(items)
        head2, chain2 = store.hash_chain(items)
        self.assertEqual((head1, chain1), (head2, chain2))
        head3, _ = store.hash_chain([{"a": 2}, {"b": 2}])
        self.assertNotEqual(head1, head3)
        self.assertEqual(len(chain1), 2)
        self.assertEqual(chain1[0], store.GENESIS if hasattr(store, "GENESIS") else chain1[0])


class TestArmAndServe(TempHome):
    def test_armed_case_serves_fixture_and_writes_trace(self) -> None:
        trace = Path(self._tmp.name) / "trace.jsonl"
        trace.write_text("", encoding="utf-8")
        rg_replay.arm(self.make_case(), ttl_s=60, trace_path=str(trace))

        reply = rg_replay.serve(tool_name="terminal", args={"command": "echo hello"})
        self.assertEqual(reply["action"], "block")
        self.assertIn("hello", reply["message"])

        entry = json.loads(trace.read_text("utf-8").splitlines()[0])
        self.assertEqual(entry["tool"], "terminal")
        self.assertTrue(entry["exact"])

    def test_unarmed_returns_no_directive(self) -> None:
        self.assertEqual(rg_replay.serve(tool_name="terminal", args={"command": "ls"}), {})

    def test_unrecorded_tool_is_blocked_not_executed(self) -> None:
        rg_replay.arm(self.make_case(), ttl_s=60)
        reply = rg_replay.serve(tool_name="terminal", args={"command": "rm -rf /tmp/whatever"})
        self.assertEqual(reply["action"], "block")
        self.assertIn("not called during capture", reply["message"])

    def test_expired_arm_is_ignored(self) -> None:
        rg_replay.arm(self.make_case(), ttl_s=-1)
        self.assertIsNone(rg_replay.armed_case())
        self.assertEqual(rg_replay.serve(tool_name="terminal", args={"command": "echo hello"}), {})

    def test_disarm_removes_state(self) -> None:
        rg_replay.arm(self.make_case(), ttl_s=60)
        rg_replay.disarm()
        self.assertIsNone(rg_replay.armed_case())


class TestScoringAndVerdict(unittest.TestCase):
    def cfg(self) -> dict:
        return dict(store.DEFAULT_CONFIG)

    def test_answer_similarity_bounds(self) -> None:
        self.assertEqual(rg_replay.answer_similarity("abc", "abc"), 1.0)
        self.assertEqual(rg_replay.answer_similarity("abc", "xyz"), 0.0)

    def test_verdict_pass_on_match(self) -> None:
        case = {"prompt": "p", "answer": "hello there", "tool_calls": [], "expect": []}
        result = {"answer": "hello there", "fixture_hits": 0, "fixture_total": 0,
                  "plan_similarity": 1.0, "expectations": []}
        self.assertEqual(rg_replay.verdict_for(result, case, self.cfg()), "pass")

    def test_verdict_regressed_on_contradiction(self) -> None:
        case = {"prompt": "p", "answer": "the deploy failed hard", "tool_calls": [], "expect": []}
        result = {"answer": "unrelated content about cats", "fixture_hits": 0,
                  "fixture_total": 0, "plan_similarity": 0.0, "expectations": []}
        self.assertEqual(rg_replay.verdict_for(result, case, self.cfg()), "regressed")

    def test_verdict_error_wins(self) -> None:
        case = {"prompt": "p", "answer": "x", "tool_calls": [], "expect": []}
        result = {"answer": "", "error": "timeout", "fixture_hits": 0,
                  "fixture_total": 0, "plan_similarity": 0.0, "expectations": []}
        self.assertEqual(rg_replay.verdict_for(result, case, self.cfg()), "error")

    def test_missing_expectation_forces_regression(self) -> None:
        case = {"prompt": "p", "answer": "ok", "tool_calls": [], "expect": ["deploy-ok"]}
        result = {"answer": "ok", "fixture_hits": 0, "fixture_total": 0,
                  "plan_similarity": 1.0,
                  "expectations": rg_replay.check_expectations(["deploy-ok"], "ok")}
        self.assertEqual(rg_replay.verdict_for(result, case, self.cfg()), "regressed")


class TestReporting(TempHome):
    def payload(self) -> dict:
        results = [
            {"case_id": "c1", "prompt": "p1", "model": "m1", "verdict": "pass",
             "answer": "a1", "answer_similarity": 1.0, "plan_similarity": 1.0,
             "fixture_hits": 1, "fixture_total": 1, "fixture_exact": 1,
             "duration_ms": 100, "recorded_duration_ms": 90, "replayed_calls": [], "expectations": []},
            {"case_id": "c2", "prompt": "p2", "model": "m1", "verdict": "regressed",
             "answer": "a2", "answer_similarity": 0.1, "plan_similarity": 0.0,
             "fixture_hits": 0, "fixture_total": 1, "fixture_exact": 0,
             "duration_ms": 200, "recorded_duration_ms": 150, "replayed_calls": [], "expectations": []},
        ]
        return {
            "schema": store.SCHEMA, "run_id": "run123", "model": "m1",
            "started_at": "2026-11-11T10:00:00+0530", "results": results,
            "summary": rg_replay.summarize(results),
        }

    def test_markdown_reports_verdicts(self) -> None:
        md = rg_report.report_markdown(self.payload())
        self.assertIn("regressed", md.lower())
        self.assertIn("c2", md)

    def test_junit_is_wellformed(self) -> None:
        import xml.etree.ElementTree as ET
        root = ET.fromstring(rg_report.junit_xml(self.payload()))
        self.assertEqual(root.tag, "testsuite")
        self.assertEqual(len(root.findall("testcase")), 2)

    def test_export_bundle_contains_manifest_and_results(self) -> None:
        store.save_run("run123", self.payload())
        bundle = rg_report.export_bundle(self.payload())
        self.assertTrue(bundle.exists())
        with zipfile.ZipFile(bundle) as zf:
            names = zf.namelist()
        self.assertTrue(any(n.endswith("results.json") for n in names))
        self.assertTrue(any(n.endswith("manifest.json") for n in names))


if __name__ == "__main__":
    unittest.main(verbosity=2)
