# hermes-regress

Regression tests for your agent, built from the sessions you already had.

Every Hermes plugin, prompt tweak, model swap and config change alters how your agent behaves. Nothing tells you whether it still does the things you actually rely on. `hermes-regress` watches your real turns, turns each finished turn into a test case, and replays those cases against a different model or setup with every tool side effect stubbed out.

```
/regress run --model anthropic/claude-sonnet-4
```

```
run 20261111T081500  model anthropic/claude-sonnet-4  cases 12
  pass 9   drift 2   regressed 1   error 0
  regressed  a1b2c3d4  "deploy the staging build and confirm the health check"
```

## Why this exists

A Hermes install accumulates trust in behaviour, not code. You know your agent checks the log file before restarting a service, writes the changelog in your format, and asks before touching production. Change one thing and that trust is silently invalid: no test fails, no error appears, the agent just answers differently.

Benchmarks do not cover this, because the tasks you care about are the ones you invented: your repo layout, your deploy script, your house style. `hermes-regress` builds the test suite from those.

## Install

```bash
git clone https://github.com/harishkotra/hermes-regress ~/hermes-regress
ln -s ~/hermes-regress ~/.hermes/plugins/hermes-regress
hermes plugins enable regress
```

Restart Hermes. Plugins are loaded when the runtime starts, so a freshly installed or freshly edited plugin is invisible until then. `/regress selftest` proves the runtime is intercepting before you trust a replay.

## Use

Capture is passive. There is nothing to switch on: install, restart, and work as usual.

```
/regress status                    capture on/off, case count, last run
/regress list 10                   newest cases
/regress show a1b2c3d4             turn summary; --full adds arguments and outputs
/regress expect a1b2c3d4 "tests pass"    add a must-appear assertion
/regress run --model gpt-5         replay every case against another model
/regress run --case a1b2c3d4 --limit 5
/regress report                    markdown report for the last run
/regress export                    zip + hash-chained manifest
/regress prune --keep 200
```

Headless, for CI:

```bash
python3 rg_cli.py run --model anthropic/claude-sonnet-4 --fail-on-regressed
python3 rg_cli.py report
```

Exit codes: `0` clean, `1` error, `2` a case regressed.

## What a case is

A plain JSON file under `$HERMES_HOME/plugin-data/hermes-regress/cases/` (mode 0600). Readable, diffable, safe to commit once you have read it.

```json
{
  "case_id": "a1b2c3d4e5f60718",
  "created_at": "2026-11-11T08:05:16+0530",
  "model": "anthropic/claude-sonnet-4",
  "prompt": "deploy the staging build and confirm the health check",
  "answer": "Staging deployed. Health check returns 200 after 3 retries.",
  "tool_calls": [
    {"tool": "terminal", "args": {"command": "make deploy ENV=staging"},
     "result": "deployed build 481 to staging", "source": "post_tool_call"},
    {"tool": "terminal", "args": {"command": "curl -sf localhost:8080/healthz"},
     "result": "200", "source": "post_tool_call"}
  ],
  "expect": [],
  "redactions": 0
}
```

`tool_calls` is the fixture set for replay. Each entry is the tool name, its exact arguments, and what it returned during capture.

## Safety model

Replay runs the real agent loop but answers every tool call from the capture. Three rules enforce that:

1. **Every tool call is vetoed at `pre_tool_call`.** Nothing reaches your shell, filesystem, network or APIs while a case is armed.
2. **The veto only applies to the replay's own turn.** The replay runs inside your gateway, so an unscoped veto would freeze every other chat, messaging session and cron job for minutes. When arming, the runner tags the child process with a random token; the hook answers `{}` for anything that does not carry it. Your other sessions keep working normally.
3. **The hook fails closed.** If fixture lookup raises while a case is armed, `serve()` still returns a block, so a bug in this plugin cannot turn a replay into a live run.
4. **A replay never becomes a case.** Capture is skipped whenever a case is armed, so `/regress run` cannot feed its own output back into the suite.
2. **Only an arg-identical recorded call is served.** A different command with the same tool name is not answered from a fixture, it is blocked and recorded as a gap: replaying the output of a different command would answer the wrong question.
5. **Armed replay state expires.** `plugin-data/hermes-regress/replay/armed.json` carries `expires_at` (15 min default), and the hook re-reads it on every call. A crashed or abandoned run disarms itself.
6. **A runtime that skips the hook is caught, not trusted.** Every armed call appends a line to `replay/hook.log`. If a case has tool calls and the trace shows no hook activity, that case is marked `error` and the suite stops instead of replaying the rest with live tools. This is how you find out that the process serving your turns does not load plugins.

Before any run, `preflight()` replays a probe case. If the runtime does not intercept it, the run aborts with an explanation instead of executing real tools. `/regress selftest` runs the same probe on demand.

Everything written to disk passes through a redaction layer first: the prompt, the answer, every tool result and every set of tool arguments (a `curl -H "Authorization: Bearer ..."` never lands raw). Provider keys (`sk-`, `pk-`, `rk-`), GitHub tokens, Slack tokens, AWS access keys, JWTs, inline `password=` / `token=` / `api_key=` values and long hex strings become `[REDACTED]`. Case files are written 0600, and the redaction count lands in the case as `redactions`.

## Verdicts

| Verdict | Meaning |
| --- | --- |
| `pass` | Answer matches the recording and the tool plan matched. |
| `drift` | Answer or plan moved but stayed above the similarity threshold. Worth reading. |
| `regressed` | An expectation missing, a tool step dropped, or similarity under threshold. |
| `error` | The replay itself failed: timeout, provider error, runtime not intercepting. |

Thresholds live in `plugin-data/hermes-regress/config.json` (`threshold`, `plan_threshold`). Expectations always win over similarity: add one with `/regress expect` for the part that must never change.

## Evidence bundle

`/regress export` writes a zip with `results.json`, `report.md`, `junit.xml` and a `manifest.json` whose hash chain covers every case. Change one result by hand and the chain no longer verifies. JUnit output drops straight into any CI renderer.

## Verified

- `python3 -m unittest discover -s tests` runs 23 tests covering redaction, the case store, hash chain, arming and expiry, fixture matching, unrecorded-tool blocking, verdicts, expectations, markdown, JUnit and the export bundle.
- Capture is exercised against real turns: 28 cases recorded from live sessions while this plugin was developed, including turns that ran shell commands.
- Hook serving is exercised through the real dispatch path (`get_pre_tool_call_block_message` returns a block with the recorded output and writes the trace line).

## Limits, stated plainly

- Replay sends the recorded prompt, not the full original conversation, so a case measures decision-making on that task, not session-history effects.
- Answers are compared by normalised similarity plus expectations. There is no semantic judge, so unusually phrased but correct answers can read as `drift`.
- Fixtures match on exact arguments. A case whose tool calls embed timestamps or run ids will show as a plan gap, which is usually the honest signal that the task is non-deterministic.
- Replay proves the agent still chooses the same steps with the same data. It does not prove the live tool still works.

## Layout

```
plugin.yaml      manifest: hooks, command, metadata
__init__.py      register(ctx): hooks + /regress
rg_capture.py    post_tool_call / post_llm_call / on_session_end capture and redaction
rg_replay.py     signature capture, fixture serving, preflight, scoring, verdicts
rg_store.py      case and run storage, config, hash chain
rg_report.py     markdown, JUnit XML, export bundle
rg_cli.py        /regress command surface and headless entry point
tests/           unittest suite
```

## License

MIT
