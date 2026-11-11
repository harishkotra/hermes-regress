"""hermes-regress — regression tests for your agent, built from your real sessions.

Hooks in, one command. Capture is passive: every finished turn becomes a case
with its prompt, tool calls and tool outputs. Replay re-runs a case through the
real agent loop with each tool call answered from the capture, so nothing on
your machine is touched.

    /regress status
    /regress run --model anthropic/claude-sonnet-4
    /regress report
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rg_capture  # noqa: E402
import rg_cli  # noqa: E402
import rg_replay  # noqa: E402


def register(ctx) -> None:
    from rg_replay import _debug

    _debug("register() called — plugin loaded")
    ctx.register_hook("pre_tool_call", rg_replay.serve)
    ctx.register_hook("post_tool_call", rg_capture.on_post_tool_call)
    ctx.register_hook("post_llm_call", rg_capture.on_post_llm_call)
    ctx.register_hook("on_session_end", rg_capture.on_session_end)
    ctx.register_command(
        "regress",
        handler=rg_cli.handle,
        description="Capture real turns as cases, replay them against another model without side effects.",
    )
