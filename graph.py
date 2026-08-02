"""
graph.py — LangGraph wrapper (scan → curate → write, with optional review loop).

  • A single `organize` node runs the full CrewAI crew.
  • An optional `review` node (enabled via --review / ENABLE_REVIEW_LOOP) checks
    for ambiguous files flagged with ⚠️ — if any remain, the graph loops back
    to `organize` for a second pass (capped at 2 iterations).
  • DRY_RUN and ENABLE_REVIEW_LOOP are read at call-time (not import time) so
    that main.py can set them via os.environ before importing this module.
"""

import os
from pathlib import Path
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from mcp_setup import get_fs_tools
from agents import build_crew
from extract_tool import set_target_folder


def _dry_run() -> bool:
    return os.getenv("DRY_RUN", "true").lower() == "true"


def _enable_review() -> bool:
    return os.getenv("ENABLE_REVIEW_LOOP", "false").lower() == "true"


# ── State schema ──────────────────────────────────────────────────────────────
class State(TypedDict):
    folder:    str   # absolute path to the target folder
    result:    str   # crew output (raw string)
    iteration: int   # loop counter — capped at 2


# ── Nodes ─────────────────────────────────────────────────────────────────────
def organize(state: State) -> dict:
    """Run the 3-agent CrewAI crew inside the MCP filesystem context."""
    folder    = state["folder"]
    iteration = state.get("iteration", 0) + 1

    # Resolve absolute path; create folder in live mode before crew starts
    abs_folder = str(Path(folder).resolve())
    if not _dry_run():
        Path(abs_folder).mkdir(parents=True, exist_ok=True)

    set_target_folder(abs_folder)          # so extract_text resolves relative paths
    with get_fs_tools(abs_folder) as fs_tools:
        crew = build_crew(fs_tools)
        out  = crew.kickoff(inputs={"folder": abs_folder})

    return {
        "folder":    abs_folder,
        "result":    str(out),
        "iteration": iteration,
    }


def review(state: State) -> dict:
    """Pass-through node; routing logic lives in the conditional edge."""
    return state


# ── Routing ───────────────────────────────────────────────────────────────────
def should_retry(state: State) -> str:
    """Loop back to organize if ⚠️ flags remain and we haven't hit 2 passes."""
    if state.get("iteration", 1) >= 2:
        return END
    if "⚠️" in state.get("result", ""):
        return "organize"
    return END


# ── Graph builder ─────────────────────────────────────────────────────────────
def build_app():
    """Compile and return the LangGraph application.

    The review loop is wired in only when ENABLE_REVIEW_LOOP=true, keeping the
    graph topology dead-simple for the default (single-pass) case.
    """
    g = StateGraph(State)
    g.add_node("organize", organize)

    if _enable_review():
        g.add_node("review", review)
        g.add_edge(START, "organize")
        g.add_edge("organize", "review")
        g.add_conditional_edges("review", should_retry)
    else:
        g.add_edge(START, "organize")
        g.add_edge("organize", END)

    return g.compile()
