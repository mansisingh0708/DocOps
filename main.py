"""
main.py — Entry point for DocOps: the 3-agent file organizer.

Usage
-----
    # Dry run only — prints the proposed plan, changes nothing:
    python main.py ./test_folder

    # Live — prints plan, asks y/N, then moves real files:
    python main.py ./test_folder --live

    # Live + review loop (re-runs crew if ⚠️ flags remain):
    python main.py ./test_folder --live --review

    # Skip the confirmation prompt (useful in scripts):
    python main.py ./test_folder --live --yes
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()   # reads .env for OPENAI_API_KEY etc.


def _run(folder: str, *, live: bool, review: bool) -> str:
    """Build the graph and invoke the crew once.

    DRY_RUN and ENABLE_REVIEW_LOOP are set here (before importing graph)
    so build_crew() and build_app() always read the correct mode.
    """
    os.environ["DRY_RUN"]            = "false" if live   else "true"
    os.environ["ENABLE_REVIEW_LOOP"] = "true"  if review else "false"

    from graph import build_app   # imported AFTER env vars are set
    app = build_app()
    result = app.invoke({
        "folder":    folder,
        "result":    "",
        "iteration": 0,
    })
    return result["result"]


def main():
    parser = argparse.ArgumentParser(
        description="DocOps — 3-agent AI file organizer (CrewAI + LangGraph + MCP)"
    )
    parser.add_argument(
        "folder",
        help="Local folder to organise. Relative or absolute path.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Actually move / rename files (shows plan first, then asks y/N).",
    )
    parser.add_argument(
        "--review",
        action="store_true",
        help="Re-run the crew if ⚠️ ambiguous files remain after the first pass.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the y/N confirmation prompt. Use with care.",
    )
    args = parser.parse_args()

    # ── Validate folder ────────────────────────────────────────────────────
    folder = Path(args.folder).expanduser().resolve()
    if not folder.is_dir():
        sys.exit(f"❌  Not a valid directory: {folder}")

    mode_label = (
        "🔴  LIVE — files WILL be moved/renamed (with confirmation)"
        if args.live else
        "🟢  DRY RUN — no files will change"
    )

    print(f"\n{'='*62}")
    print( "  📂  DocOps — 3-Agent File Organizer")
    print(f"  Target : {folder}")
    print(f"  Mode   : {mode_label}")
    print(f"{'='*62}\n")

    # ── Step 1: Always run a dry-run preview first ─────────────────────────
    print("🔍  Running dry-run preview (nothing will change yet)…\n")
    plan = _run(str(folder), live=False, review=args.review)

    print(f"\n{'─'*62}")
    print( "  📋  PROPOSED PLAN  (dry run — nothing changed)")
    print(f"{'─'*62}\n")
    print(plan)

    # ── Step 2: Dry-run only? We're done ──────────────────────────────────
    if not args.live:
        print(f"\n{'='*62}")
        print( "  🟢  Dry run complete. No files were changed.")
        print( "      Re-run with --live to apply this plan.")
        print(f"{'='*62}\n")
        return

    # ── Step 3: Confirm before touching real files ─────────────────────────
    if not args.yes:
        print(f"\n{'!'*62}")
        print(f"  ⚠️  About to MOVE / RENAME real files in:")
        print(f"      {folder}")
        print(f"{'!'*62}")
        answer = input("\n  Proceed with these changes? Type 'y' to continue [y/N]: ").strip().lower()
        if answer != "y":
            print("\n  ❌  Aborted. No files were changed.\n")
            return

    # ── Step 4: Execute for real ───────────────────────────────────────────
    print(f"\n{'='*62}")
    print( "  🔴  Applying changes… (open Explorer to watch files move!)")
    print(f"{'='*62}\n")

    final = _run(str(folder), live=True, review=args.review)

    print(f"\n{'='*62}")
    print( "  ✅  Done! Files organized.")
    print(f"{'='*62}\n")
    print(final)


if __name__ == "__main__":
    main()
