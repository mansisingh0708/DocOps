"""
agents.py — 3 CrewAI agents + tasks + crew.

Agents
------
🔍 Scanner  — walks the target folder recursively; calls extract_text on
              every file (text/code/PDF/DOCX/XLSX). For images/binaries,
              records the category hint so nothing is skipped.
🗂️ Curator  — groups files by theme, proposes folder structure + new names,
              assigns EVERY file (uses category hint for unreadable files),
              flags ambiguity with ⚠️.
✍️ Writer   — DRY RUN: prints the plan only.
              LIVE: creates folders, moves files (collision-safe), writes README.

DRY_RUN is read at crew-build time (not module-import time) so that main.py
can set os.environ["DRY_RUN"] before calling build_crew().
"""

import os
from crewai import Agent, Task, Crew, Process

from extract_tool import extract_text   # noqa: F401


def build_crew(fs_tools: list) -> Crew:
    """Construct and return a sequential Scanner → Curator → Writer crew.

    Args:
        fs_tools: MCP filesystem tools (list_directory, move_file, …)
                  yielded by get_fs_tools().
    """
    dry_run = os.getenv("DRY_RUN", "true").lower() == "true"

    # ── Agent 1: Scanner ───────────────────────
    scanner = Agent(
        role="File Scanner",
        goal=(
            "Discover and summarise EVERY file in the target folder "
            "(recurse into subfolders). For each file call extract_text(path) "
            "to obtain its content or a category hint. "
            "Never skip a file — images and binaries get their type recorded.\n"
            "IMPORTANT: Your summary must describe WHAT the file is about "
            "(its topic, purpose, domain) — not just its file type. "
            "e.g. 'Invoice for cloud hosting — $2,400 paid' is good; "
            "'A text file' is useless."
        ),
        backstory=(
            "You are a meticulous archivist who cares about MEANING over format. "
            "A .txt might be an invoice, a recipe, meeting notes, or a love letter — "
            "you read the content and report what it's actually about. "
            "Nothing escapes your catalogue."
        ),
        tools=[*fs_tools, extract_text],
        verbose=True,
    )

    # ── Agent 2: Curator ──────────────────────
    curator = Agent(
        role="Curator",
        goal=(
            "Analyse the Scanner's content summaries and propose a SEMANTIC "
            "folder structure (depth ≤ 2) based on WHAT each file is about, "
            "NOT its file extension.\n\n"
            "CRITICAL RULES:\n"
            "• Group by TOPIC / PURPOSE / DOMAIN — not by file type.\n"
            "  An invoice (.txt) goes into 'finance/', NOT 'documents/'.\n"
            "  A grocery list (.md) goes into 'personal/', NOT 'documents/'.\n"
            "  A meeting note (.txt) goes into 'meetings/', NOT 'documents/'.\n"
            "• NEVER create generic buckets like 'documents/', 'text_files/', "
            "  or 'misc/' — always pick a meaningful theme name.\n"
            "• For binary/image files where content can't be read, use the "
            "  'suggested folder' hint from the Scanner.\n"
            "• Every file MUST have a destination.\n"
            "• Never rename files whose name contains DONT, KEEP, or DO_NOT.\n"
            "• Flag genuinely ambiguous files with ⚠️."
        ),
        backstory=(
            "You are a knowledge organiser, not a file-type sorter. "
            "You read WHAT each file contains and group related topics together. "
            "An invoice and a receipt belong together in 'finance/' regardless "
            "of whether one is .pdf and the other is .txt. A Python tax calculator "
            "and a tax invoice might both belong under 'finance/'. "
            "You think like a librarian categorising by subject, not by binding."
        ),
        verbose=True,   # no tools needed; reasons over Scanner's output
    )

    # ── Agent 3: Writer (mode depends on DRY_RUN) ────────────────────────
    if dry_run:
        writer = Agent(
            role="Organizer (DRY RUN)",
            goal=(
                "Only PRINT the reorganisation plan. "
                "Do NOT move, rename, create, or write any files."
            ),
            backstory=(
                "You are in dry-run mode. You never touch the filesystem — "
                "you only show exactly what WOULD happen if the plan ran live."
            ),
            tools=[],       # ← physically cannot change anything
            verbose=True,
        )
    else:
        writer = Agent(
            role="Organizer",
            goal=(
                "Apply the Curator's plan precisely and safely:\n"
                "1. For each destination folder, call create_directory first "
                "   so move_file never fails on a missing parent.\n"
                "2. Call move_file(source, destination) for every mapping. "
                "   If destination already exists, append '_1', '_2', … before "
                "   the extension to avoid overwriting.\n"
                "3. Skip any move where source path == destination path.\n"
                "4. Write a README.md at the root of the target folder "
                "   describing each subfolder and its contents."
            ),
            backstory=(
                "You execute filesystem reorganisations safely. You always "
                "create folders before moving files into them, handle name "
                "collisions gracefully, and never overwrite existing content."
            ),
            tools=fs_tools,
            verbose=True,
        )

    # ── Tasks ─────────────────────────────────────────────────────────────
    scan_task = Task(
        description=(
            "TARGET FOLDER: {folder}\n\n"
            "Step 1: call list_directory(path='{folder}') to get the top-level "
            "contents.  Recurse into any subdirectories.\n"
            "Step 2: For EVERY file found, call "
            "extract_text(path=<absolute_path>). Do NOT use read_text_file.\n\n"
            "Output a Markdown list:\n"
            "  `<relative_path>` → <content-based summary>\n\n"
            "Your summary MUST describe the file's TOPIC and PURPOSE, not "
            "its format. Bad: 'a text file with data'. "
            "Good: 'Invoice #2024-0847 for CloudHost Pro annual hosting — $2,400 PAID'."
        ),
        expected_output=(
            "A Markdown list of ALL files with a content-based one-line summary each. "
            "Each summary describes WHAT the file is about (topic/domain). "
            "Binary/image files show their category hint. No file skipped."
        ),
        agent=scanner,
    )

    curate_task = Task(
        description=(
            "From the Scanner's content summaries, propose a SEMANTICALLY "
            "reorganised folder structure.\n\n"
            "Output a Markdown table:\n"
            "| Old Path | New Path | Theme / Folder |\n"
            "|----------|----------|----------------|\n\n"
            "CRITICAL — Group by MEANING, not by extension:\n"
            "• Read each file's summary and decide its DOMAIN/TOPIC.\n"
            "• Files about money (invoices, budgets, tax code) → 'finance/'\n"
            "• Files about projects (status updates, dev guides) → 'project/'\n"
            "• Files about meetings (agendas, notes) → 'meetings/'\n"
            "• Personal items (grocery lists, to-do) → 'personal/'\n"
            "• Data files (CSVs, datasets) → use a domain name like 'hr_data/' "
            "  if it's employee data, not just 'data/'\n"
            "• Code/scripts → 'scripts/' is fine since code IS the content\n\n"
            "NEVER use generic folders like 'documents/' or 'text_files/'.\n\n"
            "Other rules:\n"
            "• Every file must appear in the table — no file left unassigned.\n"
            "• For binary/image files, use the 'suggested folder' hint.\n"
            "• Folder depth ≤ 2.\n"
            "• Do not rename files whose name contains DONT, KEEP, or DO_NOT.\n"
            "• Mark ambiguous files with ⚠️ in the Theme column."
        ),
        expected_output=(
            "A Markdown table with old_path → new_path for EVERY file. "
            "Folders named by TOPIC (e.g. finance/, project/, meetings/) — "
            "never by file type. Ambiguous files flagged ⚠️. No file missing."
        ),
        agent=curator,
    )

    if dry_run:
        write_task = Task(
            description=(
                "DRY RUN — Do NOT move, rename, create, or write any files.\n\n"
                "Output:\n"
                "1. Proposed moves as `old_path → new_path` (one per line).\n"
                "2. The full README.md content you WOULD have written.\n"
                "End with: 'DRY RUN COMPLETE — no files were changed.'"
            ),
            expected_output=(
                "Printed plan of all proposed moves + proposed README.md. "
                "No files changed."
            ),
            agent=writer,
        )
    else:
        write_task = Task(
            description=(
                "Execute every row in the Curator's table:\n"
                "1. call create_directory for each unique destination folder.\n"
                "2. call move_file(old_path, new_path) for each mapping.\n"
                "   If new_path already exists, use new_path_stem + '_1' + ext.\n"
                "3. Skip any move where old_path == new_path (already in place).\n"
                "4. Finally, write_file a README.md at the target folder root "
                "   listing each subfolder with a short description."
            ),
            expected_output=(
                "A confirmation line per move, a list of skipped/failed moves "
                "with reasons, and the README.md content."
            ),
            agent=writer,
        )

    # ── Crew ──────────────────────────────────────────────────────────────
    return Crew(
        agents=[scanner, curator, writer],
        tasks=[scan_task, curate_task, write_task],
        process=Process.sequential,
        verbose=True,
    )
