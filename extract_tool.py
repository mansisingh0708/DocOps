"""
extract_tool.py — Custom CrewAI tool that extracts readable text from any file.

Supports
--------
• Plain text / code (txt, md, csv, json, xml, html, java, cpp, py, js, …)
• PDF     — via pypdf
• DOCX    — via python-docx
• XLSX    — via openpyxl
• Images / binaries — returns a category hint; no crash, no content read

The deterministic EXT_CATEGORY map is the safety-net: even if the LLM
Curator can't read a file's content, it always gets a suggested destination
folder so *every* file ends up somewhere.
"""

import os
from pathlib import Path

try:
    from crewai.tools import tool          # CrewAI ≥ 0.80 / 1.x
except ImportError:
    from crewai_tools import tool          # older fallback


# ── Target folder for path resolution ─────────────────────────────────────────
# MCP tools resolve relative paths against the allowed directory, but our
# custom extract_text uses Python's open() which resolves against the process
# CWD.  This variable lets us match the MCP behaviour.
_TARGET_FOLDER: str | None = None


def set_target_folder(folder: str) -> None:
    """Set the base folder that relative paths in extract_text resolve against."""
    global _TARGET_FOLDER
    _TARGET_FOLDER = str(Path(folder).resolve())


def _resolve(path: str) -> str:
    """Resolve *path* to an absolute path.

    If *path* is already absolute, return it as-is.
    Otherwise, resolve it relative to _TARGET_FOLDER (the same directory
    the MCP server is scoped to) instead of the Python CWD.
    """
    p = Path(path)
    if p.is_absolute():
        return str(p)
    if _TARGET_FOLDER:
        return str(Path(_TARGET_FOLDER) / p)
    return str(p.resolve())       # fallback: behave like before


# ── Extension sets ────────────────────────────────────────────────────────────
TEXT_EXT = {
    "txt", "md", "csv", "json", "xml", "html", "htm",
    "java", "cpp", "c", "h", "hpp",
    "py", "js", "ts", "tsx", "css",
    "yaml", "yml", "ini", "log", "sh", "bat", "ps1",
    "sql", "r", "go", "rs", "kt", "rb", "php", "swift",
}

# Deterministic fallback: extension → category folder name
EXT_CATEGORY: dict[str, str] = {
    # documents
    "pdf": "documents", "docx": "documents", "doc": "documents",
    "txt": "documents", "md": "documents", "rtf": "documents", "odt": "documents",
    # spreadsheets
    "xlsx": "spreadsheets", "xls": "spreadsheets", "csv": "spreadsheets",
    "ods": "spreadsheets",
    # images
    "png": "images", "jpg": "images", "jpeg": "images",
    "gif": "images", "svg": "images", "webp": "images",
    "bmp": "images", "tiff": "images", "ico": "images",
    # code
    "py": "code", "js": "code", "ts": "code", "tsx": "code",
    "java": "code", "cpp": "code", "c": "code", "h": "code", "hpp": "code",
    "html": "code", "htm": "code", "css": "code", "xml": "code",
    "go": "code", "rs": "code", "kt": "code", "rb": "code",
    "sh": "code", "bat": "code", "ps1": "code", "sql": "code",
    # data / config
    "json": "data", "yaml": "data", "yml": "data", "ini": "data", "toml": "data",
    # archives
    "zip": "archives", "tar": "archives", "gz": "archives",
    "rar": "archives", "7z": "archives",
    # presentations
    "pptx": "presentations", "ppt": "presentations", "odp": "presentations",
}


def _ext(path: str) -> str:
    """Return the lowercase extension of *path* (no leading dot)."""
    return path.lower().rsplit(".", 1)[-1] if "." in path else ""


def category_for(path: str) -> str:
    """Deterministic extension → folder name (used as fallback by the LLM)."""
    return EXT_CATEGORY.get(_ext(path), "misc")


# ── Tool definition ───────────────────────────────────────────────────────────
@tool("extract_text")
def extract_text(path: str) -> str:
    """Extract readable text from a file (plain-text, code, PDF, DOCX, XLSX).

    For file types whose content cannot be read (images, archives, legacy
    binaries) a short category hint is returned instead — the LLM Curator
    can use this hint to place the file without crashing.

    Args:
        path: Absolute or relative path to the file to read.

    Returns:
        Up to 4 000 characters of the file's text content, or a category hint
        for unreadable binary files.
    """
    resolved = _resolve(path)
    ext = _ext(resolved)
    try:
        # ── plain text / source code ──────────────────────────────────────
        if ext in TEXT_EXT:
            with open(resolved, "r", encoding="utf-8", errors="ignore") as fh:
                return fh.read()[:4_000]

        # ── PDF ───────────────────────────────────────────────────────────
        if ext == "pdf":
            from pypdf import PdfReader  # type: ignore
            reader = PdfReader(resolved)
            pages = reader.pages[:5]
            return "\n".join(page.extract_text() or "" for page in pages)[:4_000]

        # ── DOCX ─────────────────────────────────────────────────────────
        if ext == "docx":
            import docx  # type: ignore
            doc = docx.Document(resolved)
            return "\n".join(p.text for p in doc.paragraphs)[:4_000]

        # ── XLSX / XLS ───────────────────────────────────────────────────
        if ext in ("xlsx", "xls"):
            import openpyxl  # type: ignore
            wb = openpyxl.load_workbook(resolved, read_only=True, data_only=True)
            ws = wb.active
            rows = []
            for row in ws.iter_rows(max_row=20, values_only=True):  # type: ignore[union-attr]
                rows.append(" | ".join(str(cell) for cell in row if cell is not None))
            return "\n".join(rows)[:4_000]

    except Exception as exc:  # noqa: BLE001
        # Readable type but something went wrong (corrupt file, locked, …)
        return (
            f"[unreadable .{ext}: {exc}] "
            f"-> suggested folder: {category_for(path)}/"
        )

    # ── images, archives, unknown binaries — return a category hint ───────
    return (
        f"[binary .{ext}, content cannot be read as text] "
        f"-> suggested folder: {category_for(path)}/"
    )
