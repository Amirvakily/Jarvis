"""Workspace file skills: list, read and write files confined to the workspace."""

from __future__ import annotations

import csv
import io
import json
from html.parser import HTMLParser
from pathlib import Path

from jarvis.skills.base import SkillContext, SkillError, skill

MAX_READ_BYTES = 20 * 1024 * 1024
MAX_LIST_ENTRIES = 200
TABLE_PREVIEW_ROWS = 20


@skill(tags=["files"])
def list_files(
    path: str = ".",
    pattern: str = "*",
    recursive: bool = False,
    *,
    ctx: SkillContext,
) -> dict:
    """List files and folders in a workspace folder.

    Use this to see what is in a folder before reading or writing files. Names that
    start with a dot are skipped.

    Args:
        path: Folder relative to the workspace. Defaults to the workspace root.
        pattern: Glob pattern matched against names, such as "*.csv".
        recursive: Also search subfolders when true.
    """
    root = ctx.resolve_path(path, must_exist=True)
    if not root.is_dir():
        raise SkillError(f"'{path}' is not a folder.")
    if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
        raise SkillError(f"Pattern '{pattern}' must be a relative name pattern such as '*.txt'.")
    workspace = ctx.workspace.resolve()
    try:
        candidates = sorted(root.rglob(pattern) if recursive else root.glob(pattern))
    except (ValueError, NotImplementedError) as exc:
        raise SkillError(f"Pattern '{pattern}' is not valid: {exc}.") from exc

    entries: list[dict] = []
    for item in candidates:
        if any(part.startswith(".") for part in item.relative_to(root).parts):
            continue
        if not item.exists() or not item.resolve().is_relative_to(workspace):
            continue
        is_dir = item.is_dir()
        entries.append(
            {
                "path": _rel(item, workspace),
                "type": "dir" if is_dir else "file",
                "size": None if is_dir else item.stat().st_size,
            }
        )
    return {
        "directory": _rel(root, workspace),
        "entries": entries[:MAX_LIST_ENTRIES],
        "truncated": len(entries) > MAX_LIST_ENTRIES,
    }


@skill(tags=["files"])
def read_file(path: str, max_chars: int = 8000, *, ctx: SkillContext) -> dict:
    """Read a workspace file as text, a table, or extracted document text.

    Use this to open a document or data file. JSON is pretty-printed, CSV and TSV are
    returned as a table preview, PDF and Word files have their text extracted, and HTML
    is reduced to its visible text. Files over 20 MB are refused.

    Args:
        path: File path relative to the workspace.
        max_chars: Maximum characters of text to return.
    """
    target = ctx.resolve_path(path, must_exist=True)
    if not target.is_file():
        raise SkillError(f"'{path}' is not a file.")
    size = target.stat().st_size
    if size > MAX_READ_BYTES:
        raise SkillError(f"'{path}' is {size / 1048576:.1f} MB; files over 20 MB cannot be read.")
    if max_chars < 1:
        raise SkillError("max_chars must be at least 1.")

    suffix = target.suffix.lower()
    if suffix == ".pdf":
        body = _read_pdf(target, max_chars)
    elif suffix == ".docx":
        body = _read_docx(target, max_chars)
    else:
        text = _decode(target, path)
        if suffix == ".json":
            body = _read_json(text, path, max_chars)
        elif suffix in (".csv", ".tsv"):
            body = _read_table(text, path, "," if suffix == ".csv" else "\t")
        elif suffix in (".html", ".htm"):
            body = _read_html(text, max_chars)
        else:
            content, cut = _truncate(text, max_chars)
            body = {"type": "text", "content": content, "truncated": cut}
    return {"path": _rel(target, ctx.workspace.resolve()), "size_bytes": size, **body}


@skill(tags=["files"])
def write_file(path: str, content: str, append: bool = False, *, ctx: SkillContext) -> dict:
    """Create or overwrite a UTF-8 text file in the workspace.

    Use this to save notes, data or generated text. Missing parent folders are created.
    Set append to add to the end of an existing file instead of replacing it.

    Args:
        path: File path relative to the workspace.
        content: Text to write.
        append: Add the text to the end of the file instead of replacing it.
    """
    target = ctx.resolve_path(path)
    if target.is_dir():
        raise SkillError(f"'{path}' is a folder, not a file.")
    try:
        data = content.encode("utf-8")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("ab" if append else "wb") as handle:
            handle.write(data)
    except (OSError, UnicodeEncodeError) as exc:
        raise SkillError(f"Could not write '{path}': {exc}.") from exc
    return {"path": _rel(target, ctx.workspace.resolve()), "bytes_written": len(data)}


# -- helpers ------------------------------------------------------------------


def _rel(path: Path, workspace: Path) -> str:
    return path.relative_to(workspace).as_posix()


def _truncate(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[:limit], True


def _binary_error(name: str) -> SkillError:
    return SkillError(f"'{name}' looks like a binary file, so it cannot be read as text.")


def _decode(target: Path, name: str) -> str:
    raw = target.read_bytes()
    if b"\x00" in raw:
        raise _binary_error(name)
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise _binary_error(name) from exc


def _read_json(text: str, name: str, max_chars: int) -> dict:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SkillError(f"'{name}' is not valid JSON: {exc.msg} at line {exc.lineno}.") from exc
    content, cut = _truncate(json.dumps(data, indent=2, ensure_ascii=False), max_chars)
    return {"type": "json", "content": content, "truncated": cut}


def _read_table(text: str, name: str, delimiter: str) -> dict:
    try:
        rows = list(csv.reader(io.StringIO(text, newline=""), delimiter=delimiter))
    except csv.Error as exc:
        raise SkillError(f"'{name}' could not be parsed as a table: {exc}.") from exc
    columns = rows[0] if rows else []
    body = rows[1:]
    return {
        "type": "table",
        "columns": columns,
        "row_count": len(body),
        "rows": body[:TABLE_PREVIEW_ROWS],
        "truncated": len(body) > TABLE_PREVIEW_ROWS,
    }


class _VisibleText(HTMLParser):
    """Collects text nodes outside script and style elements."""

    _HIDDEN = frozenset({"script", "style"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._hidden_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in self._HIDDEN:
            self._hidden_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self._HIDDEN and self._hidden_depth:
            self._hidden_depth -= 1

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if text and not self._hidden_depth:
            self.parts.append(text)


def _read_html(text: str, max_chars: int) -> dict:
    parser = _VisibleText()
    parser.feed(text)
    parser.close()
    content, cut = _truncate("\n".join(parser.parts), max_chars)
    return {"type": "html", "content": content, "truncated": cut}


def _missing_library(kind: str, library: str) -> SkillError:
    return SkillError(
        f"Reading {kind} files needs {library}. Install with: pip install 'jarvis-assistant[files]'"
    )


def _read_pdf(target: Path, max_chars: int) -> dict:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise _missing_library("PDF", "pypdf") from exc
    try:
        reader = PdfReader(str(target))
        total_pages = len(reader.pages)
        chunks: list[str] = []
        used = 0
        for page in reader.pages:
            chunk = page.extract_text() or ""
            chunks.append(chunk)
            used += len(chunk)
            if used > max_chars:
                break
    except Exception as exc:
        raise SkillError(f"Could not read PDF '{target.name}': {exc}.") from exc
    content, cut = _truncate("\n\n".join(chunks), max_chars)
    return {
        "type": "pdf",
        "pages": total_pages,
        "text": content,
        "truncated": cut or len(chunks) < total_pages,
    }


def _read_docx(target: Path, max_chars: int) -> dict:
    try:
        from docx import Document
    except ImportError as exc:
        raise _missing_library("Word", "python-docx") from exc
    try:
        paragraphs = [p.text for p in Document(str(target)).paragraphs if p.text.strip()]
    except Exception as exc:
        raise SkillError(f"Could not read Word file '{target.name}': {exc}.") from exc
    content, cut = _truncate("\n".join(paragraphs), max_chars)
    return {
        "type": "docx",
        "paragraph_count": len(paragraphs),
        "text": content,
        "truncated": cut,
    }
