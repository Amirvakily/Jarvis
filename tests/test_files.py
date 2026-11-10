"""Tests for the workspace file skills."""

from __future__ import annotations

import json
import sys

import pytest

from jarvis.skills.base import SkillError
from jarvis.skills.builtin.files import list_files, read_file, write_file


def _put(ctx, rel: str, data: str | bytes) -> None:
    target = ctx.workspace / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, bytes):
        target.write_bytes(data)
    else:
        target.write_text(data, encoding="utf-8")


def test_write_then_read_round_trip(ctx):
    written = write_file("notes/todo.txt", "buy milk", ctx=ctx)
    assert written == {"path": "notes/todo.txt", "bytes_written": 8}

    read = read_file("notes/todo.txt", ctx=ctx)
    assert read == {
        "path": "notes/todo.txt",
        "size_bytes": 8,
        "type": "text",
        "content": "buy milk",
        "truncated": False,
    }


def test_write_append_adds_to_end(ctx):
    write_file("log.txt", "one", ctx=ctx)
    result = write_file("log.txt", "\ntwo", append=True, ctx=ctx)
    assert result["bytes_written"] == 4
    assert read_file("log.txt", ctx=ctx)["content"] == "one\ntwo"


def test_list_files_sorted_hidden_skipped(ctx):
    _put(ctx, "b.txt", "b")
    _put(ctx, "a.txt", "aa")
    _put(ctx, ".secret", "hidden")
    _put(ctx, "sub/c.csv", "x")

    listing = list_files(ctx=ctx)
    assert listing["directory"] == "."
    assert listing["truncated"] is False
    assert [e["path"] for e in listing["entries"]] == ["a.txt", "b.txt", "sub"]
    assert listing["entries"][0] == {"path": "a.txt", "type": "file", "size": 2}
    assert listing["entries"][2]["type"] == "dir"


def test_list_files_recursive_pattern(ctx):
    _put(ctx, "a.txt", "a")
    _put(ctx, "sub/deeper/c.csv", "x")
    _put(ctx, "sub/d.txt", "x")

    listing = list_files(pattern="*.csv", recursive=True, ctx=ctx)
    assert [e["path"] for e in listing["entries"]] == ["sub/deeper/c.csv"]


def test_list_files_subfolder_path(ctx):
    _put(ctx, "sub/c.csv", "x")
    listing = list_files("sub", ctx=ctx)
    assert listing["directory"] == "sub"
    assert [e["path"] for e in listing["entries"]] == ["sub/c.csv"]


def test_list_files_truncates_at_200(ctx):
    for index in range(205):
        _put(ctx, f"many/f{index:03}.txt", "")
    listing = list_files("many", ctx=ctx)
    assert len(listing["entries"]) == 200
    assert listing["truncated"] is True


def test_list_files_rejects_non_folder(ctx):
    _put(ctx, "a.txt", "a")
    with pytest.raises(SkillError, match="not a folder"):
        list_files("a.txt", ctx=ctx)


def test_read_json_is_pretty_printed(ctx):
    data = {"b": 1, "a": [1, 2]}
    _put(ctx, "data.json", json.dumps(data))
    result = read_file("data.json", ctx=ctx)
    assert result["type"] == "json"
    assert result["content"] == json.dumps(data, indent=2)
    assert result["truncated"] is False


def test_read_json_truncates(ctx):
    _put(ctx, "data.json", json.dumps({"key": "x" * 500}))
    result = read_file("data.json", max_chars=20, ctx=ctx)
    assert len(result["content"]) == 20
    assert result["truncated"] is True


def test_read_invalid_json_is_plain_error(ctx):
    _put(ctx, "bad.json", "{not json")
    with pytest.raises(SkillError, match="not valid JSON"):
        read_file("bad.json", ctx=ctx)


def test_read_csv_table(ctx):
    _put(ctx, "scores.csv", "name,score\nann,3\nbob,5\n")
    result = read_file("scores.csv", ctx=ctx)
    assert result["type"] == "table"
    assert result["columns"] == ["name", "score"]
    assert result["row_count"] == 2
    assert result["rows"] == [["ann", "3"], ["bob", "5"]]
    assert result["truncated"] is False


def test_read_tsv_table(ctx):
    _put(ctx, "pairs.tsv", "a\tb\n1\t2\n")
    result = read_file("pairs.tsv", ctx=ctx)
    assert result["columns"] == ["a", "b"]
    assert result["rows"] == [["1", "2"]]


def test_read_csv_preview_is_capped(ctx):
    lines = ["n"] + [str(i) for i in range(25)]
    _put(ctx, "long.csv", "\n".join(lines) + "\n")
    result = read_file("long.csv", ctx=ctx)
    assert result["row_count"] == 25
    assert len(result["rows"]) == 20
    assert result["truncated"] is True


def test_read_html_returns_visible_text_only(ctx):
    page = (
        "<html><head><style>p{color:red}</style><script>var hidden=1;</script></head>"
        "<body><h1>Hello</h1><p>World &amp; more</p></body></html>"
    )
    _put(ctx, "page.html", page)
    result = read_file("page.html", ctx=ctx)
    assert result["type"] == "html"
    assert "Hello" in result["content"]
    assert "World & more" in result["content"]
    assert "hidden" not in result["content"]
    assert "color" not in result["content"]


def test_read_text_truncates(ctx):
    _put(ctx, "long.txt", "x" * 100)
    result = read_file("long.txt", max_chars=10, ctx=ctx)
    assert result["content"] == "x" * 10
    assert result["truncated"] is True
    assert result["size_bytes"] == 100


def test_read_binary_is_refused(ctx):
    _put(ctx, "image.bin", b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")
    with pytest.raises(SkillError, match="binary"):
        read_file("image.bin", ctx=ctx)


def test_read_undecodable_text_is_refused_as_binary(ctx):
    _put(ctx, "odd.txt", b"\xff\xfe\xfa\xf9")
    with pytest.raises(SkillError, match="binary"):
        read_file("odd.txt", ctx=ctx)


def test_read_refuses_files_over_20mb(ctx):
    target = ctx.workspace / "big.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        handle.truncate(21 * 1024 * 1024)
    with pytest.raises(SkillError, match="20 MB"):
        read_file("big.txt", ctx=ctx)


def test_read_missing_file(ctx):
    with pytest.raises(SkillError, match="does not exist"):
        read_file("nope.txt", ctx=ctx)


def test_pdf_without_pypdf_gives_install_hint(ctx, monkeypatch):
    monkeypatch.setitem(sys.modules, "pypdf", None)
    _put(ctx, "doc.pdf", b"%PDF-1.4")
    with pytest.raises(SkillError, match="pip install 'jarvis-assistant\\[files\\]'"):
        read_file("doc.pdf", ctx=ctx)


def test_docx_without_python_docx_gives_install_hint(ctx, monkeypatch):
    monkeypatch.setitem(sys.modules, "docx", None)
    _put(ctx, "doc.docx", b"PK\x03\x04")
    with pytest.raises(SkillError, match="pip install 'jarvis-assistant\\[files\\]'"):
        read_file("doc.docx", ctx=ctx)


def test_paths_outside_workspace_are_refused(ctx):
    outside = ctx.workspace.parent / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    absolute = str(outside)

    with pytest.raises(SkillError, match="outside the workspace"):
        list_files("../outside.txt", ctx=ctx)
    with pytest.raises(SkillError, match="outside the workspace"):
        list_files(absolute, ctx=ctx)
    with pytest.raises(SkillError, match="relative name pattern"):
        list_files(pattern="../*", ctx=ctx)

    with pytest.raises(SkillError, match="outside the workspace"):
        read_file("../outside.txt", ctx=ctx)
    with pytest.raises(SkillError, match="outside the workspace"):
        read_file(absolute, ctx=ctx)

    with pytest.raises(SkillError, match="outside the workspace"):
        write_file("../outside.txt", "overwritten", ctx=ctx)
    with pytest.raises(SkillError, match="outside the workspace"):
        write_file(absolute, "overwritten", ctx=ctx)

    assert outside.read_text(encoding="utf-8") == "secret"
