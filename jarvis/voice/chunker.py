"""Turns a token stream into speakable sentences.

Speech starts as soon as the first sentence is complete instead of waiting
for the whole reply, which is most of what makes a voice assistant feel
responsive.
"""

from __future__ import annotations

import re

# A sentence ends at . ! ? … (plus closing quotes/brackets) followed by
# whitespace, or at a line break. Requiring the whitespace keeps "3.14" and
# "example.com" intact while the stream is still arriving.
_BOUNDARY = re.compile(r"([.!?…]+[\"')\]]*)(\s+)|(\n+)")
_ABBREVIATIONS = (
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.", "vs.", "etc.", "e.g.", "i.e.",
    "inc.", "ltd.", "no.", "approx.", "u.s.", "a.m.", "p.m.",
)  # fmt: skip

MIN_CHARS = 12  # avoid speaking tiny fragments such as "Sure."
FIRST_MIN_CHARS = 4  # ...except at the start, where a fast first word matters


def clean_for_speech(text: str) -> str:
    """Strip markup a speech engine would read aloud ("asterisk asterisk")."""
    text = re.sub(r"```.*?```", " (code omitted) ", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"https?://\S+", "a link", text)
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*+•]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*>\s?", "", text, flags=re.MULTILINE)
    text = re.sub(r"(\*\*|__|\*|~~)(?=\S)(.+?)(?<=\S)\1", r"\2", text)
    text = re.sub(r"[*_#|~]+", " ", text)  # snake_case reads better as words
    return re.sub(r"\s+", " ", text).strip()


class SentenceChunker:
    def __init__(self) -> None:
        self._buffer = ""
        self._emitted = 0

    def feed(self, text: str) -> list[str]:
        """Add streamed text; return any sentences that are now complete."""
        self._buffer += text
        sentences: list[str] = []
        position = 0
        for match in _BOUNDARY.finditer(self._buffer):
            end = match.end(1) if match.group(1) else match.start(3)
            candidate = self._buffer[position:end]
            words = candidate.split()
            if match.group(1) and words and words[-1].lower().lstrip("(\"'") in _ABBREVIATIONS:
                continue
            minimum = FIRST_MIN_CHARS if self._emitted == 0 and not sentences else MIN_CHARS
            if len(candidate.strip()) < minimum:
                continue
            # Never cut inside an unfinished code block.
            if candidate.count("```") % 2 or self._buffer[:position].count("```") % 2:
                continue
            spoken = clean_for_speech(candidate)
            if spoken:
                sentences.append(spoken)
            position = match.end()
        self._buffer = self._buffer[position:]
        self._emitted += len(sentences)
        return sentences

    def flush(self) -> list[str]:
        """Return whatever is left when the reply (or a pause) ends."""
        spoken = clean_for_speech(self._buffer)
        self._buffer = ""
        if not spoken:
            return []
        self._emitted += 1
        return [spoken]
