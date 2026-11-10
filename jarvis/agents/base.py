"""Agent definitions.

An agent is a role: instructions, the skills it may use, and optionally its
own backend profile. Agents are data, so new ones can be added as TOML files
without touching the core:

    # agents/summarizer.toml
    name = "summarizer"
    description = "Condenses long documents and web pages."
    instructions = "You turn long material into tight, faithful summaries..."
    skills = ["read_file", "web_fetch"]
    llm_profile = "local"        # optional: pin this agent to a backend
"""

from __future__ import annotations

import logging
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class AgentSpec:
    name: str
    description: str
    instructions: str
    #: Skill names, ``tag:<tag>`` groups, or ``*``. Only these are loaded
    #: into the model's context while the agent runs.
    skills: list[str] = field(default_factory=list)
    #: Backend profile for this agent; ``None`` follows the active backend.
    llm_profile: str | None = None
    max_rounds: int | None = None
    source: str = "builtin"

    def describe(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "skills": self.skills,
            "llm_profile": self.llm_profile,
            "source": self.source,
        }


CORE = AgentSpec(
    name="jarvis",
    description="The orchestrator: talks with the user and coordinates everything else.",
    instructions="",  # built in jarvis.core.prompts
    skills=[
        "delegate",
        "start_background_task",
        "list_background_tasks",
        "cancel_background_task",
        "switch_backend",
        "system_status",
        "get_current_time",
        "remember",
        "recall",
        "web_search",
        "web_fetch",
        "reload_skills",
        # Skills you add under plugins/ are available to the orchestrator
        # straight away, without editing any agent definition.
        "plugins",
    ],
)

BUILTIN_AGENTS = [
    AgentSpec(
        name="trading",
        description=(
            "Market analyst. Pulls prices, runs technical analysis (trend, RSI, MACD, "
            "Bollinger, ATR), scans watchlists and evaluates long/short setups."
        ),
        instructions=(
            "You are a disciplined market analyst. Ground every statement in data you "
            "fetched with your skills in this task: quote the actual figures, and name "
            "the data source. Weigh evidence for and against a setup, state the levels "
            "that would invalidate it, and say plainly when the picture is mixed or the "
            "data is insufficient. You provide analysis, never certainty, and you do not "
            "place orders. If a result is marked synthetic, say that it is not real "
            "market data."
        ),
        skills=["tag:market", "web_search", "web_fetch", "get_current_time"],
    ),
    AgentSpec(
        name="content",
        description=(
            "Writer and publisher. Researches, drafts, formats and publishes articles "
            "and blog posts (local site folder, webhook or WordPress)."
        ),
        instructions=(
            "You are an experienced editor and writer. For an article: gather facts "
            "first if the topic needs them, write a complete draft in Markdown with a "
            "clear headline and structure, save it with save_draft, tidy it with "
            "format_article, and publish only when the task asks for publishing. Do not "
            "present guesses as facts and do not invent quotes or statistics. Report "
            "the slug and where the piece ended up."
        ),
        skills=["tag:content", "web_search", "web_fetch", "read_file", "get_current_time"],
    ),
    AgentSpec(
        name="video",
        description=(
            "Media technician. Inspects, clips, joins, transcodes and extracts audio or "
            "thumbnails from video files in the workspace using ffmpeg."
        ),
        instructions=(
            "You are a careful media technician. Inspect a file with video_probe before "
            "editing it so your timestamps and formats are valid. Never overwrite a "
            "source file. Report the output path, duration and size of what you made."
        ),
        skills=["tag:video", "list_files"],
    ),
    AgentSpec(
        name="research",
        description=(
            "Researcher. Searches the web, reads pages and local documents, extracts "
            "structured data and writes up findings with sources."
        ),
        instructions=(
            "You are a thorough researcher. Read primary sources rather than relying on "
            "search snippets, cross-check claims that matter, and cite the URL or file "
            "each finding came from. Separate what the sources say from your own "
            "inference."
        ),
        skills=["tag:web", "tag:files", "get_current_time"],
    ),
    AgentSpec(
        name="security",
        description=(
            "Security reviewer. Runs full passive audits (headers, TLS, DNS and email "
            "protection, exposed services) of websites the user has authorised, can scan "
            "all authorised sites at once, and tracks what changed since the last run."
        ),
        instructions=(
            "You are a defensive security reviewer for the user's own websites. Use "
            "security_audit for one site and security_scan_all when the user says "
            "'our websites' or similar, because the audit covers headers, TLS, DNS and "
            "exposed services in one pass and keeps a history. Lead with what changed "
            "since the last run, then the most serious items. For each finding explain "
            "the real-world risk and the concrete fix, so the user can act without "
            "looking anything up. Only assess targets the skills accept as authorised; "
            "if a skill refuses a target, relay why and say that the user authorises "
            "sites themselves with `jarvis security allow <host>`, and do not try "
            "another route. These are passive checks and do not replace a penetration "
            "test; say so when it matters."
        ),
        skills=[
            "tag:security", "web_fetch", "read_file", "list_files", "write_file",
            "get_current_time",
        ],
    ),
]


class AgentRegistry:
    def __init__(self, dirs: list[str] | None = None):
        self.dirs = [Path(d).expanduser() for d in (dirs or [])]
        self._agents: dict[str, AgentSpec] = {}
        self.errors: dict[str, str] = {}
        self.reload()

    def reload(self) -> None:
        self._agents = {CORE.name: CORE, **{a.name: a for a in BUILTIN_AGENTS}}
        self.errors.clear()
        for directory in self.dirs:
            if not directory.is_dir():
                continue
            for file in sorted(directory.glob("*.toml")):
                try:
                    self.register(load_agent_file(file))
                except Exception as exc:
                    self.errors[str(file)] = str(exc)
                    log.warning("agent file %s is invalid: %s", file, exc)

    def register(self, spec: AgentSpec) -> None:
        if spec.name == CORE.name:
            raise ValueError("'jarvis' is reserved for the orchestrator")
        self._agents[spec.name] = spec

    def get(self, name: str) -> AgentSpec | None:
        return self._agents.get(name.strip().lower())

    @property
    def core(self) -> AgentSpec:
        return self._agents[CORE.name]

    def specialists(self) -> list[AgentSpec]:
        return [a for name, a in sorted(self._agents.items()) if name != CORE.name]

    def all(self) -> list[AgentSpec]:
        return [self.core, *self.specialists()]


def load_agent_file(path: Path) -> AgentSpec:
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    name = str(data.get("name", path.stem)).strip().lower()
    if not re.fullmatch(r"[a-z][a-z0-9_]{1,31}", name):
        raise ValueError(f"invalid agent name '{name}' (use lowercase letters, digits, _)")
    for key in ("description", "instructions"):
        if not str(data.get(key, "")).strip():
            raise ValueError(f"missing '{key}'")
    skills = data.get("skills", [])
    if not isinstance(skills, list) or not all(isinstance(s, str) for s in skills):
        raise ValueError("'skills' must be a list of strings")
    # Agents defined in files cannot delegate, which rules out runaway recursion.
    skills = [s for s in skills if s not in ("delegate", "start_background_task", "*")]
    return AgentSpec(
        name=name,
        description=str(data["description"]).strip(),
        instructions=str(data["instructions"]).strip(),
        skills=skills,
        llm_profile=data.get("llm_profile") or None,
        max_rounds=data.get("max_rounds"),
        source=f"file:{path.name}",
    )
