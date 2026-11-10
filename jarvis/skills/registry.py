"""Skill discovery and execution.

Skills come from three places, later sources overriding earlier ones:

1. ``jarvis.skills.builtin`` - shipped with the core.
2. Entry points in the ``jarvis.skills`` group - installed packages.
3. Plugin directories (``./plugins`` by default) - drop a ``.py`` file in
   and call :meth:`SkillRegistry.reload` (or say "reload skills").
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import inspect
import json
import logging
import pkgutil
import sys
from importlib import metadata
from pathlib import Path
from types import ModuleType
from typing import Any

from jarvis.core.types import ToolSpec
from jarvis.skills.base import Skill, SkillContext, SkillError, validate_arguments

log = logging.getLogger(__name__)

MAX_RESULT_CHARS = 12000


class SkillRegistry:
    def __init__(self, plugin_dirs: list[str] | None = None, disabled: list[str] | None = None):
        self.plugin_dirs = [Path(p).expanduser() for p in (plugin_dirs or [])]
        self.disabled = set(disabled or [])
        self._skills: dict[str, Skill] = {}
        self.errors: dict[str, str] = {}

    # -- registration -------------------------------------------------------
    def register(self, item: Any, source: str = "runtime") -> Skill:
        skill = item if isinstance(item, Skill) else getattr(item, "__skill__", None)
        if skill is None:
            raise TypeError(f"{item!r} is not a skill; decorate it with @skill")
        skill.source = source
        if skill.name not in self.disabled:
            self._skills[skill.name] = skill
        return skill

    def register_module(self, module: ModuleType, source: str) -> int:
        count = 0
        for _, member in inspect.getmembers(module):
            found = getattr(member, "__skill__", None)
            if isinstance(found, Skill) and getattr(member, "__module__", None) == module.__name__:
                self.register(found, source)
                count += 1
        return count

    def load_all(self) -> SkillRegistry:
        self.errors.clear()
        self._load_builtin()
        self._load_entry_points()
        for directory in self.plugin_dirs:
            self._load_plugin_dir(directory)
        return self

    def reload(self) -> dict:
        """Re-import every plugin so edits take effect without a restart."""
        kept = {n: s for n, s in self._skills.items() if s.source in ("runtime", "core")}
        self._skills = kept
        self.load_all()
        return {"skills": len(self._skills), "errors": dict(self.errors)}

    def _load_builtin(self) -> None:
        import jarvis.skills.builtin as package

        for info in pkgutil.iter_modules(package.__path__):
            name = f"{package.__name__}.{info.name}"
            try:
                self.register_module(importlib.import_module(name), "builtin")
            except Exception as exc:
                self.errors[name] = repr(exc)
                log.warning("builtin skill module %s failed to load", name, exc_info=True)

    def _load_entry_points(self) -> None:
        for entry in metadata.entry_points(group="jarvis.skills"):
            try:
                self.register_module(entry.load(), f"package:{entry.name}")
            except Exception as exc:
                self.errors[f"entry_point:{entry.name}"] = repr(exc)

    def _load_plugin_dir(self, directory: Path) -> None:
        if not directory.is_dir():
            return
        for file in sorted(directory.glob("*.py")):
            if file.name.startswith("_"):
                continue
            module_name = f"jarvis_plugin_{file.stem}"
            try:
                spec = importlib.util.spec_from_file_location(module_name, file)
                module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
                sys.modules[module_name] = module
                spec.loader.exec_module(module)  # type: ignore[union-attr]
                self.register_module(module, f"plugin:{file.name}")
            except Exception as exc:
                self.errors[str(file)] = repr(exc)
                log.warning("plugin %s failed to load", file, exc_info=True)

    # -- lookup -------------------------------------------------------------
    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def all(self) -> list[Skill]:
        return sorted(self._skills.values(), key=lambda s: s.name)

    def select(self, selectors: list[str] | None) -> list[Skill]:
        """Resolve an agent's skill list.

        Entries are skill names, ``tag:<tag>`` for every skill carrying a
        tag, ``plugins`` for everything added from plugin files or installed
        packages, or ``*`` for everything. Only the selected skills are put
        in front of the model, which keeps prompts small for local models.
        """
        if selectors is None:
            return []
        chosen: dict[str, Skill] = {}
        for selector in selectors:
            if selector == "*":
                chosen.update(self._skills)
            elif selector == "plugins":
                chosen.update(
                    {n: s for n, s in self._skills.items() if s.source.startswith(("plugin:", "package:"))}
                )
            elif selector.startswith("tag:"):
                tag = selector[4:]
                chosen.update({n: s for n, s in self._skills.items() if tag in s.tags})
            elif selector in self._skills:
                chosen[selector] = self._skills[selector]
        return sorted(chosen.values(), key=lambda s: s.name)

    def specs(self, selectors: list[str] | None) -> list[ToolSpec]:
        return [s.spec() for s in self.select(selectors)]

    # -- execution ----------------------------------------------------------
    async def execute(self, name: str, arguments: Any, ctx: SkillContext) -> tuple[str, bool]:
        """Run a skill. Returns ``(result_text, is_error)`` and never raises
        for skill failures: errors go back to the model as text."""
        skill = self._skills.get(name)
        if skill is None:
            return f"Unknown skill '{name}'. Available: {', '.join(sorted(self._skills))}.", True
        try:
            args = validate_arguments(skill.parameters, arguments)
            if skill.wants_context:
                args["ctx"] = ctx
            if inspect.iscoroutinefunction(skill.fn):
                call = skill.fn(**args)
            else:
                call = asyncio.to_thread(skill.fn, **args)
            result = await asyncio.wait_for(call, timeout=skill.timeout)
            return _render(result), False
        except SkillError as exc:
            return str(exc), True
        except TimeoutError:
            return f"Skill '{name}' timed out after {skill.timeout:.0f}s.", True
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.exception("skill %s failed", name)
            return f"Skill '{name}' failed: {type(exc).__name__}: {exc}", True


def _render(result: Any) -> str:
    if result is None:
        text = "Done."
    elif isinstance(result, str):
        text = result
    else:
        try:
            text = json.dumps(result, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            text = str(result)
    if len(text) > MAX_RESULT_CHARS:
        omitted = len(text) - MAX_RESULT_CHARS
        text = text[:MAX_RESULT_CHARS] + f"\n... [truncated {omitted} characters]"
    return text
