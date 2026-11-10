"""Skill definition: a typed Python function a model can call.

    from jarvis import skill

    @skill(tags=["demo"])
    def add(a: int, b: int = 0) -> int:
        '''Add two numbers.

        Args:
            a: First number.
            b: Second number.
        '''
        return a + b

The JSON schema shown to the model is derived from the signature and the
``Args:`` section of the docstring. A parameter named ``ctx`` receives a
:class:`SkillContext` and is hidden from the model.
"""

from __future__ import annotations

import enum
import inspect
import re
import types
import typing
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin

from jarvis.core.types import ToolSpec

if typing.TYPE_CHECKING:  # pragma: no cover
    from jarvis.config import Config


class SkillError(Exception):
    """Raise from a skill to report a failure to the model in plain words."""


@dataclass
class SkillContext:
    """Runtime services handed to skills that declare a ``ctx`` parameter."""

    config: Config
    agent: str = "jarvis"
    emit: Callable[[dict], Awaitable[None]] | None = None
    services: dict[str, Any] = field(default_factory=dict)

    @property
    def workspace(self) -> Path:
        return self.config.workspace

    def resolve_path(self, relative: str, *, must_exist: bool = False) -> Path:
        """Resolve a model-supplied path, refusing anything outside the workspace."""
        root = self.workspace
        target = (root / relative).resolve()
        if not target.is_relative_to(root):
            raise SkillError(f"Path '{relative}' is outside the workspace ({root}).")
        if must_exist and not target.exists():
            raise SkillError(f"'{relative}' does not exist in the workspace ({root}).")
        return target


@dataclass
class Skill:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., Any]
    tags: list[str] = field(default_factory=list)
    timeout: float = 120.0
    source: str = "builtin"
    wants_context: bool = False

    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.parameters)


_PRIMITIVES = {str: "string", int: "integer", float: "number", bool: "boolean"}


def _schema_for(annotation: Any) -> dict[str, Any]:
    if annotation in (inspect.Parameter.empty, Any):
        return {}
    if annotation in _PRIMITIVES:
        return {"type": _PRIMITIVES[annotation]}
    origin, args = get_origin(annotation), get_args(annotation)
    if origin is Literal:
        kind = _PRIMITIVES.get(type(args[0]), "string")
        return {"type": kind, "enum": list(args)}
    if origin in (Union, types.UnionType):
        inner = [a for a in args if a is not type(None)]
        return _schema_for(inner[0]) if len(inner) == 1 else {}
    if origin in (list, tuple, set) or annotation in (list, tuple, set):
        schema: dict[str, Any] = {"type": "array"}
        if args:
            schema["items"] = _schema_for(args[0])
        return schema
    if origin is dict or annotation is dict:
        return {"type": "object"}
    if inspect.isclass(annotation) and issubclass(annotation, enum.Enum):
        return {"type": "string", "enum": [member.value for member in annotation]}
    return {}


def _parse_docstring(doc: str) -> tuple[str, dict[str, str]]:
    """Split a Google-style docstring into (summary, {param: description})."""
    doc = inspect.cleandoc(doc or "")
    match = re.search(r"^Args?:\s*$", doc, flags=re.MULTILINE)
    if not match:
        return doc.strip(), {}
    summary, rest = doc[: match.start()].strip(), doc[match.end() :]
    params: dict[str, str] = {}
    current: str | None = None
    for line in rest.splitlines():
        if re.match(r"^\S", line):  # next section such as "Returns:"
            break
        entry = re.match(r"^\s+(\w+)(?:\s*\([^)]*\))?:\s*(.*)$", line)
        if entry:
            current = entry.group(1)
            params[current] = entry.group(2).strip()
        elif current and line.strip():
            params[current] += " " + line.strip()
    return summary, params


def build_schema(fn: Callable) -> tuple[str, dict[str, Any], bool]:
    summary, param_docs = _parse_docstring(fn.__doc__ or "")
    try:
        hints = typing.get_type_hints(fn)
    except Exception:
        hints = {}
    properties: dict[str, Any] = {}
    required: list[str] = []
    wants_context = False
    for name, param in inspect.signature(fn).parameters.items():
        if name == "ctx":
            wants_context = True
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        schema = _schema_for(hints.get(name, param.annotation))
        if name in param_docs:
            schema["description"] = param_docs[name]
        if param.default is inspect.Parameter.empty:
            required.append(name)
        elif param.default is not None and isinstance(param.default, (str, int, float, bool)):
            schema["default"] = param.default
        properties[name] = schema
    parameters = {"type": "object", "properties": properties, "required": required}
    return summary, parameters, wants_context


def skill(
    fn: Callable | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    tags: list[str] | None = None,
    timeout: float = 120.0,
):
    """Mark a function as a skill. Usable bare (``@skill``) or with options."""

    def decorate(func: Callable) -> Callable:
        summary, parameters, wants_context = build_schema(func)
        func.__skill__ = Skill(  # type: ignore[attr-defined]
            name=name or func.__name__,
            description=description or summary or func.__name__,
            parameters=parameters,
            fn=func,
            tags=list(tags or []),
            timeout=timeout,
            wants_context=wants_context,
        )
        return func

    return decorate(fn) if fn is not None else decorate


# -- argument validation ------------------------------------------------------

_CHECKS: dict[str, Callable[[Any], bool]] = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
}


def _coerce(value: Any, kind: str) -> Any:
    """Small local models often send numbers and booleans as strings."""
    if isinstance(value, str):
        text = value.strip()
        try:
            if kind == "integer":
                return int(text)
            if kind == "number":
                return float(text)
        except ValueError:
            return value
        if kind == "boolean" and text.lower() in ("true", "false"):
            return text.lower() == "true"
        if kind == "array" and text and not text.startswith("["):
            return [part.strip() for part in text.split(",") if part.strip()]
    if kind == "number" and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if kind == "integer" and isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def validate_arguments(schema: dict[str, Any], arguments: Any) -> dict[str, Any]:
    """Check model-supplied arguments against a skill schema.

    Returns cleaned arguments or raises :class:`SkillError` describing every
    problem, so the model can correct itself on the next round.
    """
    if not isinstance(arguments, dict):
        raise SkillError("Arguments must be a JSON object.")
    properties = schema.get("properties", {})
    problems: list[str] = []
    cleaned: dict[str, Any] = {}
    for key in schema.get("required", []):
        if arguments.get(key) is None:
            problems.append(f"missing required argument '{key}'")
    for key, value in arguments.items():
        if key not in properties:
            problems.append(f"unknown argument '{key}'")
            continue
        if value is None:
            continue
        spec = properties[key]
        kind = spec.get("type")
        if kind:
            value = _coerce(value, kind)
            if not _CHECKS[kind](value):
                problems.append(f"'{key}' must be of type {kind}")
                continue
        if "enum" in spec and value not in spec["enum"]:
            problems.append(f"'{key}' must be one of {spec['enum']}")
            continue
        cleaned[key] = value
    if problems:
        raise SkillError("Invalid arguments: " + "; ".join(problems) + ".")
    return cleaned
