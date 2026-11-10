"""Core behaviour: skills, agents, the reasoning engine, conversation and tasks."""

import asyncio
import contextlib
import json
import threading
import time
from typing import Literal

import pytest

from jarvis.agents.base import CORE, AgentRegistry, AgentSpec, load_agent_file
from jarvis.app import Jarvis
from jarvis.core.engine import INTERRUPTED
from jarvis.core.session import Session
from jarvis.core.types import LLMError, Message, ToolCall
from jarvis.llm.mock import MockProvider
from jarvis.skills.base import SkillContext, SkillError, skill, validate_arguments
from jarvis.skills.registry import MAX_RESULT_CHARS, SkillRegistry

# --------------------------------------------------------------------------- helpers


def make_app(config, script=None, **settings):
    provider = MockProvider(
        "mock", {"provider": "mock", "model": "mock", **settings}, config, script=script
    )
    return Jarvis(config, providers=[provider]), provider


def tool_msg(*calls: tuple[str, str, dict]) -> Message:
    """An assistant message that calls skills: (call_id, skill_name, arguments)."""
    return Message("assistant", "", tool_calls=[ToolCall(i, n, a) for i, n, a in calls])


async def wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached in time")
        await asyncio.sleep(0.005)


def grant(app, *functions):
    """Register throwaway skills and give the orchestrator access to them."""
    for fn in functions:
        app.skills.register(fn)
        app.agents.core.skills.append(fn.__skill__.name)


def assert_well_formed(messages):
    """Every assistant tool call must be answered by a following tool message."""
    for index, message in enumerate(messages):
        if message.role == "assistant" and message.tool_calls:
            wanted = {call.id for call in message.tool_calls}
            answered = set()
            for later in messages[index + 1 :]:
                if later.role != "tool":
                    break
                answered.add(later.tool_call_id)
            assert wanted <= answered, f"unanswered tool calls: {wanted - answered}"


@pytest.fixture(autouse=True)
def restore_core_skills():
    original = list(CORE.skills)
    yield
    CORE.skills[:] = original


@pytest.fixture
def recorder():
    """Collects emitted engine events."""
    events: list[dict] = []

    async def emit(event):
        events.append(event)

    emit.events = events
    return emit


# --------------------------------------------------------------------------- throwaway skills


@skill
async def nap(seconds: float, label: str) -> str:
    """Sleep a bit, then return the label."""
    await asyncio.sleep(seconds)
    return label


@skill
def boom() -> str:
    """Always fails with a SkillError."""
    raise SkillError("the flux capacitor is offline")


@skill
def crash() -> str:
    """Fails with an unexpected exception."""
    raise ValueError("kaboom")


@skill
async def slow_forever() -> str:
    """Never finishes in time."""
    await asyncio.sleep(30)
    return "never"


@skill
def echo_tool(text: str = "x") -> str:
    """Return the text."""
    return f"echo:{text}"


# --------------------------------------------------------------------------- skills: schema


def test_schema_from_type_hints_and_docstring():
    @skill
    def sample(
        count: int,
        ratio: float,
        flag: bool,
        title: str,
        tags: list[str],
        mode: Literal["fast", "slow"],
        ctx: SkillContext,
        limit: int = 5,
        label: str | None = None,
        note: str = "hi",
    ) -> str:
        """Do the sample thing.

        Args:
            count: How many
                items to make.
            ratio (float): The ratio.
            mode: Speed mode.

        Returns:
            Something that is not a parameter.
        """

    found = sample.__skill__
    assert found.description == "Do the sample thing."
    assert found.wants_context is True
    props = found.parameters["properties"]
    assert "ctx" not in props
    assert props["count"] == {"type": "integer", "description": "How many items to make."}
    assert props["ratio"]["type"] == "number" and props["ratio"]["description"] == "The ratio."
    assert props["flag"]["type"] == "boolean"
    assert props["title"]["type"] == "string"
    assert props["tags"] == {"type": "array", "items": {"type": "string"}}
    assert props["mode"]["type"] == "string" and props["mode"]["enum"] == ["fast", "slow"]
    assert props["limit"] == {"type": "integer", "default": 5}
    assert props["label"] == {"type": "string"}  # Optional collapses to its inner type, no default
    assert props["note"]["default"] == "hi"
    # Only parameters without defaults (and not ctx) are required.
    assert found.parameters["required"] == ["count", "ratio", "flag", "title", "tags", "mode"]
    assert "Returns" not in json.dumps(found.parameters)


def test_skill_decorator_options_and_default_description():
    @skill(name="renamed", tags=["a", "b"], timeout=3)
    def original_name():
        pass

    found = original_name.__skill__
    assert (found.name, found.tags, found.timeout) == ("renamed", ["a", "b"], 3)
    assert found.description == "original_name"  # falls back to the function name


# --------------------------------------------------------------------------- skills: validation

SCHEMA = {
    "type": "object",
    "properties": {
        "n": {"type": "integer"},
        "x": {"type": "number"},
        "ok": {"type": "boolean"},
        "name": {"type": "string"},
        "items": {"type": "array", "items": {"type": "string"}},
        "mode": {"type": "string", "enum": ["a", "b"]},
    },
    "required": ["name"],
}


def test_validate_missing_required_and_unknown_key():
    with pytest.raises(SkillError) as info:
        validate_arguments(SCHEMA, {"bogus": 1})
    message = str(info.value)
    assert "missing required argument 'name'" in message
    assert "unknown argument 'bogus'" in message  # every problem is reported at once


def test_validate_wrong_type_and_enum():
    with pytest.raises(SkillError, match="'n' must be of type integer"):
        validate_arguments(SCHEMA, {"name": "x", "n": "abc"})
    with pytest.raises(SkillError, match="'n' must be of type integer"):
        validate_arguments(SCHEMA, {"name": "x", "n": True})  # bool is not an int
    with pytest.raises(SkillError, match="'mode' must be one of"):
        validate_arguments(SCHEMA, {"name": "x", "mode": "c"})
    with pytest.raises(SkillError, match="JSON object"):
        validate_arguments(SCHEMA, ["not", "a", "dict"])


def test_validate_coerces_strings_for_small_models():
    cleaned = validate_arguments(
        SCHEMA,
        {"name": "x", "n": "7", "x": "2.5", "ok": "TRUE", "items": "a, b ,c", "mode": "a"},
    )
    assert cleaned == {"name": "x", "n": 7, "x": 2.5, "ok": True, "items": ["a", "b", "c"], "mode": "a"}
    assert isinstance(cleaned["n"], int)
    # Whole floats are accepted for integers and ints for numbers.
    cleaned = validate_arguments(SCHEMA, {"name": "x", "n": 3.0, "x": 4})
    assert cleaned["n"] == 3 and isinstance(cleaned["n"], int)
    assert cleaned["x"] == 4.0 and isinstance(cleaned["x"], float)
    # A real list passes through; None for an optional is dropped.
    assert validate_arguments(SCHEMA, {"name": "x", "items": ["q"], "n": None}) == {
        "name": "x",
        "items": ["q"],
    }


# --------------------------------------------------------------------------- skills: registry


def test_registry_select_names_tags_and_star():
    @skill(tags=["x"])
    def zeta():
        pass

    @skill(tags=["x", "y"])
    def alpha():
        pass

    @skill(tags=["y"])
    def mid():
        pass

    registry = SkillRegistry()
    for fn in (zeta, alpha, mid):
        registry.register(fn)
    names = lambda sel: [s.name for s in registry.select(sel)]  # noqa: E731
    assert names(["zeta"]) == ["zeta"]
    assert names(["tag:x"]) == ["alpha", "zeta"]  # sorted by name
    assert names(["tag:y", "zeta", "missing"]) == ["alpha", "mid", "zeta"]
    assert names(["*"]) == ["alpha", "mid", "zeta"]
    assert names(["alpha", "tag:x"]) == ["alpha", "zeta"]  # no duplicates
    assert names([]) == [] and names(None) == []
    assert [spec.name for spec in registry.specs(["mid"])] == ["mid"]


def test_registry_register_rejects_non_skill():
    with pytest.raises(TypeError):
        SkillRegistry().register(lambda: None)


async def test_execute_success_and_dict_result_is_json(config):
    @skill
    def as_dict(n: int) -> dict:
        """Return a dict."""
        return {"n": n, "name": "café"}

    @skill
    def nothing() -> None:
        """Return nothing."""

    registry = SkillRegistry()
    registry.register(as_dict)
    registry.register(nothing)
    ctx = SkillContext(config)
    text, is_error = await registry.execute("as_dict", {"n": "3"}, ctx)
    assert not is_error and json.loads(text) == {"n": 3, "name": "café"}
    assert await registry.execute("nothing", {}, ctx) == ("Done.", False)


async def test_execute_errors_are_text_not_exceptions(config):
    registry = SkillRegistry()
    for fn in (boom, crash, echo_tool):
        registry.register(fn)
    ctx = SkillContext(config)

    text, is_error = await registry.execute("boom", {}, ctx)
    assert (text, is_error) == ("the flux capacitor is offline", True)

    text, is_error = await registry.execute("crash", {}, ctx)
    assert is_error and "ValueError" in text and "kaboom" in text

    text, is_error = await registry.execute("echo_tool", {"nope": 1}, ctx)
    assert is_error and "unknown argument 'nope'" in text

    text, is_error = await registry.execute("ghost", {}, ctx)
    assert is_error and "Unknown skill 'ghost'" in text and "echo_tool" in text


async def test_execute_timeout(config):
    @skill(timeout=0.05)
    async def sleepy() -> str:
        """Sleeps too long."""
        await asyncio.sleep(2)
        return "late"

    registry = SkillRegistry()
    registry.register(sleepy)
    started = time.monotonic()
    text, is_error = await registry.execute("sleepy", {}, SkillContext(config))
    assert is_error and "timed out" in text
    assert time.monotonic() - started < 1.0


async def test_sync_skills_run_off_the_event_loop(config):
    seen = {}

    @skill
    def blocking() -> str:
        """Blocks its thread for a moment."""
        seen["thread"] = threading.get_ident()
        time.sleep(0.15)
        return "slept"

    registry = SkillRegistry()
    registry.register(blocking)
    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    task = asyncio.create_task(ticker())
    text, is_error = await registry.execute("blocking", {}, SkillContext(config))
    task.cancel()
    assert (text, is_error) == ("slept", False)
    assert seen["thread"] != threading.get_ident()
    assert ticks >= 3  # the loop kept running while the skill blocked


async def test_long_results_are_truncated(config):
    @skill
    def huge() -> str:
        """Returns a lot."""
        return "x" * (MAX_RESULT_CHARS + 500)

    registry = SkillRegistry()
    registry.register(huge)
    text, is_error = await registry.execute("huge", {}, SkillContext(config))
    assert not is_error
    assert text.startswith("x" * MAX_RESULT_CHARS)
    assert "truncated 500 characters" in text
    assert len(text) < MAX_RESULT_CHARS + 100


async def test_context_is_injected_only_for_ctx_skills(config):
    @skill
    def who(ctx: SkillContext, greeting: str = "hi") -> str:
        """Report the calling agent."""
        return f"{greeting} {ctx.agent}"

    registry = SkillRegistry()
    registry.register(who)
    text, _ = await registry.execute("who", {"greeting": "yo"}, SkillContext(config, agent="trading"))
    assert text == "yo trading"
    # A model cannot smuggle its own ctx argument in.
    text, is_error = await registry.execute("who", {"ctx": "evil"}, SkillContext(config))
    assert is_error and "unknown argument 'ctx'" in text


# --------------------------------------------------------------------------- skills: plugins

PLUGIN_V1 = '''
from jarvis.skills.base import skill

@skill
def plug_hello(name: str) -> str:
    """Greet someone."""
    return "hi " + name
'''

PLUGIN_V2 = '''
from jarvis.skills.base import skill

@skill
def plug_hello(name: str) -> str:
    """Greet someone, louder."""
    return "HELLO, " + name.upper() + "!"

@skill
def plug_extra() -> str:
    """A new skill."""
    return "extra"
'''


async def test_plugin_directory_loading_and_reload(tmp_path, config):
    plugins = tmp_path / "plugins"
    plugins.mkdir()
    (plugins / "hello_plugin.py").write_text(PLUGIN_V1, encoding="utf-8")
    (plugins / "_ignored.py").write_text(PLUGIN_V2, encoding="utf-8")  # underscore files are skipped

    registry = SkillRegistry([str(plugins)]).load_all()
    assert registry.errors == {}
    found = registry.get("plug_hello")
    assert found is not None and found.source == "plugin:hello_plugin.py"
    assert registry.get("plug_extra") is None
    ctx = SkillContext(config)
    assert await registry.execute("plug_hello", {"name": "bob"}, ctx) == ("hi bob", False)

    @skill
    def runtime_one() -> str:
        """Added at runtime."""
        return "r"

    registry.register(runtime_one)

    (plugins / "hello_plugin.py").write_text(PLUGIN_V2, encoding="utf-8")  # different size
    report = registry.reload()
    assert report["errors"] == {}
    assert await registry.execute("plug_hello", {"name": "bob"}, ctx) == ("HELLO, BOB!", False)
    assert registry.get("plug_extra") is not None
    assert registry.get("runtime_one") is not None  # runtime registrations survive a reload


async def test_broken_plugin_is_reported_without_breaking_others(tmp_path, config):
    plugins = tmp_path / "plugins"
    plugins.mkdir()
    (plugins / "good_plugin.py").write_text(PLUGIN_V1, encoding="utf-8")
    broken = plugins / "broken_plugin.py"
    broken.write_text("def oops(:\n    pass\n", encoding="utf-8")
    (plugins / "raises_plugin.py").write_text("raise RuntimeError('import time failure')\n", encoding="utf-8")

    registry = SkillRegistry([str(plugins)]).load_all()
    assert str(broken) in registry.errors and "SyntaxError" in registry.errors[str(broken)]
    assert any("import time failure" in message for message in registry.errors.values())
    assert registry.get("plug_hello") is not None
    # Builtin skills are unaffected.
    assert registry.get("remember") is not None

    # Fixing the file and reloading clears the error.
    broken.write_text(PLUGIN_V2.replace("plug_", "fixed_"), encoding="utf-8")
    registry.reload()
    assert str(broken) not in registry.errors
    assert registry.get("fixed_hello") is not None


def test_disabled_skills_are_not_registered():
    registry = SkillRegistry(disabled=["remember"]).load_all()
    assert registry.get("remember") is None
    assert registry.get("recall") is not None


# --------------------------------------------------------------------------- agents


def test_builtin_agents_exist():
    registry = AgentRegistry()
    names = {agent.name for agent in registry.specialists()}
    assert {"trading", "content", "video", "research", "security"} <= names
    assert registry.core.name == "jarvis"
    assert "jarvis" not in names
    assert [a.name for a in registry.all()][0] == "jarvis"
    assert registry.get("  Trading ") is registry.get("trading")  # lookup is forgiving
    assert registry.get("nobody") is None
    assert "delegate" in registry.core.skills


def test_toml_agent_loads(tmp_path):
    (tmp_path / "summarizer.toml").write_text(
        'name = "Summarizer"\n'
        'description = "Condenses things."\n'
        'instructions = "Be brief."\n'
        'skills = ["read_file", "web_fetch"]\n'
        'llm_profile = "mock"\n'
        "max_rounds = 3\n",
        encoding="utf-8",
    )
    registry = AgentRegistry([str(tmp_path)])
    assert registry.errors == {}
    agent = registry.get("summarizer")
    assert agent is not None
    assert (agent.description, agent.instructions) == ("Condenses things.", "Be brief.")
    assert agent.skills == ["read_file", "web_fetch"]
    assert (agent.llm_profile, agent.max_rounds) == ("mock", 3)
    assert agent.source == "file:summarizer.toml"
    assert agent in registry.specialists()


def test_invalid_agent_files_land_in_errors(tmp_path):
    files = {
        "no_instructions.toml": 'name = "noinstr"\ndescription = "d"\n',
        "bad_name.toml": 'name = "Bad Name!"\ndescription = "d"\ninstructions = "i"\n',
        "bad_skills.toml": 'name = "badskills"\ndescription = "d"\ninstructions = "i"\nskills = "read_file"\n',
        "syntax.toml": "this is = not [valid toml\n",
        "reserved.toml": 'name = "jarvis"\ndescription = "d"\ninstructions = "i"\n',
        "fine.toml": 'name = "fine"\ndescription = "d"\ninstructions = "i"\n',
    }
    for name, text in files.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    registry = AgentRegistry([str(tmp_path)])
    bad = {name for name in files if name != "fine.toml"}
    assert {path.rsplit("/", 1)[-1] for path in registry.errors} == bad
    assert registry.get("fine") is not None
    assert registry.get("noinstr") is None and registry.get("badskills") is None
    assert registry.core.source == "builtin"  # a file cannot replace the orchestrator
    # Reloading is idempotent and rebuilds the error list instead of growing it.
    registry.reload()
    assert len(registry.errors) == len(bad)


def test_file_agents_cannot_list_delegate(tmp_path):
    path = tmp_path / "sneaky.toml"
    path.write_text(
        'name = "sneaky"\ndescription = "d"\ninstructions = "i"\n'
        'skills = ["delegate", "start_background_task", "*", "read_file"]\n',
        encoding="utf-8",
    )
    assert load_agent_file(path).skills == ["read_file"]


def test_file_agents_cannot_reach_delegate_through_tags(tmp_path, config):
    path = tmp_path / "tagged.toml"
    path.write_text(
        'name = "tagged"\ndescription = "d"\ninstructions = "i"\nskills = ["tag:core"]\n',
        encoding="utf-8",
    )
    agent = load_agent_file(path)
    app = Jarvis(config)
    names = {tool.name for tool in app.engine._tools_for(agent)}
    assert "get_current_time" in names  # the tag still expands...
    assert not names & {"delegate", "start_background_task", "cancel_background_task"}
    # ...while the orchestrator keeps its orchestration skills.
    assert "delegate" in {tool.name for tool in app.engine._tools_for(app.agents.core)}


# --------------------------------------------------------------------------- engine


async def test_plain_reply_is_stored_in_session(config, recorder):
    app, provider = make_app(config, script=["Hello there!"])
    reply = await app.chat("hi", emit=recorder)
    assert reply == "Hello there!"
    assert [(m.role, m.content) for m in app.session().messages] == [
        ("user", "hi"),
        ("assistant", "Hello there!"),
    ]
    request = provider.calls[0]
    assert "Jarvis" in request["system"] and '"mock" backend' in request["system"]
    assert "delegate" in {tool.name for tool in request["tools"]}
    assert [e["text"] for e in recorder.events if e["type"] == "text_delta"] == ["Hello ", "there!"]


async def test_tool_call_runs_and_result_goes_back(config, recorder):
    app, provider = make_app(config, script=[tool_msg(("c1", "echo_tool", {"text": "yo"})), "Got it."])
    grant(app, echo_tool)
    reply = await app.chat("use the tool", emit=recorder)
    assert reply == "Got it."
    sent = provider.calls[1]["messages"]
    assert [m.role for m in sent] == ["user", "assistant", "tool"]
    assert sent[1].tool_calls[0].id == "c1"
    assert (sent[2].tool_call_id, sent[2].name, sent[2].content) == ("c1", "echo_tool", "echo:yo")
    assert not sent[2].is_error
    types = [e["type"] for e in recorder.events]
    assert types.index("tool_start") < types.index("tool_end")
    assert [m.role for m in app.session().messages] == ["user", "assistant", "tool", "assistant"]


async def test_parallel_tool_calls_run_concurrently_and_keep_call_order(config):
    app, provider = make_app(
        config,
        script=[
            tool_msg(("c1", "nap", {"seconds": 0.2, "label": "first"}), ("c2", "nap", {"seconds": 0.02, "label": "second"})),
            "both done",
        ],
    )
    grant(app, nap)
    started = time.monotonic()
    reply = await app.chat("go")
    elapsed = time.monotonic() - started
    assert reply == "both done"
    assert elapsed < 0.38  # sequential execution would need >= 0.22 plus; concurrent ~0.2
    tools = [m for m in app.session().messages if m.role == "tool"]
    assert [(m.tool_call_id, m.content) for m in tools] == [("c1", "first"), ("c2", "second")]
    assert [m.tool_call_id for m in provider.calls[1]["messages"] if m.role == "tool"] == ["c1", "c2"]


async def test_two_equal_sleeps_overlap(config):
    app, _ = make_app(
        config,
        script=[
            tool_msg(("c1", "nap", {"seconds": 0.2, "label": "a"}), ("c2", "nap", {"seconds": 0.2, "label": "b"})),
            "ok",
        ],
    )
    grant(app, nap)
    started = time.monotonic()
    await app.chat("go")
    assert time.monotonic() - started < 0.38


async def test_failing_skill_yields_error_message_and_loop_continues(config):
    app, provider = make_app(
        config,
        script=[
            tool_msg(("c1", "boom", {}), ("c2", "crash", {}), ("c3", "echo_tool", {"text": "fine"})),
            "Some failed, one worked.",
        ],
    )
    grant(app, boom, crash, echo_tool)
    reply = await app.chat("try them")
    assert reply == "Some failed, one worked."
    tools = {m.tool_call_id: m for m in provider.calls[1]["messages"] if m.role == "tool"}
    assert tools["c1"].is_error and tools["c1"].content == "the flux capacitor is offline"
    assert tools["c2"].is_error and "kaboom" in tools["c2"].content
    assert not tools["c3"].is_error and tools["c3"].content == "echo:fine"


async def test_tool_round_cap_ends_with_a_request_that_offers_no_tools(config):
    config.set("assistant.max_tool_rounds", 2)
    again = tool_msg(("c1", "echo_tool", {}))
    app, provider = make_app(config, script=[again, again, "Wrapping up."])
    grant(app, echo_tool)
    reply = await app.chat("loop")
    assert reply == "Wrapping up."
    assert len(provider.calls) == 3
    assert provider.calls[0]["tools"] and provider.calls[1]["tools"]
    assert provider.calls[2]["tools"] == []


async def test_stubborn_model_cannot_loop_past_the_cap(config):
    config.set("assistant.max_tool_rounds", 2)
    again = tool_msg(("c1", "echo_tool", {}))
    app, provider = make_app(config, script=[again, again, again, again, again])
    grant(app, echo_tool)
    await asyncio.wait_for(app.chat("loop"), 5)
    assert len(provider.calls) == 3  # two rounds with tools plus one final tool-less request
    assert provider.calls[-1]["tools"] == []
    assert_well_formed(app.session().messages)


async def test_llm_error_rolls_session_back_to_pre_turn_length(config):
    app, _ = make_app(config, script=["first", LLMError("backend down")])
    await app.chat("one")
    assert len(app.session().messages) == 2
    with pytest.raises(LLMError, match="backend down"):
        await app.chat("two")
    assert [m.content for m in app.session().messages] == ["one", "first"]


async def test_llm_error_after_tool_round_removes_the_whole_turn(config):
    app, _ = make_app(
        config, script=["first", tool_msg(("c1", "echo_tool", {})), LLMError("died mid turn")]
    )
    grant(app, echo_tool)
    await app.chat("one")
    with pytest.raises(LLMError):
        await app.chat("two")
    assert len(app.session().messages) == 2


# --------------------------------------------------------------------------- delegation


async def test_delegate_runs_specialist_in_fresh_session(config, recorder):
    app, provider = make_app(
        config,
        script=[
            tool_msg(("d1", "delegate", {"agent": "trading", "task": "analyse AAPL fully"})),
            "AAPL looks range bound.",  # the specialist's report
            "Trading says AAPL is range bound.",  # the orchestrator's answer
        ],
    )
    app.session().add(Message("user", "earlier chatter SECRET-CONTEXT"))
    app.session().add(Message("assistant", "earlier reply"))
    reply = await app.chat("what about apple?", emit=recorder)
    assert reply == "Trading says AAPL is range bound."

    orchestrator, specialist, final = provider.calls
    assert "specialist on a team" in specialist["system"] and '"trading"' in specialist["system"]
    assert "specialist on a team" not in orchestrator["system"]
    # The specialist sees only its task: none of the orchestrator's conversation.
    assert [(m.role, m.content) for m in specialist["messages"]] == [("user", "analyse AAPL fully")]
    assert "SECRET-CONTEXT" not in json.dumps([m.to_dict() for m in specialist["messages"]])
    assert "delegate" not in {t.name for t in specialist["tools"]}
    # The report comes back to the orchestrator as the tool result.
    tool_result = final["messages"][-1]
    assert (tool_result.role, tool_result.tool_call_id) == ("tool", "d1")
    assert tool_result.content == "AAPL looks range bound." and not tool_result.is_error
    kinds = [e["type"] for e in recorder.events]
    assert "agent_start" in kinds and "agent_end" in kinds
    assert next(e for e in recorder.events if e["type"] == "agent_end")["ok"] is True
    # The specialist's text is activity, not speech.
    assert "agent_text" in kinds


async def test_delegate_unknown_agent_lists_available_agents(config):
    app, provider = make_app(
        config, script=[tool_msg(("d1", "delegate", {"agent": "wizard", "task": "x"})), "no such agent"]
    )
    await app.chat("summon")
    result = provider.calls[1]["messages"][-1]
    assert result.is_error
    assert "No agent named 'wizard'" in result.content
    assert "trading" in result.content and "research" in result.content
    assert "jarvis" not in result.content.split("Available agents:")[1]


async def test_delegating_to_the_orchestrator_itself_is_refused(config):
    app, provider = make_app(
        config, script=[tool_msg(("d1", "delegate", {"agent": "jarvis", "task": "x"})), "ok"]
    )
    await app.chat("loop me")
    assert provider.calls[1]["messages"][-1].is_error


async def test_specialist_cannot_delegate_further(config):
    app, provider = make_app(
        config,
        script=[
            tool_msg(("d1", "delegate", {"agent": "deepdive", "task": "dig"})),
            tool_msg(("e1", "delegate", {"agent": "trading", "task": "nested"})),  # specialist
            "could not delegate",  # specialist's report
            "finished",
        ],
    )
    app.agents.register(
        AgentSpec("deepdive", "Digs deep.", "You dig.", skills=["delegate"], source="test")
    )
    reply = await app.chat("dig deep")
    assert reply == "finished"
    nested_attempt = provider.calls[2]["messages"][-1]
    assert nested_attempt.tool_call_id == "e1" and nested_attempt.is_error
    assert "not one of your skills" in nested_attempt.content
    assert len(provider.calls) == 4  # the nested specialist never ran


async def test_run_agent_depth_guard(config):
    app, _ = make_app(config)
    with pytest.raises(SkillError, match="cannot delegate further"):
        await app.run_agent("trading", "x", depth=2)


async def test_run_agents_runs_concurrently_and_isolates_failures(config):
    def reply(messages):
        task = messages[-1].content
        return LLMError("model exploded") if task == "bad" else f"report for {task}"

    app, _ = make_app(config, script=[reply] * 3)
    results = await app.run_agents([("trading", "alpha"), ("research", "bad"), ("nobody", "beta")])
    assert [r["agent"] for r in results] == ["trading", "research", "nobody"]
    assert results[0] == {"agent": "trading", "task": "alpha", "ok": True, "report": "report for alpha"}
    assert results[1]["ok"] is False and "model exploded" in results[1]["report"]
    assert results[2]["ok"] is False and "No agent named 'nobody'" in results[2]["report"]


async def test_run_agents_overlap_in_time(config):
    app, _ = make_app(config, delay=0.1)  # echo "You said: a" = 3 words = 0.3 s each
    started = time.monotonic()
    results = await app.run_agents([("trading", "a"), ("research", "b")])
    elapsed = time.monotonic() - started
    assert all(r["ok"] for r in results)
    assert results[0]["report"] == "You said: a" and results[1]["report"] == "You said: b"
    assert elapsed < 0.55  # sequential would take ~0.6 s


# --------------------------------------------------------------------------- history and epochs


async def test_history_trimming_cuts_at_a_user_message_and_bumps_epoch(config):
    config.set("assistant.max_history_messages", 4)
    app, provider = make_app(config, script=["r1", "r2", "r3", "r4"])
    session = app.session()
    for index in range(3):
        await app.chat(f"q{index + 1}")
    assert len(session.messages) == 6  # trimming happens before the next turn
    epoch_before = session.epoch
    await app.chat("q4")
    assert session.epoch == epoch_before + 1
    assert session.messages[0].role == "user"
    assert [m.content for m in session.messages] == ["q2", "r2", "q3", "r3", "q4", "r4"]
    sent = provider.calls[3]["messages"]
    assert [m.content for m in sent] == ["q2", "r2", "q3", "r3", "q4"]


def test_trim_never_separates_a_tool_call_from_its_result():
    session = Session()
    session.add(Message("user", "old"))
    session.add(tool_msg(("c1", "x", {})))
    session.add(Message("tool", "r", tool_call_id="c1", name="x"))
    session.add(Message("assistant", "done"))
    session.add(Message("user", "new"))
    session.add(Message("assistant", "ok"))
    # Naive slicing at index 3 would start with an orphan assistant; the cut moves to the user.
    assert session.trim(3) is True
    assert [m.content for m in session.messages] == ["new", "ok"]
    assert session.trim(10) is False


async def test_provider_payloads_from_an_older_epoch_are_stripped(config):
    def with_payload(text):
        return Message("assistant", text, meta={"anthropic": {"marker": text}})

    app, provider = make_app(
        config, script=[with_payload("one"), with_payload("two"), "three", "four"]
    )
    await app.chat("a")
    await app.chat("b")
    # Nothing changed between the turns, so the payload from turn 1 is replayed intact.
    first_assistant = provider.calls[1]["messages"][1]
    assert first_assistant.meta["anthropic"] == {"marker": "one"}

    # A different system prompt starts a new epoch: old payloads are sent as plain messages.
    config.set("assistant.name", "Friday")
    await app.chat("c")
    sent = provider.calls[2]["messages"]
    assert "Friday" in provider.calls[2]["system"]
    assert [m.content for m in sent[:4]] == ["a", "one", "b", "two"]
    assert "anthropic" not in sent[1].meta and "anthropic" not in sent[3].meta
    # ...but the stored history keeps them untouched.
    assert app.session().messages[1].meta["anthropic"] == {"marker": "one"}

    # Within the new epoch, payloads written now are kept again.
    session = app.session()
    session.messages[-1].meta["anthropic"] = {"marker": "three"}
    await app.chat("d")
    sent = provider.calls[3]["messages"]
    assert "anthropic" not in sent[1].meta
    assert sent[5].content == "three" and sent[5].meta["anthropic"] == {"marker": "three"}


async def test_changed_tool_set_also_strips_payloads(config):
    app, provider = make_app(
        config,
        script=[Message("assistant", "one", meta={"anthropic": {"m": 1}}), "two"],
    )
    await app.chat("a")
    grant(app, echo_tool)  # the orchestrator now offers one more tool
    await app.chat("b")
    assert "anthropic" not in provider.calls[1]["messages"][1].meta


# --------------------------------------------------------------------------- cancellation


async def test_cancel_mid_stream_stores_partial_text_as_interrupted(config, recorder):
    app, _ = make_app(config, delay=0.05)
    task = asyncio.create_task(app.chat("one two three four five six seven eight", emit=recorder))
    await wait_until(lambda: sum(e["type"] == "text_delta" for e in recorder.events) >= 3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    last = app.session().messages[-1]
    assert last.role == "assistant"
    assert last.meta["interrupted"] is True
    assert last.content.startswith("You said:")
    assert "eight" not in last.content  # only what had been streamed so far
    assert [m.role for m in app.session().messages] == ["user", "assistant"]


async def test_cancel_during_slow_skill_leaves_a_valid_session(config, recorder):
    app, provider = make_app(
        config,
        script=[
            Message(
                "assistant",
                "",
                tool_calls=[ToolCall("c1", "slow_forever", {}), ToolCall("c2", "slow_forever", {})],
            ),
            "Back to normal.",
        ],
    )
    grant(app, slow_forever)
    task = asyncio.create_task(app.chat("start it", emit=recorder))
    await wait_until(lambda: sum(e["type"] == "tool_start" for e in recorder.events) == 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    messages = app.session().messages
    assert_well_formed(messages)
    placeholders = [m for m in messages if m.role == "tool"]
    assert [m.tool_call_id for m in placeholders] == ["c1", "c2"]
    assert all(m.is_error and m.content == INTERRUPTED for m in placeholders)

    # The follow-up request is accepted: well-formed history, and the turn completes.
    reply = await asyncio.wait_for(app.chat("never mind, hello"), 5)
    assert reply == "Back to normal."
    assert_well_formed(provider.calls[-1]["messages"])
    assert provider.calls[-1]["messages"][-1].content == "never mind, hello"


# --------------------------------------------------------------------------- conversation


@pytest.fixture
def events():
    return []


def watch(conversation, events):
    conversation.subscribe(events.append)


async def test_conversation_event_order(config, events):
    app, _ = make_app(config, script=["Hello there my friend. How are you doing today? Fine."])
    conversation = app.conversation()
    watch(conversation, events)
    turn_id = await conversation.submit("hi")
    await conversation.wait()
    types = [e["type"] for e in events]
    assert types[0] == "turn_start" and types[-1] == "turn_end"
    assert "text_delta" in types and "sentence" in types
    assert max(i for i, t in enumerate(types) if t == "text_delta") < types.index("turn_end")
    sentences = [e for e in events if e["type"] == "sentence"]
    assert [s["text"] for s in sentences] == [
        "Hello there my friend.",
        "How are you doing today?",
        "Fine.",
    ]
    assert [s["index"] for s in sentences] == [0, 1, 2]
    assert all(e["turn_id"] == turn_id for e in events if e["type"] != "voice_state")
    end = events[-1]
    assert end["interrupted"] is False and end["error"] is None
    assert end["text"].startswith("Hello there")


async def test_empty_input_does_not_start_a_turn(config, events):
    app, _ = make_app(config)
    conversation = app.conversation()
    watch(conversation, events)
    assert await conversation.submit("   ") is None
    assert events == [] and not conversation.busy


async def test_submit_during_a_running_turn_interrupts_it(config, events):
    app, _ = make_app(config, delay=0.05)
    conversation = app.conversation()
    watch(conversation, events)
    first = await conversation.submit("alpha beta gamma delta epsilon zeta eta theta")
    await wait_until(lambda: any(e["type"] == "text_delta" for e in events))
    second = await conversation.submit("short")
    await conversation.wait()
    ends = [e for e in events if e["type"] == "turn_end"]
    assert [(e["turn_id"], e["interrupted"]) for e in ends] == [(first, True), (second, False)]
    types = [(e["type"], e.get("turn_id")) for e in events]
    assert types.index(("turn_end", first)) < types.index(("turn_start", second))
    assert ends[1]["text"] == "You said: short"
    roles = [(m.role, bool(m.meta.get("interrupted"))) for m in app.session().messages]
    assert roles == [("user", False), ("assistant", True), ("user", False), ("assistant", False)]


async def test_interrupt_rewrites_history_to_what_was_heard(config, events):
    app, _ = make_app(config, delay=0.05)
    conversation = app.conversation()
    watch(conversation, events)
    await conversation.submit("one two three four five six seven eight nine ten")
    await wait_until(lambda: any(e["type"] == "text_delta" for e in events))
    epoch = app.session().epoch
    assert await conversation.interrupt(heard="  You said:  ") is True
    last = app.session().messages[-1]
    assert (last.role, last.content) == ("assistant", "You said:")
    assert last.meta["interrupted"] is True
    assert app.session().epoch == epoch + 1  # history changed, provider payloads are stale
    assert not conversation.busy
    assert await conversation.interrupt() is False  # nothing left to interrupt


async def test_interrupt_with_nothing_heard_removes_the_reply(config, events):
    app, _ = make_app(config, delay=0.05)
    conversation = app.conversation()
    watch(conversation, events)
    await conversation.submit("one two three four five six seven eight nine ten")
    await wait_until(lambda: any(e["type"] == "text_delta" for e in events))
    await conversation.interrupt(heard="")
    messages = app.session().messages
    assert [m.role for m in messages] == ["user"]


async def test_interrupt_without_heard_keeps_partial_reply(config, events):
    app, _ = make_app(config, delay=0.05)
    conversation = app.conversation()
    watch(conversation, events)
    await conversation.submit("one two three four five six seven eight nine ten")
    await wait_until(lambda: sum(e["type"] == "text_delta" for e in events) >= 2)
    await conversation.interrupt()
    last = app.session().messages[-1]
    assert last.role == "assistant" and last.meta["interrupted"] and last.content.startswith("You")


async def test_stop_phrase_interrupts_without_starting_a_turn(config, events):
    app, _ = make_app(config, delay=0.05)
    conversation = app.conversation()
    watch(conversation, events)
    first = await conversation.submit("one two three four five six seven eight nine ten")
    await wait_until(lambda: any(e["type"] == "text_delta" for e in events))
    assert await conversation.submit("Stop.", source="voice") is None
    types = [e["type"] for e in events]
    assert "stopped" in types
    assert types.count("turn_start") == 1
    end = next(e for e in events if e["type"] == "turn_end")
    assert end["turn_id"] == first and end["interrupted"] is True
    assert not conversation.busy


async def test_stop_phrase_when_idle_only_emits_stopped(config, events):
    app, provider = make_app(config)
    conversation = app.conversation()
    watch(conversation, events)
    assert await conversation.submit("be quiet", source="voice") is None
    assert [e["type"] for e in events] == ["stopped"]
    assert provider.calls == []


async def test_stop_word_typed_as_text_is_a_normal_message(config, events):
    app, _ = make_app(config)
    conversation = app.conversation()
    watch(conversation, events)
    assert await conversation.submit("stop", source="text") is not None
    await conversation.wait()
    assert next(e for e in events if e["type"] == "turn_end")["text"] == "You said: stop"


async def test_llm_error_event_apology_and_turn_end(config, events):
    app, _ = make_app(config, script=[LLMError("Cannot reach Ollama: it is not running")])
    conversation = app.conversation()
    watch(conversation, events)
    await conversation.submit("hello")
    await conversation.wait()
    types = [e["type"] for e in events]
    assert types.index("error") < types.index("sentence") < types.index("turn_end")
    error = next(e for e in events if e["type"] == "error")
    assert "Cannot reach Ollama" in error["message"]
    apology = next(e for e in events if e["type"] == "sentence")["text"]
    assert apology.startswith("I couldn't reach my language model.")
    assert "Ollama doesn't seem to be running" in apology
    end = events[-1]
    assert end["error"] == error["message"] and end["interrupted"] is False
    assert app.session().messages == []  # the failed turn left no trace


async def test_unexpected_crash_is_reported_and_conversation_survives(config, events):
    app, _ = make_app(config, script=[RuntimeError("provider bug"), "recovered"])
    conversation = app.conversation()
    watch(conversation, events)
    await conversation.submit("one")
    await conversation.wait()
    end = [e for e in events if e["type"] == "turn_end"][0]
    assert end["error"] == "RuntimeError: provider bug"
    await conversation.submit("two")
    await conversation.wait()
    assert [e for e in events if e["type"] == "turn_end"][1]["text"] == "recovered"


async def test_reset_clears_session_and_emits_event(config, events):
    app, _ = make_app(config, script=["hello"])
    conversation = app.conversation()
    watch(conversation, events)
    await conversation.submit("hi")
    await conversation.wait()
    assert app.session().transcript()
    await conversation.reset()
    assert app.session().messages == [] and app.session().transcript() == []
    assert events[-1]["type"] == "reset"


async def test_session_survives_restart(config):
    app, _ = make_app(config, script=["stored reply"])
    conversation = app.conversation()
    await conversation.submit("remember me")
    await conversation.wait()
    fresh = Jarvis(config)
    assert [m.content for m in fresh.session().messages] == ["remember me", "stored reply"]


# --------------------------------------------------------------------------- background tasks


async def test_background_task_announces_and_injects_note_into_next_turn(config, events):
    app, provider = make_app(config, script=["Bullish outlook for the week.", "Noted, thanks."])
    conversation = app.conversation()
    watch(conversation, events)
    job = app.tasks.start("trading", "analyse BTC weekly")
    await wait_until(lambda: any(e["type"] == "announcement" for e in events))
    updates = [e for e in events if e["type"] == "task_update"]
    assert updates and updates[-1]["task"]["status"] == "done"
    assert updates[-1]["task"]["result"] == "Bullish outlook for the week."
    announcement = next(e for e in events if e["type"] == "announcement")
    assert "trading agent has finished" in announcement["text"] and announcement["task_id"] == job.id

    await conversation.submit("what happened?")
    await conversation.wait()
    prompt = provider.calls[1]["messages"][-1].content
    assert prompt.startswith("[System notes since your last reply]")
    assert f"Background task {job.id} (trading: analyse BTC weekly) reported" in prompt
    assert "Bullish outlook for the week." in prompt
    assert prompt.endswith("[User]\nwhat happened?")

    # The note is consumed: it is not repeated on the following turn.
    provider.script.append("third")
    await conversation.submit("and now?")
    await conversation.wait()
    assert "System notes" not in provider.calls[2]["messages"][-1].content


async def test_failed_background_task_is_announced_as_failed(config, events):
    app, _ = make_app(config, script=[LLMError("no model")])
    conversation = app.conversation()
    watch(conversation, events)
    app.tasks.start("research", "dig")
    await wait_until(lambda: any(e["type"] == "announcement" for e in events))
    assert "failed" in next(e for e in events if e["type"] == "announcement")["text"]
    assert any(e["type"] == "task_update" and e["task"]["status"] == "failed" for e in events)


async def test_recurring_task_repeats_and_can_be_cancelled(config, monkeypatch):
    monkeypatch.setattr("jarvis.tasks.MIN_INTERVAL_MINUTES", 0.0001)
    app, _ = make_app(config)
    updates = []
    app.subscribe(updates.append)
    job = app.tasks.start("research", "ping", interval_minutes=0.0005)  # 30 ms
    assert job.interval_minutes == 0.0005
    await wait_until(lambda: job.runs >= 2)
    assert job.result == "You said: ping"
    assert app.tasks.cancel(job.id) is True
    with contextlib.suppress(asyncio.CancelledError):
        await job.handle
    assert job.status == "cancelled"
    runs = job.runs
    await asyncio.sleep(0.12)
    assert job.runs == runs  # really stopped
    assert app.tasks.cancel(job.id) is False
    assert app.tasks.cancel("t999") is False
    assert [u["task"]["status"] for u in updates if u["type"] == "task_update"][-1] == "cancelled"
    await app.tasks.shutdown()


async def test_interval_below_minimum_is_raised_to_minimum(config, monkeypatch):
    monkeypatch.setattr("jarvis.tasks.MIN_INTERVAL_MINUTES", 5.0)
    app, _ = make_app(config)
    job = app.tasks.start("research", "ping", interval_minutes=0.01)
    assert job.interval_minutes == 5.0
    await app.tasks.shutdown()


async def test_start_background_task_skill(config):
    app, _ = make_app(config)  # rule-based: "!skill {json}" calls a skill directly
    first = await app.chat('!start_background_task {"agent": "trading", "task": "watch ETH"}')
    assert first.startswith("start_background_task returned:")
    info = json.loads(first.split("returned: ", 1)[1])
    assert info["status"] == "started" and info["task_id"] == "t1"
    second = await app.chat('!start_background_task {"agent": "ghost", "task": "x"}')
    assert "No agent named 'ghost'" in second
    await app.tasks.shutdown()


# --------------------------------------------------------------------------- core skills


async def test_switch_backend_skill_changes_and_persists_the_profile(config):
    config.set("llm.profiles.second", {"provider": "mock", "model": "mock-two"})
    app, provider = make_app(
        config, script=[tool_msg(("s1", "switch_backend", {"target": "second"})), "unused"]
    )
    changes = []
    app.subscribe(changes.append)
    assert app.router.active == "mock"
    reply = await app.chat("go to second")
    assert app.router.active == "second"
    # The follow-up request in the same turn was served by the new backend.
    assert len(provider.calls) == 1
    assert reply.startswith("switch_backend returned:") and '"active": "second"' in reply
    assert any(e["type"] == "backend" and e["active"] == "second" for e in changes)
    state = json.loads((config.data_dir / "state.json").read_text(encoding="utf-8"))
    assert state["active_backend"] == "second"
    # A brand new app on the same data dir starts on that profile.
    assert Jarvis(config).router.active == "second"


async def test_switch_backend_unknown_target_is_a_tool_error(config):
    app, provider = make_app(
        config, script=[tool_msg(("s1", "switch_backend", {"target": "warp-drive"})), "ok"]
    )
    await app.chat("switch")
    result = provider.calls[1]["messages"][-1]
    assert result.is_error and "No backend matches" in result.content
    assert app.router.active == "mock"


async def test_remember_and_recall_round_trip(config):
    app, _ = make_app(config)
    ctx = SkillContext(config, services={"jarvis": app})
    text, is_error = await app.skills.execute("remember", {"key": " Fav Colour ", "value": "blue"}, ctx)
    assert (text, is_error) == ("Saved 'fav colour'.", False)
    await app.skills.execute("remember", {"key": "pet", "value": "a cat named Tom"}, ctx)

    everything = json.loads((await app.skills.execute("recall", {}, ctx))[0])
    assert everything == {"fav colour": "blue", "pet": "a cat named Tom"}
    by_value = json.loads((await app.skills.execute("recall", {"query": "TOM"}, ctx))[0])
    assert by_value == {"pet": "a cat named Tom"}
    assert json.loads((await app.skills.execute("recall", {"query": "zzz"}, ctx))[0]) == {}

    # Notes persist across instances; an empty value forgets.
    assert Jarvis(config).notes()["fav colour"] == "blue"
    text, _ = await app.skills.execute("remember", {"key": "pet", "value": ""}, ctx)
    assert text == "Forgot 'pet'."
    assert "pet" not in app.notes()
    text, is_error = await app.skills.execute("remember", {"key": "  ", "value": "x"}, ctx)
    assert is_error and "key is required" in text


async def test_core_skills_need_a_running_instance(config):
    registry = SkillRegistry().load_all()
    text, is_error = await registry.execute("recall", {}, SkillContext(config))
    assert is_error and "needs a running Jarvis" in text


def test_message_round_trip_through_dict():
    original = Message(
        "assistant", "hi", [ToolCall("c1", "x", {"a": 1})], meta={"epoch": 3, "interrupted": True}
    )
    assert Message.from_dict(original.to_dict()) == original
    tool = Message("tool", "r", tool_call_id="c1", name="x", is_error=True)
    assert Message.from_dict(tool.to_dict()) == tool
