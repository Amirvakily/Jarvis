"""Model backends: OpenAI-compatible, Ollama, Anthropic, mock and the router."""

from __future__ import annotations

import json
from types import SimpleNamespace

import anthropic
import pytest

from jarvis._http import httpx
from jarvis.config import Config
from jarvis.core.types import LLMError, LLMResponse, Message, TextDelta, ToolCall, ToolSpec
from jarvis.llm.anthropic_provider import (
    FALLBACK_BETA,
    AnthropicProvider,
    to_anthropic_messages,
)
from jarvis.llm.mock import MockProvider
from jarvis.llm.ollama import OllamaProvider
from jarvis.llm.openai_compat import OpenAICompatProvider
from jarvis.llm.router import LLMRouter
from jarvis.resources import ResourceManager

TOOL = ToolSpec(
    "get_weather",
    "Weather lookup",
    {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
)
HISTORY = [
    Message("user", "weather in Paris?"),
    Message("assistant", "Checking.", [ToolCall("call_1", "get_weather", {"city": "Paris"})]),
    Message("tool", "sunny", tool_call_id="call_1", name="get_weather"),
]


async def collect(provider, **kwargs):
    kwargs.setdefault("system", "be brief")
    kwargs.setdefault("messages", [Message("user", "hi")])
    kwargs.setdefault("tools", [])
    return [e async for e in provider.stream(**kwargs)]


def sse(*chunks) -> bytes:
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks)
    return (body + "data: [DONE]\n\n").encode()


def ndjson(*chunks) -> bytes:
    return ("\n".join(json.dumps(c) for c in chunks) + "\n").encode()


# --------------------------------------------------------------------------
# OpenAI-compatible
# --------------------------------------------------------------------------
def make_openai(handler, **settings):
    settings = {"provider": "openai", "model": "gpt-x", "base_url": "http://llm.test/v1", **settings}
    return OpenAICompatProvider(
        "oa", settings, Config(), transport=httpx.MockTransport(handler)
    )


def delta(**kw):
    return {"choices": [{"delta": kw, "finish_reason": None}]}


def finish(reason, **extra):
    return {"choices": [{"delta": {}, "finish_reason": reason}], **extra}


async def test_openai_request_body_and_auth(monkeypatch):
    monkeypatch.setenv("MY_KEY", "sekret")
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, content=sse(delta(content="ok"), finish("stop")))

    provider = make_openai(handler, api_key_env="MY_KEY", max_tokens=123)
    history = [*HISTORY, Message("assistant", "It is sunny."), Message("user", "thanks")]
    await collect(provider, messages=history, tools=[TOOL])

    assert seen["url"] == "http://llm.test/v1/chat/completions"
    assert seen["auth"] == "Bearer sekret"
    body = seen["body"]
    assert body["model"] == "gpt-x" and body["stream"] is True
    assert body["max_tokens"] == 123
    msgs = body["messages"]
    assert msgs[0] == {"role": "system", "content": "be brief"}
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "tool", "assistant", "user"]
    call = msgs[2]["tool_calls"][0]
    assert call["id"] == "call_1" and call["type"] == "function"
    assert call["function"]["name"] == "get_weather"
    assert json.loads(call["function"]["arguments"]) == {"city": "Paris"}
    assert isinstance(call["function"]["arguments"], str)
    assert msgs[3] == {"role": "tool", "tool_call_id": "call_1", "content": "sunny"}
    assert body["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Weather lookup",
                "parameters": TOOL.parameters,
            },
        }
    ]


async def test_openai_no_auth_header_when_env_unset(monkeypatch):
    monkeypatch.delenv("MY_KEY", raising=False)
    seen = {}

    def handler(request):
        seen["headers"] = request.headers
        return httpx.Response(200, content=sse(finish("stop")))

    await collect(make_openai(handler, api_key_env="MY_KEY"))
    assert "authorization" not in seen["headers"]


async def test_openai_max_tokens_argument_overrides_setting():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, content=sse(finish("stop")))

    await collect(make_openai(handler, max_tokens=5), max_tokens=77)
    assert seen["body"]["max_tokens"] == 77


async def test_openai_text_deltas_and_usage():
    def handler(request):
        return httpx.Response(
            200,
            content=sse(
                delta(content="Hel"),
                delta(content="lo "),
                delta(content="world"),
                finish("stop"),
                {"choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": 3}},
            ),
        )

    events = await collect(make_openai(handler))
    assert [e.text for e in events[:-1]] == ["Hel", "lo ", "world"]
    final = events[-1]
    assert isinstance(final, LLMResponse)
    assert final.message.content == "Hello world"
    assert final.stop_reason == "end"
    assert final.usage == {"input_tokens": 11, "output_tokens": 3}
    assert final.model == "gpt-x"


async def test_openai_split_and_parallel_tool_calls():
    def handler(request):
        return httpx.Response(
            200,
            content=sse(
                delta(tool_calls=[{"index": 0, "id": "a", "function": {"name": "get_weather", "arguments": ""}}]),
                delta(tool_calls=[{"index": 1, "id": "b", "function": {"name": "get_time", "arguments": ""}}]),
                delta(tool_calls=[{"index": 0, "function": {"arguments": '{"city": '}}]),
                delta(tool_calls=[{"index": 1, "function": {"arguments": '{"tz":'}}]),
                delta(tool_calls=[{"index": 0, "function": {"arguments": '"Paris"}'}}]),
                delta(tool_calls=[{"index": 1, "function": {"arguments": ' "UTC"}'}}]),
                finish("tool_calls"),
            ),
        )

    final = (await collect(make_openai(handler), tools=[TOOL]))[-1]
    assert final.stop_reason == "tool_use"
    assert [(c.id, c.name, c.arguments) for c in final.message.tool_calls] == [
        ("a", "get_weather", {"city": "Paris"}),
        ("b", "get_time", {"tz": "UTC"}),
    ]


async def test_openai_finish_length_is_max_tokens():
    def handler(request):
        return httpx.Response(200, content=sse(delta(content="cut"), finish("length")))

    assert (await collect(make_openai(handler)))[-1].stop_reason == "max_tokens"


async def test_openai_malformed_arguments_marked_invalid():
    def handler(request):
        return httpx.Response(
            200,
            content=sse(
                delta(tool_calls=[{"index": 0, "id": "a", "function": {"name": "t", "arguments": '{"x": '}}]),
                finish("tool_calls"),
            ),
        )

    final = (await collect(make_openai(handler)))[-1]
    assert final.message.tool_calls[0].arguments == {"INVALID_JSON": '{"x": '}


async def test_openai_http_500_retryable():
    provider = make_openai(lambda r: httpx.Response(500, content=b"oops"))
    with pytest.raises(LLMError) as err:
        await collect(provider)
    assert err.value.retryable is True


async def test_openai_http_401_not_retryable():
    provider = make_openai(lambda r: httpx.Response(401, content=b"nope"))
    with pytest.raises(LLMError) as err:
        await collect(provider)
    assert err.value.retryable is False


async def test_openai_connection_error_retryable():
    def handler(request):
        raise httpx.ConnectError("boom")

    with pytest.raises(LLMError) as err:
        await collect(make_openai(handler))
    assert err.value.retryable is True


# --------------------------------------------------------------------------
# Ollama
# --------------------------------------------------------------------------
def make_ollama(handler, **settings):
    settings = {
        "provider": "ollama",
        "model": "llama3.1:8b",
        "base_url": "http://ollama.test",
        "keep_alive": "7m",
        "num_ctx": 4096,
        **settings,
    }
    return OllamaProvider("loc", settings, Config(), transport=httpx.MockTransport(handler))


async def test_ollama_request_body():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, content=ndjson({"message": {"content": "x"}, "done": True}))

    history = [*HISTORY, Message("user", "ok")]
    await collect(make_ollama(handler), messages=history, tools=[TOOL], max_tokens=50)
    body = seen["body"]
    assert seen["url"] == "http://ollama.test/api/chat"
    assert body["model"] == "llama3.1:8b"
    assert body["stream"] is True
    assert body["keep_alive"] == "7m"
    assert body["options"] == {"num_ctx": 4096, "num_predict": 50}
    assert body["tools"][0]["function"]["name"] == "get_weather"
    msgs = body["messages"]
    assert msgs[0] == {"role": "system", "content": "be brief"}
    # Ollama takes object arguments, not JSON strings.
    assert msgs[2]["tool_calls"] == [
        {"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}
    ]
    assert msgs[3] == {"role": "tool", "content": "sunny", "tool_name": "get_weather"}


async def test_ollama_streaming_text_and_usage():
    def handler(request):
        return httpx.Response(
            200,
            content=ndjson(
                {"message": {"content": "Hi "}, "done": False},
                {"message": {"content": "there"}, "done": False},
                {"message": {"content": ""}, "done": True, "done_reason": "stop",
                 "prompt_eval_count": 9, "eval_count": 4},
            ),
        )

    events = await collect(make_ollama(handler))
    assert [e.text for e in events[:-1]] == ["Hi ", "there"]
    final = events[-1]
    assert final.message.content == "Hi there"
    assert final.stop_reason == "end"
    assert final.usage == {"input_tokens": 9, "output_tokens": 4}


async def test_ollama_tool_call_chunk():
    def handler(request):
        return httpx.Response(
            200,
            content=ndjson(
                {"message": {"content": "", "tool_calls": [
                    {"function": {"name": "get_weather", "arguments": {"city": "Rome"}}}]},
                 "done": False},
                {"message": {"content": ""}, "done": True, "done_reason": "stop"},
            ),
        )

    final = (await collect(make_ollama(handler), tools=[TOOL]))[-1]
    assert final.stop_reason == "tool_use"
    (call,) = final.message.tool_calls
    assert call.name == "get_weather" and call.arguments == {"city": "Rome"}
    assert call.id.startswith("call_")


async def test_ollama_done_reason_length():
    def handler(request):
        return httpx.Response(
            200, content=ndjson({"message": {"content": "a"}, "done": True, "done_reason": "length"})
        )

    assert (await collect(make_ollama(handler)))[-1].stop_reason == "max_tokens"


async def test_ollama_404_mentions_pull():
    provider = make_ollama(lambda r: httpx.Response(404, content=b"model not found"))
    with pytest.raises(LLMError) as err:
        await collect(provider)
    assert "ollama pull llama3.1:8b" in str(err.value)
    assert err.value.retryable is False


async def test_ollama_connection_failure_retryable():
    def handler(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(LLMError) as err:
        await collect(make_ollama(handler))
    assert err.value.retryable is True


async def test_ollama_load_and_unload_post_keep_alive():
    posts = []

    def handler(request):
        posts.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200, json={})

    provider = make_ollama(handler)
    await provider.load()
    await provider.unload()
    assert posts == [
        ("/api/generate", {"model": "llama3.1:8b", "keep_alive": "7m"}),
        ("/api/generate", {"model": "llama3.1:8b", "keep_alive": 0}),
    ]


async def test_ollama_health_states():
    ok = make_ollama(lambda r: httpx.Response(200, json={"models": [{"name": "llama3.1:8b"}]}))
    assert (await ok.health())["ok"] is True

    missing = make_ollama(lambda r: httpx.Response(200, json={"models": [{"name": "other:1b"}]}))
    result = await missing.health()
    assert result["ok"] is False and "ollama pull llama3.1:8b" in result["detail"]

    def boom(request):
        raise httpx.ConnectError("down")

    down = await make_ollama(boom).health()
    assert down["ok"] is False and "not reachable" in down["detail"]


# --------------------------------------------------------------------------
# Anthropic
# --------------------------------------------------------------------------
def text_block(text):
    return SimpleNamespace(type="text", text=text)


def thinking_block():
    return SimpleNamespace(type="thinking", thinking="", signature="sig")


def tool_block(id_="tu_1", name="get_weather", input_=None):
    return SimpleNamespace(type="tool_use", id=id_, name=name, input=input_ or {"city": "Paris"})


def final_message(content, stop_reason="end_turn"):
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=2),
        model="claude-served",
    )


class FakeStream:
    def __init__(self, outcome):
        self.outcome = outcome
        self._events = []

    async def __aenter__(self):
        if isinstance(self.outcome, Exception):
            raise self.outcome
        self._events = [
            SimpleNamespace(type="text", text=b.text) for b in self.outcome.content if b.type == "text"
        ]
        return self

    async def __aexit__(self, *exc):
        return False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._events:
            raise StopAsyncIteration
        return self._events.pop(0)

    async def get_final_message(self):
        return self.outcome


class FakeAPI:
    def __init__(self, owner, label):
        self.owner, self.label = owner, label

    def stream(self, **params):
        self.owner.calls.append((self.label, params))
        return FakeStream(self.owner.outcomes.pop(0))


class FakeAnthropic:
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls: list[tuple[str, dict]] = []
        self.messages = FakeAPI(self, "plain")
        self.beta = SimpleNamespace(messages=FakeAPI(self, "beta"))


def make_anthropic(*outcomes, **settings):
    fake = FakeAnthropic(*outcomes)
    settings = {"provider": "anthropic", "model": "claude-test", **settings}
    return AnthropicProvider("cl", settings, Config(), client=fake), fake


def status_error(cls, status, message="bad"):
    request = httpx.Request("POST", "https://x")
    return cls(message, response=httpx.Response(status, request=request), body=None)


def test_anthropic_merges_consecutive_tool_results():
    msgs = [
        Message("user", "go"),
        Message("assistant", "", [ToolCall("a", "t", {}), ToolCall("b", "t", {})]),
        Message("tool", "r1", tool_call_id="a"),
        Message("tool", "boom", tool_call_id="b", is_error=True),
        Message("tool", "", tool_call_id="c"),
    ]
    out = to_anthropic_messages(msgs)
    assert [m["role"] for m in out] == ["user", "assistant", "user"]
    results = out[2]["content"]
    assert [r["tool_use_id"] for r in results] == ["a", "b", "c"]
    assert "is_error" not in results[0]
    assert results[1]["is_error"] is True
    assert results[2]["content"] == "(no output)"


def test_anthropic_native_blocks_replayed_or_rebuilt():
    blocks = [
        {"type": "thinking", "thinking": "hm", "signature": "sig"},
        {"type": "text", "text": "Checking."},
        {"type": "tool_use", "id": "tu_1", "name": "get_weather", "input": {"city": "Paris"}},
    ]
    msg = Message(
        "assistant", "Checking.", [ToolCall("tu_1", "get_weather", {"city": "Paris"})],
        meta={"anthropic": blocks},
    )
    native = to_anthropic_messages([msg], native=True)
    assert native == [{"role": "assistant", "content": blocks}]
    rebuilt = to_anthropic_messages([msg], native=False)
    assert rebuilt[0]["content"] == [
        {"type": "text", "text": "Checking."},
        {"type": "tool_use", "id": "tu_1", "name": "get_weather", "input": {"city": "Paris"}},
    ]


def test_anthropic_drops_empty_messages():
    out = to_anthropic_messages(
        [Message("user", "hi"), Message("assistant", "  "), Message("user", "  "), Message("user", "yo")]
    )
    assert out == [{"role": "user", "content": "hi"}, {"role": "user", "content": "yo"}]


async def test_anthropic_request_params_with_fallback():
    provider, fake = make_anthropic(
        final_message([text_block("hi")]), effort="low", refusal_fallback=True
    )
    await collect(provider, tools=[TOOL])
    label, params = fake.calls[0]
    assert label == "beta"
    assert params["model"] == "claude-test"
    assert params["thinking"] == {"type": "adaptive"}
    assert params["output_config"] == {"effort": "low"}
    assert params["betas"] == [FALLBACK_BETA]
    assert params["fallbacks"] == "default"
    assert params["system"] == "be brief"
    assert "temperature" not in params
    (tool,) = params["tools"]
    assert tool["input_schema"] == TOOL.parameters
    assert tool["eager_input_streaming"] is True


async def test_anthropic_request_params_without_fallback():
    provider, fake = make_anthropic(final_message([text_block("hi")]), refusal_fallback=False)
    await collect(provider, tools=[TOOL])
    label, params = fake.calls[0]
    assert label == "plain"
    assert "betas" not in params and "fallbacks" not in params
    assert "temperature" not in params


async def test_anthropic_normal_text_reply():
    provider, _ = make_anthropic(final_message([text_block("Hello!")]))
    events = await collect(provider)
    assert [e.text for e in events[:-1]] == ["Hello!"]
    final = events[-1]
    assert final.message.content == "Hello!"
    assert final.stop_reason == "end"
    assert final.usage["input_tokens"] == 10 and final.usage["output_tokens"] == 5
    assert final.model == "claude-served"


async def test_anthropic_tool_use_reply():
    provider, _ = make_anthropic(
        final_message([thinking_block(), text_block("Let me look."), tool_block()], "tool_use")
    )
    final = (await collect(provider, tools=[TOOL]))[-1]
    assert final.stop_reason == "tool_use"
    (call,) = final.message.tool_calls
    assert (call.id, call.name, call.arguments) == ("tu_1", "get_weather", {"city": "Paris"})
    assert final.message.meta["anthropic"] == [
        {"type": "thinking", "thinking": "", "signature": "sig"},
        {"type": "text", "text": "Let me look."},
        {"type": "tool_use", "id": "tu_1", "name": "get_weather", "input": {"city": "Paris"}},
    ]


async def test_anthropic_refusal():
    provider, _ = make_anthropic(final_message([tool_block()], "refusal"))
    events = await collect(provider)
    final = events[-1]
    assert final.stop_reason == "refusal"
    assert final.message.tool_calls == []
    assert final.message.content
    assert isinstance(events[0], TextDelta)


async def test_anthropic_max_tokens_discards_tool_calls():
    provider, _ = make_anthropic(
        final_message([text_block("partial"), tool_block()], "max_tokens")
    )
    final = (await collect(provider, tools=[TOOL]))[-1]
    assert final.stop_reason == "max_tokens"
    assert final.message.tool_calls == []


async def test_anthropic_bad_request_retries_without_native_blocks():
    history = [
        Message("user", "q"),
        Message(
            "assistant", "a", meta={"anthropic": [
                {"type": "thinking", "thinking": "x", "signature": "s"},
                {"type": "text", "text": "a"},
            ]},
        ),
        Message("user", "follow up"),
    ]
    provider, fake = make_anthropic(
        status_error(anthropic.BadRequestError, 400, "invalid signature"),
        final_message([text_block("ok")]),
    )
    final = (await collect(provider, messages=history))[-1]
    assert final.message.content == "ok"
    assert len(fake.calls) == 2

    def kinds(params):
        return [
            b["type"]
            for m in params["messages"]
            if isinstance(m["content"], list)
            for b in m["content"]
        ]

    assert "thinking" in kinds(fake.calls[0][1])
    assert "thinking" not in kinds(fake.calls[1][1])


async def test_anthropic_bad_request_without_native_blocks_not_retried():
    provider, fake = make_anthropic(status_error(anthropic.BadRequestError, 400, "nope"))
    with pytest.raises(LLMError) as err:
        await collect(provider)
    assert len(fake.calls) == 1
    assert err.value.retryable is False


@pytest.mark.parametrize(
    ("error", "retryable"),
    [
        (status_error(anthropic.AuthenticationError, 401), False),
        (status_error(anthropic.RateLimitError, 429), True),
        (status_error(anthropic.InternalServerError, 500), True),
        (status_error(anthropic.PermissionDeniedError, 403), False),
        (anthropic.APIConnectionError(request=httpx.Request("POST", "https://x")), True),
    ],
)
async def test_anthropic_errors_map_to_llm_error(error, retryable):
    provider, _ = make_anthropic(error)
    with pytest.raises(LLMError) as err:
        await collect(provider)
    assert err.value.retryable is retryable


# --------------------------------------------------------------------------
# Mock
# --------------------------------------------------------------------------
async def test_mock_scripted_replies_in_order():
    provider = MockProvider(script=["one", Message("assistant", "two"), "three"])
    texts = [(await collect(provider))[-1].message.content for _ in range(3)]
    assert texts == ["one", "two", "three"]
    with pytest.raises(LLMError):
        await collect(provider)


async def test_mock_scripted_exception_and_callable():
    provider = MockProvider(script=[LLMError("x", retryable=True), lambda msgs: f"n={len(msgs)}"])
    with pytest.raises(LLMError):
        await collect(provider)
    assert (await collect(provider))[-1].message.content == "n=1"


async def test_mock_skill_rule():
    tool = ToolSpec("note", "n", {"type": "object"})
    provider = MockProvider()
    final = (await collect(
        provider, messages=[Message("user", '!note {"text": "milk"}')], tools=[tool]
    ))[-1]
    assert final.stop_reason == "tool_use"
    assert final.message.tool_calls[0].name == "note"
    assert final.message.tool_calls[0].arguments == {"text": "milk"}
    # Unknown tool names fall through to echo.
    echoed = (await collect(provider, messages=[Message("user", "!note {}")], tools=[]))[-1]
    assert echoed.stop_reason == "end"


async def test_mock_agent_delegation_rule():
    tool = ToolSpec("delegate", "d", {"type": "object"})
    final = (await collect(
        MockProvider(), messages=[Message("user", "@coder fix the bug")], tools=[tool]
    ))[-1]
    assert final.message.tool_calls[0].arguments == {"agent": "coder", "task": "fix the bug"}


async def test_mock_echo_and_tool_result_summary():
    provider = MockProvider()
    assert (await collect(provider, messages=[Message("user", "hello")]))[-1].message.content == "You said: hello"
    msgs = [Message("user", "x"), Message("tool", "42", tool_call_id="1", name="calc")]
    assert (await collect(provider, messages=msgs))[-1].message.content == "calc returned: 42"


# --------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------
def make_config(profiles=None, **llm):
    base = {"mock": {"provider": "mock", "model": "mock"}}
    config = Config({"llm": {"active": "mock", "profiles": profiles or base, **llm}})
    # Config deep-merges the built-in profiles; keep only the ones given.
    wanted = profiles or base
    config.data["llm"]["profiles"] = {k: v for k, v in config.data["llm"]["profiles"].items() if k in wanted}
    return config


def make_router(config=None, resources=None):
    return LLMRouter(config or make_config(), resources)


def scripted(profile, script, **settings):
    settings = {"provider": "mock", "model": profile, **settings}
    return MockProvider(profile, settings, None, script=script)


def test_router_unknown_active_raises():
    with pytest.raises(LLMError):
        LLMRouter(Config({"llm": {"active": "nope"}}))


def test_router_switch_by_name_and_keywords():
    config = make_config(
        {
            "mock": {"provider": "mock", "model": "mock"},
            "loc": {"provider": "ollama", "model": "m"},
            "flag": {"provider": "mock", "model": "f", "local": True},
            "api": {"provider": "openai", "model": "gpt"},
        }
    )
    router = make_router(config)
    heard = []
    router.on_switch(heard.append)

    assert router.switch("api")["profile"] == "api"
    assert router.switch("local")["profile"] == "loc"
    assert router.active == "loc"
    assert router.switch("cloud")["profile"] == "api"
    assert heard == ["api", "loc", "api"]
    with pytest.raises(LLMError):
        router.switch("does-not-exist")
    assert router.active == "api"


def test_router_local_flag_counts_as_local():
    config = make_config(
        {
            "mock": {"provider": "mock", "model": "mock"},
            "flag": {"provider": "openai", "model": "f", "local": True},
        }
    )
    assert make_router(config).switch("local")["profile"] == "flag"


async def test_router_stream_active_and_explicit_profile():
    router = make_router()
    router.add_provider(scripted("mock", ["from active"]))
    router.add_provider(scripted("other", ["from other"]))
    events = [e async for e in router.stream(system="", messages=[Message("user", "x")], tools=[])]
    assert events[-1].message.content == "from active"
    events = [
        e async for e in router.stream(
            system="", messages=[Message("user", "x")], tools=[], profile="other"
        )
    ]
    assert events[-1].message.content == "from other"


async def _run(router, **kw):
    return [e async for e in router.stream(system="", messages=[Message("user", "x")], tools=[], **kw)]


async def test_router_fallback_on_retryable_error_before_output():
    router = make_router(make_config(fallback="backup"))
    primary = scripted("mock", [LLMError("down", retryable=True)])
    backup = scripted("backup", ["rescued"])
    router.add_provider(primary)
    router.add_provider(backup)
    events = await _run(router)
    assert events[-1].message.content == "rescued"
    assert len(primary.calls) == 1 and len(backup.calls) == 1


async def test_router_no_fallback_on_non_retryable_error():
    router = make_router(make_config(fallback="backup"))
    router.add_provider(scripted("mock", [LLMError("bad key", retryable=False)]))
    backup = scripted("backup", ["rescued"])
    router.add_provider(backup)
    with pytest.raises(LLMError, match="bad key"):
        await _run(router)
    assert backup.calls == []


async def test_router_no_fallback_after_output_started():
    class Partial(MockProvider):
        async def stream(self, **kw):
            yield TextDelta("partial ")
            raise LLMError("dropped", retryable=True)

    router = make_router(make_config(fallback="backup"))
    router.add_provider(Partial("mock", {"provider": "mock", "model": "m"}, None))
    backup = scripted("backup", ["rescued"])
    router.add_provider(backup)
    seen = []
    with pytest.raises(LLMError, match="dropped"):
        async for event in router.stream(system="", messages=[Message("user", "x")], tools=[]):
            seen.append(event)
    assert [e.text for e in seen] == ["partial "]
    assert backup.calls == []


async def test_router_last_candidate_error_propagates():
    router = make_router(make_config(fallback="backup"))
    router.add_provider(scripted("mock", [LLMError("a", retryable=True)]))
    router.add_provider(scripted("backup", [LLMError("b", retryable=True)]))
    with pytest.raises(LLMError, match="b"):
        await _run(router)


async def test_router_local_provider_managed_by_resources():
    resources = ResourceManager()
    router = make_router(resources=resources)
    local = scripted("mock", ["hello there"], local=True, memory_mb=1234)
    router.add_provider(local)

    status = {c["name"]: c for c in resources.status()["components"]}
    assert status["llm:mock"]["memory_mb"] == 1234
    assert status["llm:mock"]["loaded"] is False

    mid = []
    async for event in router.stream(system="", messages=[Message("user", "x")], tools=[]):
        if isinstance(event, TextDelta):
            mid.append(resources.status()["components"][0].copy())
    assert mid and mid[0]["loaded"] and mid[0]["in_use"]
    after = resources.status()["components"][0]
    assert after["loaded"] is True and after["in_use"] is False


async def test_router_remote_provider_not_registered():
    resources = ResourceManager()
    router = make_router(resources=resources)
    router.add_provider(scripted("mock", ["x"]))
    assert "llm:mock" not in resources


async def test_router_releases_resource_when_provider_fails():
    resources = ResourceManager()
    router = make_router(resources=resources)
    router.add_provider(scripted("mock", [LLMError("x")], local=True))
    with pytest.raises(LLMError):
        await _run(router)
    assert resources.status()["components"][0]["in_use"] is False
