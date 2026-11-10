"""Command line entry point.

    jarvis                 open the desktop app (window + PC microphone)
    jarvis voice           hands-free in the terminal, no window
    jarvis chat            typed conversation in the terminal
    jarvis ask "..."       one question, one answer
    jarvis serve           local server only (use from a browser or phone)
    jarvis agents ...      list agents / run several at once
    jarvis skills ...      list, scaffold, test and reload skills
    jarvis backend [name]  show backends or switch the default one
    jarvis doctor          check that everything needed is installed
    jarvis setup-voice     download the local voice model
    jarvis init            write a starter jarvis.toml, plugins/ and agents/
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import importlib.util
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

from jarvis import __version__
from jarvis.config import Config

DIM, BOLD, RESET = "\033[2m", "\033[1m", "\033[0m"


def _style(text: str, code: str) -> str:
    return f"{code}{text}{RESET}" if sys.stdout.isatty() else text


def _load(args: argparse.Namespace) -> Config:
    return Config.load(args.config)


def _make(args: argparse.Namespace):
    from jarvis.app import Jarvis

    app = Jarvis(_load(args))
    override = getattr(args, "backend_override", None)
    if override:  # this run only; `jarvis backend NAME` changes the default
        if override not in app.router.profiles:
            raise SystemExit(f"No backend profile '{override}'. Have: {', '.join(app.router.profiles)}")
        app.router.active = override
    return app


# -- interactive front ends ---------------------------------------------------


class TerminalView:
    """Prints conversation events as they happen."""

    def __init__(self, name: str, show_deltas: bool = True):
        self.name = name
        self.show_deltas = show_deltas
        self._open = False

    def __call__(self, event: dict) -> None:
        kind = event["type"]
        if kind == "transcript":
            print(f"\n{_style('you>', BOLD)} {event['text']}")
        elif kind == "text_delta" and self.show_deltas:
            if not self._open:
                print(f"{_style(self.name.lower() + '>', BOLD)} ", end="")
                self._open = True
            print(event["text"], end="", flush=True)
        elif kind == "sentence" and not self.show_deltas:
            print(f"{_style(self.name.lower() + '>', BOLD)} {event['text']}")
        elif kind == "tool_start":
            self._break()
            who = "" if event.get("agent") == "jarvis" else f"{event.get('agent')}: "
            args = json.dumps(event.get("arguments", {}), ensure_ascii=False)
            print(_style(f"  [{who}{event['name']} {args[:120]}]", DIM))
        elif kind == "agent_start":
            self._break()
            print(_style(f"  [agent {event['agent']} started]", DIM))
        elif kind == "agent_end":
            print(_style(f"  [agent {event['agent']} {'done' if event.get('ok') else 'failed'}]", DIM))
        elif kind == "announcement":
            self._break()
            print(_style(f"  [{event['text']}]", DIM))
        elif kind == "error":
            self._break()
            print(f"  error: {event['message']}")
        elif kind == "backend":
            self._break()
            print(_style(f"  [backend: {event['active']}]", DIM))
        elif kind == "turn_end":
            self._break()
            if event.get("interrupted"):
                print(_style("  [interrupted]", DIM))

    def _break(self) -> None:
        if self._open:
            print()
            self._open = False


CHAT_HELP = """\
/backend [name]     show or switch the backend (local, cloud, or a profile name)
/agents             list specialist agents
/skills             list skills
/run AGENT TASK     run one agent directly
/reload             reload skill plugins and agent files
/reset              forget this conversation
/quit               leave"""


async def _chat(args: argparse.Namespace) -> int:
    app = _make(args)
    conversation = app.conversation(args.session)
    conversation.subscribe(TerminalView(app.config.get("assistant.name", "Jarvis")))
    print(f"{app.config.get('assistant.name')} - backend: {app.router.active}. /help for commands.")
    try:
        while True:
            try:
                line = (await asyncio.to_thread(input, _style("you> ", BOLD))).strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                continue
            if line.startswith("/"):
                if not await _chat_command(app, conversation, line):
                    break
                continue
            await conversation.submit(line, source="text", voice=False)
            try:
                await conversation.wait()
            except KeyboardInterrupt:
                await conversation.interrupt()
    finally:
        await app.aclose()
    return 0


async def _chat_command(app, conversation, line: str) -> bool:
    command, _, rest = line[1:].partition(" ")
    rest = rest.strip()
    if command in ("quit", "exit", "q"):
        return False
    if command == "help":
        print(CHAT_HELP)
    elif command == "backend":
        if rest:
            try:
                app.router.switch(rest)
            except Exception as exc:
                print(f"  {exc}")
        else:
            _print_backends(app.router.describe(), await app.router.health())
    elif command == "agents":
        for agent in app.agents.specialists():
            print(f"  {agent.name:10} {agent.description}")
    elif command == "skills":
        for skill in app.skills.all():
            print(f"  {skill.name:24} {skill.description.splitlines()[0][:80]}")
    elif command == "run":
        name, _, task = rest.partition(" ")
        if not task:
            print("  usage: /run AGENT TASK")
        else:
            results = await app.run_agents([(name, task)])
            print(results[0]["report"])
    elif command == "reload":
        print(f"  {app.skills.reload()}")
        app.agents.reload()
    elif command == "reset":
        await conversation.reset()
        print("  conversation cleared")
    else:
        print(f"  unknown command /{command}; try /help")
    return True


async def _voice(args: argparse.Namespace) -> int:
    from jarvis.voice.native import create_native_voice

    app = _make(args)
    conversation = app.conversation(args.session)
    name = app.config.get("assistant.name", "Jarvis")
    conversation.subscribe(TerminalView(name, show_deltas=False))
    try:
        voice = create_native_voice(conversation)
        await voice.start()
    except Exception as exc:
        print(f"Voice is not available: {exc}\nRun `jarvis doctor` for details.")
        await app.aclose()
        return 1
    wake = app.config.get("voice.wake_word")
    hint = f'say "{wake}" to get my attention' if wake else "just start talking"
    print(f"{name} is listening on your microphone ({hint}). Backend: {app.router.active}. Ctrl+C to quit.")
    sweeper = asyncio.create_task(app.resources.run_sweeper())
    try:
        await voice.run()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        sweeper.cancel()
        await app.aclose()
    return 0


async def _ask(args: argparse.Namespace) -> int:
    app = _make(args)
    from jarvis.core.session import Session

    try:
        reply = await app.chat(" ".join(args.text), session=Session("ask"))
        print(reply)
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        await app.aclose()


def _app(args: argparse.Namespace) -> int:
    from jarvis.desktop import launch

    return launch(
        _load(args),
        window=not getattr(args, "no_window", False),
        voice=not getattr(args, "no_voice", False),
        host=getattr(args, "host", None),
        port=getattr(args, "port", None),
    )


def _serve(args: argparse.Namespace) -> int:
    from jarvis.desktop import launch

    return launch(
        _load(args), window=False, open_browser=False, voice=args.voice, host=args.host, port=args.port
    )


# -- agents, skills, backends ---------------------------------------------------


async def _agents(args: argparse.Namespace) -> int:
    app = _make(args)
    try:
        if args.action == "list":
            for agent in app.agents.all():
                profile = agent.llm_profile or "(active backend)"
                print(f"{_style(agent.name, BOLD):20} {agent.description}")
                print(_style(f"    skills: {', '.join(agent.skills)}   backend: {profile}", DIM))
            for path, error in app.agents.errors.items():
                print(f"  invalid agent file {path}: {error}")
            return 0
        pairs = args.jobs
        if len(pairs) < 2 or len(pairs) % 2:
            print('usage: jarvis agents run AGENT "TASK" [AGENT "TASK" ...]')
            return 2
        jobs = list(zip(pairs[::2], pairs[1::2], strict=True))
        view = TerminalView("agent", show_deltas=False)

        async def emit(event: dict) -> None:
            view(event)

        results = await app.run_agents(jobs, emit=emit)
        for result in results:
            status = "ok" if result["ok"] else "FAILED"
            print(f"\n{_style(result['agent'], BOLD)} [{status}]\n{result['report']}")
        return 0 if all(r["ok"] for r in results) else 1
    finally:
        await app.aclose()


def _parse_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except ValueError:
        return raw


async def _skills(args: argparse.Namespace) -> int:
    if args.action == "new":
        return _new_skill(_load(args), args.name)
    app = _make(args)
    try:
        if args.action in ("list", "reload"):
            for skill in app.skills.all():
                tags = ",".join(skill.tags)
                print(f"{_style(skill.name, BOLD):34} [{tags}] {_style(skill.source, DIM)}")
                print(f"    {skill.description.splitlines()[0][:100]}")
            for path, error in app.skills.errors.items():
                print(f"  FAILED TO LOAD {path}: {error}")
            return 1 if app.skills.errors else 0
        # test: run one skill directly, no model involved
        from jarvis.skills.base import SkillContext

        arguments = {}
        for pair in args.args:
            key, sep, value = pair.partition("=")
            if not sep:
                print(f"arguments must be key=value (got '{pair}')")
                return 2
            arguments[key] = _parse_value(value)
        ctx = SkillContext(app.config, services=app.services())
        text, is_error = await app.skills.execute(args.name, arguments, ctx)
        print(text)
        return 1 if is_error else 0
    finally:
        await app.aclose()


def _new_skill(config: Config, name: str) -> int:
    if not name.isidentifier() or not name.islower():
        print("Skill names must be lowercase Python identifiers, e.g. weather_report")
        return 2
    directory = Path((config.get("skills.plugin_dirs") or ["./plugins"])[0]).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{name}.py"
    if target.exists():
        print(f"{target} already exists")
        return 1
    template = (Path(__file__).parent / "skills" / "templates" / "skill.py.tmpl").read_text()
    target.write_text(template.replace("__NAME__", name), encoding="utf-8")
    print(f"Created {target}\nTry it:  jarvis skills test {name} text=hello repeat=2")
    return 0


def _print_backends(description: dict, health: dict) -> None:
    for profile in description["profiles"]:
        marker = "*" if profile["name"] == description["active"] else " "
        where = "local" if profile["local"] else "cloud"
        status = health.get(profile["name"], {})
        state = "ready" if status.get("ok") else f"not ready: {status.get('detail', '')}"
        print(f" {marker} {profile['name']:12} {where:6} {profile['provider']:10} {profile['model']:28} {state}")


async def _backend(args: argparse.Namespace) -> int:
    app = _make(args)
    try:
        if args.name:
            try:
                app.router.switch(args.name)
            except Exception as exc:
                print(exc)
                return 1
            print(f"Default backend is now '{app.router.active}'.")
        _print_backends(app.router.describe(), await app.router.health())
        return 0
    finally:
        await app.aclose()


# -- setup helpers ----------------------------------------------------------------


async def _doctor(args: argparse.Namespace) -> int:
    app = _make(args)
    problems = 0

    def row(ok: bool | None, label: str, detail: str = "") -> None:
        nonlocal problems
        mark = {True: "ok ", False: "!! ", None: "-- "}[ok]
        problems += ok is False
        print(f"  {mark} {label:28} {detail}")

    try:
        print(f"Jarvis {__version__} - Python {sys.version.split()[0]} - config: {app.config.path or 'defaults'}")
        print("\nBackends")
        health = await app.router.health()
        description = app.router.describe()
        for profile in description["profiles"]:
            status = health[profile["name"]]
            active = profile["name"] == description["active"]
            label = f"{profile['name']}{' (active)' if active else ''}"
            row(status["ok"] if active else (True if status["ok"] else None), label,
                status["detail"] or f"{profile['provider']} {profile['model']}")

        print("\nVoice")
        has = lambda module: importlib.util.find_spec(module) is not None  # noqa: E731
        row(True if has("sounddevice") else None, "microphone (sounddevice)",
            "" if has("sounddevice") else "pip install 'jarvis-assistant[voice]'")
        row(True if has("faster_whisper") else None, "speech-to-text (Whisper)",
            "" if has("faster_whisper") else "pip install 'jarvis-assistant[voice]'")
        from jarvis.voice.tts import SystemTTS, find_piper_voice

        voice = find_piper_voice(app.config) if has("piper") else None
        row(True if voice else None, "text-to-speech (Piper)",
            str(voice) if voice else "run: jarvis setup-voice")
        row(True if SystemTTS.available() else None, "text-to-speech (OS voice)")
        if has("sounddevice"):
            try:
                import sounddevice

                default_in = sounddevice.query_devices(kind="input")["name"]
                row(True, "default microphone", default_in)
            except Exception as exc:
                row(False, "default microphone", str(exc))

        print("\nDesktop and tools")
        row(True if has("webview") else None, "native window (pywebview)",
            "" if has("webview") else "optional: pip install 'jarvis-assistant[desktop]'")
        row(True if shutil.which("ffmpeg") else None, "ffmpeg (video agent)",
            "" if shutil.which("ffmpeg") else "install ffmpeg to use the video agent")
        row(not app.skills.errors, f"skills loaded: {len(app.skills.all())}",
            "; ".join(app.skills.errors) or "")
        row(not app.agents.errors, f"agents loaded: {len(app.agents.all())}",
            "; ".join(app.agents.errors) or "")
        print(f"\n{'All required pieces are in place.' if not problems else f'{problems} problem(s) need attention.'}")
        return 1 if problems else 0
    finally:
        await app.aclose()


async def _setup_voice(args: argparse.Namespace) -> int:
    config = _load(args)
    missing = [m for m in ("sounddevice", "faster_whisper", "piper") if not importlib.util.find_spec(m)]
    if missing:
        print("Install the voice packages first:\n    pip install 'jarvis-assistant[voice]'")
        return 1
    from jarvis.voice.tts import download_piper_voice

    print(f"Downloading voice '{args.voice}' ...")
    try:
        path = await download_piper_voice(config, args.voice)
    except Exception as exc:
        print(f"Download failed: {exc}")
        return 1
    print(f"Voice saved to {path}")
    print("The speech recognition model downloads itself the first time you speak.")
    return 0


def _init(args: argparse.Namespace) -> int:
    example = Path(__file__).parent / "config.example.toml"
    target = Path("jarvis.toml")
    if target.exists():
        print("jarvis.toml already exists; leaving it alone.")
    else:
        shutil.copyfile(example, target)
        print("Wrote jarvis.toml")
    for folder in ("plugins", "agents"):
        Path(folder).mkdir(exist_ok=True)
    print("Created plugins/ and agents/ - drop skill files and agent definitions there.")
    return 0


# -- argument parsing ---------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jarvis", description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__doc__.split("\n\n", 1)[1])
    parser.add_argument("--version", action="version", version=f"jarvis {__version__}")
    parser.add_argument("-c", "--config", help="path to a jarvis.toml")
    parser.add_argument("-b", "--backend", dest="backend_override", metavar="PROFILE",
                        help="use this backend profile for this run")
    parser.add_argument("-v", "--verbose", action="store_true", help="log more")
    sub = parser.add_subparsers(dest="command")

    app = sub.add_parser("app", help="open the desktop app (default)")
    app.add_argument("--no-voice", action="store_true", help="do not open the microphone")
    app.add_argument("--no-window", action="store_true", help="use the browser instead of a window")
    app.add_argument("--port", type=int)
    app.add_argument("--host")

    for name, text in (("voice", "hands-free in the terminal"), ("chat", "typed chat in the terminal")):
        cmd = sub.add_parser(name, help=text)
        cmd.add_argument("--session", default="main", help="conversation to continue")

    ask = sub.add_parser("ask", help="ask one question")
    ask.add_argument("text", nargs="+")

    serve = sub.add_parser("serve", help="run the local server without a window")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    serve.add_argument("--voice", action="store_true", help="also listen on this machine's microphone")

    agents = sub.add_parser("agents", help="list agents or run several concurrently")
    agents.add_argument("action", choices=["list", "run"])
    agents.add_argument("jobs", nargs="*", metavar="AGENT TASK")

    skills = sub.add_parser("skills", help="list, create, test or reload skills")
    skills.add_argument("action", choices=["list", "new", "test", "reload"])
    skills.add_argument("name", nargs="?")
    skills.add_argument("args", nargs="*", metavar="key=value")

    backend = sub.add_parser("backend", help="show backends or set the default")
    backend.add_argument("name", nargs="?")

    sub.add_parser("doctor", help="check the installation")
    setup = sub.add_parser("setup-voice", help="download the local voice")
    setup.add_argument("--voice", default="en_US-lessac-medium", help="Piper voice name")
    sub.add_parser("init", help="write a starter config in this folder")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    command = args.command or "app"
    if command in ("skills",) and args.action in ("new", "test") and not args.name:
        parser.error(f"skills {args.action} needs a skill name")
    sync_commands = {"app": _app, "serve": _serve, "init": _init}
    async_commands = {
        "chat": _chat, "voice": _voice, "ask": _ask, "agents": _agents, "skills": _skills,
        "backend": _backend, "doctor": _doctor, "setup-voice": _setup_voice,
    }  # fmt: skip
    try:
        if command in sync_commands:
            return sync_commands[command](args)
        with contextlib.suppress(KeyboardInterrupt):
            return asyncio.run(async_commands[command](args))
        return 130
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
