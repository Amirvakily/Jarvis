"""Desktop launcher: Jarvis as an app on your PC.

Everything runs in this one process on your machine: the assistant, the
voice loop on your microphone and speakers, and a small local server that
the window talks to. The window is a native one (via pywebview) when that
package is installed; otherwise your default browser is opened instead.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import socket
import threading
import webbrowser
from typing import Any

from jarvis.app import Jarvis
from jarvis.config import Config

log = logging.getLogger(__name__)


def _port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, port))
            return True
        except OSError:
            return False


def _free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((host, 0))
        return probe.getsockname()[1]


async def start_native_voice(jarvis: Jarvis) -> tuple[Any, str]:
    """Try to bring up microphone/speaker voice. Returns ``(voice, note)``;
    ``voice`` is None when unavailable and ``note`` says why."""
    mode = jarvis.config.get("voice.mode", "auto")
    if mode in ("off", "browser"):
        return None, f"voice.mode is '{mode}'"
    from jarvis.voice.native import create_native_voice

    try:
        voice = create_native_voice(jarvis.conversation())
        await voice.start()
        return voice, "listening on the PC microphone"
    except Exception as exc:  # missing engine, no microphone, driver trouble...
        return None, str(exc)


class Runtime(threading.Thread):
    """Runs the assistant, the voice loop and the local server on a
    background event loop so the main thread stays free for the window."""

    def __init__(self, config: Config, *, host: str, port: int, voice: bool = True):
        super().__init__(name="jarvis-runtime", daemon=True)
        self.config = config
        self.host = host
        self.port = port
        self.want_voice = voice
        self.ready = threading.Event()
        self.error: BaseException | None = None
        self.voice_note = ""
        self.token = ""
        self._server: Any = None
        self._loop: asyncio.AbstractEventLoop | None = None

    @property
    def url(self) -> str:
        host = "127.0.0.1" if self.host in ("0.0.0.0", "") else self.host
        suffix = f"/?token={self.token}" if self.token else "/"
        return f"http://{host}:{self.port}{suffix}"

    def run(self) -> None:
        try:
            asyncio.run(self._main())
        except BaseException as exc:
            self.error = exc
        finally:
            self.ready.set()

    async def _main(self) -> None:
        import uvicorn

        from jarvis.server.app import create_app

        self._loop = asyncio.get_running_loop()
        jarvis = Jarvis(self.config)
        native, self.voice_note = (None, "voice disabled")
        if self.want_voice:
            native, self.voice_note = await start_native_voice(jarvis)
        app = create_app(jarvis, native)
        self.token = app.state.token
        self._server = uvicorn.Server(
            uvicorn.Config(app, host=self.host, port=self.port, log_level="warning")
        )
        jobs = [asyncio.create_task(jarvis.resources.run_sweeper())]
        if native is not None:
            jobs.append(asyncio.create_task(native.run()))
        jobs.append(asyncio.create_task(self._signal_ready()))
        try:
            await self._server.serve()
        finally:
            for job in jobs:
                job.cancel()
            await asyncio.gather(*jobs, return_exceptions=True)
            await jarvis.aclose()

    async def _signal_ready(self) -> None:
        while not self._server.started:
            await asyncio.sleep(0.05)
        self.ready.set()

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        self.join(timeout=10)


def launch(
    config: Config,
    *,
    window: bool = True,
    open_browser: bool = True,
    voice: bool = True,
    host: str | None = None,
    port: int | None = None,
) -> int:
    host = host or config.get("server.host", "127.0.0.1")
    port = port or int(config.get("server.port", 8765))
    if not _port_free(host, port):
        fallback = _free_port(host)
        print(f"Port {port} is in use; using {fallback} instead.")
        port = fallback

    runtime = Runtime(config, host=host, port=port, voice=voice)
    runtime.start()
    runtime.ready.wait(timeout=60)
    if runtime.error is not None:
        print(f"Jarvis could not start: {runtime.error}")
        return 1

    name = config.get("assistant.name", "Jarvis")
    print(f"{name} is running at {runtime.url}")
    print(f"Voice: {runtime.voice_note}")
    try:
        if window and _open_window(name, runtime.url):
            return 0
        if window:
            print("(For a native window: pip install 'jarvis-assistant[desktop]')")
        if open_browser:
            webbrowser.open(runtime.url)
        print("Press Ctrl+C to quit.")
        while runtime.is_alive():
            runtime.join(timeout=0.5)
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        runtime.stop()


def _open_window(title: str, url: str) -> bool:
    """Show the native window and block until it is closed."""
    try:
        import webview
    except ImportError:
        return False
    with contextlib.suppress(Exception):
        webview.create_window(title, url, width=1180, height=780, min_size=(820, 560))
        webview.start()
        return True
    return False
