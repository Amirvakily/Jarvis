"""One live conversation: turn-taking, interruption and event fan-out.

Every front end (desktop window, terminal, native microphone loop, browser)
talks to the same Conversation object, so a sentence spoken into the PC
microphone shows up in the window and a typed message can be answered out
loud. Front ends subscribe to a stream of plain dict events:

    turn_start   {turn_id, text, source}
    text_delta   {turn_id, text}             streamed reply text
    sentence     {turn_id, text, index}      a complete sentence, ready to speak
    tool_start / tool_end / agent_start / agent_end / agent_text / usage
    turn_end     {turn_id, text, interrupted, error}
    announcement {text}                      background task finished, etc.
    backend      {...}                       the active backend changed
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import itertools
import logging
import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from jarvis.core.session import Session
from jarvis.core.types import LLMError
from jarvis.voice.chunker import SentenceChunker

if TYPE_CHECKING:  # pragma: no cover
    from jarvis.app import Jarvis

log = logging.getLogger(__name__)

Subscriber = Callable[[dict], Any]

#: Said on their own, these stop the assistant instead of starting a new turn.
STOP_PHRASES = {
    "stop", "stop it", "stop talking", "be quiet", "quiet", "shut up", "cancel",
    "cancel that", "never mind", "nevermind", "enough", "that's enough", "hold on",
}  # fmt: skip


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z' ]", "", text.lower()).strip()


class Conversation:
    def __init__(self, app: Jarvis, session: Session):
        self.app = app
        self.session = session
        self._subscribers: list[Subscriber] = []
        self._turn: asyncio.Task | None = None
        self._turn_ids = itertools.count(1)
        self._notes: list[str] = []
        self._lock = asyncio.Lock()
        self._unsubscribe_app = app.subscribe(self._on_app_event)

    # -- events ---------------------------------------------------------------
    def subscribe(self, subscriber: Subscriber) -> Callable[[], None]:
        """Register a callback for conversation events. Callbacks must return
        quickly (queue the event); slow ones would stall the reply stream."""
        self._subscribers.append(subscriber)

        def unsubscribe() -> None:
            if subscriber in self._subscribers:
                self._subscribers.remove(subscriber)

        return unsubscribe

    async def emit(self, event: dict) -> None:
        for subscriber in list(self._subscribers):
            try:
                result = subscriber(event)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                log.exception("conversation subscriber failed")

    def _on_app_event(self, event: dict) -> None:
        if event["type"] == "task_update":
            task = event["task"]
            if task["status"] in ("done", "failed", "waiting"):
                outcome = "failed" if task["status"] == "failed" else "reported"
                self._notes.append(
                    f"Background task {task['id']} ({task['agent']}: {task['task'][:120]}) "
                    f"{outcome}:\n{task['result'][:3000]}"
                )
                spoken = (
                    f"The {task['agent']} agent has finished a background task."
                    if task["status"] != "failed"
                    else f"A background task for the {task['agent']} agent failed."
                )
                self._schedule({"type": "announcement", "text": spoken, "task_id": task["id"]})
        self._schedule(event)

    def _schedule(self, event: dict) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = loop.create_task(self.emit(event))
        task.add_done_callback(lambda t: t.exception())

    # -- turn-taking ------------------------------------------------------------
    @property
    def busy(self) -> bool:
        return self._turn is not None and not self._turn.done()

    async def submit(self, text: str, *, source: str = "text", voice: bool | None = None) -> int | None:
        """Start a turn for ``text``. Anything still in progress is interrupted
        first, which is what makes barge-in work. Returns the turn id, or
        ``None`` if the input was empty or only a stop command."""
        text = text.strip()
        if not text:
            return None
        async with self._lock:
            if source == "voice" and _normalise(text) in STOP_PHRASES:
                await self._cancel_turn()
                await self.emit({"type": "stopped", "text": text})
                return None
            await self._cancel_turn()
            turn_id = next(self._turn_ids)
            spoken = (source == "voice") if voice is None else voice
            self._turn = asyncio.create_task(
                self._run_turn(turn_id, text, source, spoken), name=f"jarvis-turn-{turn_id}"
            )
            return turn_id

    async def interrupt(self, heard: str | None = None) -> bool:
        """Stop the current turn. ``heard`` is the part of the reply the user
        actually listened to; history is corrected to match, so the assistant
        does not later assume the user heard things that were cut off."""
        async with self._lock:
            interrupted = await self._cancel_turn()
        if interrupted and heard is not None:
            last = self.session.messages[-1] if self.session.messages else None
            if last and last.role == "assistant" and last.meta.get("interrupted"):
                if heard.strip():
                    last.content = heard.strip()
                else:
                    self.session.messages.pop()
                self.session.epoch += 1
                self.session.save()
        return interrupted

    async def _cancel_turn(self) -> bool:
        turn = self._turn
        if turn is None or turn.done():
            return False
        turn.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await turn
        return True

    async def wait(self) -> None:
        """Wait for the current turn, if any, to finish."""
        if self._turn is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await self._turn

    async def _run_turn(self, turn_id: int, text: str, source: str, voice: bool) -> None:
        await self.emit({"type": "turn_start", "turn_id": turn_id, "text": text, "source": source})
        chunker = SentenceChunker()
        index = itertools.count()

        async def speak(sentences: list[str]) -> None:
            for sentence in sentences:
                await self.emit(
                    {"type": "sentence", "turn_id": turn_id, "text": sentence, "index": next(index)}
                )

        async def on_event(event: dict) -> None:
            event["turn_id"] = turn_id
            await self.emit(event)
            if event["type"] == "text_delta":
                await speak(chunker.feed(event["text"]))
            elif event["type"] == "tool_start":
                await speak(chunker.flush())  # say "one moment" before the tool runs

        prompt = text
        if self._notes:
            notes, self._notes = self._notes, []
            prompt = "[System notes since your last reply]\n" + "\n\n".join(notes) + f"\n\n[User]\n{text}"

        reply, interrupted, error = "", False, None
        try:
            reply = await self.app.chat(prompt, session=self.session, emit=on_event, voice=voice)
            await speak(chunker.flush())
        except asyncio.CancelledError:
            interrupted = True
            raise
        except LLMError as exc:
            error = str(exc)
            log.warning("turn %s failed: %s", turn_id, exc)
            await self.emit({"type": "error", "turn_id": turn_id, "message": error})
            await speak(["I couldn't reach my language model. " + _spoken_hint(error)])
        except Exception as exc:  # keep the conversation alive whatever happens
            error = f"{type(exc).__name__}: {exc}"
            log.exception("turn %s crashed", turn_id)
            await self.emit({"type": "error", "turn_id": turn_id, "message": error})
        finally:
            self.session.save()
            await asyncio.shield(
                self.emit(
                    {
                        "type": "turn_end",
                        "turn_id": turn_id,
                        "text": reply,
                        "interrupted": interrupted,
                        "error": error,
                    }
                )
            )

    async def reset(self) -> None:
        async with self._lock:
            await self._cancel_turn()
        self.session.clear()
        await self.emit({"type": "reset"})

    def close(self) -> None:
        self._unsubscribe_app()


def _spoken_hint(error: str) -> str:
    lowered = error.lower()
    if "ollama" in lowered and ("reach" in lowered or "running" in lowered):
        return "Ollama doesn't seem to be running. Start it, or switch me to a cloud backend."
    if "pull" in lowered or "not installed" in lowered:
        return "The local model isn't downloaded yet."
    if "credential" in lowered or "api_key" in lowered or "api key" in lowered:
        return "The API key for this backend is missing or was rejected."
    return "Check the backend settings, or switch to another backend."
