"""Voice pipeline: sentence chunking, endpointing, speech queue and the native loop."""

import asyncio
import contextlib
import math
import time
from array import array

import pytest

from jarvis.app import Jarvis
from jarvis.llm.mock import MockProvider
from jarvis.voice.audio import Audio
from jarvis.voice.chunker import SentenceChunker, clean_for_speech
from jarvis.voice.native import NativeVoice
from jarvis.voice.speaker import SpeechQueue
from jarvis.voice.stt import MockSTT
from jarvis.voice.tts import MockTTS
from jarvis.voice.vad import Endpointer, Utterance

# --------------------------------------------------------------------------- helpers

FRAME_SAMPLES = 480  # 30 ms at 16 kHz


def pcm_frame(amplitude: int = 0) -> bytes:
    """One 30 ms PCM16 frame: silence, or a 440 Hz sine of the given amplitude."""
    if amplitude == 0:
        return bytes(FRAME_SAMPLES * 2)
    samples = array(
        "h", (int(amplitude * math.sin(2 * math.pi * 440 * i / 16000)) for i in range(FRAME_SAMPLES))
    )
    return samples.tobytes()


SILENCE = pcm_frame(0)
SPEECH = pcm_frame(8000)
QUIET = pcm_frame(700)  # rms ~0.015: above the base threshold, below the boosted one


def feed_all(endpointer, frame, count):
    return [endpointer.feed(frame) for _ in range(count)]


async def wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached in time")
        await asyncio.sleep(0.005)


# --------------------------------------------------------------------------- SentenceChunker

TEXT = "Hello there, my friend. How are you doing today? I am doing just fine!"
EXPECTED = ["Hello there, my friend.", "How are you doing today?", "I am doing just fine!"]


def run_chunker(pieces):
    chunker = SentenceChunker()
    out = []
    for piece in pieces:
        out.extend(chunker.feed(piece))
    out.extend(chunker.flush())
    return out


def test_sentences_are_emitted_as_soon_as_they_complete():
    chunker = SentenceChunker()
    assert chunker.feed("Hello there, my friend.") == []  # not known to be finished yet
    assert chunker.feed(" How are") == ["Hello there, my friend."]
    assert chunker.feed(" you doing today?") == []
    assert chunker.feed("  I am doing just fine!") == ["How are you doing today?"]
    assert chunker.flush() == ["I am doing just fine!"]
    assert chunker.flush() == []


@pytest.mark.parametrize("size", [1, 2, 3, 5, 7, 11, 100])
def test_chunking_is_independent_of_delta_boundaries(size):
    pieces = [TEXT[i : i + size] for i in range(0, len(TEXT), size)]
    assert run_chunker(pieces) == EXPECTED


def test_decimals_and_domains_are_not_split():
    text = "Pi is roughly 3.14159 and you can read more at example.com today. Next sentence here."
    for size in (1, 4, 1000):
        pieces = [text[i : i + size] for i in range(0, len(text), size)]
        assert run_chunker(pieces) == [
            "Pi is roughly 3.14159 and you can read more at example.com today.",
            "Next sentence here.",
        ]


def test_abbreviations_are_not_split():
    out = run_chunker(["Dr. Smith will see you now. Mr. Jones is waiting outside."])
    assert out == ["Dr. Smith will see you now.", "Mr. Jones is waiting outside."]
    assert run_chunker(["Use fruits, e.g. apples and pears, for this. Done now."]) == [
        "Use fruits, e.g. apples and pears, for this.",
        "Done now.",
    ]


def test_words_ending_like_abbreviations_still_end_a_sentence():
    # 'fast.' ends with 'st.'; observed: ['Run it fast. Then stop it now.'] (one merged chunk)
    assert run_chunker(["Run it fast. Then stop it now. "]) == ["Run it fast.", "Then stop it now."]


def test_flush_returns_the_unfinished_tail():
    chunker = SentenceChunker()
    assert chunker.feed("First thing is done. And then the tail") == ["First thing is done."]
    assert chunker.flush() == ["And then the tail"]
    assert chunker.feed("Fresh start of text. ") == ["Fresh start of text."]


def test_line_breaks_end_a_sentence():
    chunker = SentenceChunker()
    assert chunker.feed("Here is the first line\nand the second") == ["Here is the first line"]
    assert chunker.flush() == ["and the second"]


def test_code_blocks_are_not_cut_in_the_middle():
    chunker = SentenceChunker()
    out = chunker.feed("Try this: ```print(1). print(2).\nprint(3)``` and then stop. ")
    assert len(out) == 1 and "(code omitted)" in out[0] and out[0].endswith("stop.")


def test_clean_for_speech_strips_markdown():
    assert clean_for_speech("This is **bold** and *italic* and ~~gone~~.") == "This is bold and italic and gone."
    assert clean_for_speech("# Heading\nSome text") == "Heading Some text"
    assert clean_for_speech("- one\n- two\n* three") == "one two three"
    assert clean_for_speech("> quoted line") == "quoted line"
    assert clean_for_speech("See [the docs](https://example.com/docs) now") == "See the docs now"
    assert clean_for_speech("![alt text](http://x.y/i.png)") == "alt text"
    assert clean_for_speech("Run `ls -la` please") == "Run ls -la please"
    out = clean_for_speech("Before\n```python\nprint('x')\n```\nAfter")
    assert out == "Before (code omitted) After"
    assert "print" not in out


def test_clean_for_speech_replaces_bare_urls():
    out = clean_for_speech("Visit https://example.com/a/b?c=1 or http://foo.bar for details.")
    assert "http" not in out and "example.com" not in out
    assert out == "Visit a link or a link for details."
    assert clean_for_speech("   \n  ") == ""


# --------------------------------------------------------------------------- Endpointer


def test_start_after_sustained_speech():
    ep = Endpointer()
    assert feed_all(ep, SILENCE, 5) == [None] * 5
    results = feed_all(ep, SPEECH, 5)  # start_ms=150 -> 5 frames
    assert results == [None, None, None, None, "start"]
    assert ep.active


def test_utterance_after_trailing_silence_includes_pre_roll():
    ep = Endpointer()
    feed_all(ep, SILENCE, 12)  # more than the 10-frame pre-roll
    results = feed_all(ep, SPEECH, 20)
    assert results.count("start") == 1
    trailing = feed_all(ep, SILENCE, 25)  # end_silence_ms=750 -> 25 frames
    assert trailing[:-1] == [None] * 24
    utterance = trailing[-1]
    assert isinstance(utterance, Utterance)
    # pre-roll (5 silent + 5 triggering speech frames) + 15 more speech + 25 silent
    assert len(utterance.pcm) == (10 + 15 + 25) * len(SPEECH)
    assert utterance.pcm[: 5 * len(SPEECH)] == SILENCE * 5  # the lead-in before the voice
    assert utterance.pcm[5 * len(SPEECH) : 6 * len(SPEECH)] == SPEECH
    assert utterance.sample_rate == 16000
    assert utterance.duration == pytest.approx(50 * 0.03)
    assert not ep.active  # ready for the next utterance
    assert feed_all(ep, SILENCE, 3) == [None] * 3


def test_short_pause_inside_speech_does_not_end_the_utterance():
    ep = Endpointer()
    feed_all(ep, SPEECH, 10)
    assert feed_all(ep, SILENCE, 10) == [None] * 10  # 300 ms < 750 ms
    feed_all(ep, SPEECH, 10)
    results = feed_all(ep, SILENCE, 25)
    assert isinstance(results[-1], Utterance)


def test_blips_do_not_produce_utterances():
    ep = Endpointer()
    # Two loud frames never even start.
    assert feed_all(ep, SPEECH, 2) == [None, None]
    assert feed_all(ep, SILENCE, 30) == [None] * 30

    # A burst long enough to start but shorter than min_speech_ms is cancelled.
    ep = Endpointer()
    results = feed_all(ep, SPEECH, 6) + feed_all(ep, SILENCE, 25)
    assert results[4] == "start"
    assert results[-1] == "cancel"
    assert not any(isinstance(r, Utterance) for r in results)
    assert not ep.active


def test_boost_raises_the_threshold_so_quiet_speech_no_longer_triggers():
    plain = Endpointer()
    assert "start" in feed_all(plain, QUIET, 10)

    boosted = Endpointer()
    boosted.boost = 2.5
    assert boosted.threshold == pytest.approx(plain.threshold * 2.5)
    assert feed_all(boosted, QUIET, 20) == [None] * 20
    # Clearly loud speech still gets through (barge-in must remain possible).
    assert "start" in feed_all(boosted, SPEECH, 6)


def test_noise_floor_adapts_to_a_steady_background():
    ep = Endpointer()
    start_threshold = ep.threshold
    hum = pcm_frame(300)  # below threshold, so it is learned as background
    assert feed_all(ep, hum, 150) == [None] * 150
    assert ep.threshold > start_threshold
    assert feed_all(ep, QUIET, 10) == [None] * 10  # no longer counts as speech


def test_maximum_length_forces_an_utterance():
    ep = Endpointer(max_seconds=1.0)  # 33 frames
    results = feed_all(ep, SPEECH, 60)
    first = next(r for r in results if isinstance(r, Utterance))
    assert results.index(first) == 32
    assert len(first.pcm) == 33 * len(SPEECH)
    assert first.duration == pytest.approx(0.99)


# --------------------------------------------------------------------------- SpeechQueue


class FakeTTS:
    name = "fake"

    def __init__(self, fail_on=(), empty_on=()):
        self.fail_on = set(fail_on)
        self.empty_on = set(empty_on)
        self.calls = []

    async def synthesize(self, text):
        self.calls.append(text)
        if text in self.fail_on:
            raise RuntimeError("tts engine crashed")
        if text in self.empty_on:
            return Audio(b"", 16000)
        return Audio(b"\x01\x00" * 160, 16000)


class Player:
    def __init__(self, duration=0.1):
        self.duration = duration
        self.started = []
        self.finished = []

    async def __call__(self, audio, text):
        self.started.append(text)
        await asyncio.sleep(self.duration)
        self.finished.append(text)


async def idle(queue, timeout=3.0):
    await asyncio.wait_for(queue.wait_idle(), timeout)


async def test_speech_queue_plays_in_order_and_tracks_speaking():
    player, states = Player(), []
    queue = SpeechQueue(FakeTTS(), player, states.append)
    assert not queue.speaking
    queue.say("one")
    queue.say("two")
    queue.say("   ")  # ignored
    queue.say("three")
    assert queue.speaking
    await idle(queue)
    assert player.finished == ["one", "two", "three"]
    assert not queue.speaking
    assert states == [True, False]  # one continuous burst, not one per sentence
    assert await queue.cancel() == "one two three"  # everything had been heard
    await queue.aclose()


async def test_speaking_is_true_while_busy_and_wait_idle_returns_after():
    player = Player(0.15)
    queue = SpeechQueue(FakeTTS(), player)
    await asyncio.wait_for(queue.wait_idle(), 1)  # idle from the start
    queue.say("hello")
    waiter = asyncio.create_task(queue.wait_idle())
    await asyncio.sleep(0.05)
    assert queue.speaking and not waiter.done()
    await asyncio.wait_for(waiter, 2)
    assert player.finished == ["hello"] and not queue.speaking
    await queue.aclose()


async def test_next_sentence_is_synthesised_while_the_current_one_plays():
    tts, player = FakeTTS(), Player(0.3)
    queue = SpeechQueue(tts, player)
    queue.say("first")
    queue.say("second")
    await wait_until(lambda: player.started == ["first"])
    await asyncio.sleep(0.02)
    assert tts.calls == ["first", "second"] and player.finished == []
    await queue.aclose()


async def test_cancel_during_first_item_drops_the_rest_and_reports_nothing_heard():
    player, states = Player(0.3), []
    queue = SpeechQueue(FakeTTS(), player, states.append)
    for text in ("one", "two", "three"):
        queue.say(text)
    await wait_until(lambda: player.started == ["one"])
    heard = await queue.cancel()
    assert heard == ""  # the first sentence never finished playing
    await idle(queue, 1)
    await asyncio.sleep(0.4)  # long enough for dropped items to (wrongly) play
    assert player.started == ["one"] and player.finished == []
    assert not queue.speaking and states == [True, False]
    await queue.aclose()


async def test_cancel_returns_only_fully_played_sentences():
    player = Player(0.1)
    queue = SpeechQueue(FakeTTS(), player)
    for text in ("one", "two", "three"):
        queue.say(text)
    await wait_until(lambda: player.started == ["one", "two"])
    heard = await queue.cancel()
    assert heard == "one"
    await idle(queue, 1)
    assert player.finished == ["one"] and player.started == ["one", "two"]
    assert await queue.cancel() == ""  # the record was reset
    await queue.aclose()


async def test_queue_is_reusable_after_cancel():
    player = Player(0.2)
    queue = SpeechQueue(FakeTTS(), player)
    queue.say("old")
    await wait_until(lambda: player.started)
    await queue.cancel()
    queue.begin_turn()
    queue.say("new")
    await idle(queue)
    assert player.finished == ["new"]
    assert await queue.cancel() == "new"
    await queue.aclose()


async def test_failing_or_empty_synthesis_is_skipped_without_killing_the_queue():
    tts = FakeTTS(fail_on={"bad"}, empty_on={"mute"})
    player = Player(0.02)
    queue = SpeechQueue(tts, player)
    for text in ("good one", "bad", "mute", "good two"):
        queue.say(text)
    await idle(queue)
    assert player.finished == ["good one", "good two"]
    assert tts.calls == ["good one", "bad", "mute", "good two"]
    assert await queue.cancel() == "good one good two"  # skipped text was never "heard"
    queue.say("still alive")
    await idle(queue)
    assert player.finished[-1] == "still alive"
    await queue.aclose()


async def test_begin_turn_forgets_what_was_heard():
    player = Player(0.01)
    queue = SpeechQueue(FakeTTS(), player)
    queue.say("previous")
    await idle(queue)
    queue.begin_turn()
    queue.say("current")
    await idle(queue)
    assert await queue.cancel() == "current"
    await queue.aclose()


# --------------------------------------------------------------------------- NativeVoice


class FakeIO:
    """Scripted microphone and speaker implementing the AudioIO protocol."""

    sample_rate = 16000

    def __init__(self, play_seconds=0.3):
        self.play_seconds = play_seconds
        self._frames: asyncio.Queue[bytes] = asyncio.Queue()
        self.started = False
        self.stopped = False
        self.plays: list[Audio] = []
        self.play_began = asyncio.Event()
        self.cancelled_plays = 0
        self.completed_plays = 0

    def feed(self, *frames: bytes) -> None:
        for frame in frames:
            self._frames.put_nowait(frame)

    def say(self, speech_frames=20, silence_frames=30) -> None:
        self.feed(*([SPEECH] * speech_frames), *([SILENCE] * silence_frames))

    async def start(self):
        self.started = True

    async def stop(self):
        self.stopped = True

    async def frames(self):
        while True:
            yield await self._frames.get()
            await asyncio.sleep(0.001)

    async def play(self, audio):
        self.plays.append(audio)
        self.play_began.set()
        try:
            await asyncio.sleep(self.play_seconds)
            self.completed_plays += 1
        except asyncio.CancelledError:
            self.cancelled_plays += 1
            raise


class RecordingSTT(MockSTT):
    def __init__(self, config, texts=None):
        super().__init__(config)
        self.texts = list(texts or [])
        self.calls = []

    async def transcribe(self, audio, language=None):
        self.calls.append(audio)
        return self.texts.pop(0) if self.texts else self.text


def make_voice(config, *, script=None, delay=0.0, stt_texts=None, play_seconds=0.3, **options):
    provider = MockProvider(
        "mock", {"provider": "mock", "model": "mock", "delay": delay}, config, script=script
    )
    app = Jarvis(config, providers=[provider])
    conversation = app.conversation()
    io = FakeIO(play_seconds)
    stt = RecordingSTT(config, stt_texts)
    voice = NativeVoice(conversation, stt, MockTTS(config), io, **options)
    events = []
    conversation.subscribe(events.append)
    return app, provider, conversation, io, stt, voice, events


@pytest.fixture
async def run_voice():
    """Start voice.run() as a task; cancel it again when the test is over."""
    tasks = []

    def start(voice):
        task = asyncio.create_task(voice.run())
        tasks.append(task)
        return task

    yield start
    for task in tasks:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


async def test_speech_produces_transcript_turn_and_playback(config, run_voice):
    app, provider, conversation, io, stt, voice, events = make_voice(config)
    task = run_voice(voice)
    io.say()
    await asyncio.wait_for(io.play_began.wait(), 5)
    types = [e["type"] for e in events]
    assert io.started
    assert {"transcript", "turn_start", "sentence"} <= set(types)
    assert types.index("transcript") < types.index("turn_start") < types.index("sentence")
    transcript = next(e for e in events if e["type"] == "transcript")
    assert transcript["text"] == "hello jarvis"
    start = next(e for e in events if e["type"] == "turn_start")
    assert start["source"] == "voice" and start["text"] == "hello jarvis"
    assert len(stt.calls) == 1 and stt.calls[0].sample_rate == 16000
    assert stt.calls[0].duration > 1.0
    assert io.plays and io.plays[0].duration > 0
    assert "# Speaking" in provider.calls[0]["system"]  # voice turns get the spoken-style prompt
    states = [e["state"] for e in events if e["type"] == "voice_state"]
    assert "listening" in states and "transcribing" in states and "speaking" in states
    await conversation.wait()
    await asyncio.wait_for(voice.speech.wait_idle(), 5)
    assert io.completed_plays == 1 and voice.state == "idle"
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert io.stopped  # the audio device is released on shutdown


async def test_silence_and_blips_start_nothing(config, run_voice):
    app, provider, conversation, io, stt, voice, events = make_voice(config)
    run_voice(voice)
    io.feed(*([SILENCE] * 40), *([SPEECH] * 2), *([SILENCE] * 40))  # a click, not speech
    io.feed(*([SPEECH] * 6), *([SILENCE] * 30))  # a cough: starts, then is cancelled
    await wait_until(lambda: voice.state == "idle" and io._frames.empty())
    await asyncio.sleep(0.1)
    assert stt.calls == [] and provider.calls == []
    assert not any(e["type"] in ("transcript", "turn_start") for e in events)


async def test_wake_word_ignores_utterances_without_it(config, run_voice):
    app, provider, conversation, io, stt, voice, events = make_voice(
        config, wake_word="jarvis", stt_texts=["what time is it"]
    )
    run_voice(voice)
    io.say()
    await wait_until(lambda: len(stt.calls) == 1)
    await asyncio.sleep(0.15)
    assert provider.calls == []
    assert not any(e["type"] in ("transcript", "turn_start") for e in events)
    assert voice.state == "idle"


async def test_wake_word_is_accepted_and_stripped(config, run_voice):
    app, provider, conversation, io, stt, voice, events = make_voice(
        config, wake_word="Jarvis", stt_texts=["Jarvis, what time is it"]
    )
    run_voice(voice)
    io.say()
    await wait_until(lambda: any(e["type"] == "turn_start" for e in events))
    assert next(e for e in events if e["type"] == "transcript")["text"] == "what time is it"
    await conversation.wait()
    assert app.session().messages[0].content == "what time is it"
    assert provider.calls[0]["messages"][-1].content == "what time is it"


async def test_follow_up_without_wake_word_is_accepted_shortly_after_a_reply(config, run_voice):
    app, provider, conversation, io, stt, voice, events = make_voice(
        config, wake_word="jarvis", stt_texts=["jarvis hello", "and another thing"], play_seconds=0.05
    )
    run_voice(voice)
    io.say()
    await wait_until(lambda: sum(e["type"] == "turn_end" for e in events) == 1)
    await asyncio.wait_for(voice.speech.wait_idle(), 5)
    io.say()
    await wait_until(lambda: sum(e["type"] == "turn_end" for e in events) == 2)
    texts = [e["text"] for e in events if e["type"] == "transcript"]
    assert texts == ["hello", "and another thing"]


def test_wake_word_text_rules(config):
    app, provider, conversation, io, stt, voice, events = make_voice(config, wake_word="Jarvis")
    assert voice.wake_word == "jarvis"
    assert voice._apply_wake_word("") == ""
    assert voice._apply_wake_word("turn on the lights") == ""
    assert voice._apply_wake_word("Hey JARVIS, turn on the lights.") == "Hey turn on the lights"
    assert voice._apply_wake_word("jarvis") == "jarvis"  # nothing else said: keep it
    assert voice._apply_wake_word("jarvisian things") == ""  # whole word only
    voice._awake_until = time.monotonic() + 5
    assert voice._apply_wake_word("turn on the lights") == "turn on the lights"
    plain = make_voice(config)[5]
    assert plain._apply_wake_word("anything at all") == "anything at all"


async def test_barge_in_cancels_playback_and_interrupts_the_conversation(config, run_voice):
    long_reply = (
        "Here is the first sentence of my answer. "
        + "And then I keep on talking for quite a while about many other things. " * 5
    )
    app, provider, conversation, io, stt, voice, events = make_voice(
        config, script=[long_reply, "Okay, going on."], delay=0.03, play_seconds=0.6
    )
    run_voice(voice)
    io.say()
    await asyncio.wait_for(io.play_began.wait(), 5)
    assert conversation.busy and voice.speech.speaking  # still generating while the first sentence plays

    io.feed(*([SPEECH] * 8))  # the user talks over the assistant
    await wait_until(lambda: io.cancelled_plays == 1)
    await wait_until(lambda: any(e["type"] == "turn_end" for e in events))
    end = next(e for e in events if e["type"] == "turn_end")
    assert end["interrupted"] is True
    assert io.completed_plays == 0
    # turn_end fires before interrupt() has finished correcting the history, so wait for it.
    await wait_until(lambda: not conversation.busy)
    # The user never heard a complete sentence, so the half reply is dropped from history.
    await wait_until(lambda: [m.role for m in app.session().messages] == ["user"])

    # The barge-in speech is then handled as the next utterance.
    io.feed(*([SPEECH] * 12), *([SILENCE] * 30))
    await wait_until(lambda: sum(e["type"] == "turn_end" for e in events) == 2)
    assert sum(e["type"] == "transcript" for e in events) == 2
    assert [e["interrupted"] for e in events if e["type"] == "turn_end"] == [True, False]
    assert app.session().messages[-1].content == "Okay, going on."


async def test_barge_in_records_what_was_actually_heard(config, run_voice):
    reply = "Short one. " + "A very long second part that keeps on going for ages and ages. " * 8
    app, provider, conversation, io, stt, voice, events = make_voice(
        config, script=[reply], delay=0.02, play_seconds=0.15
    )
    run_voice(voice)
    io.say()
    await wait_until(lambda: len(io.plays) >= 1 and io.completed_plays >= 1)
    await wait_until(lambda: len(io.plays) >= 2)  # the second sentence is now playing
    io.feed(*([SPEECH] * 8))
    await wait_until(lambda: any(e["type"] == "turn_end" for e in events))
    # History is rewritten to what was heard just after turn_end is emitted.
    await wait_until(lambda: app.session().messages[-1].content == "Short one.")
    last = app.session().messages[-1]
    assert last.role == "assistant" and last.meta.get("interrupted")


async def test_speech_during_playback_is_ignored_when_barge_in_is_off(config, run_voice):
    app, provider, conversation, io, stt, voice, events = make_voice(
        config, barge_in=False, play_seconds=0.4
    )
    run_voice(voice)
    io.say()
    await asyncio.wait_for(io.play_began.wait(), 5)
    io.feed(*([SPEECH] * 15), *([SILENCE] * 30))
    await wait_until(lambda: io.completed_plays == 1)
    assert io.cancelled_plays == 0
    assert len(stt.calls) == 1  # the talking-over was never transcribed


async def test_muted_voice_ignores_speech(config, run_voice):
    app, provider, conversation, io, stt, voice, events = make_voice(config)
    run_voice(voice)
    voice.set_muted(True)
    await wait_until(lambda: any(e["type"] == "voice_muted" for e in events))
    assert next(e for e in events if e["type"] == "voice_muted")["muted"] is True
    io.say()
    await wait_until(lambda: io._frames.empty())
    await asyncio.sleep(0.15)
    assert stt.calls == [] and provider.calls == []
    assert not any(e["type"] in ("transcript", "turn_start") for e in events)

    voice.set_muted(False)
    io.say()
    await wait_until(lambda: any(e["type"] == "turn_start" for e in events))
    assert len(stt.calls) == 1


async def test_transcription_failure_is_reported_and_loop_survives(config, run_voice):
    app, provider, conversation, io, stt, voice, events = make_voice(config)

    class Broken(RecordingSTT):
        async def transcribe(self, audio, language=None):
            self.calls.append(audio)
            if len(self.calls) == 1:
                raise RuntimeError("model not loaded")
            return "second try"

    voice.stt = Broken(config)
    run_voice(voice)
    io.say()
    await wait_until(lambda: any(e["type"] == "error" for e in events))
    assert "Speech recognition failed" in next(e for e in events if e["type"] == "error")["message"]
    io.say()
    await wait_until(lambda: any(e["type"] == "turn_start" for e in events))
    assert next(e for e in events if e["type"] == "transcript")["text"] == "second try"


async def test_stop_phrase_by_voice_silences_playback(config, run_voice):
    reply = "Here is the first sentence of my answer. " + "More and more words follow here. " * 10
    app, provider, conversation, io, stt, voice, events = make_voice(
        config, script=[reply], delay=0.02, play_seconds=0.6, barge_in=False
    )
    run_voice(voice)
    io.say()
    await asyncio.wait_for(io.play_began.wait(), 5)
    # With barge-in off the microphone is deaf; the stop command arrives via the conversation.
    await conversation.submit("stop", source="voice")
    await wait_until(lambda: io.cancelled_plays == 1)
    assert any(e["type"] == "stopped" for e in events)
    assert next(e for e in events if e["type"] == "turn_end")["interrupted"] is True
