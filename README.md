# Jarvis

A voice-enabled, modular multi-agent AI assistant. You talk to it, it answers out loud, and it can hand work to specialist agents, run tasks in the background and use skills you write yourself.

Everything runs on your own machine. The "window" is a local app talking to a local process over a loopback WebSocket; nothing is hosted, and the only traffic that leaves your PC is what you configure (a cloud model API, web lookups made by skills).

## What you get

- **Voice-first conversation.** Speech in, speech out, interruptible mid-sentence, with context kept across turns. Works through the PC microphone and speakers or through the window's browser engine.
- **Hybrid backend.** A local 8B model through Ollama (or a GGUF file run in-process), or cloud APIs: Anthropic, OpenAI, or any OpenAI-compatible endpoint. Switch at runtime by voice, from the app, or from the CLI.
- **Multi-agent framework.** An orchestrator plus specialist agents. Specialists get a fresh context, can run in parallel and in the background, and you can define your own as TOML files.
- **Skills and plugin hub.** A skill is a typed Python function with a docstring. Drop a file into `plugins/` and the orchestrator can use it.

## Quick start

### Windows

You need Python 3.11 or newer (from python.org, with "Add to PATH" ticked).

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install.ps1
```

The script creates `.venv`, installs Jarvis with the `all` extras (voice, desktop window, file parsers), writes `jarvis.toml` if you have none, downloads a local Piper voice, and runs `jarvis doctor`.

Then double-click `start-jarvis.bat`. It opens the Jarvis window and the PC microphone. Arguments pass through, so `start-jarvis.bat -b claude` starts on the `claude` profile.

### macOS / Linux

```sh
sh scripts/install.sh
.venv/bin/jarvis
```

On Linux, microphone access needs PortAudio (`sudo apt install libportaudio2`).

### Give it a brain

Jarvis needs a language model. Pick one.

**A. Local (private, free).** Install [Ollama](https://ollama.com), then:

```sh
ollama pull llama3.1:8b
```

The default `local` profile already points at `http://localhost:11434` with that model.

**B. Cloud.** Set a key in your environment, then make that profile the default:

```sh
# Anthropic
export ANTHROPIC_API_KEY=...        # Windows PowerShell: $env:ANTHROPIC_API_KEY = "..."
jarvis backend claude

# or OpenAI
export OPENAI_API_KEY=...
jarvis backend openai
```

Run `jarvis doctor` at any point to check backends, microphone, speech engines, ffmpeg and loaded skills and agents.

### Try it with no model at all

A `mock` provider exists so you can exercise the whole app (window, voice loop, agents, skills) with nothing installed. In `jarvis.toml`, set `active = "mock"` in the existing `[llm]` section and add the profile:

```toml
[llm]
active = "mock"

[llm.profiles.mock]
provider = "mock"
```

The mock backend is rule-based:

- `!skill_name {json}` calls that skill directly, e.g. `!roll_dice {"sides": 20}`.
- `@agent task` delegates to an agent, e.g. `@research what is a GGUF file`.
- Anything else is echoed back.

Use `jarvis chat` to type these, or speak them.

## Ways to use it

| Command | What it does |
|---|---|
| `jarvis` (or `jarvis app`) | Desktop app: a window plus the PC microphone. `--no-voice` skips the microphone, `--no-window` uses your browser, `--host` and `--port` override the server address. |
| `jarvis voice` | Hands-free in the terminal, no window. |
| `jarvis chat` | Typed conversation in the terminal. Slash commands: `/backend [name]`, `/agents`, `/skills`, `/run AGENT TASK`, `/reload`, `/reset`, `/quit`. |
| `jarvis ask "..."` | One question, one answer. |
| `jarvis serve` | Local server only, no window; use from a browser. `--voice` also listens on this machine's microphone. |
| `jarvis agents list` | List agents with their skills and pinned backend. |
| `jarvis agents run AGENT "TASK" [AGENT "TASK" ...]` | Run one or several agents concurrently. |
| `jarvis skills list` / `new NAME` / `test NAME key=value ...` / `reload` | List, scaffold, run and reload skills. |
| `jarvis backend [NAME]` | Show backends and their health, or set the default (remembered across restarts). |
| `jarvis doctor` | Check the installation. |
| `jarvis setup-voice` | Download the local Piper voice (`--voice` picks another; default `en_US-lessac-medium`). |
| `jarvis init` | Write a starter `jarvis.toml`, plus `plugins/` and `agents/` folders, in the current directory. |

`voice` and `chat` take `--session NAME` to continue a named conversation (default `main`). Global options go before the subcommand: `-c CONFIG` (path to a `jarvis.toml`), `-b PROFILE` (backend for this run only), `-v` (log more), `--version`.

Things to say out loud:

- "Give me a technical analysis of NVDA." (trading agent)
- "Write an article about home battery storage and publish it." (content agent: draft, format, publish)
- "Clip the first ninety seconds of interview.mp4 in the workspace." (video agent)
- "Switch to the cloud." / "Go local."
- "Have research look into the new Ollama release while trading scans my watchlist." (two agents at once)
- "Every thirty minutes, check the BTC chart and tell me if the trend flips." (recurring background task, announced aloud when it reports)
- "Remember that I prefer short answers."
- "Stop."

## Voice

`voice.mode` decides where speech happens:

| Mode | Meaning |
|---|---|
| `auto` | Use the PC microphone and speakers when the speech engines are installed; otherwise the window's browser engine handles it. |
| `native` | PC microphone and speakers only. |
| `browser` | The window or browser tab does the listening and speaking. |
| `off` | Text only. |

**Engines.** Listening uses `voice.stt.provider`: `faster-whisper` (local Whisper, model set by `voice.stt.model`, default `base.en`; `tiny.en` is faster, `small.en` more accurate), `openai` (an OpenAI-compatible transcription endpoint), `browser`, or `auto`. Speaking uses `voice.tts.provider`: `piper` (local neural voice, found in `~/.jarvis/voices` or at `voice.tts.model`), `system` (the OS voice: PowerShell on Windows, `say` on macOS, `espeak-ng` on Linux), `openai` (an OpenAI-compatible speech endpoint), `browser`, or `auto`.

With `auto`, STT picks local Whisper if faster-whisper is installed, otherwise the OpenAI-compatible endpoint if its key is set. TTS picks Piper if it is installed and a voice has been downloaded, then the OS voice, then the OpenAI-compatible endpoint if its key is set. The endpoint URLs and key variable names for the OpenAI-compatible engines are `voice.stt.base_url`, `voice.stt.api_key_env`, `voice.tts.base_url` and `voice.tts.api_key_env`.

**Barge-in.** With `voice.barge_in = true` the microphone stays open while Jarvis talks. Speak over it and playback stops, the reply is cancelled, and your new sentence becomes the next turn. The conversation history records only the part of the reply you actually heard, so Jarvis does not assume you heard words that were cut off. Use headphones: with open speakers the assistant can occasionally hear itself.

**Stop phrases.** Said on their own, these stop Jarvis instead of starting a new turn: stop, stop it, stop talking, be quiet, quiet, shut up, cancel, cancel that, never mind, nevermind, enough, that's enough, hold on.

**Wake word and follow-up.** Set `voice.wake_word = "jarvis"` and speech that does not contain the word is ignored. After Jarvis finishes speaking, you can answer without the wake word for `voice.follow_up_seconds` (default 20). Other settings: `voice.language`, `voice.input_device`, `voice.output_device`.

## Switching between local and cloud

Backends are profiles under `[llm.profiles.NAME]` in `jarvis.toml`. One is active at a time (`llm.active`). The example config ships `local` (Ollama), `claude`, and `openai`, with commented templates for a GGUF file (`provider = "llamacpp"`, install `jarvis-assistant[llamacpp]`), a lighter `claude-fast`, and any OpenAI-compatible endpoint (`provider = "openai"` with your `base_url`; add `local = true` to count it as local).

Three ways to switch:

1. **By voice:** "switch to the cloud", "go local", "use Claude".
2. **In the app:** the Backend menu in the window.
3. **From the terminal:** `jarvis backend claude` sets the default and it is remembered across restarts (stored in `state.json` under `assistant.data_dir`). `jarvis -b claude ...` uses a profile for one run only.

`llm.fallback = "claude"` names a profile to try once if the active backend cannot be reached before it produces output.

**Pin an agent to a profile.** Set `llm_profile` in an agent's TOML file so routine specialists run on a small or cheap model while the orchestrator uses a stronger one. For example, a summarizer pinned to `local` or `claude-fast` while `jarvis` itself runs on `claude`.

### Memory management

A laptop cannot keep an 8B model, a Whisper model and a voice resident at once, so heavy local components are managed:

- Each component (local LLM, Whisper, Piper) loads the first time a task needs it.
- When loading something would exceed `resources.memory_budget_mb` (default 8000), the least recently used idle component is evicted first. Each component declares an estimate (`memory_mb` on the profile, `voice.stt.memory_mb`, `voice.tts.memory_mb`).
- Anything idle longer than `resources.idle_unload_seconds` (default 600) is unloaded.
- An agent only loads the skills it lists, which keeps prompts small for local models.

## Agents

An agent is a role: instructions, the skills it may use, and optionally a pinned backend profile.

| Agent | What it does | Skills |
|---|---|---|
| `jarvis` | The orchestrator. Talks to you and coordinates everything else. | `delegate`, `start_background_task`, `list_background_tasks`, `cancel_background_task`, `switch_backend`, `system_status`, `get_current_time`, `remember`, `recall`, `web_search`, `web_fetch`, `reload_skills`, plus everything from `plugins/` |
| `trading` | Market analyst: prices, technical analysis, watchlist scans, long/short setups. | `tag:market`, `web_search`, `web_fetch`, `get_current_time` |
| `content` | Writer and publisher: researches, drafts, formats and publishes articles. | `tag:content`, `web_search`, `web_fetch`, `read_file`, `get_current_time` |
| `video` | Media technician: inspect, clip, join, transcode, extract audio and thumbnails with ffmpeg. | `tag:video`, `list_files` |
| `research` | Searches the web, reads pages and local documents, writes up findings with sources. | `tag:web`, `tag:files`, `get_current_time` |
| `security` | Passive security posture checks on sites you have authorised. | `security_scan`, `web_fetch`, `write_file` |
| `summarizer` | Condenses long documents and pages (defined in `agents/summarizer.toml`). | `read_file`, `web_fetch`, `list_files` |

**How delegation works.**

- The orchestrator calls `delegate` with an agent name and a task. The specialist starts with a fresh context containing just that task, and its report comes back to the orchestrator.
- When the orchestrator issues several `delegate` calls in one step, they run concurrently. `jarvis agents run` and the server's `/api/agents/run` also run several agents at once; one failing does not stop the others.
- `start_background_task` runs an agent without blocking the conversation. Set `repeat_every_minutes` for a recurring task (minimum one minute, at most 12 active tasks). Results are announced aloud when they arrive. `list_background_tasks` and `cancel_background_task` manage them.
- Delegation is one level deep: specialists cannot delegate further.

**Add your own agent** by dropping a `.toml` file into `agents/` (the `agents.dirs` setting). No code needed. This is `agents/summarizer.toml`:

```toml
name = "summarizer"
description = "Condenses long documents, transcripts and web pages into faithful, tight summaries."

instructions = """
You turn long material into summaries a busy person can act on. Read the whole \
source before writing. Keep every claim traceable to the source, keep the \
numbers exact, and say so when the source is unclear or contradicts itself. \
Lead with the single most important point, then the supporting points in \
order of importance.
"""

# Skill names, or "tag:<tag>" for a whole group (see `jarvis skills list`).
skills = ["read_file", "web_fetch", "list_files"]

# Optional: pin this agent to a specific backend profile from jarvis.toml.
# llm_profile = "local"
```

`name` must be lowercase letters, digits and underscores, and `description` and `instructions` are required. Agents defined in files cannot hold the `delegate`, `start_background_task` or `*` entries; those are stripped. An optional `max_rounds` limits tool rounds. Say "reload skills" or run `jarvis skills reload` to pick up edits; `jarvis agents list` reports files that failed to load. `agents/experimental_template.toml.example` is a starting point for an experimental agent: copy it to `<name>.toml`, edit it and pair it with your own skills in `plugins/`.

The project ships no deepfake or impersonation generator. Media-synthesis experiments are left to skills you write yourself, and should only use your own likeness or that of people who have consented.

## Skills

Built-in skills, grouped by tag (`jarvis skills list` prints the live inventory):

| Tag | Skill | What it does |
|---|---|---|
| core | `delegate` | Hand a task to a specialist agent and wait for its report. |
| core | `start_background_task` | Start an agent on a task in the background, optionally recurring. |
| core | `list_background_tasks` | List background tasks with status and latest result. |
| core | `cancel_background_task` | Stop a running or recurring background task. |
| core | `switch_backend` | Switch the language model backend. |
| core | `system_status` | Report the active backend, what is loaded in memory, and available agents and skills. |
| core | `reload_skills` | Reload plugins and agent definitions from disk. |
| core, util | `get_current_time` | Current local date, time and timezone. |
| core, memory | `remember` / `recall` | Save and look up facts across conversations. |
| web | `web_search` | Search the web (DuckDuckGo). |
| web | `web_fetch` | Download a page and return its readable text. |
| web | `web_scrape` | Extract links, headings, tables, images and metadata from a page. |
| files | `list_files` | List files and folders in a workspace folder. |
| files | `read_file` | Read a workspace file as text, a table, or extracted document text. |
| files | `write_file` | Create or overwrite a UTF-8 text file in the workspace. |
| market | `market_quote` | Latest price, daily change and volume for a stock, ETF, index or crypto pair. |
| market | `technical_analysis` | Trend, RSI, MACD, Bollinger, ATR and support/resistance for a symbol. |
| market | `market_scan` | Scan several symbols and rank them by technical score. |
| market | `evaluate_setup` | Evaluate a hypothetical long or short: entry, ATR-based stop, target, confluence. |
| content | `save_draft` / `read_draft` / `list_drafts` | Keep Markdown article drafts in the workspace. |
| content | `format_article` | Tidy a draft in place and check readability. |
| content | `publish_article` | Publish a draft locally, to a webhook, or to WordPress. |
| video | `video_probe` | Describe a video or audio file. |
| video | `video_clip` | Cut one segment out of a video or audio file. |
| video | `video_concat` | Join videos end to end without re-encoding. |
| video | `video_transcode` | Re-encode into another format, optionally scaled to a new height. |
| video | `video_extract_audio` | Save a video's audio track as its own file. |
| video | `video_thumbnail` | Save one frame as a JPEG. |
| security | `security_scan` | Passive security check of a website you own or may assess. |
| custom, fun | `roll_dice` | The example plugin in `plugins/example_dice.py`. |

The video skills need `ffmpeg` and `ffprobe` on your PATH (for example `winget install ffmpeg` on Windows). `read_file` handles PDF and Word documents when the `files` extra is installed.

### Write your own in two minutes

```sh
jarvis skills new weather_report
```

This writes `plugins/weather_report.py` (the first entry of `skills.plugin_dirs`). Names must be lowercase Python identifiers. The generated file looks like this:

```python
from jarvis import SkillContext, skill
from jarvis.skills.base import SkillError


@skill(tags=["custom"])
def weather_report(text: str, ctx: SkillContext, repeat: int = 1) -> dict:
    """Describe in one or two sentences what this skill does and when to use it.

    The model reads this description to decide when to call the skill, so be
    specific. Parameters are documented below and typed in the signature.

    Args:
        text: What this parameter means.
        repeat: How many times to repeat the text.
    """
    if repeat < 1:
        raise SkillError("repeat must be at least 1.")  # reported back to the model
    return {"result": " ".join([text] * repeat)}
```

How it works:

- The function's type hints and `Args:` docstring become the schema the model sees, and the docstring summary becomes the description. Say clearly when to use it.
- Add a `ctx: SkillContext` parameter if you need the config or workspace. It is hidden from the model.
- Raise `SkillError` for problems the model should hear about; any other exception is reported as a failure. The return value (dict, string or `None`) goes back to the model.
- `@skill` also accepts `name=`, `description=`, `tags=[...]` and `timeout=` (seconds, default 120). Both sync and `async` functions work.
- `ctx.resolve_path("notes.txt")` returns a path inside the workspace and refuses anything outside it.
- For network access, use `http_client(ctx)` and `await check_url(ctx, url)` from `jarvis.skills.util`. `check_url` rejects non-http(s) URLs and, unless `web.allow_private_hosts` is set, hosts that resolve to private or loopback addresses.

Test without a model, then use it:

```sh
jarvis skills test weather_report text=hello repeat=2
```

Values are parsed as JSON when possible (`repeat=2` is a number) and otherwise passed as strings. After editing, say "reload skills" or run `jarvis skills reload`.

Skills in `plugins/` are available to the orchestrator automatically, through the `plugins` selector in its skill list. To give a plugin skill to a specialist, list its name or one of its tags in that agent's `skills`. A selector can be a skill name, `tag:<tag>`, `plugins` (everything from plugin files and installed packages), or `*`.

To ship skills as a package, expose a module that defines `@skill` functions under the `jarvis.skills` entry-point group in your `pyproject.toml`:

```toml
[project.entry-points."jarvis.skills"]
my_pack = "my_package.skills"
```

## Safety model

- The server binds to `127.0.0.1` by default. Requests carrying a cross-origin `Origin` header are refused, for both HTTP and the WebSocket. If you bind to a non-loopback address a token is required: set `server.token`, or one is generated and logged at startup. Pass it as `?token=...` or as a bearer token.
- File skills and video skills only touch `assistant.workspace` (default `./workspace`). Paths outside it are refused.
- Web skills refuse hosts that resolve to private, loopback or link-local addresses unless `web.allow_private_hosts = true`. Timeout is `web.timeout_seconds`.
- `security_scan` is passive and only runs against hosts in `security.authorized_targets` (exact names or `*.example.com` patterns; loopback is always allowed). Other targets are refused.
- The trading agent analyses and never places orders. If market data comes from the `synthetic` source (`trading.data_source = "synthetic"`, for testing), results are labelled "synthetic (not real market data)".
- API keys come from environment variables (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `WORDPRESS_APP_PASSWORD`, or whatever `api_key_env` names), never from the config file.

## Configuration

Run `jarvis init` to copy the annotated starter config to `./jarvis.toml`, or read `jarvis/config.example.toml`. Every value is optional; anything omitted falls back to a built-in default. Jarvis looks for `./jarvis.toml`, then `~/.config/jarvis/jarvis.toml`, or you pass `-c PATH`.

| Section | Controls |
|---|---|
| `[assistant]` | `name`, `user_name`, `workspace`, `data_dir` (conversations, memory, voices, remembered backend). |
| `[llm]` | `active` profile, `fallback` profile. |
| `[llm.profiles.NAME]` | One backend: `provider` (`ollama`, `llamacpp`, `anthropic`, `openai`, `mock`), `model`, `base_url`, `api_key_env`, and provider-specific options. |
| `[resources]` | `memory_budget_mb`, `idle_unload_seconds`. |
| `[voice]` | `mode`, `language`, `barge_in`, `wake_word`, `follow_up_seconds`. |
| `[voice.stt]` / `[voice.tts]` | Speech engine, model, device, voice. |
| `[server]` | `host`, `port`, `token`. |
| `[skills]` | `plugin_dirs`, `disabled`. |
| `[agents]` | `dirs`. |
| `[trading]` | `data_source` (`auto`, `binance`, `yahoo`, `synthetic`), `watchlist`. |
| `[publish]` | `output_dir`, `webhook_url`, `wordpress_url`, `wordpress_user`, `wordpress_password_env`. |
| `[security]` | `authorized_targets`. |
| `[web]` | `allow_private_hosts`, `timeout_seconds`. |

## Architecture

```
 mic -> VAD -> STT (Whisper) -> Conversation -> Engine <-> LLMRouter --+-- local: Ollama / GGUF
                                     |            |                    +-- cloud: Anthropic / OpenAI / compatible
                                     |            +--> skills (concurrent tool calls)
                                     |            +--> delegate --> specialist agents (fresh context, parallel)
                                     |                              \--> background tasks (recurring, announced)
                                     +--> sentences -> TTS (Piper / OS / API) -> speakers

 window (pywebview) or browser <== local WebSocket / HTTP (127.0.0.1) ==> same Conversation
```

Every front end (window, terminal, microphone loop, browser) shares one `Conversation`, so a sentence spoken into the PC microphone shows up in the window and a typed message can be answered aloud.

```
jarvis/
  cli.py            command line entry point
  desktop.py        launcher: runtime thread + native window or browser
  app.py            assembles backends, skills, agents, sessions and tasks
  conversation.py   turn-taking, interruption, stop phrases, event fan-out
  config.py         defaults merged with jarvis.toml
  resources.py      on-demand loading and eviction of heavy components
  tasks.py          background and recurring agent runs
  core/             engine (model loop, tool calls), prompts, session history
  agents/           agent specs, built-in agents, TOML loader
  llm/              router and providers: ollama, anthropic, openai_compat, llamacpp, mock
  skills/           @skill decorator, registry, helpers; builtin/ holds the shipped skills
  voice/            microphone loop (native.py), VAD, STT, TTS, barge-in speech queue
  analysis/         market data fetching and technical indicators
  server/           FastAPI app, WebSocket protocol, static UI
agents/             your agent TOML files (summarizer.toml, experimental_template.toml.example)
plugins/            your skill files (example_dice.py)
scripts/            install.ps1, install.sh
start-jarvis.bat    Windows launcher
tests/              test suite
```

## Development

```sh
pip install -e ".[dev]"
pytest
ruff check .
```

Tests use fakes and recorded-shape responses, so they run offline with no audio hardware.

## Status and known limits

Jarvis was developed and tested on Linux in a sandbox with no audio hardware and no access to model or market-data hosts. The following are implemented but were not exercised against the real thing:

- Live Ollama, Anthropic and OpenAI calls (covered by wire-format tests).
- The Whisper and Piper engines and microphone/speaker I/O (covered with fakes).
- The Windows and macOS system voices.
- The pywebview window.
- The live Yahoo, Binance and DuckDuckGo endpoints (covered with recorded-shape tests).

Run `jarvis doctor` first and report what breaks.
