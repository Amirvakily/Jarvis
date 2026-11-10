"use strict";
(() => {
  const $ = (id) => document.getElementById(id);
  const el = {
    orb: $("orb"), status: $("status"), hint: $("hint"), mic: $("mic-btn"), stop: $("stop-btn"),
    speak: $("speak-btn"), transcript: $("transcript"), input: $("text-input"), send: $("send-btn"),
    composer: $("composer"), backend: $("backend-select"), backendTag: $("backend-tag"),
    activity: $("activity"), agents: $("agents"), tasks: $("tasks"), memBar: $("mem-bar"),
    memText: $("mem-text"), memList: $("mem-list"), reset: $("reset-btn"),
  };

  const params = new URLSearchParams(location.search);
  const token = params.get("token");
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  const synth = "speechSynthesis" in window ? window.speechSynthesis : null;

  const S = {
    ws: null, connected: false, backoff: 1000, reconnectTimer: null,
    voice: { native: false }, busy: false, turnId: null, silencedTurn: null,
    listenMode: "none",   // native | server | browser | none
    speakMode: "none",    // server | browser | none
    speakOn: true, listening: false, nativeMuted: false, nativeState: "idle",
    userSpeaking: false, transcribing: false, transcribingTimer: null,
    level: 0, turnSentences: [], finished: [],
    pendingHeader: null,
  };
  try { if (localStorage.getItem("jarvis.speakOn") === "0") S.speakOn = false; } catch (e) { /* storage unavailable */ }

  /* ------------------------------------------------------------------ helpers */
  function authHeaders() { return token ? { Authorization: "Bearer " + token } : {}; }
  async function api(path, opts = {}) {
    const res = await fetch(path, { ...opts, headers: { ...authHeaders(), ...(opts.headers || {}) } });
    if (!res.ok) throw new Error(path + " " + res.status);
    return res.json();
  }
  function send(obj) {
    if (S.ws && S.ws.readyState === WebSocket.OPEN) { S.ws.send(JSON.stringify(obj)); return true; }
    return false;
  }
  function node(tag, cls, text) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  }
  function clear(n) { while (n.firstChild) n.removeChild(n.firstChild); }
  function trunc(s, n) { s = String(s == null ? "" : s).replace(/\s+/g, " ").trim(); return s.length > n ? s.slice(0, n - 1) + "…" : s; }
  function words(s) { return String(s).toLowerCase().replace(/[^a-z0-9' ]+/g, " ").split(/\s+/).filter(Boolean); }

  /* ------------------------------------------------------- tiny safe markdown */
  function inline(parent, text) {
    const re = /(\*\*[^*\n]+\*\*|`[^`\n]+`)/g;
    let last = 0, m;
    while ((m = re.exec(text))) {
      if (m.index > last) parent.appendChild(document.createTextNode(text.slice(last, m.index)));
      const tok = m[0];
      if (tok[0] === "`") parent.appendChild(node("code", "", tok.slice(1, -1)));
      else parent.appendChild(node("strong", "", tok.slice(2, -2)));
      last = m.index + tok.length;
    }
    if (last < text.length) parent.appendChild(document.createTextNode(text.slice(last)));
  }
  function renderMarkdown(target, text) {
    clear(target);
    const lines = String(text).split("\n");
    let para = [], code = null;
    const flush = () => {
      if (!para.length) return;
      const p = node("p");
      para.forEach((l, i) => { if (i) p.appendChild(document.createElement("br")); inline(p, l); });
      target.appendChild(p); para = [];
    };
    const endCode = () => { const pre = node("pre"); pre.appendChild(node("code", "", code.join("\n"))); target.appendChild(pre); code = null; };
    for (const line of lines) {
      if (/^\s*```/.test(line)) { if (code) endCode(); else { flush(); code = []; } continue; }
      if (code) { code.push(line); continue; }
      if (!line.trim()) flush(); else para.push(line);
    }
    if (code) endCode();
    flush();
  }

  /* --------------------------------------------------------------- transcript */
  const turns = new Map();       // turn_id -> {user, assistant, text, raf}
  let pendingUser = null, pendingTimer = null;

  function nearBottom() { const t = el.transcript; return t.scrollHeight - t.scrollTop - t.clientHeight < 80; }
  function scrollDown(force) { if (force || stick) el.transcript.scrollTop = el.transcript.scrollHeight; }
  let stick = true;
  el.transcript.addEventListener("scroll", () => { stick = nearBottom(); });

  function addMsg(role, text, extra) {
    const m = node("div", "msg " + role + (extra ? " " + extra : ""));
    if (role === "user") m.textContent = text;
    else renderMarkdown(m, text);
    el.transcript.appendChild(m);
    scrollDown();
    return m;
  }
  function setAssistantText(rec, text) {
    rec.text = text;
    if (rec.raf) return;
    rec.raf = requestAnimationFrame(() => {
      rec.raf = 0;
      renderMarkdown(rec.assistant, rec.text);
      scrollDown();
    });
  }
  function ensureTurn(id) {
    let rec = turns.get(id);
    if (!rec) { rec = { assistant: null, text: "", raf: 0 }; turns.set(id, rec); }
    if (!rec.assistant) { rec.assistant = addMsg("assistant", "", "streaming"); }
    return rec;
  }
  function showPending(text) {
    clearTimeout(pendingTimer);
    if (!text) { removePending(); return; }
    if (!pendingUser) { pendingUser = node("div", "msg user pending"); el.transcript.appendChild(pendingUser); }
    pendingUser.textContent = text;
    scrollDown();
  }
  function removePending() { clearTimeout(pendingTimer); if (pendingUser) { pendingUser.remove(); pendingUser = null; } }

  function rebuildTranscript(items) {
    clear(el.transcript); turns.clear(); pendingUser = null; stick = true;
    for (const it of items || []) {
      const m = addMsg(it.role === "user" ? "user" : "assistant", it.text || "");
      if (it.interrupted) m.classList.add("interrupted");
    }
    scrollDown(true);
  }

  /* ----------------------------------------------------------------- activity */
  const actRows = new Map();
  let actTurn = null;
  function actTrim() { while (el.activity.children.length > 100) el.activity.removeChild(el.activity.firstChild); }
  function actHeader(turnId) {
    if (actTurn === turnId) return;
    actTurn = turnId;
    const t = turns.get(turnId);
    const li = node("li", "turn", "Turn " + (turnId == null ? "" : turnId) + (t && t.userText ? " · " + trunc(t.userText, 40) : ""));
    el.activity.appendChild(li);
  }
  function actAdd(key, cls, label, meta, title) {
    const li = node("li", "act running " + cls);
    li.appendChild(node("span", "ic", "…"));
    li.appendChild(node("span", "label", label));
    li.appendChild(node("span", "meta", meta || ""));
    if (title) li.title = title;
    el.activity.appendChild(li);
    actRows.set(key, li);
    actTrim();
    el.activity.scrollTop = el.activity.scrollHeight;
  }
  function actEnd(key, ok, meta, title) {
    const li = actRows.get(key);
    if (!li) return;
    li.classList.remove("running"); li.classList.add(ok ? "ok" : "fail");
    li.querySelector(".ic").textContent = ok ? "✓" : "✗";
    if (meta != null) li.querySelector(".meta").textContent = meta;
    if (title) li.title = title;
    actRows.delete(key);
  }

  /* ------------------------------------------------------------------ sidebar */
  function renderBackend(b) {
    if (!b) return;
    clear(el.backend);
    for (const p of b.profiles || []) {
      const o = node("option", "", p.name + " — " + (p.model || p.provider) + (p.local ? " (local)" : " (cloud)"));
      o.value = p.name;
      el.backend.appendChild(o);
    }
    el.backend.value = b.active;
    const cur = (b.profiles || []).find((p) => p.name === b.active);
    el.backendTag.hidden = !cur;
    if (cur) {
      el.backendTag.textContent = cur.local ? "local" : "cloud";
      el.backendTag.className = "tag" + (cur.local ? "" : " cloud");
    }
  }
  function renderAgents(list) {
    clear(el.agents);
    for (const a of list || []) {
      const li = node("li");
      li.appendChild(node("span", "name", a.name));
      li.appendChild(node("span", "desc", a.description || ""));
      el.agents.appendChild(li);
    }
  }
  const tasks = new Map();
  function resultText(r) {
    if (r == null) return "";
    if (typeof r === "string") return r;
    try { return JSON.stringify(r); } catch (e) { return String(r); }
  }
  function renderTasks() {
    clear(el.tasks);
    for (const t of tasks.values()) {
      const li = node("li", "task");
      const top = node("div", "top");
      top.appendChild(node("span", "name", (t.agent ? t.agent + ": " : "") + trunc(t.task, 60)));
      const st = String(t.status || "");
      const active = !/^(done|completed|complete|finished|failed|error|cancelled|canceled|ok)$/i.test(st);
      const cls = active ? "running" : /fail|error|cancel/i.test(st) ? "failed" : "done";
      top.appendChild(node("span", "st " + cls, st));
      li.appendChild(top);
      if (t.result != null && t.result !== "") li.appendChild(node("span", "res", trunc(resultText(t.result), 240)));
      if (active) {
        const b = node("button", "btn", "Cancel");
        b.type = "button"; b.setAttribute("aria-label", "Cancel task " + trunc(t.task, 30));
        b.addEventListener("click", () => {
          b.disabled = true;
          api("/api/tasks/" + encodeURIComponent(t.id), { method: "DELETE" }).catch(() => { b.disabled = false; });
        });
        li.appendChild(b);
      }
      el.tasks.appendChild(li);
    }
  }
  function setTasks(list) { tasks.clear(); for (const t of list || []) tasks.set(t.id, t); renderTasks(); }
  function renderResources(r) {
    if (!r) return;
    const used = Number(r.used_mb) || 0, budget = Number(r.budget_mb) || 0;
    const pct = budget ? Math.min(100, (used / budget) * 100) : 0;
    el.memBar.style.width = pct + "%";
    el.memBar.classList.toggle("high", pct > 85);
    el.memText.textContent = Math.round(used) + " / " + Math.round(budget) + " MB used";
    clear(el.memList);
    for (const c of r.components || []) {
      const li = node("li", c.loaded ? "" : "off");
      li.appendChild(node("span", "", c.name + (c.in_use ? " · in use" : "")));
      li.appendChild(node("span", "", c.loaded ? Math.round(c.memory_mb) + " MB" : "unloaded"));
      el.memList.appendChild(li);
    }
  }
  async function refreshState() {
    try {
      const st = await api("/api/state");
      renderResources(st.resources);
      if (st.tasks) setTasks(st.tasks);
    } catch (e) { /* offline; reconnect logic handles it */ }
  }
  setInterval(() => { if (S.connected) refreshState(); }, 10000);

  /* ----------------------------------------------------------- status and orb */
  function assistantPlaying() { return playing || audioQueue.length > 0 || (synth && (synth.speaking || synth.pending) && S.speakMode === "browser"); }

  function updateUI() {
    let orb, text;
    if (!S.connected) { orb = "offline"; text = "Disconnected — retrying"; }
    else if (S.voice.native) {
      const st = S.nativeState;
      if (S.nativeMuted) { orb = "muted"; text = "Muted"; }
      else if (st === "listening") { orb = "listening"; text = "Listening…"; }
      else if (st === "transcribing") { orb = "transcribing"; text = "Transcribing…"; }
      else if (st === "thinking" || (S.busy && st === "idle")) { orb = "thinking"; text = "Thinking…"; }
      else if (st === "speaking") { orb = "speaking"; text = "Speaking…"; }
      else { orb = "idle"; text = "Ready"; }
    } else if (assistantPlaying()) { orb = "speaking"; text = "Speaking…"; }
    else if (S.busy) { orb = "thinking"; text = "Thinking…"; }
    else if (S.transcribing) { orb = "transcribing"; text = "Transcribing…"; }
    else if (S.listening) { orb = "listening"; text = S.userSpeaking ? "Listening… (hearing you)" : "Listening…"; }
    else { orb = "idle"; text = "Ready"; }
    el.orb.dataset.state = orb;
    el.status.textContent = text;

    const isNative = S.voice.native;
    const micOn = isNative ? !S.nativeMuted : S.listening;
    el.mic.setAttribute("aria-pressed", String(micOn));
    el.mic.classList.toggle("muted", isNative && S.nativeMuted);
    const label = isNative ? (micOn ? "Mute microphone" : "Unmute microphone") : micOn ? "Stop listening" : "Start listening";
    el.mic.setAttribute("aria-label", label);
    el.mic.title = label + " (Space)";
    el.mic.querySelector(".lbl").textContent = isNative ? (micOn ? "Mute" : "Unmute") : micOn ? "Listening" : "Mic";
    el.mic.disabled = !S.connected || S.listenMode === "none";

    el.speak.hidden = isNative || S.speakMode === "none" && !S.speakAvailable;
    el.speak.setAttribute("aria-pressed", String(S.speakOn));
    el.speak.setAttribute("aria-label", S.speakOn ? "Spoken replies on" : "Spoken replies off");
    el.stop.disabled = !S.connected || !(S.busy || assistantPlaying() || (isNative && S.nativeState === "speaking"));
    setLevel(S.level);
  }
  function setLevel(v) { S.level = v; el.orb.style.setProperty("--level", String(Math.max(0, Math.min(1, v)).toFixed(3))); }

  /* ------------------------------------------------------------ audio playback */
  let actx = null;
  const audioQueue = [];   // {text, p: Promise<AudioBuffer|null>}
  let playing = false, currentSource = null, playGen = 0;

  function audioCtx() {
    if (!actx) { const C = window.AudioContext || window.webkitAudioContext; if (C) actx = new C(); }
    if (actx && actx.state === "suspended") actx.resume().catch(() => {});
    return actx;
  }
  ["pointerdown", "keydown"].forEach((ev) => window.addEventListener(ev, () => { if (actx) audioCtx(); }, { passive: true }));

  function enqueueWav(header, buf) {
    if (!S.speakOn || S.speakMode !== "server") return;
    if (header.turn_id != null && header.turn_id === S.silencedTurn) return;
    const c = audioCtx();
    if (!c) return;
    const p = new Promise((resolve) => {
      c.decodeAudioData(buf, resolve, () => resolve(null));
    }).catch(() => null);
    audioQueue.push({ text: header.text || "", p, gen: playGen });
    pump();
  }
  async function pump() {
    if (playing) return;
    playing = true; updateUI();
    while (audioQueue.length) {
      const item = audioQueue.shift();
      const gen = playGen;
      const buf = await item.p;
      if (gen !== playGen) break;
      if (!buf) continue;
      await new Promise((resolve) => {
        const src = actx.createBufferSource();
        src.buffer = buf; src.connect(actx.destination);
        currentSource = src;
        src.onended = () => { if (currentSource === src) currentSource = null; resolve(); };
        src.start();
      });
      if (gen !== playGen) break;
      if (item.text) S.finished.push(item.text);
    }
    playing = false; updateUI();
  }
  function speakBrowser(text) {
    if (!synth || !S.speakOn || S.speakMode !== "browser" || !text) return;
    const u = new SpeechSynthesisUtterance(text);
    if (S.voice.language) u.lang = S.voice.language.length === 2 ? S.voice.language : S.voice.language;
    const gen = playGen;
    u.onend = () => { if (gen === playGen) S.finished.push(text); updateUI(); };
    u.onerror = () => updateUI();
    synth.speak(u);
    updateUI();
  }
  function stopPlayback() {
    playGen++;
    audioQueue.length = 0;
    if (currentSource) { try { currentSource.stop(); } catch (e) { /* already stopped */ } currentSource = null; }
    playing = false;
    if (synth) synth.cancel();
    S.silencedTurn = S.turnId;
    updateUI();
  }
  function bargeIn() {
    const heard = S.finished.join(" ").trim();
    stopPlayback();
    send({ type: "interrupt", heard: heard || null });
  }

  /* -------------------------------------------------- server-side STT capture */
  class Resampler {
    constructor(inRate, outRate) { this.ratio = inRate / outRate; this.acc = 0; this.cnt = 0; this.next = 0; }
    process(input) {
      if (this.ratio === 1) return input;
      const out = [];
      for (let i = 0; i < input.length; i++) {
        this.acc += input[i]; this.cnt++; this.next += 1;
        if (this.next >= this.ratio) { out.push(this.acc / this.cnt); this.acc = 0; this.cnt = 0; this.next -= this.ratio; }
      }
      return Float32Array.from(out);
    }
  }
  const RATE = 16000;
  const cap = { stream: null, ctx: null, node: null, src: null, sink: null, rs: null };
  const vad = { speaking: false, voicedMs: 0, silenceMs: 0, totalMs: 0, pre: [], preMs: 0, chunks: [], floor: 0.004 };
  function vadReset() { Object.assign(vad, { speaking: false, voicedMs: 0, silenceMs: 0, totalMs: 0, pre: [], preMs: 0, chunks: [] }); S.userSpeaking = false; }

  function toPcm16(f) {
    const o = new Int16Array(f.length);
    for (let i = 0; i < f.length; i++) { const v = Math.max(-1, Math.min(1, f[i])); o[i] = v < 0 ? v * 0x8000 : v * 0x7fff; }
    return o;
  }
  function onFrames(frames) {
    if (!S.listening) return;
    const f = cap.rs.process(frames);
    if (!f.length) return;
    let sum = 0;
    for (let i = 0; i < f.length; i++) sum += f[i] * f[i];
    const rms = Math.sqrt(sum / f.length);
    const ms = (f.length / RATE) * 1000;
    setLevel(Math.min(1, rms * 8));
    // Without barge-in, stay deaf while the assistant is talking.
    if (assistantPlaying() && !S.voice.barge_in) { vadReset(); return; }
    const thr = Math.max(0.006, vad.floor * 2.8);
    const voiced = rms > (vad.speaking ? thr * 0.7 : thr);
    if (!voiced) vad.floor += (rms - vad.floor) * (rms < vad.floor ? 0.2 : 0.03);
    vad.floor = Math.max(0.0005, Math.min(vad.floor, 0.05));
    const pcm = toPcm16(f);

    if (!vad.speaking) {
      vad.pre.push({ pcm, ms }); vad.preMs += ms;
      while (vad.preMs > 450 && vad.pre.length > 1) vad.preMs -= vad.pre.shift().ms;
      vad.voicedMs = Math.max(0, vad.voicedMs + (voiced ? ms : -ms * 0.5));  // leaky: tolerates syllable gaps
      if (vad.voicedMs >= 120) {
        vad.speaking = true; S.userSpeaking = true;
        vad.chunks = vad.pre.map((c) => c.pcm); vad.totalMs = vad.preMs; vad.silenceMs = 0;
        vad.pre = []; vad.preMs = 0;
        if (assistantPlaying() && S.voice.barge_in) bargeIn();
        updateUI();
      }
      return;
    }
    vad.chunks.push(pcm); vad.totalMs += ms;
    vad.silenceMs = voiced ? 0 : vad.silenceMs + ms;
    if (vad.silenceMs >= 750 || vad.totalMs >= 55000) endUtterance();
  }
  function endUtterance() {
    const speechMs = vad.totalMs - vad.silenceMs;
    const chunks = vad.chunks;
    vadReset();
    if (speechMs < 250) { updateUI(); return; }
    send({ type: "audio_start", sample_rate: RATE });
    for (const c of chunks) if (S.ws && S.ws.readyState === WebSocket.OPEN) S.ws.send(c.buffer);
    send({ type: "audio_end" });
    S.transcribing = true;
    clearTimeout(S.transcribingTimer);
    S.transcribingTimer = setTimeout(() => { S.transcribing = false; updateUI(); }, 8000);
    updateUI();
  }

  const WORKLET = `class JarvisCapture extends AudioWorkletProcessor {
    constructor(){ super(); this.buf = new Float32Array(1024); this.n = 0; }
    process(inputs){
      const ch = inputs[0] && inputs[0][0];
      if (ch) { for (let i = 0; i < ch.length; i++) { this.buf[this.n++] = ch[i];
        if (this.n === this.buf.length) { const out = this.buf; this.port.postMessage(out, [out.buffer]); this.buf = new Float32Array(1024); this.n = 0; } } }
      return true;
    }
  }
  registerProcessor('jarvis-capture', JarvisCapture);`;

  async function startCapture() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) throw new Error("no microphone API");
    cap.stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
    const C = window.AudioContext || window.webkitAudioContext;
    cap.ctx = new C();
    if (cap.ctx.state === "suspended") await cap.ctx.resume();
    cap.rs = new Resampler(cap.ctx.sampleRate, RATE);
    cap.src = cap.ctx.createMediaStreamSource(cap.stream);
    cap.sink = cap.ctx.createGain(); cap.sink.gain.value = 0; cap.sink.connect(cap.ctx.destination);
    let ok = false;
    if (cap.ctx.audioWorklet && window.AudioWorkletNode) {
      try {
        const url = URL.createObjectURL(new Blob([WORKLET], { type: "application/javascript" }));
        await cap.ctx.audioWorklet.addModule(url);
        URL.revokeObjectURL(url);
        cap.node = new AudioWorkletNode(cap.ctx, "jarvis-capture");
        cap.node.port.onmessage = (e) => onFrames(e.data);
        ok = true;
      } catch (e) { cap.node = null; }
    }
    if (!ok) {
      cap.node = cap.ctx.createScriptProcessor(2048, 1, 1);
      cap.node.onaudioprocess = (e) => onFrames(new Float32Array(e.inputBuffer.getChannelData(0)));
    }
    cap.src.connect(cap.node); cap.node.connect(cap.sink);
    vadReset();
  }
  function stopCapture() {
    try { if (cap.node) { cap.node.disconnect(); if (cap.node.port) cap.node.port.onmessage = null; cap.node.onaudioprocess = null; } } catch (e) { /* ignore */ }
    try { if (cap.src) cap.src.disconnect(); } catch (e) { /* ignore */ }
    if (cap.stream) cap.stream.getTracks().forEach((t) => t.stop());
    if (cap.ctx) cap.ctx.close().catch(() => {});
    cap.stream = cap.ctx = cap.node = cap.src = cap.sink = null;
    vadReset(); setLevel(0);
  }

  /* ------------------------------------------------ browser speech recognition */
  let rec = null, recWanted = false, recRestart = null;
  function isEcho(text) {
    const heard = words(text);
    if (!heard.length) return true;
    const said = new Set(words(S.turnSentences.join(" ")));
    if (!said.size) return false;
    const hit = heard.filter((w) => said.has(w)).length;
    return hit / heard.length >= 0.6;
  }
  function startRecognition() {
    rec = new SR();
    rec.continuous = true; rec.interimResults = true;
    const lang = S.voice.language;
    rec.lang = lang && lang.includes("-") ? lang : lang === "en" || !lang ? (navigator.language || "en-US") : lang;
    rec.onresult = (e) => {
      let interim = "";
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const r = e.results[i], t = r[0].transcript.trim();
        if (!t) continue;
        if (r.isFinal) {
          if (assistantPlaying() && isEcho(t)) continue;
          if (assistantPlaying() && S.voice.barge_in) bargeIn();
          removePending();
          stopPlaybackLocalOnly();
          send({ type: "user_text", text: t, spoken: true });
        } else interim += (interim ? " " : "") + t;
      }
      if (interim) {
        if (assistantPlaying() && isEcho(interim)) return;
        if (assistantPlaying() && S.voice.barge_in) bargeIn();
        S.userSpeaking = true; showPending(interim);
      } else if (pendingUser && !interim) { /* keep until final/turn_start */ }
      updateUI();
    };
    rec.onerror = (e) => {
      if (e.error === "not-allowed" || e.error === "service-not-allowed" || e.error === "audio-capture") {
        recWanted = false; S.listening = false;
        showHint("Microphone is unavailable (" + e.error + ") — type below, or run `jarvis setup-voice`.");
        updateUI();
      }
    };
    rec.onend = () => {
      S.userSpeaking = false;
      if (recWanted) { clearTimeout(recRestart); recRestart = setTimeout(() => { if (recWanted) { try { rec.start(); } catch (e) { /* already started */ } } }, 300); }
      updateUI();
    };
    recWanted = true;
    rec.start();
  }
  function stopRecognition() {
    recWanted = false; clearTimeout(recRestart);
    if (rec) { rec.onend = null; try { rec.abort(); } catch (e) { /* ignore */ } rec = null; }
    S.userSpeaking = false; removePending();
  }
  function stopPlaybackLocalOnly() { if (assistantPlaying()) stopPlayback(); }

  /* ------------------------------------------------------------------- listening */
  async function setListening(on) {
    if (S.voice.native) {
      S.nativeMuted = !on;
      send({ type: "native_mute", muted: !on });
      updateUI(); return;
    }
    if (on === S.listening) return;
    if (on) {
      audioCtx();
      try {
        if (S.listenMode === "server") await startCapture();
        else if (S.listenMode === "browser") startRecognition();
        else return;
        S.listening = true;
      } catch (e) {
        S.listening = false;
        showHint("Microphone unavailable (" + (e && e.name ? e.name : e) + ") — type below.");
        stopCapture();
      }
    } else {
      S.listening = false; S.transcribing = false;
      if (S.listenMode === "server") stopCapture(); else stopRecognition();
    }
    updateUI();
  }
  function toggleMic() {
    if (S.voice.native) setListening(S.nativeMuted); else setListening(!S.listening);
  }
  function showHint(text) {
    clear(el.hint);
    const parts = text.split(/`([^`]+)`/);
    parts.forEach((p, i) => { if (i % 2) el.hint.appendChild(node("code", "", p)); else if (p) el.hint.appendChild(document.createTextNode(p)); });
    el.hint.hidden = !text;
  }

  /* --------------------------------------------------------------- user actions */
  function submitText() {
    const text = el.input.value.trim();
    if (!text || !S.connected) return;
    if (assistantPlaying()) stopPlayback();
    send({ type: "user_text", text, spoken: false });
    el.input.value = ""; autosize();
    stick = true;
  }
  function stopAssistant() {
    if (S.voice.native) { send({ type: "interrupt", heard: null }); return; }
    const heard = S.finished.join(" ").trim();
    stopPlayback();
    send({ type: "interrupt", heard: heard || null });
  }
  function autosize() { el.input.style.height = "auto"; el.input.style.height = Math.min(el.input.scrollHeight, 140) + "px"; }

  el.composer.addEventListener("submit", (e) => { e.preventDefault(); submitText(); });
  el.input.addEventListener("input", autosize);
  el.input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); submitText(); }
  });
  el.mic.addEventListener("click", toggleMic);
  el.stop.addEventListener("click", stopAssistant);
  el.speak.addEventListener("click", () => {
    S.speakOn = !S.speakOn;
    try { localStorage.setItem("jarvis.speakOn", S.speakOn ? "1" : "0"); } catch (e) { /* ignore */ }
    if (!S.speakOn) stopPlayback();
    sendHello(); updateUI();
  });
  el.reset.addEventListener("click", () => { stopPlayback(); send({ type: "reset" }); });
  el.backend.addEventListener("change", () => send({ type: "set_backend", profile: el.backend.value }));
  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape") { if (S.busy || assistantPlaying() || S.nativeState === "speaking") stopAssistant(); return; }
    if (e.code === "Space" && !e.repeat && !e.ctrlKey && !e.metaKey && !e.altKey) {
      const t = e.target, tag = t && t.tagName;
      if (tag === "TEXTAREA" || tag === "INPUT" || tag === "SELECT" || tag === "BUTTON" || (t && t.isContentEditable)) return;
      e.preventDefault(); toggleMic();
    }
  });

  /* ------------------------------------------------------------- server events */
  function sendHello() {
    let speech = "none";
    if (!S.voice.native && S.speakOn) speech = S.speakMode === "server" ? "server" : S.speakMode === "browser" ? "browser" : "none";
    send({ type: "hello", speech });
  }

  function onReady(m) {
    const v = m.voice || {};
    S.voice = v;
    stopPlayback(); S.silencedTurn = null;
    S.busy = !!m.busy; S.turnId = null; S.finished = []; S.turnSentences = [];
    S.nativeMuted = !!v.native_muted; S.nativeState = v.native_state || "idle";
    actRows.clear(); actTurn = null; clear(el.activity);

    if (v.native) { S.listenMode = "native"; S.speakMode = "none"; }
    else {
      S.listenMode = v.server_stt ? "server" : SR ? "browser" : "none";
      S.speakMode = v.server_tts ? "server" : synth ? "browser" : "none";
    }
    S.speakAvailable = S.speakMode !== "none";
    if (S.listening && S.listenMode === "none") S.listening = false;
    showHint(!v.native && (S.listenMode === "none" || S.speakMode === "none")
      ? "Voice input needs setup — run `jarvis setup-voice`, or type below." : "");
    sendHello();

    const st = m.state || {};
    renderBackend(st.backend); renderAgents(st.agents); setTasks(st.tasks); renderResources(st.resources);
    rebuildTranscript(m.transcript);
    updateUI();
  }

  function handle(m) {
    const id = m.turn_id;
    switch (m.type) {
      case "ready": onReady(m); break;
      case "turn_start": {
        removePending(); S.transcribing = false; S.busy = true;
        S.turnId = id; S.silencedTurn = null; S.finished = []; S.turnSentences = [];
        if (assistantPlaying() && S.listenMode !== "native") stopPlayback();
        S.silencedTurn = null;
        addMsg("user", m.text || "", m.source === "voice" ? "voice" : "");
        const rec = ensureTurn(id); rec.userText = m.text || "";
        stick = true; scrollDown(true);
        break;
      }
      case "text_delta": {
        const r = ensureTurn(id);
        setAssistantText(r, r.text + (m.text || ""));
        break;
      }
      case "sentence":
        if (m.text) { S.turnSentences.push(m.text); if (id == null || id !== S.silencedTurn) speakBrowser(m.text); }
        break;
      case "tool_start":
        actHeader(id);
        actAdd((id || "") + ":" + m.id, "tool", m.name || "skill", m.agent ? "via " + m.agent : trunc(JSON.stringify(m.arguments || {}), 60));
        break;
      case "tool_end":
        actEnd((id || "") + ":" + m.id, m.ok !== false, m.preview ? trunc(m.preview, 80) : null, m.preview ? String(m.preview) : "");
        break;
      case "agent_start":
        actHeader(id); actAdd((id || "") + "@" + m.agent, "agent", m.agent, trunc(m.task, 80), m.task);
        break;
      case "agent_end":
        actEnd((id || "") + "@" + m.agent, m.ok !== false);
        break;
      case "turn_end": {
        S.busy = false;
        const r = turns.get(id);
        if (r) {
          if (!r.assistant) r.assistant = addMsg("assistant", "");
          r.assistant.classList.remove("streaming");
          const finalText = m.text != null && m.text !== "" ? m.text : r.text;
          if (r.raf) { cancelAnimationFrame(r.raf); r.raf = 0; }
          renderMarkdown(r.assistant, finalText);
          if (m.interrupted) r.assistant.classList.add("interrupted");
          if (!finalText) { if (m.error) r.assistant.remove(); else if (!m.interrupted) r.assistant.remove(); }
          scrollDown();
        }
        if (m.error) addMsg("system", "Error: " + m.error, "system");
        actRows.forEach((li, k) => { if (k.startsWith((id || "") + ":") || k.startsWith((id || "") + "@")) { li.classList.remove("running"); li.querySelector(".ic").textContent = "·"; actRows.delete(k); } });
        refreshState();
        break;
      }
      case "error": addMsg("system", m.message || "Error", "system"); break;
      case "transcript":
        S.transcribing = false; showPending(m.text);
        pendingTimer = setTimeout(removePending, 3000);
        break;
      case "announcement":
        if (m.text) {
          const a = addMsg("assistant", m.text, "announcement");
          void a;
          if (!S.busy && S.speakMode === "browser") speakBrowser(m.text);
        }
        break;
      case "backend": renderBackend(m); break;
      case "task_update": if (m.task) { tasks.set(m.task.id, m.task); renderTasks(); } break;
      case "voice_state": S.nativeState = m.state; break;
      case "voice_muted": S.nativeMuted = !!m.muted; break;
      case "mic_level": if (S.voice.native) setLevel(Number(m.level) || 0); return updateUI();
      case "speech_mode":
        if (m.mode === "browser" || m.mode === "server" || m.mode === "none") { S.speakMode = m.mode === "server" && !S.voice.server_tts ? "browser" : m.mode; }
        break;
      case "stopped": S.busy = false; stopPlayback(); break;
      case "reset":
        clear(el.transcript); turns.clear(); removePending(); clear(el.activity); actRows.clear(); actTurn = null;
        S.busy = false; stopPlayback(); break;
      case "audio": S.pendingHeader = m; return;
      default: break;
    }
    updateUI();
  }

  /* ----------------------------------------------------------------- connection */
  function connect() {
    clearTimeout(S.reconnectTimer);
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(proto + "//" + location.host + "/ws" + (token ? "?token=" + encodeURIComponent(token) : ""));
    ws.binaryType = "arraybuffer";
    S.ws = ws;
    ws.onopen = () => { S.connected = true; S.backoff = 1000; updateUI(); };
    ws.onmessage = (ev) => {
      if (typeof ev.data !== "string") {
        const h = S.pendingHeader; S.pendingHeader = null;
        if (h) enqueueWav(h, ev.data);
        return;
      }
      let m;
      try { m = JSON.parse(ev.data); } catch (e) { return; }
      handle(m);
    };
    ws.onclose = () => {
      if (S.ws !== ws) return;
      S.connected = false; S.busy = false; S.pendingHeader = null;
      stopPlayback(); updateUI();
      S.reconnectTimer = setTimeout(connect, S.backoff);
      S.backoff = Math.min(10000, Math.round(S.backoff * 1.6));
    };
    ws.onerror = () => { try { ws.close(); } catch (e) { /* ignore */ } };
  }

  updateUI();
  connect();
})();
