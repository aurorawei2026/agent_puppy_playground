/**
 * Puppy Playground — replay viewer.
 *
 * Loads a run.json (meta + frames[]) and renders:
 *   - Animated puppies on a grass canvas with fence + landmarks
 *   - Speech bubbles whenever a puppy talks
 *   - Three live memory cards that grow as you scrub through time
 *   - Play / pause / step / scrub controls
 *
 * Run JSON contract (produced by sim/run.py):
 *   meta: { grid:[w,h], landmarks, bone_pos, completed, use_llm, ... }
 *   frames: [{
 *     t, done, bone:{pos,found},
 *     puppies: { name: { role, emoji, breed, pos:[x,y], action, memory:{observations,messages,knowledge} } },
 *     messages: [{ from, to, text }],
 *   }, ...]
 */

(() => {
  const CANVAS_PX = 600;
  const BUBBLE_MS = 2200;          // how long a speech bubble lingers
  const ROLES = ["scout", "sniffer", "digger"];
  const ROLE_COLORS = {
    scout:   "#4a6fa5",
    sniffer: "#b45e89",
    digger:  "#6a9955",
  };
  const LANDMARK_EMOJI = {
    tree:       "🌳",
    water_bowl: "💧",
    toy_chest:  "🧸",
    rock:       "🪨",
  };

  // ------------------------------------------------------------ state
  const state = {
    run: null,                     // full loaded run JSON
    frameIdx: 0,
    playing: false,
    speedMs: 400,
    bubbles: [],                   // { puppy, text, bornAt (ms) }
    lastAnimatedAt: 0,
    prevMemCounts: { scout: 0, sniffer: 0, digger: 0 },
  };

  // ------------------------------------------------------------ DOM
  const $ = (id) => document.getElementById(id);
  const canvas = $("grid");
  const ctx = canvas.getContext("2d");
  const speechLayer = $("speech-layer");
  const scrubber = $("scrubber");
  const tickLabel = $("tick-label");
  const metaEl = $("meta");
  const memoryCards = $("memory-cards");
  const filePicker = $("file-picker");
  const runSelect = $("run-select");
  const inputSeed = $("input-seed");
  const inputTicks = $("input-ticks");
  const inputLlm = $("input-llm");
  const cliCommand = $("cli-command");
  const btnCopy = $("btn-copy");

  // ------------------------------------------------------------ loading
  async function loadRun(url) {
    try {
      const resp = await fetch(url);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      return await resp.json();
    } catch (e) {
      metaEl.innerHTML =
        `<span style="color:#b45e89">Couldn't load <code>${url}</code> — ` +
        `run <code>python -m sim.run --no-llm</code> first, or drop a JSON file below. (${e.message})</span>`;
      return null;
    }
  }

  function pickInitialRunUrl() {
    const params = new URLSearchParams(window.location.search);
    const named = params.get("run");
    if (named) return `runs/${named}`;
    // Default to whichever run is selected in the dropdown.
    return `runs/${runSelect.value}`;
  }

  async function loadRunByName(filename) {
    const run = await loadRun(`runs/${filename}`);
    if (run) {
      clearBubbles();
      setRun(run);
    }
  }

  function setRun(run) {
    state.run = run;
    state.frameIdx = 0;
    state.bubbles = [];
    state.prevMemCounts = { scout: 0, sniffer: 0, digger: 0 };
    scrubber.min = 0;
    scrubber.max = run.frames.length - 1;
    scrubber.value = 0;
    renderMeta();
    renderMemoryCards(true);
    draw();
  }

  // ------------------------------------------------------------ geometry
  function cellPx() {
    const [w] = state.run.meta.grid;
    return CANVAS_PX / w;
  }

  function cellToPx(x, y) {
    const c = cellPx();
    return { x: x * c + c / 2, y: y * c + c / 2 };
  }

  // ------------------------------------------------------------ rendering
  function draw() {
    if (!state.run) return;
    const frame = state.run.frames[state.frameIdx];
    drawBackground();
    drawLandmarks();
    drawBone(frame);
    drawPuppies(frame);
    updateHud(frame);
  }

  function drawBackground() {
    const [w, h] = state.run.meta.grid;
    const c = cellPx();

    // Grass with alternating checker tint
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        ctx.fillStyle = (x + y) % 2 === 0 ? "#b8d8a8" : "#a4c994";
        ctx.fillRect(x * c, y * c, c, c);
      }
    }

    // Fence border
    ctx.strokeStyle = "#7c4a1e";
    ctx.lineWidth = 4;
    ctx.strokeRect(2, 2, CANVAS_PX - 4, CANVAS_PX - 4);

    // Dashed inner line for a playful "fence post" feel
    ctx.strokeStyle = "rgba(124, 74, 30, 0.35)";
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 6]);
    ctx.strokeRect(6, 6, CANVAS_PX - 12, CANVAS_PX - 12);
    ctx.setLineDash([]);
  }

  function drawLandmarks() {
    const c = cellPx();
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.font = `${Math.floor(c * 0.85)}px system-ui`;
    for (const [name, pos] of Object.entries(state.run.meta.landmarks)) {
      const emoji = LANDMARK_EMOJI[name] || "⬜";
      const { x, y } = cellToPx(pos[0], pos[1]);
      ctx.fillText(emoji, x, y);
    }
  }

  function drawBone(frame) {
    const c = cellPx();
    const [bx, by] = frame.bone.pos;
    const { x, y } = cellToPx(bx, by);
    ctx.save();
    if (!frame.bone.found) {
      // Reveal buried bone location as a subtle dashed ring (cheating for the viewer).
      ctx.strokeStyle = "rgba(217, 119, 6, 0.45)";
      ctx.setLineDash([3, 4]);
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(x, y, c * 0.42, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
    } else {
      ctx.font = `${Math.floor(c * 0.9)}px system-ui`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("🦴", x, y);
    }
    ctx.restore();
  }

  function drawPuppies(frame) {
    const c = cellPx();
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.font = `${Math.floor(c * 0.95)}px system-ui`;
    for (const name of ROLES) {
      const p = frame.puppies[name];
      if (!p) continue;
      const pos = interpolatedPos(name);
      ctx.save();
      // soft role-colored shadow disk under the puppy
      ctx.fillStyle = ROLE_COLORS[p.role] + "33";
      ctx.beginPath();
      ctx.arc(pos.x, pos.y + c * 0.18, c * 0.35, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillText(p.emoji, pos.x, pos.y);
      ctx.restore();

      // If action is dig, pulse a little rings
      if (p.action && p.action.startsWith("dig")) {
        ctx.save();
        ctx.strokeStyle = p.action === "dig:success" ? "#d97706" : "#8a8380";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, c * 0.55, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();
      }
    }
  }

  /**
   * Smoothly interpolate a puppy's drawn position between the previous frame
   * and the current frame, based on how long we've been on the current frame.
   */
  function interpolatedPos(name) {
    const cur = state.run.frames[state.frameIdx];
    const prev = state.frameIdx > 0 ? state.run.frames[state.frameIdx - 1] : cur;
    const [cx, cy] = cur.puppies[name].pos;
    const [px, py] = prev.puppies[name].pos;

    let t = 1;
    if (state.playing && state.speedMs > 0) {
      const now = performance.now();
      t = Math.min(1, (now - state.lastAnimatedAt) / state.speedMs);
    }
    const x = px + (cx - px) * t;
    const y = py + (cy - py) * t;
    return cellToPx(x, y);
  }

  // ------------------------------------------------------------ speech bubbles
  function spawnBubblesFor(frame) {
    // Bubbles are short-lived DOM nodes anchored above the sender's *canvas* pos.
    if (!frame.messages) return;
    for (const m of frame.messages) {
      const div = document.createElement("div");
      const senderRole = frame.puppies[m.from]?.role || "scout";
      div.className = `bubble ${senderRole}`;
      div.innerHTML = `<div class="who">${m.from} → ${m.to}</div>${escapeHtml(m.text)}`;
      speechLayer.appendChild(div);
      state.bubbles.push({
        el: div,
        sender: m.from,
        bornAt: performance.now(),
      });
    }
  }

  function updateBubbles() {
    const now = performance.now();
    const survivors = [];
    for (const b of state.bubbles) {
      const age = now - b.bornAt;
      if (age > BUBBLE_MS) {
        b.el.remove();
        continue;
      }
      // Position over the sender's current pixel position
      const pos = interpolatedPos(b.sender);
      b.el.style.left = `${pos.x}px`;
      b.el.style.top = `${pos.y}px`;
      b.el.style.opacity = String(1 - Math.max(0, (age - (BUBBLE_MS - 400)) / 400));
      survivors.push(b);
    }
    state.bubbles = survivors;
  }

  function clearBubbles() {
    for (const b of state.bubbles) b.el.remove();
    state.bubbles = [];
  }

  // ------------------------------------------------------------ memory cards
  function renderMemoryCards(rebuild = false) {
    if (!state.run) return;
    const frame = state.run.frames[state.frameIdx];
    if (rebuild) memoryCards.innerHTML = "";

    for (const name of ROLES) {
      const p = frame.puppies[name];
      if (!p) continue;
      const id = `card-${name}`;
      let card = document.getElementById(id);
      if (!card) {
        card = document.createElement("div");
        card.id = id;
        card.className = `card ${name}`;
        memoryCards.appendChild(card);
      }
      const mem = p.memory;
      const obsCount = mem.observations.length;
      const msgCount = mem.messages.length;
      const totalCount = obsCount + msgCount;
      const prev = state.prevMemCounts[name] || 0;

      const obsHtml = mem.observations.length
        ? `<ul>${mem.observations
            .slice(-8)
            .map(
              (o) =>
                `<li><span class="t">[t=${o.tick}]</span> ${iconFor(o.kind)} ${escapeHtml(o.content)}</li>`
            )
            .join("")}</ul>`
        : `<div class="empty">nothing yet</div>`;

      const msgHtml = mem.messages.length
        ? `<ul>${mem.messages
            .slice(-6)
            .map(
              (m) =>
                `<li><span class="t">[t=${m.tick}]</span> <b>${escapeHtml(
                  m.sender
                )}:</b> "${escapeHtml(m.text)}"</li>`
            )
            .join("")}</ul>`
        : `<div class="empty">no chat yet</div>`;

      const knowHtml = Object.keys(mem.knowledge).length
        ? `<ul>${Object.entries(mem.knowledge)
            .map(
              ([k, v]) =>
                `<li><code>${escapeHtml(k)}</code>: ${escapeHtml(JSON.stringify(v))}</li>`
            )
            .join("")}</ul>`
        : `<div class="empty">nothing yet</div>`;

      card.innerHTML = `
        <h3>${p.emoji} ${name} <span class="breed">${p.breed}</span></h3>
        <div class="section-title">Observations (${obsCount})</div>
        ${obsHtml}
        <div class="section-title">Chat received (${msgCount})</div>
        ${msgHtml}
        <div class="section-title">Knowledge</div>
        ${knowHtml}
      `;

      if (totalCount > prev) card.classList.add("mem-new");
      else card.classList.remove("mem-new");
      state.prevMemCounts[name] = totalCount;
      // Clear the flash after animation duration
      setTimeout(() => card.classList.remove("mem-new"), 600);
    }
  }

  function iconFor(kind) {
    return { see: "👀", smell: "👃", dig: "⛏", hear: "👂" }[kind] || "•";
  }

  // ------------------------------------------------------------ HUD / controls
  function renderMeta() {
    const m = state.run.meta;
    const diff = m.difficulty
      ? `difficulty <b>${m.difficulty}</b> · `
      : "";
    const senses = m.config
      ? `(vision ${m.config.scout_vision}, smell ${m.config.sniffer_range}, chat ${m.config.chat_range}) · `
      : "";
    const ranFor =
      m.ticks_run != null
        ? `ran <b>${m.ticks_run}</b> tick${m.ticks_run === 1 ? "" : "s"}${
            m.completed ? " (stopped early — bone found)" : ""
          } · `
        : "";
    metaEl.innerHTML = `
      ${diff}${senses}
      seed <b>${m.seed}</b> ·
      grid <b>${m.grid[0]}×${m.grid[1]}</b> ·
      bone at <b>(${m.bone_pos[0]}, ${m.bone_pos[1]})</b> ·
      ${ranFor}${m.use_llm ? "Claude-powered chat" : "scripted chat"} ·
      ${m.completed ? '<span class="done">✅ bone found</span>' : "in progress"}
    `;
  }

  function updateHud(frame) {
    scrubber.value = state.frameIdx;
    const total = state.run.frames.length - 1;
    tickLabel.textContent = `tick ${frame.t} / ${total}`;
  }

  // ------------------------------------------------------------ time loop
  function setFrame(i, { playBubbles = true } = {}) {
    if (!state.run) return;
    const clamped = Math.max(0, Math.min(state.run.frames.length - 1, i));
    const moving = clamped !== state.frameIdx;
    state.frameIdx = clamped;
    state.lastAnimatedAt = performance.now();
    if (playBubbles && moving) {
      spawnBubblesFor(state.run.frames[state.frameIdx]);
    }
    renderMemoryCards();
    draw();
  }

  function tick() {
    if (!state.playing) return;
    const now = performance.now();
    if (now - state.lastAnimatedAt >= state.speedMs) {
      const next = state.frameIdx + 1;
      if (next >= state.run.frames.length) {
        state.playing = false;
        $("btn-play").textContent = "▶ Play";
      } else {
        setFrame(next);
      }
    }
    updateBubbles();
    draw();
    requestAnimationFrame(tick);
  }

  // ------------------------------------------------------------ wiring
  $("btn-play").addEventListener("click", () => {
    state.playing = !state.playing;
    $("btn-play").textContent = state.playing ? "⏸ Pause" : "▶ Play";
    if (state.playing) {
      state.lastAnimatedAt = performance.now();
      requestAnimationFrame(tick);
    }
  });
  $("btn-step").addEventListener("click", () => setFrame(state.frameIdx + 1));
  $("btn-step-back").addEventListener("click", () => {
    clearBubbles();
    setFrame(state.frameIdx - 1, { playBubbles: false });
  });
  $("btn-restart").addEventListener("click", () => {
    clearBubbles();
    state.prevMemCounts = { scout: 0, sniffer: 0, digger: 0 };
    setFrame(0, { playBubbles: false });
  });

  scrubber.addEventListener("input", (e) => {
    clearBubbles();
    setFrame(Number(e.target.value), { playBubbles: false });
  });

  $("speed").addEventListener("change", (e) => {
    state.speedMs = Number(e.target.value);
  });

  filePicker.addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    try {
      const text = await file.text();
      clearBubbles();
      setRun(JSON.parse(text));
    } catch (err) {
      metaEl.textContent = `Error parsing file: ${err.message}`;
    }
  });

  runSelect.addEventListener("change", () => {
    state.playing = false;
    $("btn-play").textContent = "▶ Play";
    loadRunByName(runSelect.value);
    syncCliCommand();
  });

  // --------- live CLI command builder + copy ---------
  function diffFromFilename(f) {
    const m = f.match(/sample-(easy|medium|hard)\.json/);
    return m ? m[1] : "medium";
  }

  function syncCliCommand() {
    const diff = diffFromFilename(runSelect.value);
    const seed = Math.max(0, Number(inputSeed.value) || 0);
    const ticks = Math.max(1, Number(inputTicks.value) || 1);
    const llmFlag = inputLlm.checked ? "" : " --no-llm";
    cliCommand.textContent =
      `python -m sim.run${llmFlag} --difficulty ${diff} --seed ${seed} --ticks ${ticks}`;
  }

  [inputSeed, inputTicks, inputLlm].forEach((el) =>
    el.addEventListener("input", syncCliCommand)
  );

  btnCopy.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(cliCommand.textContent);
      btnCopy.textContent = "Copied ✓";
      btnCopy.classList.add("copied");
      setTimeout(() => {
        btnCopy.textContent = "Copy";
        btnCopy.classList.remove("copied");
      }, 1400);
    } catch (e) {
      // Fallback for older browsers / http contexts
      const r = document.createRange();
      r.selectNodeContents(cliCommand);
      const sel = window.getSelection();
      sel.removeAllRanges();
      sel.addRange(r);
      document.execCommand("copy");
      sel.removeAllRanges();
      btnCopy.textContent = "Copied ✓";
      setTimeout(() => { btnCopy.textContent = "Copy"; }, 1400);
    }
  });

  syncCliCommand();

  // ------------------------------------------------------------ utils
  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  // ------------------------------------------------------------ boot
  (async function boot() {
    // If URL has ?run=foo.json and that file matches a dropdown option,
    // sync the dropdown so the UI stays consistent with the URL.
    const params = new URLSearchParams(window.location.search);
    const named = params.get("run");
    if (named) {
      const options = Array.from(runSelect.options).map((o) => o.value);
      if (options.includes(named)) runSelect.value = named;
    }
    syncCliCommand();
    const url = pickInitialRunUrl();
    const run = await loadRun(url);
    if (run) setRun(run);
  })();
})();
