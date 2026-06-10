# CLAUDE.md

Guidance for Claude Code (and other AI assistants) when working in this repository.

---

## Project Overview

**Tune-Crafter** — an interactive web app for crafting/manipulating tunes using
**hand gestures** captured via webcam. Built for the HCI909 (Advanced Programming
of Interactive Systems) course.

- **Version:** `0.1.0` (no git tags / formal releases yet)
- **Package:** `tune-crafter-react-webapp`
- **Deployed:** GitHub Pages — https://scavalleroo.github.io/Tune-Crafter

The app reads webcam video, detects hand landmarks (MediaPipe / TensorFlow.js),
runs them through gesture state machines, and maps recognized gestures to audio
actions (play/pause, cut/loop regions, volume, effects) on a WaveSurfer waveform.
A separate Python Socket.IO server relays events (e.g. heart-rate from a Wear OS
companion app) to connected web clients.

---

## Tech Stack

- **Frontend:** React 18 + TypeScript, built with Vite 4
- **Styling:** Bootstrap 5 / react-bootstrap, SCSS (Sass)
- **Gesture / ML:** `@mediapipe/holistic`, `@mediapipe/tasks-vision`,
  `@mediapipe/hand-pose-detection` helpers, `@tensorflow/tfjs-node`
- **Audio:** `howler`, `wavesurfer-react` / `@types/wavesurfer.js`
- **Realtime:** `socket.io-client` (web) ↔ Python `python-socketio` + `eventlet` (server)
- **Analytics:** `react-ga4`
- **Server:** Python 3.10 (`src/server/SocketServer.py`)

---

## Commands

```bash
npm install            # install JS dependencies

npm run dev            # start Vite dev server (primary local workflow)
npm run build          # tsc type-check + vite production build -> dist/
npm run preview        # preview the production build
npm run lint           # eslint (ts,tsx) — max-warnings 0, fails on any warning

npm run deploy         # build + publish dist/ to GitHub Pages (gh-pages)

# Python Socket.IO server (relays events to web clients)
npm run socket-server      # macOS/Linux: python3.10 ./src/server/SocketServer.py
npm run socket-server-win  # Windows: python ./src/server/SocketServer.py
# Server deps: pip install python-socketio eventlet python-dotenv
```

### Required environment (`.env` at repo root)

```
VITE_IP_ADDRESS_SERVER_SOCKET=<machine-ip>   # e.g. 10.245.217.76
VITE_PORT_SERVER_SOCKET=<port>               # e.g. 3000
```

Both the Vite app (`src/utils/SocketClient.tsx`) and the Python server
(`SocketServer.py`) read these. The web client builds its URL as
`http://<ip>:<port>/`. If the Wear OS companion app is used, set the same IP in
its `strings.xml`.

---

## Architecture

```
Webcam ─▶ GestureComponent ─▶ MediaPipe/TF landmarks ─▶ GestureModel
                                                            │
                       GesturesFSM (per-gesture state machines)
                                                            │
                                                            ▼
                                       AudioManager ─▶ WaveSurfer / Howler
```

Key files:

- [src/App.tsx](src/App.tsx) — root; enables webcam (`getUserMedia`), wires the
  sidebar, waveform, gesture and speech components, owns the `AudioManager`.
- [src/components/GestureComponent.tsx](src/components/GestureComponent.tsx) —
  captures video, extracts hand landmark `Coordinates`, drives recognition.
  *(currently the actively-edited file)*
- [src/models/GestureModel.tsx](src/models/GestureModel.tsx) — holds the current
  state for each gesture FSM and translates transitions into `AudioManager` calls
  and WaveSurfer regions.
- [src/utils/GesturesFSM.tsx](src/utils/GesturesFSM.tsx) — enums/state machines:
  Cut, PlayPause, per-finger (Index/Middle/Ring/Picky), Volume, Effects.
- [src/AudioManager.tsx](src/AudioManager.tsx) — audio playback/region control
  over WaveSurfer + Howler.
- [src/components/AudioWaveComponent.tsx](src/components/AudioWaveComponent.tsx) —
  WaveSurfer waveform UI.
- [src/components/SpeechComponent.tsx](src/components/SpeechComponent.tsx) —
  WebSpeech API voice input.
- [src/utils/SocketClient.tsx](src/utils/SocketClient.tsx) — Socket.IO client
  (`autoConnect: false`).
- [src/server/SocketServer.py](src/server/SocketServer.py) — Socket.IO server;
  tracks web client SIDs, broadcasts `heart_beat_event` to them.
- [src/CurrentMode.tsx](src/CurrentMode.tsx) — global mutable mode flag
  (default `"normal"`; e.g. piano/christmas modes seen in history).
- [src/utils/helpers.tsx](src/utils/helpers.tsx) — geometry helpers
  (`calculateAngle`, `closedPoints`, `hasGetUserMedia`).

Default audio asset: `assets/sounds/audio.mp3` (under `public/`).

---

## Conventions & Gotchas

- **Components use the `.tsx` extension even when they contain no JSX** (e.g.
  `AudioManager.tsx`, `helpers.tsx`). Keep that convention.
- **Webcam requires a secure context** — `getUserMedia` works on `localhost` and
  HTTPS only. Grant camera permission when testing.
- **Lint is strict:** `--max-warnings 0`. A single warning fails `npm run lint`
  (and CI/deploy). Run lint before committing.
- **Port conflicts:** prior debugging notes (`src/components/random/GitHub Copilot.md`)
  cover an "address already in use" error on port 5000 — if the socket/dev server
  won't bind, find the process with `lsof -i :<port>` and kill it, or change the
  port in `.env`.
- **`.DS_Store` files** keep showing up as modified — macOS noise; do not commit
  changes to them (consider adding to `.gitignore`).
- **Browser quirks:** `App.tsx` includes Safari detection; verify gesture/audio
  behavior in Chrome (primary) and note Safari differences.
- **No automated test suite exists yet.** There is no test runner configured —
  add one (and tests) when introducing non-trivial logic (see workflow below).

---

## Working Style — Senior Engineer Plan Mode

> Adapted from Garry Tan's (YC President/CEO) Claude Code "senior engineer" prompt.
> Use this when starting any non-trivial change in this repo.

Before writing any code, review the plan thoroughly. Do **NOT** start
implementation until the review is complete and the direction is approved.

For every issue or recommendation:
- Explain the concrete tradeoffs
- Give an opinionated recommendation
- Ask for input before proceeding

**Engineering principles**
- Prefer DRY — aggressively flag duplication
- Well-tested code is mandatory (better too many tests than too few)
- Code should be "engineered enough" — not fragile or hacky, but not over-engineered
- Optimize for correctness and edge cases over speed of implementation
- Prefer explicit solutions over clever ones

**Review dimensions** (cover each for a BIG change):
1. **Architecture** — system design, component boundaries, coupling, data flow,
   bottlenecks, single points of failure, security boundaries.
2. **Code Quality** — structure/module organization, DRY violations, error
   handling & missing edge cases, tech-debt risk, over-/under-engineering.
3. **Tests** — coverage (unit/integration/e2e), assertion quality, missing edge
   cases, untested failure scenarios.
4. **Performance** — inefficient I/O, memory risks, CPU hotspots, caching
   opportunities, latency/scalability (relevant for per-frame gesture processing).

**For each issue found, provide:**
1. Clear description of the problem
2. Why it matters
3. 2–3 options (including "do nothing" if reasonable)
4. For each option: Effort / Risk / Impact / Maintenance cost
5. The recommended option and why — then ask for approval before moving forward.

**Start mode** — first ask: *Is this a BIG change or a SMALL change?*
- **BIG:** review all sections step-by-step; highlight the top 3–4 issues per section; pause for feedback after each.
- **SMALL:** ask one focused question per section; keep the review concise.

**Output style:** structured, concise, opinionated (not neutral summaries),
focused on real risks and tradeoffs — think like a Staff/Senior Engineer
reviewing a production system.

---

## Team

Alessandro Cavallotti · Matteo Fornara · Shubhankar
