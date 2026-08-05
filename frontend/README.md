# SirenSeekers — Dashboard Simulator

An AI-powered emergency vehicle detection system. This is the frontend cockpit
dashboard: a simulated premium-SUV digital instrument cluster (in the spirit of
Range Rover Pivi Pro / BMW iDrive / Mercedes MBUX) that visualises real-time
audio + vision detection of emergency vehicles.

The dashboard connects live to the FastAPI backend over WebSocket
(`/ws/detection`) by default. A built-in mock simulation is still available
for offline demos — see [Live vs. mock mode](#4-live-vs-mock-mode).

## Tech Stack

- React 19 + TypeScript
- Vite 8
- Tailwind CSS v4 (CSS-first config, `@theme` tokens in `src/index.css`)
- Framer Motion (animation)
- Lucide React (icons)
- Recharts (confidence trend chart)

## 1. Starting the backend

The FastAPI backend lives in `../backend` (relative to this frontend
directory) and exposes the detection WebSocket at `/ws/detection`.

```bash
cd ../backend
python3 -m venv venv          # skip if venv/ already exists
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The backend emits one simulated fused audio+camera detection event per
second to every connected client, and also exposes:

- `GET /api/detection/latest` — most recent event (useful before a socket connects)
- `GET /api/detection/status` — active connection count + broadcast interval

By default it accepts CORS/WebSocket connections from
`http://localhost:5173` and `http://localhost:3000` (see `CORS_ORIGINS` in
`backend/config.py`). No `.env` file is required for local development —
everything has sane defaults — but `BROADCAST_INTERVAL_SECONDS`,
`AUDIO_DETECTION_PROBABILITY`, `CAMERA_DETECTION_PROBABILITY`, and
`CORS_ORIGINS` can all be overridden as real environment variables if needed.

## 2. Starting the frontend

```bash
npm install
npm run dev
```

The app runs at `http://localhost:5173` by default and connects to the
backend automatically on load.

Other scripts:

```bash
npm run build    # type-check with tsc -b, then production build to dist/
npm run preview  # preview the production build locally
npm run lint     # oxlint
```

## 3. Required `.env` values

`.env` is gitignored (see the repo-root `.gitignore`), so copy the template
after cloning:

```bash
cp .env.example .env
```

Both files define the same two variables:

```bash
# WebSocket URL for the FastAPI detection stream.
VITE_DETECTION_WS_URL=ws://localhost:8000/ws/detection

# true  -> use the built-in simulated detection generator (no backend needed)
# false -> connect to VITE_DETECTION_WS_URL via WebSocket
VITE_USE_MOCK_DATA=false
```

Vite only picks up `VITE_`-prefixed variables, and only at dev-server
start / build time — restart `npm run dev` after changing `.env`.

## 4. Live vs. mock mode

Set `VITE_USE_MOCK_DATA` in `.env`:

- `false` (default) — `DashboardPage` uses `useDetectionWebSocket`
  (`src/hooks/useDetectionWebSocket.ts`), which connects to
  `VITE_DETECTION_WS_URL` on mount and streams live detection events into
  every panel.
- `true` — `DashboardPage` uses the original `useSimulatedDetection` hook
  instead, generating weighted-random scenarios entirely in the browser. No
  backend is contacted at all in this mode.

The dashboard never silently falls back to mock data on its own — if the
WebSocket drops in live mode, the UI stays in live mode and shows a
reconnecting/disconnected state (see below) rather than switching to the
simulator.

## 5. Testing WebSocket disconnect and reconnect

With `VITE_USE_MOCK_DATA=false`, `npm run dev` running, and the backend up:

1. Confirm the **bottom status bar** shows `Connection: Connected` and the
   **System Health** panel shows `FastAPI Backend` as `Online`.
2. Stop the backend (`Ctrl+C` in its terminal, or `kill` the `uvicorn`
   process).
3. Within a few seconds the status bar should flip to `Disconnected` /
   `Reconnecting…`, and the System Health panel's `FastAPI Backend` entry
   should turn amber/red — all other panels keep showing the **last known
   detection** rather than going blank.
4. Restart the backend (`uvicorn main:app --reload --host 0.0.0.0 --port
   8000`). The hook retries with exponential backoff (1s, 2s, 4s, ... capped
   at 30s), so reconnection happens automatically within that window —
   watch the status bar return to `Connected` and fresh events start
   flowing again without a page reload.

You can also simulate a drop from the browser devtools: open the Network
tab, find the `detection` WS connection, and close it — the same recovery
sequence should play out.

## Project Structure

```
src/
├── components/
│   ├── alerts/               WarningBanner (animated critical alert)
│   ├── camera/                CameraPanel, feed placeholder, bounding box overlay
│   ├── charts/                 Audio waveform + mic pulse visualisations
│   ├── dashboard/             Emergency panel, detection history, shared panel primitives
│   ├── layout/                  Top/bottom status bars, page shell
│   ├── radar/                  360° radar sweep + contact blip
│   └── status/                  System health panel + status lights
├── data/
│   └── mockData.ts           Detection scenarios, seed history, system health
├── hooks/
│   ├── useClock.ts              Live clock for the top bar
│   ├── useAudioWaveform.ts      Animated waveform bar generator
│   ├── useSimulatedDetection.ts   Mock-mode simulation loop
│   └── useDetectionWebSocket.ts   Live-mode WebSocket client + reconnect logic
├── pages/
│   └── DashboardPage.tsx     Picks live vs. mock data source, composes every panel
├── types/
│   └── index.ts              Shared domain types (DetectionState, AlertLevel, ConnectionStatus, ...)
├── utils/
│   ├── cn.ts                 Tiny classnames helper
│   ├── formatters.ts         Confidence/clock/alert-level/health/connection formatting + styling
│   └── normalizeDetection.ts  Safe JSON parsing + backend→frontend schema normalization
├── App.tsx
├── main.tsx
└── index.css                  Tailwind v4 theme tokens + cockpit-specific utilities
```

## How live mode works

`useDetectionWebSocket` opens a WebSocket to `VITE_DETECTION_WS_URL` on
mount and exposes `{ detection, connectionStatus, lastMessageAt, error }`.
Every incoming frame is JSON-parsed and normalized by
`src/utils/normalizeDetection.ts` before it reaches state, so a malformed or
unexpected payload is dropped (with `error` set) instead of crashing the
dashboard.

The backend's `DetectionEvent` schema doesn't map 1:1 onto the frontend's
`DetectionState`:

| Backend (`backend/models.py`)                          | Frontend (`src/types/index.ts`)                                    |
| -------------------------------------------------------- | --------------------------------------------------------------------- |
| `alertLevel`: `Critical` / `Warning` / `Info` / `None`    | `alertLevel`: `Critical` / `Confirmed` / `Possible` / `Monitoring`   |
| `vehicleType` includes `Unknown`                          | `vehicleType` includes `None` (no `Unknown`)                          |
| `direction` is nullable                                   | `direction` is always one of 8 compass values                        |
| no bounding-box field                                     | `boundingBox: BoundingBox \| null`                                   |

`normalizeDetectionEvent` reconciles these: it maps alert levels and vehicle
types 1:1 onto their frontend equivalents, defaults a missing/null
`direction` to `Front`, and — since the backend doesn't emit real
vision-model coordinates yet — derives a placeholder `boundingBox` from
`direction` + `vehicleConfidence` so the camera overlay still has something
to render against live data.

`connectionStatus` moves through `connecting → connected`, and on a drop
through `error`/`disconnected → reconnecting → connecting → connected`,
retrying with exponential backoff (capped at 30s) until the socket comes
back. The last valid `detection` is never cleared on disconnect — only a
new valid message ever replaces it — so every panel keeps showing the last
known state while the status bar/health panel surface the connection issue.

## How mock mode works

`useSimulatedDetection` (used when `VITE_USE_MOCK_DATA=true`) picks a
weighted-random scenario from `detectionScenarios` (in
`src/data/mockData.ts`) every 5 seconds — cycling between clear monitoring,
possible detections, and confirmed emergency vehicle sightings across
different vehicle types and directions. Non-monitoring scenarios are
appended to a rolling detection history (max 10 entries), same as live mode.

Every panel is a pure function of the current `DetectionState`, so the whole
dashboard reacts consistently to each tick (mock or live) — the camera
bounding box, the emergency state stepper, the radar contact, and the audio
waveform all move together.
