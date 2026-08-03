# SirenSeekers — Dashboard Simulator

An AI-powered emergency vehicle detection system. This is the frontend cockpit
dashboard: a simulated premium-SUV digital instrument cluster (in the spirit of
Range Rover Pivi Pro / BMW iDrive / Mercedes MBUX) that visualises real-time
audio + vision detection of emergency vehicles.

It currently runs entirely on **mocked data** generated in the browser. It is
built to be dropped in front of a FastAPI backend later — swap the
`useSimulatedDetection` hook for a WebSocket/REST client and the rest of the
UI needs no changes (see [Connecting a real backend](#connecting-a-real-backend)).

## Tech Stack

- React 19 + TypeScript
- Vite 8
- Tailwind CSS v4 (CSS-first config, `@theme` tokens in `src/index.css`)
- Framer Motion (animation)
- Lucide React (icons)
- Recharts (confidence trend chart)

## Getting Started

```bash
npm install
npm run dev
```

The app runs at `http://localhost:5173` by default.

Other scripts:

```bash
npm run build    # type-check with tsc -b, then production build to dist/
npm run preview  # preview the production build locally
npm run lint     # oxlint
```

## Project Structure

```
src/
├── assets/                 # static images/icons (empty by default)
├── components/
│   ├── alerts/              WarningBanner (animated critical alert)
│   ├── camera/               CameraPanel, feed placeholder, bounding box overlay
│   ├── charts/                Audio waveform + mic pulse visualisations
│   ├── dashboard/            Emergency panel, detection history, shared panel primitives
│   ├── layout/                 Top/bottom status bars, page shell
│   ├── radar/                 360° radar sweep + contact blip
│   └── status/                 System health panel + status lights
├── data/
│   └── mockData.ts          Detection scenarios, seed history, system health
├── hooks/
│   ├── useClock.ts             Live clock for the top bar
│   ├── useAudioWaveform.ts     Animated waveform bar generator
│   └── useSimulatedDetection.ts  Drives the whole simulation loop
├── pages/
│   └── DashboardPage.tsx    Composes every panel into the cockpit layout
├── types/
│   └── index.ts             Shared domain types (DetectionState, AlertLevel, ...)
├── utils/
│   ├── cn.ts                Tiny classnames helper
│   └── formatters.ts        Confidence/clock/alert-level/health formatting + styling
├── App.tsx
├── main.tsx
└── index.css                 Tailwind v4 theme tokens + cockpit-specific utilities
```

## How the simulation works

`useSimulatedDetection` (in `src/hooks/`) picks a weighted-random scenario
from `detectionScenarios` (in `src/data/mockData.ts`) every 5 seconds — cycling
between clear monitoring, possible detections, and confirmed emergency vehicle
sightings across different vehicle types and directions. Non-monitoring
scenarios are appended to a rolling detection history (max 10 entries).

Every panel is a pure function of the current `DetectionState`, so the whole
dashboard reacts consistently to each simulated tick — the camera bounding box,
the emergency state stepper, the radar contact, and the audio waveform all
move together.

## Connecting a real backend

Everything under `src/data/mockData.ts` and `src/hooks/useSimulatedDetection.ts`
is the seam meant for replacement. To wire up the FastAPI backend (TensorFlow
YAMNet for audio, OpenCV for vision):

1. Replace the body of `useSimulatedDetection` with a WebSocket or polling
   client that fetches/streams a `DetectionState` shaped payload.
2. Keep the `DetectionState`, `DetectionHistoryEntry`, and
   `SystemComponentHealth` types in `src/types/index.ts` as the contract with
   the backend (or update them together on both ends).
3. `SystemHealthPanel` currently reads static mock data
   (`systemHealthMock`) — point it at a `/health` endpoint the same way.

No other component needs to change since they only consume the typed state,
not the mock data source directly.
