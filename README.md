# Agent Puppy Off-Leash Park

A small site with a landing page and one interactive playground:

- **Landing page** (`index.html`) — the "off-leash park" hub.
- **Small-puppy playground** (`playground/`) — three puppy agents with different
  specialties cooperate to find a buried bone. Watch their memory and chat
  evolve over time.

## The puppies

| Puppy | Emoji | Breed | Specialty |
|---|---|---|---|
| **Scout** | 🐺 | Husky | Sees far, maps landmarks, relays messages |
| **Sniffer** | 🐶 | Beagle | Detects the hidden bone by smell |
| **Digger** | 🐕‍🦺 | Terrier | The only one who can dig |

Knowledge only flows through chat — no shared memory. Scout acts as the
communication hub, relaying Sniffer's bone location to Digger.

## Setup (for running new simulations)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then paste your Anthropic key into .env
```

## Running a simulation

```bash
# Scripted — no API calls, fast
python3 -m sim.run --no-llm --ticks 60 --seed 13

# Claude-powered natural-language chat
python3 -m sim.run --ticks 60 --seed 13
```

Each run writes `playground/runs/run-<timestamp>.json` and updates
`playground/runs/latest.json` (which the viewer loads by default).

## Viewing the site

Serve the folder over HTTP (the viewer uses `fetch()` which doesn't work
over `file://`):

```bash
python3 -m http.server 8000
# then open http://localhost:8000/
```

Click "Enter →" on the **Small-puppy playground** card.

## Project layout

```
.
├── index.html              # landing page (off-leash park)
├── playground/             # the small-puppy playground (replay viewer)
│   ├── index.html
│   ├── viewer.js
│   ├── style.css
│   └── runs/               # simulation logs (gitignored)
├── sim/                    # Python simulation
│   ├── world.py            # grid, entities, perception, physics
│   ├── puppy.py            # Scout / Sniffer / Digger classes
│   ├── memory.py           # per-puppy memory store
│   ├── brain.py            # Claude wrapper (with no-LLM fallback)
│   └── run.py              # main simulation loop
├── requirements.txt
├── .env.example
└── README.md
```
