"""Run a simulation and write a JSON log for the replay viewer.

Usage:
    python -m sim.run --ticks 50 --seed 42           # with LLM
    python -m sim.run --ticks 50 --seed 42 --no-llm  # without LLM

Output: playground/runs/run-<timestamp>.json
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional if user exports ANTHROPIC_API_KEY manually

from sim import puppy as puppy_mod
from sim.brain import Brain
from sim.puppy import Digger, Scout, Sniffer
from sim.world import World


RUNS_DIR = Path(__file__).resolve().parent.parent / "playground" / "runs"

# Preset sense/chat ranges per difficulty.
# Lower sense ranges = puppies have to wander more before discovering things,
# and narrower chat range means Scout's relay role matters more.
DIFFICULTIES: dict[str, dict[str, int]] = {
    "easy":   {"scout_vision": 9, "sniffer_range": 6, "chat_range": 11},
    "medium": {"scout_vision": 7, "sniffer_range": 4, "chat_range": 8},
    "hard":   {"scout_vision": 5, "sniffer_range": 3, "chat_range": 5},
}


def build_world(seed: int, use_llm: bool) -> World:
    world = World(seed=seed)
    rng = random.Random(seed + 1)
    brain = Brain(use_llm=use_llm)

    world.add_puppy(Scout(  name="scout",   pos=(2, 2),                               brain=brain, rng=rng))
    world.add_puppy(Sniffer(name="sniffer", pos=(world.width - 3, 2),                 brain=brain, rng=rng))
    # Digger starts near the middle so there's a realistic chance of ending up
    # in chat range of Scout or Sniffer at some point.
    world.add_puppy(Digger( name="digger",  pos=(world.width // 2, world.height // 2), brain=brain, rng=rng))
    return world


def snapshot_frame(world: World) -> dict:
    """Capture everything the viewer needs for this tick."""
    return {
        "t": world.tick,
        "puppies": {
            p.name: {
                "role": p.role,
                "emoji": p.emoji,
                "breed": p.breed,
                "pos": list(p.pos),
                "action": world.tick_actions.get(p.name, "stay"),
                "memory": p.memory.snapshot(),
            }
            for p in world.puppies.values()
        },
        "messages": list(world.tick_messages),
        "bone": {"pos": list(world.bone.pos), "found": world.bone.found},
        "done": world.is_done(),
    }


def run(ticks: int, seed: int, use_llm: bool, difficulty: str = "medium", verbose: bool = True) -> dict:
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"Unknown difficulty '{difficulty}'. Choose one of {list(DIFFICULTIES)}.")
    config = puppy_mod.configure(**DIFFICULTIES[difficulty])

    world = build_world(seed=seed, use_llm=use_llm)
    frames: list[dict] = []

    # Initial frame (tick 0, no actions yet, just spawn).
    frames.append(snapshot_frame(world))

    for _ in range(1, ticks + 1):
        world.tick += 1
        world.tick_messages = []
        world.tick_actions = {}

        # Phase 1: perceive (everyone senses first, in a fixed order).
        for p in world.puppies.values():
            p.perceive()

        # Phase 2: act (move / dig).
        for p in world.puppies.values():
            action = p.decide()
            world.tick_actions[p.name] = action

        # Phase 3: chat.
        for p in world.puppies.values():
            for msg in p.speak():
                world.tick_messages.append(msg)

        frames.append(snapshot_frame(world))

        if verbose:
            msg_str = ""
            if world.tick_messages:
                m = world.tick_messages[0]
                extra = f" +{len(world.tick_messages) - 1}" if len(world.tick_messages) > 1 else ""
                msg_str = f'  💬 {m["from"]}→{m["to"]}: "{m["text"]}"{extra}'
            positions = " ".join(f"{p.emoji}{tuple(p.pos)}" for p in world.puppies.values())
            print(f"t={world.tick:>3} {positions}{msg_str}")

        if world.is_done():
            if verbose:
                print(f"\n🦴 Bone found at tick {world.tick}! Simulation complete.")
            break

    return {
        "meta": {
            "seed": seed,
            "difficulty": difficulty,
            "config": config,
            "ticks_run": world.tick,
            "grid": [world.width, world.height],
            "landmarks": {k: list(v) for k, v in world.landmarks.items()},
            "bone_pos": list(world.bone.pos),
            "completed": world.is_done(),
            "use_llm": use_llm,
            "created_at": datetime.utcnow().isoformat() + "Z",
        },
        "frames": frames,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the puppy playground simulation.")
    parser.add_argument("--ticks", type=int, default=60, help="Max simulation ticks (default 60)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")
    parser.add_argument("--difficulty", choices=list(DIFFICULTIES.keys()), default="medium",
                        help="Sense/chat ranges preset (default: medium)")
    parser.add_argument("--no-llm", action="store_true", help="Skip Claude API, use scripted messages")
    parser.add_argument("--quiet", action="store_true", help="Suppress tick-by-tick console output")
    parser.add_argument("--out", type=str, default=None, help="Output JSON path (default: playground/runs/run-<timestamp>.json)")
    args = parser.parse_args()

    result = run(ticks=args.ticks, seed=args.seed, difficulty=args.difficulty,
                 use_llm=not args.no_llm, verbose=not args.quiet)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    if args.out:
        out_path = Path(args.out)
    else:
        stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        out_path = RUNS_DIR / f"run-{stamp}.json"
    out_path.write_text(json.dumps(result, indent=2))

    # Also update a stable "latest.json" so the viewer can always find something.
    latest = RUNS_DIR / "latest.json"
    latest.write_text(json.dumps(result, indent=2))

    print(f"\n✅ Wrote {out_path}")
    print(f"✅ Updated {latest}")
    print("Open playground/index.html in a browser to replay.")


if __name__ == "__main__":
    main()
