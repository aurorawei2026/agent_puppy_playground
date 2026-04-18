"""Per-puppy memory store.

A puppy's memory has three parts:
- observations: things they personally sensed
- messages:     things other puppies told them
- knowledge:    derived facts (e.g. "bone is near the tree")

Everything is timestamped with a simulation tick so the viewer can replay
memory growth over time.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Observation:
    tick: int
    kind: str          # "see" | "smell" | "dig" | "hear"
    content: str       # human-readable description
    pos: tuple[int, int] | None = None


@dataclass
class Message:
    tick: int
    sender: str
    text: str


@dataclass
class Memory:
    observations: list[Observation] = field(default_factory=list)
    messages: list[Message] = field(default_factory=list)
    knowledge: dict[str, Any] = field(default_factory=dict)

    def observe(self, tick: int, kind: str, content: str, pos: tuple[int, int] | None = None) -> None:
        self.observations.append(Observation(tick=tick, kind=kind, content=content, pos=pos))

    def receive(self, tick: int, sender: str, text: str) -> None:
        self.messages.append(Message(tick=tick, sender=sender, text=text))

    def recent_summary(self, window: int = 6) -> str:
        """Short human-readable summary for LLM prompts."""
        lines: list[str] = []
        for o in self.observations[-window:]:
            lines.append(f"[t={o.tick}] saw/sensed: {o.content}")
        for m in self.messages[-window:]:
            lines.append(f"[t={m.tick}] {m.sender} said: \"{m.text}\"")
        if not lines:
            lines.append("(nothing yet)")
        return "\n".join(lines)

    def snapshot(self) -> dict[str, Any]:
        """Viewer-friendly dict of current memory state."""
        return {
            "observations": [asdict(o) for o in self.observations],
            "messages": [asdict(m) for m in self.messages],
            "knowledge": dict(self.knowledge),
        }
