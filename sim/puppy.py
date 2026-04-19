"""Puppy agents: Scout, Sniffer, Digger.

Each puppy follows the same lifecycle every tick:

    perceive()  -> role-specific observations added to memory
    decide()    -> pick a movement / dig action (rule-based in v1)
    speak()     -> maybe generate a message to a nearby puppy (Claude in v1)

v1 keeps decision-making rule-based (cheap, deterministic) and puts the
LLM only on the chat channel. This is where memory "evolves" — puppies
swap observations in natural language and build a shared picture.
"""

from __future__ import annotations

import random
import re
from typing import TYPE_CHECKING

from sim.memory import Memory

if TYPE_CHECKING:
    from sim.brain import Brain
    from sim.world import World


SCOUT_VISION = 7       # Scout sees landmarks and other puppies within this radius
SNIFFER_RANGE = 4      # Sniffer can smell the bone within this radius
DIGGER_REACH = 0       # Digger has to be on the exact cell to dig
CHAT_RANGE = 8         # two puppies can hear each other within this radius


def configure(*, scout_vision: int | None = None,
              sniffer_range: int | None = None,
              chat_range: int | None = None) -> dict:
    """Override the default sense/chat ranges before building the world.

    Returns the full active config so it can be saved with the run.
    """
    global SCOUT_VISION, SNIFFER_RANGE, CHAT_RANGE
    if scout_vision is not None:
        SCOUT_VISION = scout_vision
    if sniffer_range is not None:
        SNIFFER_RANGE = sniffer_range
    if chat_range is not None:
        CHAT_RANGE = chat_range
    return {
        "scout_vision": SCOUT_VISION,
        "sniffer_range": SNIFFER_RANGE,
        "chat_range": CHAT_RANGE,
    }


# Extracts coordinate hints like "(4, 1)" from chat messages so Digger can learn
# where to dig *only from language*, not from reading other puppies' memory.
POS_RE = re.compile(r"\((\d+)\s*,\s*(\d+)\)")


class Puppy:
    role: str = "puppy"
    emoji: str = "🐕"
    breed: str = ""

    def __init__(
        self,
        name: str,
        pos: tuple[int, int],
        brain: "Brain",
        rng: random.Random,
    ) -> None:
        self.name = name
        self.pos = pos
        self.memory = Memory()
        self.brain = brain
        self.rng = rng
        self.world: "World | None" = None
        # Target cell this puppy wants to move toward next (overridable by role)
        self.target: tuple[int, int] | None = None
        # Track last line said to each recipient so we don't spam duplicates.
        self.last_spoken: dict[str, str] = {}

    # ---------- lifecycle ----------

    def perceive(self) -> None:
        """Role-specific — override in subclass."""
        raise NotImplementedError

    def decide(self) -> str:
        """Return a short action label (e.g. 'move:right', 'dig', 'stay').

        Default behavior: wander toward target if any, else random step.
        """
        assert self.world is not None
        if self.target is None or self.pos == self.target:
            self.target = self._pick_wander_target()

        action = self._step_toward(self.target)
        return action

    def speak(self) -> list[dict]:
        """Broadcast one message to each puppy within CHAT_RANGE.

        Each recipient gets its own composition so the puppy can tailor what
        it says to who it's saying it to. Everyone in range hears everything
        (there's no whispering in v1).
        """
        assert self.world is not None
        messages: list[dict] = []
        others = [
            p for p in self.world.puppies.values()
            if p is not self and self.world.distance(self.pos, p.pos) <= CHAT_RANGE
        ]
        others.sort(key=lambda p: self.world.distance(self.pos, p.pos))
        for recipient in others:
            text = self.brain.compose_message(sender=self, recipient=recipient)
            if not text:
                continue
            # Don't re-say the exact same thing to the same puppy.
            if self.last_spoken.get(recipient.name) == text:
                continue
            self.last_spoken[recipient.name] = text
            messages.append({"from": self.name, "to": recipient.name, "text": text})
            recipient.memory.receive(self.world.tick, self.name, text)
        return messages

    # ---------- movement helpers ----------

    def _pick_wander_target(self) -> tuple[int, int]:
        assert self.world is not None
        w = self.world
        x = self.rng.randint(1, w.width - 2)
        y = self.rng.randint(1, w.height - 2)
        return (x, y)

    def _step_toward(self, target: tuple[int, int]) -> str:
        assert self.world is not None
        cx, cy = self.pos
        tx, ty = target
        dx = (tx > cx) - (tx < cx)
        dy = (ty > cy) - (ty < cy)
        # Prefer the axis with greater remaining distance.
        if abs(tx - cx) >= abs(ty - cy) and dx != 0:
            new = (cx + dx, cy)
            return self._apply_move(new, "right" if dx > 0 else "left")
        if dy != 0:
            new = (cx, cy + dy)
            return self._apply_move(new, "down" if dy > 0 else "up")
        if dx != 0:
            new = (cx + dx, cy)
            return self._apply_move(new, "right" if dx > 0 else "left")
        return "stay"

    def _apply_move(self, new_pos: tuple[int, int], direction: str) -> str:
        assert self.world is not None
        if self.world.in_bounds(new_pos):
            self.pos = new_pos
            return f"move:{direction}"
        return "stay"


# ------------------------- Scout: Husky 🐺 -----------------------------


class Scout(Puppy):
    role = "scout"
    emoji = "🐺"
    breed = "Husky"

    def perceive(self) -> None:
        """Scout sees landmarks, notes other puppies, and relays chat coords.

        Relay: if a received message contains a coordinate and a bone keyword,
        Scout remembers it so it can pass the tip along to Digger. This is
        Scout's true contribution to the task — being the comms hub.
        """
        assert self.world is not None
        w = self.world
        for lname, lpos in w.landmarks.items():
            if w.distance(self.pos, lpos) <= SCOUT_VISION:
                key = f"landmark:{lname}"
                if key not in self.memory.knowledge:
                    self.memory.knowledge[key] = list(lpos)
                    self.memory.observe(
                        w.tick, "see",
                        f"spotted {lname} at {lpos}",
                        pos=lpos,
                    )
        for p in w.puppies.values():
            if p is self:
                continue
            if w.distance(self.pos, p.pos) <= SCOUT_VISION:
                self.memory.knowledge[f"last_seen:{p.name}"] = [p.pos[0], p.pos[1], w.tick]

        # Relay coordinates from any "bone"-related message we heard.
        for msg in reversed(self.memory.messages):
            if "bone" not in msg.text.lower() and "dig" not in msg.text.lower():
                continue
            matches = POS_RE.findall(msg.text)
            if not matches:
                continue
            x, y = matches[-1]
            x, y = int(x), int(y)
            if w.in_bounds((x, y)):
                prev = self.memory.knowledge.get("relay_coord")
                if prev != [x, y]:
                    self.memory.knowledge["relay_coord"] = [x, y]
                    self.memory.knowledge["relay_from"] = msg.sender
                    self.memory.observe(
                        w.tick, "hear",
                        f"relaying: {msg.sender} says bone at ({x}, {y})",
                        pos=(x, y),
                    )
                break

    def decide(self) -> str:
        """If we're holding a hot tip, move toward Digger so we can relay it."""
        assert self.world is not None
        w = self.world
        if "relay_coord" in self.memory.knowledge:
            digger_pos = self.memory.knowledge.get("last_seen:digger")
            if digger_pos:
                self.target = (digger_pos[0], digger_pos[1])
        return super().decide()


# ------------------------- Sniffer: Beagle 🐶 --------------------------


class Sniffer(Puppy):
    role = "sniffer"
    emoji = "🐶"
    breed = "Beagle"

    def perceive(self) -> None:
        """Sniffer senses the bone — strength depends on distance."""
        assert self.world is not None
        w = self.world
        d = w.distance(self.pos, w.bone.pos)
        if d <= SNIFFER_RANGE and not w.bone.found:
            strength = "strong" if d <= 1.5 else ("medium" if d <= 3 else "faint")
            # Direction hint (compass-ish) so messages are informative.
            dx = w.bone.pos[0] - self.pos[0]
            dy = w.bone.pos[1] - self.pos[1]
            horizontal = "east" if dx > 0 else ("west" if dx < 0 else "")
            vertical = "south" if dy > 0 else ("north" if dy < 0 else "")
            direction = (vertical + horizontal) or "right here"
            self.memory.observe(
                w.tick, "smell",
                f"bone scent ({strength}) toward the {direction}",
                pos=w.bone.pos if d <= 1.5 else None,
            )
            # Refine our own knowledge of where the bone is.
            if d <= 1.5:
                self.memory.knowledge["bone_exact"] = list(w.bone.pos)
            else:
                self.memory.knowledge["bone_near"] = list(self.pos)
                self.memory.knowledge["bone_direction"] = direction

    def decide(self) -> str:
        """If we smell the bone, home in on it instead of wandering."""
        assert self.world is not None
        w = self.world
        if not w.bone.found and w.distance(self.pos, w.bone.pos) <= SNIFFER_RANGE:
            self.target = w.bone.pos
        return super().decide()


# ------------------------- Digger: Terrier 🐕‍🦺 -------------------------


class Digger(Puppy):
    role = "digger"
    emoji = "🐕‍🦺"
    breed = "Terrier"

    def perceive(self) -> None:
        """Parse incoming chat for coordinate hints — Digger's only way to
        learn where the bone is. No peeking at other puppies' memory!
        """
        assert self.world is not None
        w = self.world
        # Walk messages newest-first so the freshest hint wins.
        for msg in reversed(self.memory.messages):
            matches = POS_RE.findall(msg.text)
            if not matches:
                continue
            # Take the last (x, y) mentioned — Sniffer puts the most precise
            # coordinate at the end of its message.
            x, y = matches[-1]
            x, y = int(x), int(y)
            if w.in_bounds((x, y)):
                prev = self.memory.knowledge.get("dig_here")
                if prev != [x, y]:
                    self.memory.knowledge["dig_here"] = [x, y]
                    self.memory.knowledge["dig_here_source"] = msg.sender
                    self.memory.knowledge["dig_here_tick"] = msg.tick
                    self.memory.observe(
                        w.tick, "hear",
                        f"going to check ({x}, {y}) — heard from {msg.sender}",
                        pos=(x, y),
                    )
                break

    def decide(self) -> str:
        """If we've heard a coordinate, head there and dig."""
        assert self.world is not None
        w = self.world

        target = None
        if "dig_here" in self.memory.knowledge:
            target = tuple(self.memory.knowledge["dig_here"])

        if target is not None:
            self.target = target
            if self.pos == target:
                if tuple(self.pos) == tuple(w.bone.pos):
                    w.bone.found = True
                    self.memory.observe(w.tick, "dig", "dug up the bone! 🦴", pos=self.pos)
                    return "dig:success"
                # Wrong spot — clear stale target so we'll act on the next hint.
                self.memory.observe(w.tick, "dig", "dug — nothing here", pos=self.pos)
                self.memory.knowledge.pop("dig_here", None)
                self.memory.knowledge.pop("dig_here_source", None)
                self.memory.knowledge.pop("dig_here_tick", None)
                return "dig:empty"

        return super().decide()
