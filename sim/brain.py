"""Brain: LLM and scripted backends for puppy speech AND decisions.

There are two classes here:

- `Brain` (chat-only): generates the puppies' *chat messages* only.
  Decisions (move / dig) are still made by rule-based logic in `puppy.py`.
  This is used by --mode rules (scripted) and --mode chat (Claude-written).

- `LlmBrain` (full decider): given a puppy's role + memory + world state,
  asks Claude to return a *complete* action choice for this tick:
      { thought, move, dig, speak }
  Used only by --mode brain. Falls back gracefully to rule-based on any
  failure so the sim always advances.
"""

from __future__ import annotations

import json
import os
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sim.puppy import Puppy


SYSTEM_PROMPT = """You are role-playing as a puppy in a playground simulation.
You are "{name}", a {breed} whose specialty is: {specialty}.
You are talking to "{recipient}", a {recipient_breed} ({recipient_specialty}).

Rules:
- Respond as the puppy would — playful, short, in first person.
- 1-2 sentences, max ~25 words.
- Share concrete facts from your recent memory so the other puppy can act.
- ALWAYS include any coordinate like (x, y) that appears in your memory —
  that's how {recipient} knows where to go. Coordinates must be in the form
  (x, y) with parentheses and a comma.
- If you have nothing useful to share, reply exactly with: (silent)
- Never mention you are an AI or refer to the simulation.
"""

SPECIALTIES = {
    "scout":   "spotting landmarks and mapping the playground from far away",
    "sniffer": "smelling things, especially the buried bone",
    "digger":  "the only one who can actually dig up buried things",
}


class Brain:
    def __init__(self, use_llm: bool = True, model: str = "claude-haiku-4-5") -> None:
        self.use_llm = use_llm
        self.model = model
        self._client = None
        if use_llm:
            self._client = self._make_client()

    def _make_client(self):
        try:
            from anthropic import Anthropic  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "Anthropic SDK not installed. Run `pip install -r requirements.txt` "
                "or use --no-llm."
            ) from e
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Add it to .env (see .env.example) "
                "or run with --no-llm."
            )
        return Anthropic(api_key=api_key)

    # ------------------------------------------------------------------

    def compose_message(self, sender: "Puppy", recipient: "Puppy") -> str | None:
        """Return the message text, or None if the puppy stays quiet."""
        recent = sender.memory.recent_summary(window=5)
        if "(nothing yet)" in recent:
            return None

        if not self.use_llm:
            return self._scripted_message(sender, recipient)

        try:
            return self._llm_message(sender, recipient, recent)
        except Exception as e:  # noqa: BLE001 - fall back on any API issue
            print(f"  [brain] LLM error, falling back to scripted: {e}")
            return self._scripted_message(sender, recipient)

    # ------------------------------------------------------------------

    def _llm_message(self, sender: "Puppy", recipient: "Puppy", recent: str) -> str | None:
        assert self._client is not None
        system = SYSTEM_PROMPT.format(
            name=sender.name,
            breed=sender.breed,
            specialty=SPECIALTIES.get(sender.role, sender.role),
            recipient=recipient.name,
            recipient_breed=recipient.breed,
            recipient_specialty=SPECIALTIES.get(recipient.role, recipient.role),
        )
        user = f"Your recent memory:\n{recent}\n\nWhat do you say to {recipient.name}?"
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=80,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = resp.content[0].text.strip() if resp.content else ""
        if not text or "(silent)" in text.lower():
            return None
        # Strip any accidental quotes
        return text.strip().strip('"').strip("'")

    # ------------------------------------------------------------------

    def _scripted_message(self, sender: "Puppy", recipient: "Puppy") -> str | None:
        """Deterministic scripted chat derived from the latest observation.

        Key invariant: whenever a puppy has useful location info, the scripted
        message MUST contain it as a (x, y) tuple so Digger can parse it.
        """
        obs_list = sender.memory.observations
        if not obs_list:
            return None
        last = obs_list[-1]

        if sender.role == "scout":
            # Relay takes priority: if Scout heard a bone tip, pass it along.
            relay = sender.memory.knowledge.get("relay_coord")
            src = sender.memory.knowledge.get("relay_from")
            if relay and recipient.role == "digger":
                return f"{recipient.name}! {src or 'Sniffer'} says the bone is at ({relay[0]}, {relay[1]}) — go dig!"
            if last.kind == "see" and last.pos is not None:
                return f"Hey {recipient.name}! I spotted the {last.content.split()[1]} at ({last.pos[0]}, {last.pos[1]})."

        if sender.role == "sniffer" and last.kind == "smell":
            if last.pos is not None:
                # Sniffer is close enough to pinpoint the bone itself.
                return f"{recipient.name}! Bone! Dig at ({last.pos[0]}, {last.pos[1]})!"
            # Otherwise broadcast sniffer's own position as a "come here" hint.
            sx, sy = sender.pos
            return f"{recipient.name}, I smell something — {last.content}. I'm at ({sx}, {sy})."

        if sender.role == "digger" and last.kind == "dig" and "dug up the bone" in last.content:
            x, y = last.pos if last.pos else sender.pos
            return f"I got it!! 🦴 Dug it up at ({x}, {y})!"

        return None


# ======================================================================
#                           LLM BRAIN (mode=brain)
# ======================================================================


ROLE_INTROS = {
    "scout":   "You are the team's Scout (husky). Your job is to map the playground and — crucially — relay coordinates from Sniffer to Digger when they're out of chat range.",
    "sniffer": "You are the team's Sniffer (beagle). You're the ONLY puppy who can smell the bone. When you catch a scent, shout concrete coordinates so Digger can act.",
    "digger":  "You are the team's Digger (terrier). You're the ONLY puppy who can dig. Move to coordinates your teammates tell you, then dig. You don't see or smell the bone yourself — you rely on chat.",
}

BRAIN_SYSTEM_PROMPT = """You control one puppy, {name} ({breed}), in a cooperative simulation.

{role_intro}

World: a 20x20 fenced grid. Your teammates are scout (husky), sniffer (beagle), and digger (terrier). You can only learn things from your OWN senses and from messages teammates send you. You CANNOT read teammates' minds.

Your abilities this run:
- vision range: {vision} cells (only Scout sees landmarks far away)
- smell range: {smell} cells (only Sniffer smells the bone)
- can_dig: {can_dig}
- chat range: {chat} cells (messages only reach puppies within this)

Each tick you choose ONE action. Respond ONLY with a compact JSON object (no prose, no markdown fences):

{{
  "thought": "one short sentence of reasoning (max ~20 words)",
  "move": [dx, dy],
  "dig": true | false,
  "speak": {{"to": "<teammate name>", "text": "<short in-character message>"}} | null
}}

Rules:
- move components are each in {{-1, 0, 1}}; [0,0] means stay.
- Set dig=true ONLY if can_dig is true AND you're at a spot you believe the bone is.
- speak.text MUST include any coordinate you want to share as "(x, y)" with parentheses and comma — that's how others learn where to go.
- If you have nothing useful to say, set speak to null.
- Never mention you are an AI or that this is a simulation.
"""

_POS_RE = re.compile(r"\((\d+)\s*,\s*(\d+)\)")


class LlmBrain:
    """LLM-driven decider. One Claude call per puppy per tick."""

    def __init__(self, model: str = "claude-haiku-4-5") -> None:
        self.model = model
        self._client = self._make_client()
        self.fallbacks = 0
        # Cache the (very verbose) system prompt per puppy so we don't rebuild it.
        self._system_cache: dict[str, str] = {}

    def _make_client(self):
        try:
            from anthropic import Anthropic  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "Anthropic SDK not installed. Run `pip install -r requirements.txt` "
                "or use --mode rules."
            ) from e
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Add it to .env (see .env.example) "
                "or run with --mode rules."
            )
        return Anthropic(api_key=api_key)

    # ------------------------------------------------------------------

    def decide(self, puppy: "Puppy") -> dict[str, Any]:
        """Ask Claude for this puppy's next action. Returns a normalized dict:

            {
              thought: str,
              move: (dx, dy),
              dig: bool,
              speak: None | {"to": str, "text": str},
              fallback: bool,   # True if we had to clamp / default anything
            }
        """
        try:
            raw = self._call(puppy)
            decision = self._parse_and_validate(raw, puppy)
            return decision
        except Exception as e:  # noqa: BLE001
            print(f"  [llm-brain] {puppy.name} error ({type(e).__name__}), falling back: {e}")
            self.fallbacks += 1
            return {
                "thought": None,
                "move": (0, 0),
                "dig": False,
                "speak": None,
                "fallback": True,
            }

    # ------------------------------------------------------------------

    def _system_for(self, puppy: "Puppy") -> str:
        if puppy.name in self._system_cache:
            return self._system_cache[puppy.name]

        # Import lazily to avoid a circular import at module load.
        from sim.puppy import CHAT_RANGE, SCOUT_VISION, SNIFFER_RANGE

        vision = SCOUT_VISION if puppy.role == "scout" else 2
        smell = SNIFFER_RANGE if puppy.role == "sniffer" else 0
        can_dig = (puppy.role == "digger")
        prompt = BRAIN_SYSTEM_PROMPT.format(
            name=puppy.name,
            breed=puppy.breed,
            role_intro=ROLE_INTROS.get(puppy.role, ""),
            vision=vision,
            smell=smell,
            can_dig=str(can_dig).lower(),
            chat=CHAT_RANGE,
        )
        self._system_cache[puppy.name] = prompt
        return prompt

    def _user_for(self, puppy: "Puppy") -> str:
        mem = puppy.memory
        obs_lines = [
            f"- [t={o.tick}] {o.kind}: {o.content}" + (f" @ {tuple(o.pos)}" if o.pos else "")
            for o in mem.observations[-5:]
        ] or ["- (nothing yet)"]
        msg_lines = [
            f'- [t={m.tick}] {m.sender}: "{m.text}"'
            for m in mem.messages[-5:]
        ] or ["- (nothing yet)"]
        know_lines = [f"- {k}: {v}" for k, v in mem.knowledge.items()] or ["- (none yet)"]

        assert puppy.world is not None
        teammates = [p.name for p in puppy.world.puppies.values() if p is not puppy]

        return (
            f"Tick: {puppy.world.tick}\n"
            f"Your position: {tuple(puppy.pos)}\n"
            f"Teammates: {', '.join(teammates)}\n\n"
            f"Recent observations:\n" + "\n".join(obs_lines) + "\n\n"
            f"Recent chat received:\n" + "\n".join(msg_lines) + "\n\n"
            f"Your derived knowledge:\n" + "\n".join(know_lines) + "\n\n"
            f"Choose your action now. Respond with JSON only."
        )

    def _call(self, puppy: "Puppy") -> str:
        system = self._system_for(puppy)
        user = self._user_for(puppy)
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=220,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return resp.content[0].text.strip() if resp.content else ""

    def _parse_and_validate(self, raw: str, puppy: "Puppy") -> dict[str, Any]:
        fallback = False

        # Be forgiving about ```json fences.
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            obj = json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to pull the first {...} blob out.
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if not match:
                raise ValueError(f"no JSON object in response: {raw!r}")
            obj = json.loads(match.group(0))
            fallback = True

        if not isinstance(obj, dict):
            raise ValueError(f"expected JSON object, got {type(obj).__name__}")

        thought = obj.get("thought")
        if thought is not None and not isinstance(thought, str):
            thought = None
            fallback = True

        move = obj.get("move", [0, 0])
        if (not isinstance(move, (list, tuple)) or len(move) != 2
                or not all(isinstance(v, int) for v in move)):
            move = (0, 0)
            fallback = True
        else:
            dx = max(-1, min(1, int(move[0])))
            dy = max(-1, min(1, int(move[1])))
            if (dx, dy) != tuple(move):
                fallback = True
            move = (dx, dy)

        dig = bool(obj.get("dig", False))
        if dig and puppy.role != "digger":
            dig = False
            fallback = True

        speak = obj.get("speak")
        if speak is not None:
            if not (isinstance(speak, dict) and isinstance(speak.get("to"), str)
                    and isinstance(speak.get("text"), str) and speak["text"].strip()):
                speak = None
                fallback = True
            else:
                assert puppy.world is not None
                if speak["to"] not in puppy.world.puppies or speak["to"] == puppy.name:
                    speak = None
                    fallback = True
                else:
                    speak = {"to": speak["to"], "text": speak["text"].strip().strip('"').strip("'")}

        return {
            "thought": thought,
            "move": move,
            "dig": dig,
            "speak": speak,
            "fallback": fallback,
        }
