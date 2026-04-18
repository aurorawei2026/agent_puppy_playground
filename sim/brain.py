"""Brain: generates the puppies' chat messages.

Two modes:
- LLM mode (default): uses Claude to compose short, in-character messages.
- Scripted mode (--no-llm): generates deterministic templated messages from
  the puppy's most recent observation. Useful for testing and for running
  the simulation with zero API cost.

Either way, if a puppy has nothing new to say, it stays quiet.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

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
