"""The fenced playground world.

A simple 2D grid with landmarks, a hidden bone, and three puppies.
The world is deterministic given a seed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sim.puppy import Puppy


GRID_W = 20
GRID_H = 20

# Fixed landmarks — gives Scout stable reference points to talk about.
LANDMARKS: dict[str, tuple[int, int]] = {
    "tree":       (5, 5),
    "water_bowl": (15, 15),
    "toy_chest":  (10, 3),
    "rock":       (3, 16),
}


@dataclass
class Bone:
    pos: tuple[int, int]
    found: bool = False   # has Digger dug it up yet?


class World:
    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        self.seed = seed
        self.width = GRID_W
        self.height = GRID_H
        self.landmarks = dict(LANDMARKS)
        self.bone = Bone(pos=self._random_empty_cell())
        self.puppies: dict[str, Puppy] = {}
        self.tick: int = 0
        self.tick_messages: list[dict] = []
        self.tick_actions: dict[str, str] = {}

    def _random_empty_cell(self) -> tuple[int, int]:
        occupied = set(self.landmarks.values())
        while True:
            x = self.rng.randint(1, self.width - 2)
            y = self.rng.randint(1, self.height - 2)
            if (x, y) not in occupied:
                return (x, y)

    def add_puppy(self, puppy: "Puppy") -> None:
        self.puppies[puppy.name] = puppy
        puppy.world = self

    def in_bounds(self, pos: tuple[int, int]) -> bool:
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height

    def distance(self, a: tuple[int, int], b: tuple[int, int]) -> float:
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    def is_done(self) -> bool:
        return self.bone.found
