from __future__ import annotations

import argparse
import base64
import collections
import dataclasses
from dataclasses import dataclass, field
from enum import IntEnum
import heapq
import json
import math
import os
import random
import sys
import textwrap
import time
import zlib
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple


# ============================================================
#  DATA-FIRST DUNGEON-LIKE WORLD SIMULATION
# ============================================================
#
# Headless world simulation with optional Tk and PPM renderers.
# WorldState holds the data updated by each simulation system.
#
# Systems:
# - macro planet bands + local volcanic patch generation
# - infernal builder tiles, score, symmetry, rituals, templates
# - creature / demon procedural genomes and entity behaviors
# - traffic / migration routes between cities, ports, and ritual hubs
# - cellular elemental sim: water, lava, acid, plasma, fire, ice cooling
# - ghost / aura / dread / soul economy feedback loop
# - optional renderer: Tk live viewer and pure-Python PPM snapshot export
# ============================================================


class Terrain(IntEnum):
    WATER = 0
    MAGMA_SEA = 1
    ASH = 2
    OBSIDIAN_RIDGE = 3
    CITY_BAND = 4
    ICEFIELD = 5
    ALIEN_BIOMASS = 6


class Ground(IntEnum):
    NONE = 0
    ROAD = 1
    EMBER_RUNE = 2
    VOID_GLASS = 3


class Structure(IntEnum):
    NONE = 0
    OBSIDIAN = 1
    SPIRE = 2
    WRAITH_TOTEM = 3
    MONOLITH = 4
    QUANTUM_CORE = 5
    FIRE_ALTAR = 6


class Fluid(IntEnum):
    NONE = 0
    WATER = 1
    LAVA = 2
    ACID = 3
    PLASMA = 4


class EntityKind(IntEnum):
    GHOST = 1
    DEMON = 2
    CREATURE = 3
    MIGRANT = 4
    SKIMMER = 5
    SOUL = 6


TERRAIN_NAME = {
    Terrain.WATER: "water",
    Terrain.MAGMA_SEA: "magma_sea",
    Terrain.ASH: "ash",
    Terrain.OBSIDIAN_RIDGE: "obsidian_ridge",
    Terrain.CITY_BAND: "city_band",
    Terrain.ICEFIELD: "icefield",
    Terrain.ALIEN_BIOMASS: "alien_biomass",
}
GROUND_NAME = {
    Ground.NONE: "none",
    Ground.ROAD: "road",
    Ground.EMBER_RUNE: "ember_rune",
    Ground.VOID_GLASS: "void_glass",
}
STRUCTURE_NAME = {
    Structure.NONE: "none",
    Structure.OBSIDIAN: "obsidian",
    Structure.SPIRE: "spire",
    Structure.WRAITH_TOTEM: "wraith_totem",
    Structure.MONOLITH: "monolith",
    Structure.QUANTUM_CORE: "quantum_core",
    Structure.FIRE_ALTAR: "fire_altar",
}
FLUID_NAME = {
    Fluid.NONE: "none",
    Fluid.WATER: "water",
    Fluid.LAVA: "lava",
    Fluid.ACID: "acid",
    Fluid.PLASMA: "plasma",
}
ENTITY_NAME = {
    EntityKind.GHOST: "ghost",
    EntityKind.DEMON: "demon",
    EntityKind.CREATURE: "creature",
    EntityKind.MIGRANT: "migrant",
    EntityKind.SKIMMER: "skimmer",
    EntityKind.SOUL: "soul",
}


@dataclass
class WorldConfig:
    width: int = 72
    height: int = 72
    seed: int = 1337
    tick_seconds: float = 0.25
    city_count: int = 6
    initial_ghosts: int = 10
    initial_demons: int = 6
    initial_creatures: int = 8
    initial_migrants: int = 18
    initial_skimmers: int = 8
    challenge_date: Optional[str] = None


@dataclass
class Challenge:
    title: str
    description: str
    requirements: Dict[str, int]
    starter_template: str


@dataclass
class City:
    name: str
    x: int
    y: int
    port_x: int
    port_y: int
    radius: int
    temperament: str


@dataclass
class CreatureGenome:
    genome_id: int
    name: str
    title: str
    element: str
    horn_style: str
    eye_style: str
    mouth_style: str
    body_style: str
    palette: Tuple[int, int, int]
    behavior: str
    appetite: float
    volatility: float
    aura_bias: float
    dread_bias: float


@dataclass
class Entity:
    entity_id: int
    kind: EntityKind
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    home_city: Optional[int] = None
    dest_city: Optional[int] = None
    state: str = "idle"
    path: List[Tuple[int, int]] = field(default_factory=list)
    path_index: int = 0
    genome_id: Optional[int] = None
    life: float = 1.0
    energy: float = 1.0
    heat_affinity: float = 0.0
    aura_affinity: float = 0.0
    corruption_affinity: float = 0.0
    age_ticks: int = 0
    cargo: float = 0.0

    def cell(self) -> Tuple[int, int]:
        return int(round(self.x)), int(round(self.y))


@dataclass
class Metrics:
    dread: int = 0
    aura: int = 0
    souls_per_minute: int = 0
    symmetry: float = 1.0
    total_tiles: int = 0
    rank: str = "Cinder Seed"
    counts: Dict[str, int] = field(default_factory=dict)
    challenge_progress: float = 0.0
    deliveries: int = 0


@dataclass
class WorldState:
    config: WorldConfig
    width: int
    height: int
    rng: random.Random
    tick: int = 0
    seconds: float = 0.0

    terrain: List[int] = field(default_factory=list)
    height_field: List[float] = field(default_factory=list)
    heat: List[float] = field(default_factory=list)
    moisture: List[float] = field(default_factory=list)
    corruption: List[float] = field(default_factory=list)
    ash: List[float] = field(default_factory=list)
    ground: List[int] = field(default_factory=list)
    structure: List[int] = field(default_factory=list)
    structure_height: List[int] = field(default_factory=list)
    fluid: List[int] = field(default_factory=list)
    fluid_amount: List[float] = field(default_factory=list)
    fire: List[float] = field(default_factory=list)
    city_index: List[int] = field(default_factory=list)

    cities: List[City] = field(default_factory=list)
    archetypes: Dict[int, CreatureGenome] = field(default_factory=dict)
    entities: Dict[int, Entity] = field(default_factory=dict)
    next_entity_id: int = 1

    deliveries_window: Deque[int] = field(default_factory=lambda: collections.deque(maxlen=240))
    recent_events: Deque[str] = field(default_factory=lambda: collections.deque(maxlen=14))
    challenge: Optional[Challenge] = None
    metrics: Metrics = field(default_factory=Metrics)


DAILY_CHALLENGES = [
    Challenge(
        title="Lava Moat",
        description="Place 16 lava cells and 8 spires to complete the challenge.",
        requirements={"lava": 16, "spire": 8},
        starter_template="citadel",
    ),
    Challenge(
        title="Phantom Court",
        description="Sustain 4 ghosts and 24 obsidian structures in one domain.",
        requirements={"ghost": 4, "obsidian": 24},
        starter_template="ritual",
    ),
    Challenge(
        title="Hellfire Choir",
        description="Maintain 6 fire altars and at least 12 active fire cells.",
        requirements={"fire_altar": 6, "fire": 12},
        starter_template="gauntlet",
    ),
    Challenge(
        title="Abyssal Convergence",
        description="Stabilize one quantum core, two monoliths, and four roads to feed them.",
        requirements={"quantum_core": 1, "monolith": 2, "road": 4},
        starter_template="maze",
    ),
]

CITY_NAMES = [
    "Aethelgard",
    "Nova Prime",
    "Solaria",
    "Terminus",
    "Zion",
    "Arcadia",
    "Babylon",
    "Citadel",
    "Elysium",
    "Midgar",
]

GENOME_NAMES = {
    "prefix": ["Glar", "Zor", "Vex", "Quib", "Snar", "Kran", "Blob", "Thal", "Xyl", "Mog", "Drak", "Brum", "Yth", "Cth"],
    "vowel": ["a", "e", "i", "o", "u", "y", "ae", "oo"],
    "suffix": ["gox", "bat", "tul", "zis", "nosh", "nip", "dor", "moth", "lith", "pod", "tron", "pus", "ling"],
    "title": ["The Destroyer", "The Cuddly", "of the Deep", "The Ancient", "The Vengeful", "The Silent", "The Gluttonous", "The Anomalous", "of the Cosmos"],
    "element": ["Fire", "Water", "Nature", "Electric", "Void", "Toxic", "Abyssal", "Ash"],
    "horns": ["Ram", "Crown", "Swept", "Jagged", "Double", "Cat Ears", "Antennae"],
    "eyes": ["Slit", "Hollow", "Spider", "Cyclops", "Beast", "Cute", "Star"],
    "mouth": ["Maw", "Tusks", "Mandibles", "Grin", "Blep", "Zigzag"],
    "body": ["Blob", "Ghost", "Spiky", "Pear", "Squircle", "Wisp"],
    "behavior": ["curious", "predatory", "reverent", "chaotic", "playful", "territorial"],
}


# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------

def clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def smoothstep(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


def distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def hash_string(text: str) -> int:
    value = 0
    for char in text:
        value = ((value << 5) - value + ord(char)) & 0xFFFFFFFF
    return value


def hash2(seed: int, x: int, y: int) -> float:
    n = x * 374761393 + y * 668265263 + seed * 69069
    n = (n ^ (n >> 13)) & 0xFFFFFFFF
    n = (n * 1274126177) & 0xFFFFFFFF
    return ((n ^ (n >> 16)) & 0xFFFFFFFF) / 0xFFFFFFFF


class NoiseField:
    def __init__(self, seed: int) -> None:
        self.seed = seed

    def value(self, x: float, y: float) -> float:
        ix = math.floor(x)
        iy = math.floor(y)
        fx = x - ix
        fy = y - iy

        ux = smoothstep(fx)
        uy = smoothstep(fy)

        a = hash2(self.seed, ix, iy)
        b = hash2(self.seed, ix + 1, iy)
        c = hash2(self.seed, ix, iy + 1)
        d = hash2(self.seed, ix + 1, iy + 1)

        ab = lerp(a, b, ux)
        cd = lerp(c, d, ux)
        return lerp(ab, cd, uy)

    def fbm(self, x: float, y: float, octaves: int = 4, lacunarity: float = 2.0, gain: float = 0.5) -> float:
        total = 0.0
        amplitude = 0.5
        frequency = 1.0
        norm = 0.0
        for _ in range(octaves):
            total += self.value(x * frequency, y * frequency) * amplitude
            norm += amplitude
            amplitude *= gain
            frequency *= lacunarity
        return total / norm if norm else 0.0

    def domain_warp(self, x: float, y: float) -> float:
        qx = self.fbm(x + 3.17, y - 7.29, 3)
        qy = self.fbm(x - 9.11, y + 2.41, 3)
        return self.fbm(x + qx * 2.3, y + qy * 2.3, 5)


# ------------------------------------------------------------
# Core simulation engine
# ------------------------------------------------------------


class InfernalWorld:
    TOOL_SEQUENCE = [
        "obsidian",
        "lava",
        "spire",
        "ember_rune",
        "wraith_totem",
        "monolith",
        "quantum_core",
        "void_glass",
        "road",
        "fire_altar",
        "water",
        "acid",
        "plasma",
        "ghost",
        "demon",
        "erase",
    ]

    def __init__(self, config: Optional[WorldConfig] = None) -> None:
        self.config = config or WorldConfig()
        self.state = WorldState(
            config=self.config,
            width=self.config.width,
            height=self.config.height,
            rng=random.Random(self.config.seed),
        )
        self._noise = NoiseField(self.config.seed)
        self._spawn_history_cooldown = 0
        self._generate_world()
        self.compute_metrics()

    # ------------------------
    # Index helpers
    # ------------------------
    def index(self, x: int, y: int) -> int:
        return y * self.state.width + x

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.state.width and 0 <= y < self.state.height

    def neighbors4(self, x: int, y: int) -> Iterable[Tuple[int, int]]:
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if self.in_bounds(nx, ny):
                yield nx, ny

    def neighbors8(self, x: int, y: int) -> Iterable[Tuple[int, int]]:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if self.in_bounds(nx, ny):
                    yield nx, ny

    # ------------------------
    # Generation
    # ------------------------
    def _generate_world(self) -> None:
        s = self.state
        count = s.width * s.height
        s.terrain = [int(Terrain.ASH)] * count
        s.height_field = [0.0] * count
        s.heat = [0.0] * count
        s.moisture = [0.0] * count
        s.corruption = [0.0] * count
        s.ash = [0.0] * count
        s.ground = [int(Ground.NONE)] * count
        s.structure = [int(Structure.NONE)] * count
        s.structure_height = [0] * count
        s.fluid = [int(Fluid.NONE)] * count
        s.fluid_amount = [0.0] * count
        s.fire = [0.0] * count
        s.city_index = [-1] * count
        s.challenge = self._get_daily_challenge()

        rng = s.rng
        crater_x = rng.uniform(0.28, 0.72)
        crater_y = rng.uniform(0.28, 0.72)
        crater_radius = rng.uniform(0.09, 0.16)
        sea_level = 0.31

        for y in range(s.height):
            for x in range(s.width):
                i = self.index(x, y)
                nx = x / max(1, s.width - 1)
                ny = y / max(1, s.height - 1)
                py = ny * 2.0 - 1.0

                band = math.sin(py * 7.8 + self.config.seed * 0.013) * 0.18
                band += math.cos(py * 15.2 + self.config.seed * 0.007) * 0.08
                warped = self._noise.domain_warp(nx * 4.3, ny * 4.3)
                raw = self._noise.fbm(nx * 5.7 + 13.1, ny * 5.7 - 4.7, octaves=5)
                detail = self._noise.fbm(nx * 16.0 - 3.0, ny * 16.0 + 9.0, octaves=3)

                dist = math.hypot(nx - crater_x, ny - crater_y)
                volcano = max(0.0, 1.0 - (dist / max(0.001, crater_radius * 3.4)))
                crater = 0.0
                if dist < crater_radius:
                    crater = (1.0 - dist / crater_radius) * 0.42
                rim = max(0.0, 1.0 - abs(dist - crater_radius * 1.2) / (crater_radius * 0.65)) * 0.08

                elevation = 0.26 + band + raw * 0.22 + warped * 0.22 + detail * 0.08 + volcano * 0.34 - crater + rim
                elevation = clamp(elevation, 0.0, 1.0)

                heat = clamp(0.1 + volcano * 0.9 + band * 0.25 + warped * 0.2, 0.0, 1.0)
                moisture = clamp(0.58 - volcano * 0.55 + (1.0 - abs(py)) * 0.18 + raw * 0.16, 0.0, 1.0)
                corruption = clamp(0.12 + warped * 0.46 + volcano * 0.22 + max(0.0, band) * 0.18, 0.0, 1.0)
                ash = clamp(0.15 + heat * 0.6 + detail * 0.2, 0.0, 1.0)

                s.height_field[i] = elevation
                s.heat[i] = heat
                s.moisture[i] = moisture
                s.corruption[i] = corruption
                s.ash[i] = ash

                if elevation < sea_level:
                    if heat > 0.54:
                        s.terrain[i] = int(Terrain.MAGMA_SEA)
                        s.fluid[i] = int(Fluid.LAVA)
                        s.fluid_amount[i] = clamp(1.0 + heat * 0.2, 0.0, 1.5)
                        s.fire[i] = 0.22 + heat * 0.28
                    else:
                        s.terrain[i] = int(Terrain.WATER)
                        s.fluid[i] = int(Fluid.WATER)
                        s.fluid_amount[i] = clamp(0.75 + moisture * 0.35, 0.0, 1.5)
                elif elevation > 0.82:
                    if heat < 0.35 and moisture > 0.42:
                        s.terrain[i] = int(Terrain.ICEFIELD)
                    else:
                        s.terrain[i] = int(Terrain.OBSIDIAN_RIDGE)
                elif band > 0.18 and elevation > sea_level + 0.07:
                    s.terrain[i] = int(Terrain.CITY_BAND)
                elif corruption > 0.58:
                    s.terrain[i] = int(Terrain.ALIEN_BIOMASS)
                else:
                    s.terrain[i] = int(Terrain.ASH)

        self._place_cities()
        self._carve_road_network()
        self._seed_ritual_core(crater_x, crater_y, crater_radius)
        self._seed_structures_from_terrain()
        self._generate_creature_archetypes(10)
        self._spawn_initial_entities()
        self._event(f"World forged from seed {self.config.seed}")

    def _city_candidates(self) -> List[Tuple[float, int, int]]:
        candidates: List[Tuple[float, int, int]] = []
        for y in range(2, self.state.height - 2):
            for x in range(2, self.state.width - 2):
                i = self.index(x, y)
                terrain = Terrain(self.state.terrain[i])
                if terrain not in (Terrain.ASH, Terrain.CITY_BAND, Terrain.ALIEN_BIOMASS):
                    continue
                if self.state.fluid[i] != int(Fluid.NONE):
                    continue
                slope = self._local_slope(x, y)
                if slope > 0.12:
                    continue
                desirability = self.state.height_field[i] + self.state.corruption[i] * 0.25 + self.state.heat[i] * 0.12
                if terrain == Terrain.CITY_BAND:
                    desirability += 0.22
                candidates.append((desirability, x, y))
        candidates.sort(reverse=True)
        return candidates

    def _place_cities(self) -> None:
        s = self.state
        candidates = self._city_candidates()
        chosen: List[Tuple[int, int]] = []
        names = CITY_NAMES.copy()
        s.rng.shuffle(names)

        for desirability, x, y in candidates:
            if len(chosen) >= self.config.city_count:
                break
            if all(distance((x, y), prev) >= 10 for prev in chosen):
                chosen.append((x, y))

        for idx, (x, y) in enumerate(chosen):
            port_x, port_y = self._find_nearest_port(x, y)
            city = City(
                name=names[idx % len(names)],
                x=x,
                y=y,
                port_x=port_x,
                port_y=port_y,
                radius=4 + (idx % 3),
                temperament=s.rng.choice(["industrial", "occult", "migratory", "ceremonial"]),
            )
            s.cities.append(city)

            for yy in range(max(0, y - city.radius), min(s.height, y + city.radius + 1)):
                for xx in range(max(0, x - city.radius), min(s.width, x + city.radius + 1)):
                    if distance((x, y), (xx, yy)) <= city.radius + 0.25:
                        ii = self.index(xx, yy)
                        s.city_index[ii] = idx
                        if distance((x, y), (xx, yy)) <= city.radius - 1:
                            if s.ground[ii] == int(Ground.NONE):
                                s.ground[ii] = int(Ground.ROAD)
                        if distance((x, y), (xx, yy)) <= 1.5 and s.structure[ii] == int(Structure.NONE):
                            s.structure[ii] = int(Structure.OBSIDIAN)
                            s.structure_height[ii] = 1 + (idx % 2)
            center_i = self.index(x, y)
            s.structure[center_i] = int(Structure.MONOLITH if idx % 2 == 0 else Structure.FIRE_ALTAR)
            s.structure_height[center_i] = 2
            self._event(f"City {city.name} seeded at ({x},{y})")

    def _find_nearest_port(self, x: int, y: int) -> Tuple[int, int]:
        best = (x, y)
        best_dist = float("inf")
        for yy in range(self.state.height):
            for xx in range(self.state.width):
                i = self.index(xx, yy)
                fluid = Fluid(self.state.fluid[i])
                if fluid in (Fluid.WATER, Fluid.LAVA):
                    d = distance((x, y), (xx, yy))
                    if d < best_dist:
                        best_dist = d
                        best = (xx, yy)
        return best

    def _carve_road_network(self) -> None:
        if not self.state.cities:
            return
        for city_a, city_b in zip(self.state.cities, self.state.cities[1:]):
            path = self.find_path((city_a.x, city_a.y), (city_b.x, city_b.y), allow_fluids=False)
            for x, y in path:
                self.state.ground[self.index(x, y)] = int(Ground.ROAD)
        for city in self.state.cities:
            path = self.find_path((city.x, city.y), (city.port_x, city.port_y), allow_fluids=True)
            for x, y in path:
                i = self.index(x, y)
                if self.state.fluid[i] == int(Fluid.NONE):
                    self.state.ground[i] = int(Ground.ROAD)

    def _seed_ritual_core(self, crater_x: float, crater_y: float, crater_radius: float) -> None:
        cx = int(round(crater_x * (self.state.width - 1)))
        cy = int(round(crater_y * (self.state.height - 1)))
        radius = max(3, int(round(crater_radius * self.state.width * 1.5)))
        for y in range(max(0, cy - radius - 1), min(self.state.height, cy + radius + 2)):
            for x in range(max(0, cx - radius - 1), min(self.state.width, cx + radius + 2)):
                d = distance((x, y), (cx, cy))
                i = self.index(x, y)
                if abs(d - radius) <= 0.75 and self.state.fluid[i] == int(Fluid.NONE):
                    self.state.ground[i] = int(Ground.EMBER_RUNE if (x + y) % 2 == 0 else Ground.VOID_GLASS)
                elif d <= 1.5:
                    self.state.structure[i] = int(Structure.QUANTUM_CORE)
                    self.state.structure_height[i] = 2
                    self.state.fire[i] = max(self.state.fire[i], 0.8)
                    self.state.heat[i] = max(self.state.heat[i], 0.98)
        self._event("Central ritual core stabilized")

    def _seed_structures_from_terrain(self) -> None:
        s = self.state
        rng = s.rng
        for y in range(s.height):
            for x in range(s.width):
                i = self.index(x, y)
                if s.structure[i] != int(Structure.NONE):
                    continue
                if s.fluid[i] != int(Fluid.NONE):
                    continue
                terrain = Terrain(s.terrain[i])
                r = rng.random()
                if terrain == Terrain.OBSIDIAN_RIDGE and r < 0.08:
                    s.structure[i] = int(Structure.SPIRE)
                    s.structure_height[i] = 2 + (r > 0.04)
                elif terrain == Terrain.ALIEN_BIOMASS and r < 0.03:
                    s.structure[i] = int(Structure.WRAITH_TOTEM)
                    s.structure_height[i] = 1
                elif terrain == Terrain.CITY_BAND and r < 0.022:
                    s.structure[i] = int(Structure.MONOLITH)
                    s.structure_height[i] = 1
                elif terrain == Terrain.ASH and r < 0.015:
                    s.structure[i] = int(Structure.FIRE_ALTAR)
                    s.structure_height[i] = 1
                    s.fire[i] = max(s.fire[i], 0.35)

    def _generate_creature_archetypes(self, count: int) -> None:
        rng = self.state.rng
        for genome_id in range(1, count + 1):
            name = rng.choice(GENOME_NAMES["prefix"])
            if rng.random() < 0.5:
                name += rng.choice(GENOME_NAMES["vowel"])
            name += rng.choice(GENOME_NAMES["suffix"])
            archetype = CreatureGenome(
                genome_id=genome_id,
                name=name,
                title=rng.choice(GENOME_NAMES["title"]),
                element=rng.choice(GENOME_NAMES["element"]),
                horn_style=rng.choice(GENOME_NAMES["horns"]),
                eye_style=rng.choice(GENOME_NAMES["eyes"]),
                mouth_style=rng.choice(GENOME_NAMES["mouth"]),
                body_style=rng.choice(GENOME_NAMES["body"]),
                palette=(rng.randint(50, 255), rng.randint(25, 180), rng.randint(25, 255)),
                behavior=rng.choice(GENOME_NAMES["behavior"]),
                appetite=rng.uniform(0.2, 1.0),
                volatility=rng.uniform(0.1, 1.0),
                aura_bias=rng.uniform(-1.0, 1.0),
                dread_bias=rng.uniform(-1.0, 1.0),
            )
            self.state.archetypes[genome_id] = archetype

    def _spawn_initial_entities(self) -> None:
        s = self.state
        for _ in range(self.config.initial_ghosts):
            x, y = self._find_spawn_cell(prefer=[Terrain.ALIEN_BIOMASS, Terrain.CITY_BAND, Terrain.ASH], near_structure=Structure.WRAITH_TOTEM)
            self.spawn_entity(EntityKind.GHOST, x, y)
        for _ in range(self.config.initial_demons):
            x, y = self._find_spawn_cell(prefer=[Terrain.MAGMA_SEA, Terrain.ALIEN_BIOMASS, Terrain.ASH], hot=True)
            self.spawn_entity(EntityKind.DEMON, x, y)
        for _ in range(self.config.initial_creatures):
            x, y = self._find_spawn_cell(prefer=[Terrain.CITY_BAND, Terrain.ALIEN_BIOMASS, Terrain.ASH])
            genome_id = self.state.rng.choice(list(self.state.archetypes.keys()))
            self.spawn_entity(EntityKind.CREATURE, x, y, genome_id=genome_id)
        for _ in range(self.config.initial_migrants):
            self.spawn_route_entity(EntityKind.MIGRANT)
        for _ in range(self.config.initial_skimmers):
            self.spawn_route_entity(EntityKind.SKIMMER)

    def _find_spawn_cell(
        self,
        prefer: Sequence[Terrain],
        near_structure: Optional[Structure] = None,
        hot: bool = False,
    ) -> Tuple[int, int]:
        candidates: List[Tuple[float, int, int]] = []
        preferred = {int(t) for t in prefer}
        for y in range(self.state.height):
            for x in range(self.state.width):
                i = self.index(x, y)
                score = 0.0
                if self.state.terrain[i] in preferred:
                    score += 1.0
                if hot:
                    score += self.state.heat[i] * 1.5
                else:
                    score += self.state.moisture[i] * 0.4 + self.state.corruption[i] * 0.4
                if near_structure is not None:
                    if self._nearest_structure_distance(x, y, near_structure) <= 6:
                        score += 1.0
                if self.state.structure[i] != int(Structure.NONE) and self.state.structure[i] != int(Structure.WRAITH_TOTEM):
                    score -= 0.5
                if self.state.fluid[i] == int(Fluid.NONE):
                    score += 0.3
                candidates.append((score + self.state.rng.random() * 0.1, x, y))
        candidates.sort(reverse=True)
        top = candidates[: max(1, min(50, len(candidates)))]
        _, x, y = self.state.rng.choice(top)
        return x, y

    def _nearest_structure_distance(self, x: int, y: int, structure: Structure) -> int:
        best = 10_000
        for yy in range(max(0, y - 8), min(self.state.height, y + 9)):
            for xx in range(max(0, x - 8), min(self.state.width, x + 9)):
                i = self.index(xx, yy)
                if self.state.structure[i] == int(structure):
                    best = min(best, int(distance((x, y), (xx, yy))))
        return best

    def _local_slope(self, x: int, y: int) -> float:
        base = self.state.height_field[self.index(x, y)]
        diffs = []
        for nx, ny in self.neighbors4(x, y):
            diffs.append(abs(base - self.state.height_field[self.index(nx, ny)]))
        return max(diffs) if diffs else 0.0

    # ------------------------
    # Serialization
    # ------------------------
    def to_json(self) -> str:
        s = self.state
        payload = {
            "version": 1,
            "config": dataclasses.asdict(self.config),
            "tick": s.tick,
            "seconds": s.seconds,
            "terrain": s.terrain,
            "height_field": s.height_field,
            "heat": s.heat,
            "moisture": s.moisture,
            "corruption": s.corruption,
            "ash": s.ash,
            "ground": s.ground,
            "structure": s.structure,
            "structure_height": s.structure_height,
            "fluid": s.fluid,
            "fluid_amount": s.fluid_amount,
            "fire": s.fire,
            "city_index": s.city_index,
            "cities": [dataclasses.asdict(c) for c in s.cities],
            "archetypes": {str(k): dataclasses.asdict(v) for k, v in s.archetypes.items()},
            "entities": {str(k): dataclasses.asdict(v) for k, v in s.entities.items()},
            "next_entity_id": s.next_entity_id,
            "deliveries_window": list(s.deliveries_window),
            "recent_events": list(s.recent_events),
            "challenge": dataclasses.asdict(s.challenge) if s.challenge else None,
        }
        return json.dumps(payload, separators=(",", ":"))

    def to_share_code(self) -> str:
        blob = self.to_json().encode("utf-8")
        packed = zlib.compress(blob, level=9)
        return base64.urlsafe_b64encode(packed).decode("ascii")

    @classmethod
    def from_json(cls, json_text: str) -> "InfernalWorld":
        payload = json.loads(json_text)
        config = WorldConfig(**payload["config"])
        world = cls(config)
        world._load_payload(payload)
        return world

    @classmethod
    def from_share_code(cls, code: str) -> "InfernalWorld":
        blob = base64.urlsafe_b64decode(code.encode("ascii"))
        return cls.from_json(zlib.decompress(blob).decode("utf-8"))

    def _load_payload(self, payload: dict) -> None:
        s = self.state
        s.tick = int(payload["tick"])
        s.seconds = float(payload["seconds"])
        for name in (
            "terrain",
            "height_field",
            "heat",
            "moisture",
            "corruption",
            "ash",
            "ground",
            "structure",
            "structure_height",
            "fluid",
            "fluid_amount",
            "fire",
            "city_index",
        ):
            setattr(s, name, list(payload[name]))
        s.cities = [City(**city) for city in payload["cities"]]
        s.archetypes = {int(k): CreatureGenome(**v) for k, v in payload["archetypes"].items()}
        s.entities = {}
        for key, value in payload["entities"].items():
            value = dict(value)
            value["kind"] = EntityKind(value["kind"])
            s.entities[int(key)] = Entity(**value)
        s.next_entity_id = int(payload["next_entity_id"])
        s.deliveries_window = collections.deque(payload.get("deliveries_window", []), maxlen=240)
        s.recent_events = collections.deque(payload.get("recent_events", []), maxlen=14)
        challenge_payload = payload.get("challenge")
        s.challenge = Challenge(**challenge_payload) if challenge_payload else self._get_daily_challenge()
        self.compute_metrics()

    # ------------------------
    # Events and metrics
    # ------------------------
    def _event(self, message: str) -> None:
        timestamp = f"t{self.state.tick:05d}"
        self.state.recent_events.appendleft(f"[{timestamp}] {message}")

    def _get_daily_challenge(self) -> Challenge:
        if self.config.challenge_date:
            key = self.config.challenge_date
        else:
            now = time.localtime()
            key = f"{now.tm_year:04d}-{now.tm_mon:02d}-{now.tm_mday:02d}"
        index = abs(hash_string(key)) % len(DAILY_CHALLENGES)
        base = DAILY_CHALLENGES[index]
        return Challenge(base.title, base.description, dict(base.requirements), base.starter_template)

    def compute_metrics(self) -> Metrics:
        counts = collections.Counter()
        symmetry_matches = 0
        symmetry_total = 0

        for y in range(self.state.height):
            for x in range(self.state.width):
                i = self.index(x, y)
                if self.state.ground[i] == int(Ground.ROAD):
                    counts["road"] += 1
                if self.state.ground[i] == int(Ground.EMBER_RUNE):
                    counts["ember_rune"] += 1
                if self.state.ground[i] == int(Ground.VOID_GLASS):
                    counts["void_glass"] += 1
                if self.state.structure[i] != int(Structure.NONE):
                    counts[STRUCTURE_NAME[Structure(self.state.structure[i])]] += 1
                if self.state.fluid[i] == int(Fluid.LAVA):
                    counts["lava"] += 1
                if self.state.fluid[i] == int(Fluid.WATER):
                    counts["water"] += 1
                if self.state.fluid[i] == int(Fluid.ACID):
                    counts["acid"] += 1
                if self.state.fluid[i] == int(Fluid.PLASMA):
                    counts["plasma"] += 1
                if self.state.fire[i] > 0.15:
                    counts["fire"] += 1

                mirror_x = self.state.width - 1 - x
                if x <= mirror_x:
                    symmetry_total += 1
                    mi = self.index(mirror_x, y)
                    if (
                        self.state.ground[i] == self.state.ground[mi]
                        and self.state.structure[i] == self.state.structure[mi]
                        and self.state.fluid[i] == self.state.fluid[mi]
                    ):
                        symmetry_matches += 1

        for entity in self.state.entities.values():
            counts[ENTITY_NAME[entity.kind]] += 1

        symmetry = (symmetry_matches / symmetry_total) if symmetry_total else 1.0
        symmetry_bonus = 1.0 + symmetry * 0.8

        dread = (
            counts["obsidian"] * 10
            + counts["spire"] * 18
            + counts["lava"] * 8
            + counts["fire"] * 6
            + counts["ember_rune"] * 12
            + counts["demon"] * 20
            + counts["quantum_core"] * 36
            + int(sum(self.state.corruption) / max(1, len(self.state.corruption)) * 160)
        )
        aura = (
            counts["ghost"] * 18
            + counts["wraith_totem"] * 22
            + counts["monolith"] * 16
            + counts["void_glass"] * 10
            + counts["quantum_core"] * 22
            + counts["creature"] * 7
            + int(sum(self.state.moisture) / max(1, len(self.state.moisture)) * 60)
        )
        total_tiles = (
            counts["obsidian"]
            + counts["spire"]
            + counts["wraith_totem"]
            + counts["monolith"]
            + counts["quantum_core"]
            + counts["fire_altar"]
            + counts["road"]
            + counts["ember_rune"]
            + counts["void_glass"]
            + counts["lava"]
            + counts["water"]
            + counts["acid"]
            + counts["plasma"]
        )
        dread = int(dread * symmetry_bonus)
        aura = int(aura * (0.75 + symmetry * 0.7))
        souls = int(sum(self.state.deliveries_window) * 60 / max(1, len(self.state.deliveries_window)))
        souls += counts["ghost"] * 2 + counts["migrant"] + counts["skimmer"]
        rank = self._rank_for(dread + aura)

        challenge = self.state.challenge or self._get_daily_challenge()
        progress_terms = []
        for key, target in challenge.requirements.items():
            progress_terms.append(min(1.0, counts[key] / max(1, target)))
        challenge_progress = sum(progress_terms) / len(progress_terms) if progress_terms else 0.0

        metrics = Metrics(
            dread=dread,
            aura=aura,
            souls_per_minute=souls,
            symmetry=symmetry,
            total_tiles=total_tiles,
            rank=rank,
            counts=dict(counts),
            challenge_progress=challenge_progress,
            deliveries=sum(self.state.deliveries_window),
        )
        self.state.metrics = metrics
        if self.state.challenge is None:
            self.state.challenge = challenge
        return metrics

    def _rank_for(self, score: int) -> str:
        thresholds = [
            (0, "Cinder Seed"),
            (200, "Bone Keep"),
            (420, "Dread Hold"),
            (760, "Infernal Citadel"),
            (1150, "Abyssal Throne"),
        ]
        label = thresholds[0][1]
        for threshold, name in thresholds:
            if score >= threshold:
                label = name
        return label

    # ------------------------
    # Routing and pathfinding
    # ------------------------
    def movement_cost(self, x: int, y: int, allow_fluids: bool, kind: Optional[EntityKind] = None) -> float:
        i = self.index(x, y)
        structure = Structure(self.state.structure[i])
        fluid = Fluid(self.state.fluid[i])
        terrain = Terrain(self.state.terrain[i])
        ground = Ground(self.state.ground[i])
        heat = self.state.heat[i]

        if structure in (Structure.OBSIDIAN, Structure.SPIRE, Structure.MONOLITH, Structure.QUANTUM_CORE) and kind not in (EntityKind.GHOST,):
            return float("inf")
        if fluid != Fluid.NONE and not allow_fluids and kind not in (EntityKind.GHOST, EntityKind.DEMON):
            return float("inf")

        cost = 1.0
        if ground == Ground.ROAD:
            cost *= 0.45
        elif ground == Ground.VOID_GLASS:
            cost *= 0.75
        elif ground == Ground.EMBER_RUNE:
            cost *= 1.15

        if terrain == Terrain.CITY_BAND:
            cost *= 0.85
        elif terrain == Terrain.ASH:
            cost *= 1.0
        elif terrain == Terrain.ALIEN_BIOMASS:
            cost *= 1.25
        elif terrain == Terrain.OBSIDIAN_RIDGE:
            cost *= 1.5
        elif terrain == Terrain.ICEFIELD:
            cost *= 0.9
        elif terrain in (Terrain.WATER, Terrain.MAGMA_SEA):
            cost *= 1.4 if allow_fluids else 999.0

        if fluid == Fluid.LAVA:
            cost *= 1.9 if allow_fluids else 999.0
        elif fluid == Fluid.WATER:
            cost *= 1.1 if allow_fluids else 999.0
        elif fluid == Fluid.ACID:
            cost *= 1.5
        elif fluid == Fluid.PLASMA:
            cost *= 2.1

        cost += self._local_slope(x, y) * 12.0
        cost += heat * 0.4
        return cost

    def find_path(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        allow_fluids: bool,
        kind: Optional[EntityKind] = None,
    ) -> List[Tuple[int, int]]:
        if start == goal:
            return [start]
        frontier: List[Tuple[float, Tuple[int, int]]] = [(0.0, start)]
        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
        cost_so_far: Dict[Tuple[int, int], float] = {start: 0.0}

        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal:
                break
            for nx, ny in self.neighbors4(*current):
                move_cost = self.movement_cost(nx, ny, allow_fluids, kind)
                if math.isinf(move_cost):
                    continue
                new_cost = cost_so_far[current] + move_cost
                next_node = (nx, ny)
                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    cost_so_far[next_node] = new_cost
                    priority = new_cost + abs(goal[0] - nx) + abs(goal[1] - ny)
                    heapq.heappush(frontier, (priority, next_node))
                    came_from[next_node] = current

        if goal not in came_from:
            return [start]

        path = []
        cursor: Optional[Tuple[int, int]] = goal
        while cursor is not None:
            path.append(cursor)
            cursor = came_from[cursor]
        path.reverse()
        return path

    # ------------------------
    # Entity management
    # ------------------------
    def spawn_entity(
        self,
        kind: EntityKind,
        x: int,
        y: int,
        genome_id: Optional[int] = None,
        home_city: Optional[int] = None,
        dest_city: Optional[int] = None,
    ) -> int:
        entity_id = self.state.next_entity_id
        self.state.next_entity_id += 1
        rng = self.state.rng

        entity = Entity(
            entity_id=entity_id,
            kind=kind,
            x=float(x),
            y=float(y),
            genome_id=genome_id,
            home_city=home_city,
            dest_city=dest_city,
            life=1.0,
            energy=0.8 + rng.random() * 0.4,
            heat_affinity=rng.uniform(-1.0, 1.0),
            aura_affinity=rng.uniform(-1.0, 1.0),
            corruption_affinity=rng.uniform(-1.0, 1.0),
        )

        if kind == EntityKind.GHOST:
            entity.state = "haunt"
            entity.life = 0.85 + rng.random() * 0.3
            entity.heat_affinity = -0.6
            entity.aura_affinity = 0.9
            entity.corruption_affinity = 0.3
        elif kind == EntityKind.DEMON:
            entity.state = "prowl"
            entity.life = 1.3 + rng.random() * 0.5
            entity.heat_affinity = 0.95
            entity.aura_affinity = -0.25
            entity.corruption_affinity = 0.8
        elif kind == EntityKind.CREATURE:
            entity.state = "wander"
            if genome_id is None:
                entity.genome_id = self.state.rng.choice(list(self.state.archetypes.keys()))
        elif kind in (EntityKind.MIGRANT, EntityKind.SKIMMER):
            entity.state = "route"
        elif kind == EntityKind.SOUL:
            entity.state = "drift"
            entity.life = 0.6

        self.state.entities[entity_id] = entity
        return entity_id

    def spawn_route_entity(self, kind: EntityKind) -> Optional[int]:
        if len(self.state.cities) < 2:
            return None
        src_idx, dst_idx = self.state.rng.sample(range(len(self.state.cities)), 2)
        src = self.state.cities[src_idx]
        dst = self.state.cities[dst_idx]
        spawn = (src.port_x, src.port_y) if kind == EntityKind.SKIMMER else (src.x, src.y)
        entity_id = self.spawn_entity(kind, spawn[0], spawn[1], home_city=src_idx, dest_city=dst_idx)
        entity = self.state.entities[entity_id]
        allow_fluids = kind == EntityKind.SKIMMER
        goal = (dst.port_x, dst.port_y) if kind == EntityKind.SKIMMER else (dst.x, dst.y)
        entity.path = self.find_path(spawn, goal, allow_fluids=allow_fluids, kind=kind)
        entity.path_index = 0
        entity.cargo = 0.5 + self.state.rng.random() * 1.0
        return entity_id

    # ------------------------
    # Tool application
    # ------------------------
    def apply_tool(self, x: int, y: int, tool: str, brush: int = 1) -> None:
        radius = max(0, brush - 1)
        for yy in range(y - radius, y + radius + 1):
            for xx in range(x - radius, x + radius + 1):
                if not self.in_bounds(xx, yy):
                    continue
                self._apply_tool_cell(xx, yy, tool)
        self.compute_metrics()

    def _apply_tool_cell(self, x: int, y: int, tool: str) -> None:
        i = self.index(x, y)
        if tool == "erase":
            self.state.ground[i] = int(Ground.NONE)
            self.state.structure[i] = int(Structure.NONE)
            self.state.structure_height[i] = 0
            self.state.fluid[i] = int(Fluid.NONE)
            self.state.fluid_amount[i] = 0.0
            self.state.fire[i] = 0.0
            dead = [eid for eid, ent in self.state.entities.items() if ent.cell() == (x, y)]
            for eid in dead:
                self.state.entities.pop(eid, None)
            return
        if tool == "road":
            self.state.ground[i] = int(Ground.ROAD)
        elif tool == "ember_rune":
            self.state.ground[i] = int(Ground.EMBER_RUNE)
            self.state.corruption[i] = clamp(self.state.corruption[i] + 0.18, 0.0, 1.0)
        elif tool == "void_glass":
            self.state.ground[i] = int(Ground.VOID_GLASS)
            self.state.moisture[i] = clamp(self.state.moisture[i] + 0.08, 0.0, 1.0)
        elif tool == "obsidian":
            self.state.structure[i] = int(Structure.OBSIDIAN)
            self.state.structure_height[i] = min(3, max(1, self.state.structure_height[i] + 1))
            self.state.fluid[i] = int(Fluid.NONE)
            self.state.fluid_amount[i] = 0.0
            self.state.fire[i] = 0.0
        elif tool == "spire":
            self.state.structure[i] = int(Structure.SPIRE)
            self.state.structure_height[i] = 2
        elif tool == "wraith_totem":
            self.state.structure[i] = int(Structure.WRAITH_TOTEM)
            self.state.structure_height[i] = 1
        elif tool == "monolith":
            self.state.structure[i] = int(Structure.MONOLITH)
            self.state.structure_height[i] = 2
        elif tool == "quantum_core":
            self.state.structure[i] = int(Structure.QUANTUM_CORE)
            self.state.structure_height[i] = 2
            self.state.fire[i] = max(self.state.fire[i], 0.65)
            self.state.heat[i] = max(self.state.heat[i], 0.9)
        elif tool == "fire_altar":
            self.state.structure[i] = int(Structure.FIRE_ALTAR)
            self.state.structure_height[i] = 1
            self.state.fire[i] = max(self.state.fire[i], 1.0)
        elif tool == "lava":
            self.state.fluid[i] = int(Fluid.LAVA)
            self.state.fluid_amount[i] = 1.2
            self.state.fire[i] = max(self.state.fire[i], 0.45)
            self.state.heat[i] = max(self.state.heat[i], 0.9)
        elif tool == "water":
            self.state.fluid[i] = int(Fluid.WATER)
            self.state.fluid_amount[i] = 1.0
            self.state.moisture[i] = max(self.state.moisture[i], 0.9)
        elif tool == "acid":
            self.state.fluid[i] = int(Fluid.ACID)
            self.state.fluid_amount[i] = 0.9
        elif tool == "plasma":
            self.state.fluid[i] = int(Fluid.PLASMA)
            self.state.fluid_amount[i] = 0.8
            self.state.fire[i] = max(self.state.fire[i], 0.8)
            self.state.heat[i] = max(self.state.heat[i], 1.0)
        elif tool == "ghost":
            self.spawn_entity(EntityKind.GHOST, x, y)
        elif tool == "demon":
            self.spawn_entity(EntityKind.DEMON, x, y)

    # ------------------------
    # Tick systems
    # ------------------------
    def step(self, ticks: int = 1) -> None:
        for _ in range(ticks):
            self.state.tick += 1
            self.state.seconds += self.config.tick_seconds
            self._update_global_fields()
            self._structure_field_system()
            self._chemistry_system()
            self._spawn_system()
            self._entity_system()
            self._cleanup_system()
            self.compute_metrics()

    def _update_global_fields(self) -> None:
        t = self.state.tick
        phase = t * 0.013
        for i in range(len(self.state.heat)):
            self.state.ash[i] = clamp(self.state.ash[i] * 0.985 + 0.01 + 0.01 * math.sin(phase + i * 0.01), 0.0, 1.0)
            self.state.heat[i] = clamp(self.state.heat[i] * 0.995 + 0.006 * math.sin(phase + i * 0.002), 0.0, 1.0)
            self.state.moisture[i] = clamp(self.state.moisture[i] * 0.998 + 0.001 * math.cos(phase * 0.5 + i * 0.005), 0.0, 1.0)
            self.state.corruption[i] = clamp(self.state.corruption[i] * 0.998, 0.0, 1.0)
        if t % 80 == 0:
            self._event("Planetary bands shifted")

    def _structure_field_system(self) -> None:
        s = self.state
        to_increase_ghost_spawn = 0
        for y in range(s.height):
            for x in range(s.width):
                i = self.index(x, y)
                structure = Structure(s.structure[i])
                ground = Ground(s.ground[i])
                if ground == Ground.EMBER_RUNE:
                    s.corruption[i] = clamp(s.corruption[i] + 0.008, 0.0, 1.0)
                    s.fire[i] = clamp(s.fire[i] + 0.004, 0.0, 1.0)
                elif ground == Ground.VOID_GLASS:
                    s.moisture[i] = clamp(s.moisture[i] + 0.004, 0.0, 1.0)

                if structure == Structure.WRAITH_TOTEM:
                    s.moisture[i] = clamp(s.moisture[i] + 0.01, 0.0, 1.0)
                    s.heat[i] = clamp(s.heat[i] - 0.01, 0.0, 1.0)
                    to_increase_ghost_spawn += 1
                elif structure == Structure.MONOLITH:
                    s.moisture[i] = clamp(s.moisture[i] + 0.006, 0.0, 1.0)
                    s.corruption[i] = clamp(s.corruption[i] - 0.005, 0.0, 1.0)
                elif structure == Structure.QUANTUM_CORE:
                    s.heat[i] = clamp(s.heat[i] + 0.02, 0.0, 1.0)
                    s.corruption[i] = clamp(s.corruption[i] + 0.012, 0.0, 1.0)
                    s.fire[i] = clamp(s.fire[i] + 0.025, 0.0, 1.0)
                    for nx, ny in self.neighbors8(x, y):
                        ni = self.index(nx, ny)
                        if s.rng.random() < 0.008:
                            s.fluid[ni] = int(Fluid.PLASMA)
                            s.fluid_amount[ni] = max(s.fluid_amount[ni], 0.35)
                elif structure == Structure.FIRE_ALTAR:
                    s.fire[i] = clamp(max(s.fire[i], 0.55) + 0.015, 0.0, 1.0)
                    s.heat[i] = clamp(s.heat[i] + 0.012, 0.0, 1.0)
                elif structure == Structure.SPIRE:
                    s.ash[i] = clamp(s.ash[i] + 0.01, 0.0, 1.0)
                    s.corruption[i] = clamp(s.corruption[i] + 0.003, 0.0, 1.0)
                elif structure == Structure.OBSIDIAN:
                    s.fire[i] = max(0.0, s.fire[i] - 0.03)
                    s.heat[i] = max(0.0, s.heat[i] - 0.01)

        if to_increase_ghost_spawn and self.state.tick % 60 == 0 and self.state.rng.random() < min(0.8, 0.05 * to_increase_ghost_spawn):
            x, y = self._find_spawn_cell(prefer=[Terrain.ALIEN_BIOMASS, Terrain.ASH, Terrain.CITY_BAND], near_structure=Structure.WRAITH_TOTEM)
            self.spawn_entity(EntityKind.GHOST, x, y)
            self._event("A ghost condensed around a wraith totem")

    def _chemistry_system(self) -> None:
        s = self.state
        new_fluid = s.fluid[:]
        new_amount = s.fluid_amount[:]
        new_fire = s.fire[:]
        new_terrain = s.terrain[:]
        new_structure = s.structure[:]
        new_structure_height = s.structure_height[:]

        for y in range(s.height):
            for x in range(s.width):
                i = self.index(x, y)
                fluid = Fluid(s.fluid[i])
                amount = s.fluid_amount[i]
                terrain = Terrain(s.terrain[i])
                structure = Structure(s.structure[i])

                if fluid != Fluid.NONE and amount > 0.02:
                    if fluid == Fluid.WATER:
                        if s.fire[i] > 0.2:
                            new_fire[i] = max(0.0, new_fire[i] - 0.25)
                        if terrain == Terrain.MAGMA_SEA:
                            new_terrain[i] = int(Terrain.OBSIDIAN_RIDGE)
                    elif fluid == Fluid.LAVA:
                        new_fire[i] = clamp(new_fire[i] + 0.04, 0.0, 1.0)
                        if s.moisture[i] > 0.65:
                            new_terrain[i] = int(Terrain.OBSIDIAN_RIDGE)
                    elif fluid == Fluid.ACID:
                        if structure in (Structure.OBSIDIAN, Structure.SPIRE, Structure.FIRE_ALTAR):
                            if s.rng.random() < 0.06:
                                new_structure[i] = int(Structure.NONE)
                                new_structure_height[i] = 0
                                self._event(f"Acid dissolved a {STRUCTURE_NAME[structure]}")
                        if Ground(s.ground[i]) in (Ground.ROAD, Ground.EMBER_RUNE):
                            if s.rng.random() < 0.08:
                                s.ground[i] = int(Ground.NONE)
                    elif fluid == Fluid.PLASMA:
                        new_fire[i] = clamp(new_fire[i] + 0.08, 0.0, 1.0)
                        s.heat[i] = clamp(s.heat[i] + 0.05, 0.0, 1.0)
                        if s.rng.random() < 0.02:
                            for nx, ny in self.neighbors4(x, y):
                                ni = self.index(nx, ny)
                                if Structure(s.structure[ni]) == Structure.NONE:
                                    new_fire[ni] = clamp(new_fire[ni] + 0.18, 0.0, 1.0)

                    target: Optional[Tuple[int, int]] = None
                    best_energy = self._flow_energy(x, y)
                    for nx, ny in self.neighbors4(x, y):
                        ni = self.index(nx, ny)
                        if math.isinf(self.movement_cost(nx, ny, allow_fluids=True)):
                            continue
                        energy = self._flow_energy(nx, ny)
                        if energy < best_energy - 0.01:
                            best_energy = energy
                            target = (nx, ny)
                    if target is not None:
                        ni = self.index(*target)
                        move = min(amount * (0.18 if fluid != Fluid.PLASMA else 0.12), 0.22)
                        if move > 0.01:
                            if Fluid(new_fluid[ni]) in (Fluid.NONE, fluid):
                                new_fluid[ni] = int(fluid)
                                new_amount[ni] += move
                                new_amount[i] -= move
                            else:
                                other = Fluid(new_fluid[ni])
                                if {fluid, other} == {Fluid.WATER, Fluid.LAVA}:
                                    new_fluid[ni] = int(Fluid.NONE)
                                    new_amount[ni] = 0.0
                                    new_terrain[ni] = int(Terrain.OBSIDIAN_RIDGE)
                                    new_fire[ni] = max(0.0, new_fire[ni] - 0.3)
                                elif fluid == Fluid.ACID and other == Fluid.WATER:
                                    new_fluid[ni] = int(Fluid.ACID)
                                    new_amount[ni] = min(1.2, new_amount[ni] + move * 0.5)
                                elif fluid == Fluid.PLASMA:
                                    new_fire[ni] = clamp(new_fire[ni] + 0.1, 0.0, 1.0)
                    new_amount[i] = clamp(new_amount[i], 0.0, 1.5)

                if s.fire[i] > 0.01:
                    volatility = self._cell_volatility(x, y)
                    cooling = 0.05 + s.moisture[i] * 0.08
                    heating = 0.02 + s.heat[i] * 0.06 + volatility * 0.04
                    new_fire[i] = clamp(new_fire[i] + heating - cooling, 0.0, 1.0)
                    if new_fire[i] > 0.18:
                        for nx, ny in self.neighbors4(x, y):
                            ni = self.index(nx, ny)
                            chance = 0.05 + self._cell_volatility(nx, ny) * 0.22
                            chance += 0.08 if Fluid(s.fluid[ni]) == Fluid.PLASMA else 0.0
                            chance -= 0.06 if Fluid(s.fluid[ni]) == Fluid.WATER else 0.0
                            if s.rng.random() < chance:
                                new_fire[ni] = clamp(new_fire[ni] + 0.14, 0.0, 1.0)
                if terrain == Terrain.ICEFIELD and s.heat[i] > 0.6:
                    new_terrain[i] = int(Terrain.WATER)
                    new_fluid[i] = int(Fluid.WATER)
                    new_amount[i] = max(new_amount[i], 0.6)

        s.fluid = new_fluid
        s.fluid_amount = [clamp(v, 0.0, 1.5) for v in new_amount]
        s.fire = [clamp(v, 0.0, 1.0) for v in new_fire]
        s.terrain = new_terrain
        s.structure = new_structure
        s.structure_height = new_structure_height

        for i in range(len(s.fluid_amount)):
            if s.fluid_amount[i] < 0.03:
                s.fluid[i] = int(Fluid.NONE)
                s.fluid_amount[i] = 0.0

    def _flow_energy(self, x: int, y: int) -> float:
        i = self.index(x, y)
        barrier = 0.0
        structure = Structure(self.state.structure[i])
        if structure == Structure.OBSIDIAN:
            barrier += 0.8
        elif structure in (Structure.SPIRE, Structure.MONOLITH, Structure.QUANTUM_CORE):
            barrier += 0.5
        return self.state.height_field[i] + barrier - self.state.heat[i] * 0.04

    def _cell_volatility(self, x: int, y: int) -> float:
        i = self.index(x, y)
        terrain = Terrain(self.state.terrain[i])
        ground = Ground(self.state.ground[i])
        structure = Structure(self.state.structure[i])
        value = 0.0
        if terrain == Terrain.ALIEN_BIOMASS:
            value += 0.45
        elif terrain == Terrain.ASH:
            value += 0.2
        elif terrain == Terrain.CITY_BAND:
            value += 0.08
        if ground == Ground.EMBER_RUNE:
            value += 0.25
        elif ground == Ground.ROAD:
            value += 0.05
        if structure == Structure.FIRE_ALTAR:
            value += 0.5
        elif structure == Structure.WRAITH_TOTEM:
            value += 0.12
        elif structure == Structure.MONOLITH:
            value -= 0.08
        elif structure == Structure.OBSIDIAN:
            value -= 0.2
        if Fluid(self.state.fluid[i]) == Fluid.WATER:
            value -= 0.35
        return clamp(value, 0.0, 1.0)

    def _spawn_system(self) -> None:
        s = self.state
        if self._spawn_history_cooldown > 0:
            self._spawn_history_cooldown -= 1

        if s.tick % 40 == 0 and len([e for e in s.entities.values() if e.kind == EntityKind.MIGRANT]) < self.config.initial_migrants + 6:
            self.spawn_route_entity(EntityKind.MIGRANT)
        if s.tick % 55 == 0 and len([e for e in s.entities.values() if e.kind == EntityKind.SKIMMER]) < self.config.initial_skimmers + 4:
            self.spawn_route_entity(EntityKind.SKIMMER)

        demon_count = sum(1 for e in s.entities.values() if e.kind == EntityKind.DEMON)
        if s.tick % 70 == 0 and demon_count < self.config.initial_demons + 8:
            x, y = self._find_spawn_cell(prefer=[Terrain.MAGMA_SEA, Terrain.ALIEN_BIOMASS, Terrain.ASH], hot=True)
            self.spawn_entity(EntityKind.DEMON, x, y)
            if self._spawn_history_cooldown == 0:
                self._event("A demon crawled from the magma bands")
                self._spawn_history_cooldown = 12

        creature_count = sum(1 for e in s.entities.values() if e.kind == EntityKind.CREATURE)
        if s.tick % 85 == 0 and creature_count < self.config.initial_creatures + 10:
            x, y = self._find_spawn_cell(prefer=[Terrain.CITY_BAND, Terrain.ALIEN_BIOMASS, Terrain.ASH])
            genome_id = s.rng.choice(list(s.archetypes.keys()))
            eid = self.spawn_entity(EntityKind.CREATURE, x, y, genome_id=genome_id)
            if self._spawn_history_cooldown == 0:
                genome = s.archetypes[self.state.entities[eid].genome_id or genome_id]
                self._event(f"{genome.name} {genome.title} entered the domain")
                self._spawn_history_cooldown = 12

        if s.tick % 30 == 0 and s.metrics.challenge_progress > 0.35:
            x, y = self._find_spawn_cell(prefer=[Terrain.CITY_BAND, Terrain.ASH], near_structure=Structure.QUANTUM_CORE)
            self.spawn_entity(EntityKind.SOUL, x, y)

    def _entity_system(self) -> None:
        s = self.state
        for entity in list(s.entities.values()):
            entity.age_ticks += 1
            if entity.kind in (EntityKind.MIGRANT, EntityKind.SKIMMER):
                self._update_route_entity(entity)
            elif entity.kind == EntityKind.GHOST:
                self._update_ghost(entity)
            elif entity.kind == EntityKind.DEMON:
                self._update_demon(entity)
            elif entity.kind == EntityKind.CREATURE:
                self._update_creature(entity)
            elif entity.kind == EntityKind.SOUL:
                self._update_soul(entity)

    def _update_route_entity(self, entity: Entity) -> None:
        s = self.state
        kind = entity.kind
        allow_fluids = kind == EntityKind.SKIMMER
        if not entity.path or entity.path_index >= len(entity.path):
            if entity.home_city is None or entity.dest_city is None:
                return
            dst = s.cities[entity.dest_city]
            start = entity.cell()
            goal = (dst.port_x, dst.port_y) if kind == EntityKind.SKIMMER else (dst.x, dst.y)
            entity.path = self.find_path(start, goal, allow_fluids=allow_fluids, kind=kind)
            entity.path_index = 0
            if len(entity.path) <= 1:
                return

        target = entity.path[min(entity.path_index, len(entity.path) - 1)]
        speed = 0.22 if kind == EntityKind.MIGRANT else 0.26
        dx = target[0] - entity.x
        dy = target[1] - entity.y
        dist = math.hypot(dx, dy)
        if dist < speed:
            entity.x = float(target[0])
            entity.y = float(target[1])
            entity.path_index += 1
            if entity.path_index >= len(entity.path):
                self._complete_delivery(entity)
        else:
            entity.vx = dx / dist * speed
            entity.vy = dy / dist * speed
            entity.x += entity.vx
            entity.y += entity.vy

    def _complete_delivery(self, entity: Entity) -> None:
        s = self.state
        cargo = int(max(1, round(entity.cargo * 3)))
        s.deliveries_window.append(cargo)
        if entity.dest_city is not None:
            city = s.cities[entity.dest_city]
            self._event(f"{ENTITY_NAME[entity.kind].title()} delivered to {city.name}")
            if entity.kind == EntityKind.MIGRANT and s.rng.random() < 0.18:
                self.spawn_entity(EntityKind.SOUL, city.x, city.y)
        if entity.home_city is None or entity.dest_city is None:
            return
        entity.home_city, entity.dest_city = entity.dest_city, entity.home_city
        dst = s.cities[entity.dest_city]
        entity.path = self.find_path(
            entity.cell(),
            (dst.port_x, dst.port_y) if entity.kind == EntityKind.SKIMMER else (dst.x, dst.y),
            allow_fluids=entity.kind == EntityKind.SKIMMER,
            kind=entity.kind,
        )
        entity.path_index = 0
        entity.cargo = 0.4 + s.rng.random() * 1.2

    def _update_ghost(self, entity: Entity) -> None:
        x, y = entity.cell()
        best = (entity.x, entity.y)
        best_score = -9999.0
        for nx, ny in self.neighbors8(x, y):
            i = self.index(nx, ny)
            score = self.state.moisture[i] * 1.2 + self.state.fire[i] * 0.1
            score += 0.8 if self.state.structure[i] == int(Structure.WRAITH_TOTEM) else 0.0
            score += 0.6 if self.state.structure[i] == int(Structure.MONOLITH) else 0.0
            score += 0.5 if self.state.ground[i] == int(Ground.VOID_GLASS) else 0.0
            score -= self.state.heat[i] * 1.6
            score += self.state.rng.random() * 0.06
            if score > best_score:
                best_score = score
                best = (nx, ny)
        self._move_entity_toward(entity, best, speed=0.18)
        i = self.index(*entity.cell())
        self.state.corruption[i] = clamp(self.state.corruption[i] - 0.02, 0.0, 1.0)
        self.state.moisture[i] = clamp(self.state.moisture[i] + 0.01, 0.0, 1.0)
        entity.life -= 0.0006 + self.state.heat[i] * 0.001

    def _update_demon(self, entity: Entity) -> None:
        x, y = entity.cell()
        best = (entity.x, entity.y)
        best_score = -9999.0
        for nx, ny in self.neighbors8(x, y):
            i = self.index(nx, ny)
            score = self.state.heat[i] * 1.4 + self.state.corruption[i] * 1.25 + self.state.fire[i] * 0.8
            score += 0.7 if self.state.fluid[i] == int(Fluid.LAVA) else 0.0
            score -= 0.3 if self.state.fluid[i] == int(Fluid.WATER) else 0.0
            score += self.state.rng.random() * 0.08
            if score > best_score:
                best_score = score
                best = (nx, ny)
        self._move_entity_toward(entity, best, speed=0.2)
        i = self.index(*entity.cell())
        self.state.corruption[i] = clamp(self.state.corruption[i] + 0.025, 0.0, 1.0)
        self.state.fire[i] = clamp(self.state.fire[i] + 0.02, 0.0, 1.0)
        self.state.heat[i] = clamp(self.state.heat[i] + 0.015, 0.0, 1.0)
        entity.life -= 0.0004

    def _update_creature(self, entity: Entity) -> None:
        genome = self.state.archetypes.get(entity.genome_id or 1)
        x, y = entity.cell()
        best = (entity.x, entity.y)
        best_score = -9999.0
        for nx, ny in self.neighbors8(x, y):
            i = self.index(nx, ny)
            score = self.state.height_field[i] * 0.15 + self.state.moisture[i] * 0.3
            score += self.state.corruption[i] * genome.dread_bias * 0.4
            score += self.state.moisture[i] * genome.aura_bias * 0.4
            if self.state.city_index[i] >= 0:
                score += 0.25 if genome.behavior in ("curious", "playful") else -0.05
            if self.state.structure[i] == int(Structure.FIRE_ALTAR):
                score += genome.volatility * 0.1
            score += self.state.rng.random() * 0.09
            if score > best_score:
                best_score = score
                best = (nx, ny)
        self._move_entity_toward(entity, best, speed=0.16)
        i = self.index(*entity.cell())
        if genome.behavior == "chaotic" and self.state.rng.random() < 0.015:
            self.state.fire[i] = clamp(self.state.fire[i] + 0.08, 0.0, 1.0)
        if genome.behavior == "reverent" and self.state.structure[i] == int(Structure.MONOLITH):
            entity.energy = clamp(entity.energy + 0.01, 0.0, 2.0)
        entity.life -= 0.0003

    def _update_soul(self, entity: Entity) -> None:
        x, y = entity.cell()
        target: Optional[Tuple[int, int]] = None
        best = float("inf")
        for yy in range(max(0, y - 10), min(self.state.height, y + 11)):
            for xx in range(max(0, x - 10), min(self.state.width, x + 11)):
                i = self.index(xx, yy)
                if self.state.structure[i] in (int(Structure.QUANTUM_CORE), int(Structure.MONOLITH), int(Structure.WRAITH_TOTEM)):
                    d = distance((x, y), (xx, yy))
                    if d < best:
                        best = d
                        target = (xx, yy)
        if target is not None:
            self._move_entity_toward(entity, target, speed=0.15)
            i = self.index(*entity.cell())
            if self.state.structure[i] in (int(Structure.QUANTUM_CORE), int(Structure.MONOLITH)):
                self.state.deliveries_window.append(2)
                entity.life = 0.0
                self._event("A soul fed the ritual network")
        else:
            entity.life -= 0.01
        entity.life -= 0.003

    def _move_entity_toward(self, entity: Entity, target: Tuple[float, float], speed: float) -> None:
        dx = target[0] - entity.x
        dy = target[1] - entity.y
        dist = math.hypot(dx, dy)
        if dist == 0:
            return
        entity.vx = dx / dist * speed
        entity.vy = dy / dist * speed
        entity.x = clamp(entity.x + entity.vx, 0.0, self.state.width - 1.0)
        entity.y = clamp(entity.y + entity.vy, 0.0, self.state.height - 1.0)

    def _cleanup_system(self) -> None:
        dead_ids = []
        for entity in self.state.entities.values():
            if entity.life <= 0.0:
                dead_ids.append(entity.entity_id)
            else:
                x, y = entity.cell()
                if not self.in_bounds(x, y):
                    dead_ids.append(entity.entity_id)
        for entity_id in dead_ids:
            entity = self.state.entities.pop(entity_id, None)
            if entity and entity.kind in (EntityKind.GHOST, EntityKind.DEMON, EntityKind.CREATURE):
                self._event(f"{ENTITY_NAME[entity.kind].title()} dissipated")

    # ------------------------
    # Summary and templates
    # ------------------------
    def summary(self) -> Dict[str, object]:
        metrics = self.compute_metrics()
        challenge = self.state.challenge or self._get_daily_challenge()
        return {
            "seed": self.config.seed,
            "tick": self.state.tick,
            "seconds": round(self.state.seconds, 2),
            "metrics": dataclasses.asdict(metrics),
            "challenge": dataclasses.asdict(challenge),
            "cities": [dataclasses.asdict(city) for city in self.state.cities],
            "recent_events": list(self.state.recent_events),
            "entity_count": len(self.state.entities),
        }

    def apply_template(self, template_name: str) -> None:
        template_name = template_name.strip().lower()
        cx = self.state.width // 2
        cy = self.state.height // 2

        if template_name == "citadel":
            for radius in (4, 7):
                for x in range(cx - radius, cx + radius + 1):
                    if self.in_bounds(x, cy - radius):
                        self._apply_tool_cell(x, cy - radius, "obsidian")
                    if self.in_bounds(x, cy + radius):
                        self._apply_tool_cell(x, cy + radius, "obsidian")
                for y in range(cy - radius, cy + radius + 1):
                    if self.in_bounds(cx - radius, y):
                        self._apply_tool_cell(cx - radius, y, "obsidian")
                    if self.in_bounds(cx + radius, y):
                        self._apply_tool_cell(cx + radius, y, "obsidian")
            for x in range(cx - 8, cx + 9):
                if self.in_bounds(x, cy - 8):
                    self._apply_tool_cell(x, cy - 8, "lava")
                if self.in_bounds(x, cy + 8):
                    self._apply_tool_cell(x, cy + 8, "lava")
            for y in range(cy - 8, cy + 9):
                if self.in_bounds(cx - 8, y):
                    self._apply_tool_cell(cx - 8, y, "lava")
                if self.in_bounds(cx + 8, y):
                    self._apply_tool_cell(cx + 8, y, "lava")
            self._apply_tool_cell(cx, cy, "quantum_core")
            self._apply_tool_cell(cx - 4, cy, "spire")
            self._apply_tool_cell(cx + 4, cy, "spire")
        elif template_name == "ritual":
            for yy in range(cy - 8, cy + 9):
                for xx in range(cx - 8, cx + 9):
                    if not self.in_bounds(xx, yy):
                        continue
                    d = distance((xx, yy), (cx, cy))
                    if abs(d - 7) < 0.8:
                        self._apply_tool_cell(xx, yy, "ember_rune")
                    elif abs(d - 4) < 0.8:
                        self._apply_tool_cell(xx, yy, "void_glass")
            self._apply_tool_cell(cx, cy, "quantum_core")
            self._apply_tool_cell(cx + 4, cy, "monolith")
            self._apply_tool_cell(cx - 4, cy, "monolith")
            self._apply_tool_cell(cx, cy + 4, "wraith_totem")
            self._apply_tool_cell(cx, cy - 4, "wraith_totem")
        elif template_name == "gauntlet":
            for i in range(-12, 13):
                if self.in_bounds(cx + i, cy + i):
                    self._apply_tool_cell(cx + i, cy + i, "lava")
                if i % 2 == 0 and self.in_bounds(cx + i, cy + i + 1):
                    self._apply_tool_cell(cx + i, cy + i + 1, "spire")
                if i % 3 == 0 and self.in_bounds(cx + i - 1, cy + i):
                    self._apply_tool_cell(cx + i - 1, cy + i, "fire_altar")
        elif template_name == "maze":
            for yy in range(cy - 10, cy + 11):
                for xx in range(cx - 10, cx + 11):
                    if not self.in_bounds(xx, yy):
                        continue
                    if (xx + yy) % 3 == 0:
                        self._apply_tool_cell(xx, yy, "obsidian")
                    if (xx * yy) % 11 == 0:
                        self._apply_tool_cell(xx, yy, "road")
                    if (xx + yy) % 8 == 0:
                        self._apply_tool_cell(xx, yy, "lava")
            self._apply_tool_cell(cx, cy, "fire_altar")
        self.compute_metrics()
        self._event(f"Template applied: {template_name}")


# ------------------------------------------------------------
# Renderer layer
# ------------------------------------------------------------


PALETTE = {
    "terrain": {
        Terrain.WATER: (52, 96, 170),
        Terrain.MAGMA_SEA: (140, 40, 12),
        Terrain.ASH: (78, 63, 56),
        Terrain.OBSIDIAN_RIDGE: (32, 35, 40),
        Terrain.CITY_BAND: (66, 72, 92),
        Terrain.ICEFIELD: (185, 215, 240),
        Terrain.ALIEN_BIOMASS: (105, 38, 50),
    },
    "ground": {
        Ground.ROAD: (190, 176, 156),
        Ground.EMBER_RUNE: (180, 30, 38),
        Ground.VOID_GLASS: (85, 70, 126),
    },
    "structure": {
        Structure.OBSIDIAN: (15, 15, 18),
        Structure.SPIRE: (140, 22, 22),
        Structure.WRAITH_TOTEM: (110, 235, 255),
        Structure.MONOLITH: (120, 90, 170),
        Structure.QUANTUM_CORE: (250, 250, 255),
        Structure.FIRE_ALTAR: (255, 190, 30),
    },
    "fluid": {
        Fluid.WATER: (70, 135, 240),
        Fluid.LAVA: (255, 92, 0),
        Fluid.ACID: (176, 230, 40),
        Fluid.PLASMA: (225, 70, 255),
    },
    "entity": {
        EntityKind.GHOST: (180, 255, 255),
        EntityKind.DEMON: (255, 80, 55),
        EntityKind.CREATURE: (255, 145, 205),
        EntityKind.MIGRANT: (255, 240, 220),
        EntityKind.SKIMMER: (120, 185, 255),
        EntityKind.SOUL: (255, 230, 120),
    },
}


def blend(base: Tuple[int, int, int], top: Tuple[int, int, int], alpha: float) -> Tuple[int, int, int]:
    alpha = clamp(alpha, 0.0, 1.0)
    return (
        int(base[0] * (1 - alpha) + top[0] * alpha),
        int(base[1] * (1 - alpha) + top[1] * alpha),
        int(base[2] * (1 - alpha) + top[2] * alpha),
    )


def shade(color: Tuple[int, int, int], amount: float) -> Tuple[int, int, int]:
    factor = clamp(1.0 + amount, 0.0, 2.0)
    return tuple(int(clamp(c * factor, 0, 255)) for c in color)


class PPMSnapshotRenderer:
    def __init__(self, cell_px: int = 8) -> None:
        self.cell_px = cell_px

    def render(self, world: InfernalWorld, path: str) -> None:
        width = world.state.width * self.cell_px
        height = world.state.height * self.cell_px
        buffer = bytearray(width * height * 3)

        def set_pixel(px: int, py: int, rgb: Tuple[int, int, int]) -> None:
            if 0 <= px < width and 0 <= py < height:
                idx = (py * width + px) * 3
                buffer[idx: idx + 3] = bytes(rgb)

        for y in range(world.state.height):
            for x in range(world.state.width):
                color = composite_cell_color(world, x, y)
                for py in range(y * self.cell_px, (y + 1) * self.cell_px):
                    for px in range(x * self.cell_px, (x + 1) * self.cell_px):
                        set_pixel(px, py, color)
        for entity in world.state.entities.values():
            ex = int(entity.x * self.cell_px + self.cell_px / 2)
            ey = int(entity.y * self.cell_px + self.cell_px / 2)
            color = PALETTE["entity"][entity.kind]
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    if dx * dx + dy * dy <= 6:
                        set_pixel(ex + dx, ey + dy, color)

        with open(path, "wb") as fh:
            header = f"P6\n{width} {height}\n255\n".encode("ascii")
            fh.write(header)
            fh.write(buffer)


class TkRenderer:
    def __init__(self, world: InfernalWorld, cell_px: int = 10) -> None:
        try:
            import tkinter as tk
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("tkinter is not available") from exc

        self.tk = tk
        self.world = world
        self.cell_px = cell_px
        self.paused = False
        self.tool = world.TOOL_SEQUENCE[0]
        self.brush = 1
        self.speed_ticks = 1
        self.snapshot_index = 0

        self.root = tk.Tk()
        self.root.title("Infernal Data-First World")
        self.root.configure(bg="#0b0b10")

        canvas_w = world.state.width * cell_px
        canvas_h = world.state.height * cell_px
        self.canvas = tk.Canvas(self.root, width=canvas_w, height=canvas_h, bg="#000000", highlightthickness=0)
        self.canvas.grid(row=0, column=0, rowspan=2, sticky="nsew")

        self.sidebar = tk.Text(
            self.root,
            width=44,
            height=48,
            bg="#111118",
            fg="#e8d8c8",
            insertbackground="#e8d8c8",
            relief="flat",
            padx=12,
            pady=12,
            font=("Courier New", 10),
        )
        self.sidebar.grid(row=0, column=1, sticky="nsew")
        self.sidebar.configure(state="disabled")

        self.status = tk.Label(
            self.root,
            text="Left click paints • Right click erases • Q/A tool • [ ] brush • Space pause",
            anchor="w",
            bg="#181820",
            fg="#c8b5a7",
            padx=10,
            pady=6,
            font=("Courier New", 10),
        )
        self.status.grid(row=1, column=1, sticky="ew")

        self.root.grid_columnconfigure(0, weight=0)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        self.canvas.bind("<Button-1>", self.on_paint)
        self.canvas.bind("<B1-Motion>", self.on_paint)
        self.canvas.bind("<Button-3>", self.on_erase)
        self.canvas.bind("<B3-Motion>", self.on_erase)
        self.root.bind("<space>", self.on_pause)
        self.root.bind("<Key-q>", self.tool_prev)
        self.root.bind("<Key-a>", self.tool_next)
        self.root.bind("<Key-bracketleft>", self.brush_down)
        self.root.bind("<Key-bracketright>", self.brush_up)
        self.root.bind("<Key-n>", self.step_once)
        self.root.bind("<Key-s>", self.save_snapshot)
        self.root.bind("<Key-t>", self.apply_random_template)
        self.root.bind("<Escape>", lambda _e: self.root.destroy())

    def canvas_to_cell(self, event) -> Tuple[int, int]:
        x = int(event.x // self.cell_px)
        y = int(event.y // self.cell_px)
        return max(0, min(self.world.state.width - 1, x)), max(0, min(self.world.state.height - 1, y))

    def on_paint(self, event) -> None:
        x, y = self.canvas_to_cell(event)
        self.world.apply_tool(x, y, self.tool, brush=self.brush)
        self.draw()

    def on_erase(self, event) -> None:
        x, y = self.canvas_to_cell(event)
        self.world.apply_tool(x, y, "erase", brush=self.brush)
        self.draw()

    def on_pause(self, _event=None) -> None:
        self.paused = not self.paused
        self.draw()

    def tool_prev(self, _event=None) -> None:
        idx = self.world.TOOL_SEQUENCE.index(self.tool)
        self.tool = self.world.TOOL_SEQUENCE[(idx - 1) % len(self.world.TOOL_SEQUENCE)]
        self.draw()

    def tool_next(self, _event=None) -> None:
        idx = self.world.TOOL_SEQUENCE.index(self.tool)
        self.tool = self.world.TOOL_SEQUENCE[(idx + 1) % len(self.world.TOOL_SEQUENCE)]
        self.draw()

    def brush_down(self, _event=None) -> None:
        self.brush = max(1, self.brush - 1)
        self.draw()

    def brush_up(self, _event=None) -> None:
        self.brush = min(4, self.brush + 1)
        self.draw()

    def step_once(self, _event=None) -> None:
        self.world.step(1)
        self.draw()

    def apply_random_template(self, _event=None) -> None:
        template = self.world.state.rng.choice(["citadel", "ritual", "gauntlet", "maze"])
        self.world.apply_template(template)
        self.draw()

    def save_snapshot(self, _event=None) -> None:
        path = os.path.abspath(f"infernal_snapshot_{self.snapshot_index:03d}.ppm")
        PPMSnapshotRenderer(self.cell_px).render(self.world, path)
        self.snapshot_index += 1
        self.draw(extra_message=f"snapshot saved to {path}")

    def draw(self, extra_message: str = "") -> None:
        self.canvas.delete("all")
        for y in range(self.world.state.height):
            for x in range(self.world.state.width):
                rgb = composite_cell_color(self.world, x, y)
                fill = rgb_to_hex(rgb)
                x0 = x * self.cell_px
                y0 = y * self.cell_px
                x1 = x0 + self.cell_px
                y1 = y0 + self.cell_px
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=fill, outline="")
                if self.world.state.ground[self.world.index(x, y)] == int(Ground.ROAD):
                    self.canvas.create_line(x0, y0 + self.cell_px // 2, x1, y0 + self.cell_px // 2, fill="#f2e3ba")

        for entity in self.world.state.entities.values():
            ex = entity.x * self.cell_px + self.cell_px / 2
            ey = entity.y * self.cell_px + self.cell_px / 2
            color = rgb_to_hex(PALETTE["entity"][entity.kind])
            radius = 2 if entity.kind in (EntityKind.MIGRANT, EntityKind.SKIMMER, EntityKind.SOUL) else 4
            self.canvas.create_oval(ex - radius, ey - radius, ex + radius, ey + radius, fill=color, outline="")

        self.sidebar.configure(state="normal")
        self.sidebar.delete("1.0", "end")
        summary = self.world.summary()
        challenge = summary["challenge"]
        metrics = summary["metrics"]
        text = textwrap.dedent(
            f"""
            INFERNAL DATA-FIRST WORLD
            =========================
            seed:        {summary['seed']}
            tick:        {summary['tick']}
            seconds:     {summary['seconds']}
            tool:        {self.tool.replace('_', ' ')}
            brush:       {self.brush}x{self.brush}
            paused:      {self.paused}

            METRICS
            -------
            dread:       {metrics['dread']}
            aura:        {metrics['aura']}
            souls/min:   {metrics['souls_per_minute']}
            symmetry:    {metrics['symmetry']:.2f}
            total tiles: {metrics['total_tiles']}
            rank:        {metrics['rank']}
            deliveries:  {metrics['deliveries']}

            CHALLENGE
            ---------
            {challenge['title']}
            {challenge['description']}
            progress:    {metrics['challenge_progress']*100:.1f}%

            COUNTS
            ------
            ghosts:      {metrics['counts'].get('ghost', 0)}
            demons:      {metrics['counts'].get('demon', 0)}
            creatures:   {metrics['counts'].get('creature', 0)}
            migrants:    {metrics['counts'].get('migrant', 0)}
            skimmers:    {metrics['counts'].get('skimmer', 0)}
            obsidian:    {metrics['counts'].get('obsidian', 0)}
            spires:      {metrics['counts'].get('spire', 0)}
            monoliths:   {metrics['counts'].get('monolith', 0)}
            wraiths:     {metrics['counts'].get('wraith_totem', 0)}
            quantum:     {metrics['counts'].get('quantum_core', 0)}
            fire altars: {metrics['counts'].get('fire_altar', 0)}
            lava:        {metrics['counts'].get('lava', 0)}
            water:       {metrics['counts'].get('water', 0)}
            acid:        {metrics['counts'].get('acid', 0)}
            plasma:      {metrics['counts'].get('plasma', 0)}
            roads:       {metrics['counts'].get('road', 0)}
            runes:       {metrics['counts'].get('ember_rune', 0)}
            void glass:  {metrics['counts'].get('void_glass', 0)}

            RECENT EVENTS
            -------------
            """
        ).strip("\n")
        self.sidebar.insert("end", text + "\n")
        for event in summary["recent_events"][:10]:
            self.sidebar.insert("end", event + "\n")
        if extra_message:
            self.sidebar.insert("end", "\n" + extra_message + "\n")
        self.sidebar.configure(state="disabled")

        self.status.configure(
            text=f"tool={self.tool.replace('_', ' ')}  brush={self.brush}  |  Q/A cycle tool  [ ] brush  N step  T template  S snapshot  {'PAUSED' if self.paused else 'RUNNING'}"
        )

    def update_loop(self) -> None:
        if not self.paused:
            self.world.step(self.speed_ticks)
            self.draw()
        self.root.after(50, self.update_loop)

    def run(self) -> None:
        self.draw()
        self.update_loop()
        self.root.mainloop()


# ------------------------------------------------------------
# Rendering helpers
# ------------------------------------------------------------


def composite_cell_color(world: InfernalWorld, x: int, y: int) -> Tuple[int, int, int]:
    i = world.index(x, y)
    terrain = Terrain(world.state.terrain[i])
    color = PALETTE["terrain"][terrain]
    height = world.state.height_field[i]
    color = shade(color, (height - 0.5) * 0.5)
    color = blend(color, (190, 90, 40), world.state.ash[i] * 0.12)

    ground = Ground(world.state.ground[i])
    if ground in PALETTE["ground"]:
        alpha = 0.35 if ground == Ground.ROAD else 0.55
        color = blend(color, PALETTE["ground"][ground], alpha)

    fluid = Fluid(world.state.fluid[i])
    if fluid in PALETTE["fluid"] and world.state.fluid_amount[i] > 0.02:
        alpha = clamp(0.35 + world.state.fluid_amount[i] * 0.35, 0.0, 0.8)
        color = blend(color, PALETTE["fluid"][fluid], alpha)

    structure = Structure(world.state.structure[i])
    if structure in PALETTE["structure"]:
        height_factor = 0.45 + world.state.structure_height[i] * 0.1
        color = blend(color, PALETTE["structure"][structure], clamp(height_factor, 0.0, 0.95))

    fire = world.state.fire[i]
    if fire > 0.02:
        color = blend(color, (255, 220, 90), clamp(fire * 0.45, 0.0, 0.75))
    return color


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    return "#%02x%02x%02x" % rgb


# ------------------------------------------------------------
# CLI entrypoint
# ------------------------------------------------------------


def run_self_test() -> int:
    import unittest

    loader = unittest.TestLoader()
    suite = loader.discover(os.path.dirname(__file__) or ".", pattern="test_infernal_data_first.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Data-first infernal world simulation")
    parser.add_argument("--seed", type=int, default=1337, help="world seed")
    parser.add_argument("--width", type=int, default=72, help="grid width")
    parser.add_argument("--height", type=int, default=72, help="grid height")
    parser.add_argument("--steps", type=int, default=0, help="run this many ticks before exiting")
    parser.add_argument("--template", type=str, default="", help="apply a named template")
    parser.add_argument("--summary", type=str, default="", help="write summary JSON to this path")
    parser.add_argument("--export-json", type=str, default="", help="write full JSON state to this path")
    parser.add_argument("--export-share", type=str, default="", help="write share code to this path")
    parser.add_argument("--snapshot", type=str, default="", help="write a PPM snapshot to this path")
    parser.add_argument("--headless", action="store_true", help="force headless mode")
    parser.add_argument("--tk", action="store_true", help="run the optional Tk renderer")
    parser.add_argument("--self-test", action="store_true", help="run the bundled test suite")
    args = parser.parse_args(argv)

    if args.self_test:
        return run_self_test()

    config = WorldConfig(width=args.width, height=args.height, seed=args.seed)
    world = InfernalWorld(config)

    if args.template:
        world.apply_template(args.template)

    if args.steps > 0:
        world.step(args.steps)

    if args.summary:
        with open(args.summary, "w", encoding="utf-8") as fh:
            json.dump(world.summary(), fh, indent=2)

    if args.export_json:
        with open(args.export_json, "w", encoding="utf-8") as fh:
            fh.write(world.to_json())

    if args.export_share:
        with open(args.export_share, "w", encoding="utf-8") as fh:
            fh.write(world.to_share_code())

    if args.snapshot:
        PPMSnapshotRenderer().render(world, args.snapshot)

    use_tk = args.tk and not args.headless
    if use_tk:
        try:
            TkRenderer(world).run()
            return 0
        except Exception as exc:
            print(f"Tk renderer unavailable, falling back to headless: {exc}", file=sys.stderr)

    if not args.tk:
        print(json.dumps(world.summary(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
