# Ultimate 2D Worldgen

Procedural 2D maps, gameplay depth layers, and tile rendering in Python.
Start with 100 predefined layouts or generate maps from 20 mathematical
families. Roguelike Depth Layers add six visual styles and a compiler for places,
navigation, environmental fields, factions, and event records.

![Four map families in four styles](docs/showcase.png)

[See the demonstration](docs/DEMONSTRATION.md) for Turtle Explorer routes,
assembled dungeons, all six styles, and Roguelike Depth Layers.

## Run locally

Use Python 3.11 or newer with the existing dependencies in
[`requirements.txt`](requirements.txt): NumPy and Pillow. Both are required by
the atlas, including headless generation, because the base engine imports
Pillow. No service, API key, external art pack, or network connection is
needed at runtime.

From the repository root, check your environment and run:

```sh
python -c "import numpy, PIL; print(numpy.__version__, PIL.__version__)"
python tilegen_full_stack_bundle/tilegen_atlas.py --size 96
python tilegen_full_stack_bundle/tilegen_atlas.py --worlds-only
```

If your platform uses `python3` or `py -3`, substitute that command for `python`.
An offline machine needs Python and these two packages provisioned beforehand;
the repository does not bundle their installers.

Open `tilegen_full_stack_bundle/tilegen_atlas_output/viewer.html` in a browser
after building. It uses local images and inline JavaScript, with no server or
CDN. Per-style images and compiled field stacks are generated locally and
excluded from the source snapshot; the overview images are included.

Outputs are written beside the generator in `tilegen_atlas_output/`.
Use `--out` with an absolute path to choose another directory. A relative
`--out` path is relative to `tilegen_full_stack_bundle`, not the shell's working
directory. Builds replace files in the chosen output directory.

## Artwork and generated files

Map textures, symbols, and simulation visuals are drawn procedurally. No
external artwork download is required. Labels use Pillow's bundled font or
system fonts in the browser and optional Tk viewer.

The `tilegen_output`, `tilegen_runtime_output`, and `tilegen_atlas_output`
directories contain generated artifacts. Change generator code or specs and
rebuild instead of editing those files by hand. The images in `docs/` are
included illustrations; this snapshot has no dedicated rebuild script for them.

To refresh compiled JSON and Markdown while keeping existing images and field
arrays, use `python tilegen_full_stack_bundle/tilegen_atlas.py --worlds-only --text-only`.
Use a full rebuild when geometry or field calculations change.

To rebuild the runtime outputs with source references relative to the bundle
root, use `python scripts/build_runtime.py`. The legacy runtime entry point
writes absolute source paths; use this wrapper for portable exports.

## Compile a world

Run this from `tilegen_full_stack_bundle` so the modules are importable:

```python
from tilegen_compiler import WorldRecipe, compile_recipe, export_world

world = compile_recipe(WorldRecipe(
    name="The Obsidian Salt Keep",
    macro="voronoi_kingdoms", structure="reaction_diffusion",
    circulation="river_delta", ornament="truchet",
    symmetry=0.72, organicity=0.84, verticality=0.30,
    erosion=0.42, secret_density=0.15,
    style="obsidian_gold", seed=93741, size=128,
))
export_world(world, "out/salt-keep")
```

The compiler builds geometry, places, and 40 simulation features in dependency
order to produce a gameplay snapshot, then runs design checks. Exports include JSON records, NumPy field
stacks, rendered maps, and level specifications. Named seed
streams support reproducible builds within the same software environment.

## Map families

| # | Species | Mathematics |
|---|---|---|
| 1 | Voronoi Kingdoms | Poisson-disc sites, Lloyd relaxation, region adjacency, gates on shared borders |
| 2 | Reaction-Diffusion | Gray–Scott, 9 presets (coral, fingerprint, mitosis, worms…) |
| 3 | Differential Growth | Rest-length springs + grid-accelerated repulsion; a loop that folds |
| 4 | Truchet Knots | Quarter-arc lattice, loop tracing, hill-climbed rotations |
| 5 | Space-Filling Curves | Hilbert, Moore, Peano, Gosper, Sierpiński |
| 6 | Quasicrystal Temples | N plane waves at π/N — rotational symmetry without period |
| 7 | Phyllotaxis Cities | Golden-angle districts |
| 8 | L-System Root Worlds | Branch depth encodes hierarchy: trunk→highway, twig→secret |
| 9 | DLA Crystal Caverns | Diffusion-limited aggregation dendrites |
| 10 | Watershed / River Delta | Priority-flood filling, D8 accumulation, traced channels |
| 11 | Tectonic Worlds | Plate motion vectors → convergent, divergent, transform boundaries |
| 12 | Flow-Field Maps | Curl-noise streamlines |
| 13 | Contour Architecture | Quantised elevation → terraces, retaining walls, stairs |
| 14 | Circuit Megacities | Manhattan traces, pads, chips, vias |
| 15 | Transit Networks | Graph first, then embedded and inflated into districts |
| 16 | Non-Euclidean Portals | A topology that exceeds its visible footprint |
| 17 | Fractal Archipelagos | Domain-warped coastlines, causeways |
| 18 | Generative Mandalas | Symmetry groups, star polygons, one deliberately broken wedge |
| 19 | Semantic Silhouettes | SDF shapes — serpent, crown, skull, sigil, machine, hand |
| 20 | Hybrid Morph | Any two pattern sources through a typed operator |

The hybrid family combines field sources through operators such as blending
and masking. Connectivity and seed behavior are checked by the test suites;
playability and game balance still need testing in a game.

## Roguelike Depth Layers

Roguelike Depth Layers cover 20 map families and 40 game simulation features.
They generate gameplay snapshots with NPCs, guard-response values, shop prices,
faction territories, and dungeon quest hooks. Movement, sound, alliances,
and rivalries use separate connection graphs. Cooperation describes how well
NPCs might work together. Investigations provide seeded clue and NPC assignments;
the encounter director makes seeded selections within encounter categories.

The town and adventure features are:

| # | Game system |
|---|---|
| 41 | Faction Territories |
| 42 | Faction Influence and Rivalries |
| 43 | Town Rules and Guard Responses |
| 44 | Shops, Trading, and Prices |
| 45 | NPC Work and Wages |
| 46 | Buildings, Ownership, and Succession |
| 47 | Smuggler NPCs and Hidden Shops |
| 48 | NPC Reputation and Disguises |
| 49 | NPC Alliances, Rivalries, and Cooperation |
| 50 | Fantasy Customs and Faction Traditions |
| 51 | Guard Vision and Stealth Detection |
| 52 | Faction Alliances and Negotiations |
| 53 | Combat Encounters and Alert Levels |
| 54 | Clues and Investigation Quest Hooks |
| 55 | Settlement Health and Environmental Hazards |
| 56 | Settlement Supplies and Resources |
| 57 | Building Durability and Damage |
| 58 | Quest Objectives and Dungeon Missions |
| 59 | World Event History |
| 60 | Encounter and Quest Director |

Six styles share the same geometry: `procedural_dark_tiles`,
`architectural_blueprint`, `parchment_ink`, `obsidian_gold`, `neon_circuit`,
and `satellite_topographic`.

The runtime helpers provide scheduled pathfinding, viewport culling,
precomputed navigation, background chunk loading, and resolution scaling.
The tests exercise these in a headless simulation; they are not integrated
into a game loop.

## Repository layout

| Path | Purpose |
|---|---|
| `tilegen_full_stack_bundle/` | Main engine, atlas, compiler, and tests |
| `tilegen_project_bundle/` | Original standalone 100-layout bundle |
| `infernal_data_first.py` | Separate standard-library world simulation; optional Tk viewer |
| `docs/` | Included overview images |
| `HANDOFF.md` | Maintenance status and known limitations |

The original `tilegen_system.py` modules remain unchanged; extensions register
additional map generators. The legacy runtime includes a small correction that
lets the turtle walk off arrival portals. The full regression suite compares
layout JSON bytes and decoded PNG pixels with the included baseline, and checks
turtle traversal across all 100 original maps.

## Verify offline

Run from the repository root:

```sh
python scripts/check_public.py
python tilegen_full_stack_bundle/test_atlas.py --full
python tilegen_full_stack_bundle/test_goals.py --full
python infernal_data_first.py --self-test
python infernal_data_first.py --headless --width 24 --height 24 --steps 2
```

The public-file check uses only the standard library and never reads Git
metadata. The regression suite puts scratch outputs under `.local/` and removes
them afterward. Verification runs locally; no hosted workflows are required.

## Portability and limits

The headless tools use Python, NumPy, Pillow, and filesystem paths rather than
OS-specific commands. They are intended for macOS, Linux, and Windows; this
cleanup was verified on macOS only. Tk is optional and may require a separately
provided Python/Tk installation. The standalone simulation's headless mode
uses only the Python standard library.

Exact PNG bytes and floating-point results can vary with dependency versions
or platforms. Keep the same environment when comparing golden outputs. Larger
worlds can be expensive, especially per-place rumour propagation and particle
based generators. The society features generate a snapshot, not a balanced,
continuously running game.

Licensed under the [MIT License](LICENSE).
