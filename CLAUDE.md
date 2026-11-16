# Ultimate 2D Worldgen — working notes

## Preserve the original map generators

Both copies of `tilegen_system.py` are **byte-for-byte unchanged from the
original upload** and must stay that way. Extensions register at import time
(`ts.GENERATORS.update(...)`, `ts.all_specs.append(...)`).

Why it matters: `python tilegen_system.py` alone must still rebuild the original
100 layout JSON files byte-identically and PNGs pixel-identically.
`test_atlas.py --full` asserts this. Add map-generation behaviour by registration.

`tilegen_lowend_runtime.py` has one intentional correction: the turtle can walk
off an arrival portal before teleporting again. Isolated portals still allow
onward jumps. Preserve the movement strategy and keep runtime fixes localized.

## Layout

```
tilegen_system.py       original 35 templates, 100 layouts, JSON/PNG export; byte-frozen
tilegen_fields.py       numpy field engine + pattern library, random-frontier regions   (no tilegen deps)
tilegen_vector.py       exact Delaunay, clipped Voronoi cells, Lloyd, proximity graphs, contour rings, RBF fields (fields only)
tilegen_grammars.py     map species 1-4, 7, 9, 10, 12, 13, 17, 20
tilegen_grammars2.py    map species 5, 6, 8, 11, 14-16, 18, 19, 21 (cellular caves)
tilegen_overworld.py    map species 22: climate overworld (landmass, erosion, wind-swept rain, Whittaker biomes, rivers, lakes, settlements, A* roads, map renderer)
tilegen_style.py        StyleProfile registry, neighbourhood renderer, Wang-16 atlas reader; map palettes via palette_mix
tilegen_autotile.py     procedural Wang-16 sheets and 47-case blob autotiles (4 skins x 4 palettes), sheet/atlas writer
tilegen_world.py        Place / Topology / World / feature registry + DAG
tilegen_layers.py       depth features 21-39
tilegen_society.py      depth features 41-60
tilegen_compiler.py     feature 40, WorldRecipe, design lint, projections
tilegen_runtime_budget.py   throttling, LOD, baked nav, chunk streaming
tilegen_lowend_runtime.py   legacy runtime: 16x16 socket-WFC worlds, npc/turtle/drone agents
tilegen_socket_ext.py   opt-in socket-WFC extensions: D4 module variants, solver hooks, count quotas, backtrack= (legacy runtime untouched)
tilegen_csp.py          bitmask constraint core shared by the WFC solvers: arc consistency, quotas, contradiction hook (stdlib only)
tilegen_backtrack.py    chronological backtracking with banning (+ optional patience backjump) over the csp bitmask domains; the socket solver's backtrack= hook
tilegen_wfc.py          overlapping-model WFC: learn N x N patterns from an exemplar, entropy collapse, restarts on named streams
tilegen_convchain.py    opt-in ConvChain: window weights learned with tilegen_wfc, Metropolis sweeps in colour classes (new maps in the style of a layout)
tilegen_routes.py       opt-in equal-cost routes that hold their heading (inertia); same cost as the legacy runtime routes
tilegen_corridors.py    opt-in dead-end retraction, corridor metrics (cycle rank), cycle insertion, diagonal-kiss cleanup for mask/TileMap corridors
tilegen_keylock.py      opt-in lock-and-key gating on chokepoints with a solvability check; world feature 'nar.gates' via register()
tilegen_atlas.py        build script + viewer
infernal_data_first.py  (repo root, not the bundle) standalone stdlib world sim, no tilegen imports
```

Import direction is strictly downward; `tilegen_fields` depends on nothing.

## Implementation constraints

- **Grammars keep the `(spec, rng) -> TileMap` signature.** That is the only
  reason the untouched runtime still consumes them.
- **4-connectivity everywhere.** The component labeller and the runtime's
  navigation are both N/E/S/W. Never carve a diagonal corridor to join regions —
  use `carve_l`. Diagonal carving does not ensure 4-connectivity.
- **Named seed streams.** Use `tf.seed_stream(seed, 'name')`, never a global
  RNG. Determinism and replay depend on it, and it means changing one grammar
  does not reshuffle every other.
- **Thresholds by target ratio, not fixed cutoff.** Composite fields pile up
  near zero; `threshold_for_ratio` exists for this.
- **Tags relative to the world, not absolute.** An absolute cutoff on a skewed
  field labels every place 'loud'.
- **Features declare `consumes`/`produces`.** Order comes from the DAG. If a
  feature must observe the finished world, mark it `last=True` — declaring only
  consumes lets the sort put it first.
- **Features must degrade.** A map with no lava/water/plaza still needs sound.
  Fall back to something meaningful rather than emitting a zero field.

## Testing

```bash
python test_goals.py --full     # 22 map families + 40 features and behaviour
python test_atlas.py --full     # regression including golden rebuild
```

Run both from `tilegen_full_stack_bundle` before finalizing changes. `--full` checks the full baseline rebuild.

## NPC relationship model

Feature 49 (`tilegen_society.npc_relationships`) generates alliances and
rivalries with signed cooperation. Pairs must reference distinct existing
NPCs, appear once, and agree with their alliance or rivalry graph. Graph
weights are positive strengths; cooperation carries the sign.

## Generated artifacts

Map artwork is procedural; labels use bundled or system fonts. Edit generators
and specs, then regenerate output directories. For text changes, use
`tilegen_atlas.py --worlds-only --text-only` to preserve images and field arrays. Use `scripts/build_runtime.py`
from the repository root for runtime exports with portable source paths.
`scripts/build_docs_images.py` rebuilds `docs/overworld.webp`, `docs/overworlds.webp` and
`docs/autotiles.webp`; `showcase.png`, `worlds.png` and `layers.png` have no rebuild script.

