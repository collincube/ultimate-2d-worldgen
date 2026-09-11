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
tilegen_fields.py       numpy field engine + pattern library   (no tilegen deps)
tilegen_grammars.py     map species 1-4, 7, 9, 10, 12, 13, 17, 20
tilegen_grammars2.py    map species 5, 6, 8, 11, 14-16, 18, 19
tilegen_style.py        StyleProfile registry, neighbourhood renderer
tilegen_world.py        Place / Topology / World / feature registry + DAG
tilegen_layers.py       depth features 21-39
tilegen_society.py      depth features 41-60
tilegen_compiler.py     feature 40, WorldRecipe, design lint, projections
tilegen_runtime_budget.py   throttling, LOD, baked nav, chunk streaming
tilegen_atlas.py        build script + viewer
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
python test_goals.py --full     # 20 map families + 40 features and behaviour
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
The `docs/` illustrations have no dedicated rebuild script in this snapshot.
