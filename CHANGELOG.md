# Changelog

## 2026-11 — Overworlds, a synthesis toolkit and level-design passes

The first update since the initial release. Everything below is additive: the
original 100-layout catalog still rebuilds byte-for-byte, every default
output of the existing families is unchanged, and each new module is opt-in
or registers itself at import time.

### New map families

- **22 · Climate Overworlds** (`tilegen_overworld.py`). Continents,
  archipelagos, highlands and peninsulas raised from warped noise and eroded
  by stream power; lakes in the uneroded basins; temperature from a latitude
  band (temperate, tropical, polar, pole to pole) and a lapse rate; an
  orographic rain sweep from any of four winds, with rain shadows behind
  ranges; 14 land biomes from Whittaker bands with coast, marsh and mountain
  overrides; rivers on rain-weighted drainage; settlements sited tier by tier
  with central-place spacing and hinterland rings, each named and given a
  setting (port, river, hill, market); roads surveyed with a Tobler-cost
  4-neighbour A* that shares trunks, bridges rivers and leaves the map at its
  edge. `render_overworld` draws a finished map with hillshade, soft biome
  colours, rivers along their drainage, roads, markers, collision-free labels
  and a legend. Also a recipe macro and a Hybrid Morph source.
- **21 · Cellular Caves** (`ca_caves`). Neighbour-count automaton with the fill
  bisected to a target open ratio, small pockets removed, rooms joined in
  Prim order with 4-connected L carves.

### Synthesis and constraint solving

- `tilegen_wfc.py` — overlapping-model wave function collapse: N×N patterns
  from an exemplar, dihedral symmetry, a ground rule, tile quotas and restarts
  on named seed streams.
- `tilegen_socket_ext.py` — socket WFC extensions without touching the legacy
  runtime: D4 module variants, solver hooks (domains, pins, weights, groups,
  reroll), per-class count quotas (at most / at least / exactly) and a
  `backtrack=` option.
- `tilegen_csp.py` — the shared bitmask core: arc consistency, quotas and a
  contradiction hook (standard library only).
- `tilegen_backtrack.py` — chronological backtracking with banning, with an
  optional patience backjump.
- `tilegen_convchain.py` — ConvChain: window weights learned from a sample,
  Metropolis sweeps in colour classes; new maps in the style of any layout.

### Geometry and fields

- `tilegen_vector.py` — exact Delaunay triangulation, clipped Voronoi cells,
  Lloyd relaxation, Delaunay/Gabriel/relative-neighbourhood/minimum spanning
  graphs, contour tracing into nested rings, and RBF fields through scattered
  pins.
- `tilegen_fields.py` — variable-radius Poisson-disc sampling, random-order
  frontier regions (and `plate_shape='frontier'` for tectonic worlds),
  stream-power and thermal erosion, lake and channel maps, 16-bit heightmap
  loading and a hillshaded relief picture.

### Level design

- `tilegen_corridors.py` — dead-end retraction, corridor metrics (cycle rank),
  loop insertion and diagonal-kiss cleanup.
- `tilegen_routes.py` — equal-cost routes that hold their heading.
- `tilegen_keylock.py` — lock-and-key gating on chokepoints with an
  independent solvability check; optional world feature `nar.gates` (61).
- `tilegen_playtest.py` — a deterministic bot plays start to goal, fetches
  keys, scores the walk and lints maps it cannot finish.
- `tilegen_blueprint.py` — engine-neutral level blueprint JSON with a reader
  and structural checks.
- `EditOverlay` (`tilegen_world.py`) — sparse hand edits that re-apply onto a
  regenerated map and report the ones whose base moved.
- `connect_components(strategy='prim')` — joins scattered regions with far
  fewer carved cells.

### Rendering

- `tilegen_autotile.py` — procedural Wang-16 edge sheets and 47-case blob
  autotiles in four skins (dungeon, cave, shore, tech) and four palettes,
  palette-limited and seamless; writes the `wang-edge-16` sheet + JSON that
  `tilegen_style.load_tile_atlas` reads.
- `tilegen_style.py` — tile atlas rendering (`render_tiles`, raw or styled),
  and `palette_mix`, which lets a style honour a map's own palette (the
  satellite and parchment styles tint overworld biomes).
- The atlas gallery now shows all 22 families: 66 maps in six styles.
- `scripts/build_docs_images.py` rebuilds the overworld and autotile pictures.

### Verification

`test_goals.py --full` now checks 22 families and 40 features plus every
toolkit module (over 660 checks); `test_atlas.py --full` still rebuilds and
compares the golden 100 layouts and runs every family at 192×192.

## 2026-09 — Initial release

100 original layouts, 20 map families, 40 simulation features, six styles,
the world compiler, design lint, runtime helpers and the standalone
simulation.
