# Ultimate 2D Worldgen — maintenance notes

See [README.md](README.md) for usage, portability, and verification commands.
[CLAUDE.md](CLAUDE.md) records implementation constraints.

## Current scope

The project has 22 map families (the 22nd is the climate overworld), 40
simulation features plus opt-in lock-and-key gating, six style profiles,
procedural autotiles, a world compiler, design checks, and runtime scheduling
helpers. The synthesis toolkit covers overlapping and socket WFC with quotas
and backtracking, ConvChain, exact Voronoi/Delaunay geometry, corridor passes,
a bot playtest and a level blueprint export. Tests cover connectivity, seed
behavior, replay, pixel-identical rendering, and compatibility with the
original layout catalog.

The depth features generate gameplay snapshots that need evaluation in a game.
No downstream game or editor consumes the compiled worlds in this repository.
The level-description export has no importer. The encounter director samples
seeded candidates within categories; investigations generate clue-linked quests.
Neither advances a running world simulation.

Compiled manifest schema 2 replaces the relationship and investigation records.
Regenerate older compiled worlds with the current generators. The standalone
simulation retains its numeric tile IDs for snapshot compatibility.

## Known limitations

- Rumour propagation runs a shortest-path search from each place; large place
  counts can dominate compilation time.
- Differential growth and diffusion-limited aggregation use particle loops.
- Overworld roads use a 4-neighbour A* per link; very large maps with many
  settlements spend most of their time there. Islands get roads within each
  landmass only (no ferries).
- Overworld climate is a single prevailing wind and a latitude band; there are
  no seasons, ocean currents or monsoons.
- The committed compiled worlds were built on macOS; rebuilding them on another
  platform can change place and entity counts slightly (floating point), so
  regenerate them only alongside a real change.
- Gameplay balance, cross-platform output equality, and a live game-loop
  integration remain unverified.
- The standalone simulation has tests for seeded ticks and snapshot round trips;
  these do not establish deterministic continuation after loading a save.
- The repository includes an MIT license; see `LICENSE`.

Empty feature outputs can be valid: maps without water have no swim network,
and enclosed maps may have zero exposure. Preserve those semantics.

## Possible next work

1. Profile large worlds before changing rumour propagation.
2. Connect exports and scheduling helpers to a game and evaluate balance.
3. Add authored recipe overrides and a level-description importer.
4. Explore incremental feature updates and simulation ticks.

Keep verification local and offline. Preserve the original generator modules
and use their registration interfaces for extensions.
