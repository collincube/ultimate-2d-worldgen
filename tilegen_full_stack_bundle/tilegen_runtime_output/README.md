# Low-End Runtime Extension

This package extends the base 100-layout tile generator with two new systems:

- agent navigation over every generated layout
- a low-end socket-WFC that composes the full 100-layout library into larger worlds

## Design goals

- data first: all runtime behavior is derived from exported tile data
- low-end friendly: small standardized modules, 4 edge sockets, bitset propagation, 4-neighbor movement
- no external art assets: all renders are procedural

## Files

- `module_library.json` — standardized 16×16 modules derived from all 100 source layouts
- `navigation_catalog.json` — per-layout agent results and render references
- `agent_profiles.json` — movement rules and costs
- `runtime_summary.json` — aggregate stats
- `module_socket_sheet.png` — visual summary of module sockets
- `agents/` — path overlays for each layout and agent type
- `worlds/` — WFC outputs and their navigation overlays
- `runtime_viewer.html` — local browser viewer

## Runtime model

Each of the 100 layouts becomes one WFC state.
Each state is downsampled to a 16×16 logic module.
Each side stores a 4-slot socket mask based on the nearest reachable edge anchors.
Adjacency is allowed when neighboring side masks overlap.

The `all100_unique` preset fills a 10×10 world using all 100 source layouts exactly once.

## Rebuild

```bash
python tilegen_lowend_runtime.py
```
