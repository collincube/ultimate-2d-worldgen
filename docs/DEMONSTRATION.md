# Ultimate 2D Worldgen — demonstration

A visual tour of the maps, Turtle Explorer routes, assembled dungeons, and
Roguelike Depth Layers. Every image below is included locally; no image host,
account, or running server is needed. Open any image for a closer look.

[Back to the README](../README.md)

## Start with the shapes

Caverns, corridors, city blocks, and branching paths come from different map
families. The same geometry can then be drawn in different styles.

![Four map families shown in four styles](showcase.png)

## One map, six styles

Follow the same bends and chambers across these panels. The renderer changes
the palette, edges, shading, and texture while keeping the layout fixed.

![One map rendered in all six styles](../tilegen_full_stack_bundle/tilegen_atlas_output/style_matrix.png)

## Meet the Turtle Explorer

The turtle follows walls using local decisions and can cross water. Its path
shows how a simple explorer behaves inside a generated map, including detours
and repeated visits. These are navigation overlays, not character animations.

![Turtle route through a circular hub](../tilegen_full_stack_bundle/tilegen_runtime_output/agents/turtle/001_a_circular_hub_with_radiating_corridors_like_spokes.png)

![Turtle route through a submerged ruin](../tilegen_full_stack_bundle/tilegen_runtime_output/agents/turtle/011_a_submerged_ruin_with_interconnected_waterlogged_halls.png)

![Turtle route through a compact winding maze](../tilegen_full_stack_bundle/tilegen_runtime_output/agents/turtle/096_a_9_9_maze_with_a_single_winding_path_from_one_corner_to_the_opposite.png)

The turtle reaches its destination on all 100 included maps, including maps
connected by teleporters. Local wall following can still take long detours;
these results do not guarantee success on every possible generated map.

![Turtle Explorer routes across the original catalog](../tilegen_full_stack_bundle/tilegen_runtime_output/agents/contact_sheet_turtle.png)

## Three ways to explore the same world

The Seeker NPC uses weighted shortest paths over walkable ground. The Turtle
Explorer follows walls and can enter water. The Survey Drone can also cross
abyss and lava tiles, with movement costs that influence its route.

Compare their routes on the same small assembled world:

### Seeker NPC

![Seeker NPC route on the small assembled world](../tilegen_full_stack_bundle/tilegen_runtime_output/worlds/tiny_fast_npc.png)

### Turtle Explorer

![Turtle Explorer route on the small assembled world](../tilegen_full_stack_bundle/tilegen_runtime_output/worlds/tiny_fast_turtle.png)

### Survey Drone

![Survey Drone route on the small assembled world](../tilegen_full_stack_bundle/tilegen_runtime_output/worlds/tiny_fast_drone.png)

## Build a larger dungeon from smaller pieces

Each source layout becomes a 16×16 module. Edge sockets describe where modules
can connect, and the assembler chooses compatible neighbors. The visible tile
borders make the construction easy to inspect.

![Module catalog with connection sockets](../tilegen_full_stack_bundle/tilegen_runtime_output/module_socket_sheet.png)

![Overview of the three assembled world presets](../tilegen_full_stack_bundle/tilegen_runtime_output/worlds/contact_sheet_worlds.png)

### A turtle in a larger world

![Turtle Explorer route through the balanced world](../tilegen_full_stack_bundle/tilegen_runtime_output/worlds/balanced_turtle.png)

### All 100 modules in one world

This preset uses each source module once in a 10×10 arrangement. Look for the
individual chambers and corridors inside the larger map.

![World assembled from all 100 source modules](../tilegen_full_stack_bundle/tilegen_runtime_output/worlds/all100_unique_base.png)

![Turtle route through the all-100-module world](../tilegen_full_stack_bundle/tilegen_runtime_output/worlds/all100_unique_turtle.png)

## Roguelike Depth Layers

A compiled world adds NPCs, faction territories, shops, guard detection,
resources, and quest hooks to the map. These are generated gameplay snapshots;
a game would still need to consume and advance them.

![Four example compiled worlds](worlds.png)

Light, sound, privacy, faction control, danger, wear, contamination, and memory
can describe different aspects of the same place. The layer views help explain
why two rooms with similar geometry might play differently.

![Nine views of the same world geometry](layers.png)

### Visit each example world

**The Obsidian Salt Keep**

![The Obsidian Salt Keep](../tilegen_full_stack_bundle/tilegen_atlas_output/worlds/salt-keep/render.png)

**Trench Station**

![Trench Station](../tilegen_full_stack_bundle/tilegen_atlas_output/worlds/trench-station/render.png)

**Obsidian Metropolis**

![Obsidian Metropolis](../tilegen_full_stack_bundle/tilegen_atlas_output/worlds/obsidian-metropolis/render.png)

**The Coral Reliquary**

![The Coral Reliquary](../tilegen_full_stack_bundle/tilegen_atlas_output/worlds/coral-reliquary/render.png)

## Browse the full catalogs

The contact sheets collect the individual renders into one image per style or
agent. The linked folders contain the individual images and generated data.

### Original layouts and tile palette

![The original 100 layouts](../tilegen_full_stack_bundle/tilegen_output/contact_sheet.png)

![Procedural tiles and overlays](../tilegen_full_stack_bundle/tilegen_output/procedural_tileset_legend.png)

### Six style catalogs

**Dark tiles**

![Dark tiles map catalog](../tilegen_full_stack_bundle/tilegen_atlas_output/contact_procedural_dark_tiles.png)

**Blueprint**

![Blueprint map catalog](../tilegen_full_stack_bundle/tilegen_atlas_output/contact_architectural_blueprint.png)

**Parchment**

![Parchment map catalog](../tilegen_full_stack_bundle/tilegen_atlas_output/contact_parchment_ink.png)

**Obsidian and gold**

![Obsidian and gold map catalog](../tilegen_full_stack_bundle/tilegen_atlas_output/contact_obsidian_gold.png)

**Neon circuit**

![Neon circuit map catalog](../tilegen_full_stack_bundle/tilegen_atlas_output/contact_neon_circuit.png)

**Topographic**

![Topographic map catalog](../tilegen_full_stack_bundle/tilegen_atlas_output/contact_satellite_topographic.png)

### Other agent routes

![Seeker NPC routes across the original catalog](../tilegen_full_stack_bundle/tilegen_runtime_output/agents/contact_sheet_npc.png)

![Survey Drone routes across the original catalog](../tilegen_full_stack_bundle/tilegen_runtime_output/agents/contact_sheet_drone.png)

[All turtle renders](../tilegen_full_stack_bundle/tilegen_runtime_output/agents/turtle/) ·
[All NPC renders](../tilegen_full_stack_bundle/tilegen_runtime_output/agents/npc/) ·
[All drone renders](../tilegen_full_stack_bundle/tilegen_runtime_output/agents/drone/) ·
[Assembled world renders](../tilegen_full_stack_bundle/tilegen_runtime_output/worlds/) ·
[Compiled worlds and layer views](../tilegen_full_stack_bundle/tilegen_atlas_output/worlds/)

## Try it locally

Use the existing Python environment described in the [README](../README.md).
From the repository root:

```sh
python tilegen_full_stack_bundle/tilegen_atlas.py --size 96
python tilegen_full_stack_bundle/tilegen_atlas.py --worlds-only
python scripts/build_runtime.py
```

The builds create local browser viewers alongside their output. The images in
this guide can be browsed without rebuilding anything.
