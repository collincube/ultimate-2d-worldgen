# Procedural Tile Generation System

This package contains a data-driven layout generator and batch render export for all 100 requested layouts.

## What is included

- `source_specs.json` — the registry of layout definitions
- `catalog.json` — generated catalog with file references and tile summaries
- `layouts/` — one raw tile matrix JSON per layout
- `renders/` — one rendered PNG per layout
- `contact_sheet.png` — overview of all 100 layouts
- `procedural_tileset_legend.png` — procedurally rendered tile/overlay legend
- `viewer.html` — local browser viewer

## Data-first structure

Each layout is defined by:
- `id`
- `name`
- `template`
- `size`
- theme/background params
- template-specific generator parameters

The generator registry turns that data into:
- base tile layer
- overlay layer
- render PNG
- exported JSON matrix

## Regenerate

Run:

```bash
python tilegen_system.py
```

Outputs will be written next to the script in `tilegen_output/`.

All textures, glyphs, and map renders are created procedurally in code. No external art assets are used.
