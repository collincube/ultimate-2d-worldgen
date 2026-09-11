"""Atlas build: the new grammars wired into the existing tilegen pipeline.

Registration is done at runtime rather than by editing tilegen_system, so
running `python tilegen_system.py` on its own still reproduces the original 100
layouts byte for byte. Importing this module adds the grammars to
ts.GENERATORS and the atlas specs to ts.all_specs, after which make_layout, the
JSON export, the socket WFC assembler and the navigation profiles treat them
exactly like the handwritten templates.

    python tilegen_atlas.py --size 96 --out tilegen_atlas_output
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import tilegen_system as ts
import tilegen_grammars as tg
import tilegen_grammars2  # noqa: F401  (registers grammars 05,06,08,11,14-16,18,19)
import tilegen_style as tsy

MACRO = 96


def _spec(i, name, template, **kw):
    d = {'id': i, 'name': name, 'template': template,
         'size': kw.pop('size', MACRO), 'theme': kw.pop('theme', 'atlas'),
         'background': 'wall'}
    d.update(kw)
    return d


def _build_specs():
    s, i = [], 101

    def add(name, template, **kw):
        nonlocal i
        s.append(_spec(i, name, template, **kw))
        i += 1

    # 01 Voronoi Kingdoms
    add('Crystalline provinces partitioned by contested borders', 'voronoi_kingdoms')
    add('A dense ward city of small irregular districts', 'voronoi_kingdoms',
        district_size=6, plaza_rate=0.30, extra_gates=0.4)
    add('Great estates divided by lake and rampart', 'voronoi_kingdoms',
        district_size=14, lake_rate=0.22, wall_thickness=2, gate_width=3)
    add('A spiral of wards seeded on a golden angle', 'voronoi_kingdoms',
        site_strategy='phyllotaxis', sites=70, relax=0)

    # 02 Reaction-Diffusion
    add('A coral labyrinth grown from chemical bloom', 'reaction_diffusion', preset='coral')
    add('Fingerprint corridors of a living archive', 'reaction_diffusion',
        preset='fingerprint', open_ratio=0.42)
    add('A mitotic warren of dividing chambers', 'reaction_diffusion',
        preset='mitosis', open_ratio=0.50)
    add('Worm-cast tunnels in soft chemical stone', 'reaction_diffusion',
        preset='worms', open_ratio=0.44)
    add('A chemical garden of pools and planting beds', 'reaction_diffusion',
        preset='coral', profile='garden')
    add('An archipelago precipitated from solution', 'reaction_diffusion',
        preset='spots', profile='islands')

    # 03 Differential Growth
    add('A folded corridor system grown under pressure', 'differential_growth')
    add('Convoluted galleries pressed into a tight shell', 'differential_growth',
        steps=280, repel_radius=6.5, corridor_width=3)
    add('An open hall veined by a single grown wall', 'differential_growth',
        invert=True, corridor_width=2)

    # 04 Truchet Knots
    add('A knot world of continuous looping circuits', 'truchet', cell=16)
    add('A fine ritual circuit of many closed loops', 'truchet',
        cell=12, target='many_loops', corridor_width=2, optimize=400)
    add('Broad ceremonial pipework on a coarse lattice', 'truchet',
        cell=22, corridor_width=4, chamber_radius=3)

    # 10 Watershed
    add('A river basin carved by its own drainage', 'watershed')
    add('A drowned coast of estuaries and tidal flats', 'watershed',
        sea_level=0.48, river_quantile=0.975)
    add('A high plateau split by canyon rivers', 'watershed',
        sea_level=0.26, mountain_level=0.70, scale=2.2)

    # 12 Flow Field
    add('Windswept streets following an invisible current', 'flow_field')
    add('A calligraphic sprawl of braided avenues', 'flow_field',
        scale=3.2, spacing=14, highway_rate=0.18)

    # 13 Contour Terraces
    add('Terraced architecture stepped along contour lines', 'contour_terraces')
    add('A stepped ziggurat of many shallow terraces', 'contour_terraces',
        levels=11, dome_weight=0.7)

    # 17 Fractal Archipelago
    add('A fractal archipelago linked by causeways', 'archipelago')
    add('A scattered reef of small wooded islets', 'archipelago',
        sea_level=0.50, scale=3.4, min_island=8)

    # 07 / 09 bonus grammars
    add('A sunflower city of spiral districts', 'phyllotaxis_city')
    add('Crystal caverns grown by mineral aggregation', 'dla_caverns')

    # 20 Hybrid Morph - the crossbreeding that multiplies families
    add('A radial temple fractured by reaction-diffusion gardens', 'hybrid_morph',
        macro='radial', structure='reaction_diffusion', op='mask', k=0.55)
    add('Voronoi wards eaten by coral growth', 'hybrid_morph',
        macro='voronoi', structure='reaction_diffusion', op='mul')
    add('A river delta crossing a spiral fortress', 'hybrid_morph',
        macro='spiral', structure='elevation', op='blend', k=0.5, hazard='water')
    add('Phyllotaxis districts warped by a flow field', 'hybrid_morph',
        macro='phyllotaxis', structure='fbm', op='warp', k=0.6)
    add('A grown labyrinth masked by a fractal island', 'hybrid_morph',
        macro='growth', structure='fractal', op='mask', k=0.45)
    add('Ridged badlands cut by checkerboard foundations', 'hybrid_morph',
        macro='ridged', structure='checker', op='sub', hazard='lava')
    add('Standing waves frozen into a voronoi crust', 'hybrid_morph',
        macro='waves', structure='voronoi', op='max', frequency=5.0)
    add('A drowned quasicrystal of spiral and noise', 'hybrid_morph',
        macro='spiral', structure='fractal', op='mul', turns=6.0, hazard='abyss')

    # 05 Space-Filling Curves
    add('A Hilbert archive where progress is near yet distant', 'space_filling',
        curve='hilbert')
    add('A Moore circuit closing on itself', 'space_filling', curve='moore')
    add('A Gosper flowsnake of hexagonal wards', 'space_filling', curve='gosper')
    add('A Sierpinski ascent of nested triangles', 'space_filling',
        curve='sierpinski')

    # 06 Quasicrystal Temples
    add('A five-fold temple that never repeats', 'quasicrystal', symmetry=5)
    add('A seven-fold reliquary of alien order', 'quasicrystal', symmetry=7,
        frequency=26.0)

    # 08 L-System Root Worlds
    add('A root world branching from a single trunk', 'lsystem_roots')
    add('A vascular canopy of arterial corridors', 'lsystem_roots',
        system='vascular')

    # 11 Tectonic Worlds
    add('Colliding plates raising a mountain chain', 'tectonic')
    add('A rifted continent opening along its faults', 'tectonic',
        plate_size=13, sea_level=0.34)

    # 14 Circuit-Board Megacities
    add('A motherboard megacity of traces and vias', 'circuit_city')
    add('A dense logic district on a fine pitch', 'circuit_city', pitch=4,
        components=14)

    # 15 Transit-Network Maps
    add('A metro network inflated into districts', 'transit_network')

    # 16 Non-Euclidean Portal Maps
    add('Chambers joined only by portals that should not meet',
        'portal_maps')

    # 18 Generative Mandalas
    add('An eightfold mandala with one profaned wedge', 'mandala', order=8)
    add('A twelvefold rosette of nested ambulatories', 'mandala', order=12,
        rings=7)

    # 19 Semantic Silhouette Maps
    add('A level in the shape of a coiled serpent', 'silhouette', shape='serpent')
    add('A level in the shape of a skull', 'silhouette', shape='skull')
    add('A level in the shape of a crown', 'silhouette', shape='crown')
    add('A level in the shape of a bound sigil', 'silhouette', shape='sigil')
    return s


ATLAS_SPECS = _build_specs()
_REGISTERED = False


def register():
    """Install grammars and specs into the existing registries (idempotent)."""
    global _REGISTERED
    if _REGISTERED:
        return
    ts.GENERATORS.update(tg.GRAMMARS)
    known = {s['id'] for s in ts.all_specs}
    for spec in ATLAS_SPECS:
        if spec['id'] not in known:
            ts.all_specs.append(spec)
    _REGISTERED = True


DEFAULT_STYLES = ['procedural_dark_tiles', 'architectural_blueprint',
                  'parchment_ink', 'obsidian_gold', 'neon_circuit',
                  'satellite_topographic']


def _save_compact(img, path, colors=128):
    """Contact sheets are large flat canvases; an adaptive palette halves them
    with no visible loss. Individual renders keep full colour."""
    img.convert('P', palette=__import__('PIL.Image', fromlist=['Image']).ADAPTIVE,
                colors=colors).save(path, optimize=True)


def build_atlas(root, styles=None, size=None, tile_px=6, specs=None, verbose=True):
    register()
    root = Path(root)
    styles = styles or DEFAULT_STYLES
    specs = specs or ATLAS_SPECS
    if size:
        specs = [dict(sp, size=size) for sp in specs]

    (root / 'layouts').mkdir(parents=True, exist_ok=True)
    for st in styles:
        (root / 'renders' / st).mkdir(parents=True, exist_ok=True)

    catalog, per_style = [], {st: [] for st in styles}
    for sp in specs:
        t0 = time.time()
        tm = ts.make_layout(sp)
        gen_s = time.time() - t0
        slug = ts.slugify(sp['name'])
        ts.export_map_json(tm, sp, root / 'layouts' / f"{sp['id']:03d}_{slug}.json")

        for st in styles:
            img = tsy.render_map(tm, sp, style=st, tile_px=tile_px,
                                 label=f"{sp['id']:03d} {sp['template']}")
            path = root / 'renders' / st / f"{sp['id']:03d}_{slug}.png"
            img.save(path)
            per_style[st].append((img, f"{sp['id']:03d} {sp['name']}"))

        base_counts, overlay_counts = ts.summarize_tiles(tm)
        catalog.append({
            'id': sp['id'], 'slug': slug, 'name': sp['name'],
            'template': sp['template'], 'theme': sp.get('theme', 'atlas'),
            'width': tm.w, 'height': tm.h,
            'fields': sorted(getattr(tm, 'fields', {})),
            'recipe': getattr(tm, 'recipe', None),
            'loops': getattr(tm, 'loops', None),
            'regions': (getattr(tm, 'regions', {}) or {}).get('count'),
            'base_counts': base_counts, 'overlay_counts': overlay_counts,
            'tile_json': f"layouts/{sp['id']:03d}_{slug}.json",
            'renders': {st: f"renders/{st}/{sp['id']:03d}_{slug}.png" for st in styles},
        })
        if verbose:
            print(f"  {sp['id']:03d} {sp['template']:20s} {tm.w}x{tm.h} "
                  f"gen {gen_s:5.2f}s  {sp['name'][:46]}")

    for st in styles:
        imgs, labs = zip(*per_style[st])
        _save_compact(tsy.contact_sheet(list(imgs), list(labs), cols=6),
                      root / f'contact_{st}.png')

    # one map through every style, to show geometry and style are independent
    demo = specs[len(specs) // 2]
    tm = ts.make_layout(demo)
    imgs = [tsy.render_map(tm, demo, style=st, tile_px=8, label=st) for st in styles]
    _save_compact(tsy.contact_sheet(imgs, styles, cols=3, cell_w=320, cell_h=340),
                  root / 'style_matrix.png')

    (root / 'catalog.json').write_text(json.dumps(catalog, indent=2), encoding='utf-8')
    (root / 'source_specs.json').write_text(json.dumps(specs, indent=2), encoding='utf-8')
    _write_viewer(catalog, styles, root / 'viewer.html')
    return catalog



def build_worlds(root, recipes=None, size=None, tile_px=5, verbose=True,
                 text_only=False):
    """Compile example gameplay snapshots with 40 depth features and export them."""
    import tilegen_compiler as tc
    root = Path(root) / 'worlds'
    root.mkdir(parents=True, exist_ok=True)
    chosen = recipes or tc.EXAMPLE_RECIPES
    index = []
    for key, recipe in chosen.items():
        if size:
            recipe = tc.WorldRecipe(**{**vars(recipe), 'size': size})
        t0 = time.time()
        world = tc.compile_recipe(recipe)
        out = root / key
        tc.export_world(world, out, style=recipe.style, tile_px=tile_px,
                        text_only=text_only)
        errors = [f for f in world.findings if f['severity'] == 'error']
        index.append({
            'key': key, 'name': recipe.name, 'macro': recipe.macro,
            'style': recipe.style, 'size': recipe.size,
            'places': len(world.places), 'entities': len(world.entities),
            'factions': len(world.factions), 'fields': len(world.fields),
            'topologies': len(world.topologies), 'events': len(world.events),
            'findings': len(world.findings), 'errors': len(errors),
            'digest': recipe.digest(),
        })
        if verbose:
            print(f"  {key:22s} {time.time() - t0:5.1f}s  {recipe.size}x{recipe.size} "
                  f"places={len(world.places):4d} entities={len(world.entities):4d} "
                  f"fields={len(world.fields):2d} findings={len(world.findings)}")
    (root / 'index.json').write_text(json.dumps(index, indent=2), encoding='utf-8')
    return index


def _write_viewer(catalog, styles, path):
    cards = '\n'.join(
        f'<figure class="card" data-t="{r["template"]}">'
        f'<img loading="lazy" data-r=\'{json.dumps(r["renders"])}\' src="{r["renders"][styles[0]]}" alt="{r["name"]}">'
        f'<figcaption><b>{r["id"]:03d}</b> {r["name"]}'
        f'<span class="t">{r["template"]} &middot; {r["width"]}&times;{r["height"]}'
        f'{" &middot; fields: " + ", ".join(r["fields"]) if r["fields"] else ""}</span>'
        f'</figcaption></figure>'
        for r in catalog)
    templates = sorted({r['template'] for r in catalog})
    opts = '\n'.join(f'<option value="{s}">{s}</option>' for s in styles)
    tops = '\n'.join(f'<option value="{t}">{t}</option>' for t in templates)
    path.write_text(f"""<!doctype html><meta charset="utf-8">
<title>Roguelike Depth Layers — Ultimate 2D Worldgen</title>
<style>
 body{{background:#0d0f13;color:#d7dae1;font:14px/1.5 system-ui,sans-serif;margin:0;padding:24px}}
 h1{{font-size:20px;margin:0 0 4px}} p.sub{{color:#8b93a3;margin:0 0 18px}}
 .bar{{display:flex;gap:12px;align-items:center;margin-bottom:18px;flex-wrap:wrap}}
 select{{background:#171a21;color:#d7dae1;border:1px solid #2a2f3a;border-radius:6px;padding:6px 10px}}
 .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:16px}}
 .card{{margin:0;background:#141720;border:1px solid #222734;border-radius:10px;overflow:hidden}}
 .card img{{width:100%;display:block;background:#0a0b0f}}
 figcaption{{padding:8px 10px;font-size:12px}} .t{{display:block;color:#7f879a;font-size:11px;margin-top:3px}}
</style>
<h1>Roguelike Depth Layers</h1>
<p class="sub">{len(catalog)} maps &middot; {len(styles)} styles &middot; {len(templates)} grammars
&mdash; the same maps rendered in different visual styles.</p>
<div class="bar">
 <label>Style <select id="s">{opts}</select></label>
 <label>Grammar <select id="g"><option value="">all</option>{tops}</select></label>
</div>
<div class="grid" id="grid">{cards}</div>
<script>
const s=document.getElementById('s'), g=document.getElementById('g');
s.onchange=()=>document.querySelectorAll('.card img').forEach(i=>{{
  i.src=JSON.parse(i.dataset.r)[s.value];}});
g.onchange=()=>document.querySelectorAll('.card').forEach(c=>{{
  c.style.display=(!g.value||c.dataset.t===g.value)?'':'none';}});
</script>
""", encoding='utf-8')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default='tilegen_atlas_output')
    ap.add_argument('--size', type=int, default=MACRO)
    ap.add_argument('--tile-px', type=int, default=6)
    ap.add_argument('--styles', nargs='*', default=DEFAULT_STYLES)
    ap.add_argument('--only', nargs='*', help='limit to these templates')
    ap.add_argument('--worlds', action='store_true',
                    help='also compile example gameplay snapshots')
    ap.add_argument('--worlds-only', action='store_true',
                    help='skip the map atlas and compile recipes only')
    ap.add_argument('--world-size', type=int, default=None)
    ap.add_argument('--text-only', action='store_true',
                    help='with --worlds-only, refresh JSON and Markdown while preserving images and arrays')
    args = ap.parse_args(argv)
    if args.text_only and not args.worlds_only:
        ap.error('--text-only requires --worlds-only')

    specs = ATLAS_SPECS
    if args.only:
        specs = [s for s in specs if s['template'] in args.only]

    root = Path(__file__).resolve().parent / args.out
    t0 = time.time()
    if not args.worlds_only:
        print(f"Building {len(specs)} atlas maps at {args.size}x{args.size} "
              f"in {len(args.styles)} styles -> {root}")
        cat = build_atlas(root, styles=args.styles, size=args.size,
                          tile_px=args.tile_px, specs=specs)
        print(f"\nBuilt {len(cat)} maps x {len(args.styles)} styles "
              f"= {len(cat) * len(args.styles)} renders in {time.time() - t0:.1f}s")
        print(f"Viewer: {root / 'viewer.html'}")
    if args.worlds or args.worlds_only:
        print(f"\nCompiling example gameplay snapshots -> {root / 'worlds'}")
        idx = build_worlds(root, size=args.world_size, text_only=args.text_only)
        print(f"Compiled {len(idx)} worlds in {time.time() - t0:.1f}s total")


if __name__ == '__main__':
    main()
