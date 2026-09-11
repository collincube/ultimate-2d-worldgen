"""Compile recipes into gameplay snapshots and export maps, records, and notes.

Features run in dependency order and publish to World registries. The compiler
validates their outputs and writes JSON, field arrays, images, and Markdown.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass, field as dc_field, asdict
from pathlib import Path

import numpy as np

import tilegen_system as ts
import tilegen_fields as tf
import tilegen_grammars as tg
import tilegen_grammars2  # noqa: F401  (registers the remaining grammars)
import tilegen_world as tw
import tilegen_layers  # noqa: F401  (registers features 21-39)
import tilegen_society  # noqa: F401  (registers features 41-60)
import tilegen_style as tsy
from tilegen_world import feature

SCHEMA_VERSION = '2.0.0'


# ------------------------------------------------------------------- recipe

@dataclass
class WorldRecipe:
    """Parameters for geometry, simulation, and presentation.

    Knobs range from 0 to 1; `to_spec` maps them to generator parameters.
    """
    macro: str = 'voronoi_kingdoms'
    structure: str = None
    circulation: str = None
    ornament: str = None
    symmetry: float = 0.5
    organicity: float = 0.5
    verticality: float = 0.3
    erosion: float = 0.4
    secret_density: float = 0.15
    style: str = 'procedural_dark_tiles'
    seed: int = 93741
    size: int = 96
    name: str = 'unnamed world'
    features: tuple = ()

    def to_spec(self):
        """Compile the knobs into generator parameters."""
        spec = {'id': self.seed % 100000, 'seed': self.seed, 'size': self.size,
                'template': self.macro, 'name': self.name, 'theme': 'recipe'}
        # organicity: how irregular the fabric is
        spec['open_ratio'] = 0.30 + 0.30 * self.organicity
        spec['relax'] = int(round(3 * (1.0 - self.organicity)))
        spec['wall_thickness'] = 1 + int(round(2 * (1.0 - self.organicity)))
        spec['warp'] = 0.15 + 0.55 * self.organicity
        # symmetry: rotational order and how strictly it is kept
        spec['order'] = max(3, int(round(4 + 8 * self.symmetry)))
        spec['break_symmetry'] = self.symmetry < 0.85
        # erosion: ruin and rubble
        spec['ruin_rate'] = 0.05 + 0.35 * self.erosion
        spec['lake_rate'] = 0.04 + 0.20 * self.erosion
        # verticality: stairs, ladders, terraces
        spec['levels'] = max(3, int(round(3 + 10 * self.verticality)))
        spec['stairs'] = int(round(8 + 40 * self.verticality))
        if self.structure:
            spec['structure'] = self.structure
        return spec

    def digest(self):
        payload = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def compile_recipe(recipe, run_features=True, validate=True, verbose=False):
    """Recipe -> geometry -> world -> features -> validation."""
    spec = recipe.to_spec()
    rng = random.Random(recipe.seed)
    tm = ts.GENERATORS[recipe.macro](spec, rng) if recipe.macro in ts.GENERATORS \
        else tg.GRAMMARS[recipe.macro](spec, rng)

    if recipe.structure:
        tm = _blend_structure(tm, recipe, spec)
    if recipe.circulation:
        tm = _apply_circulation(tm, recipe, spec)
    if recipe.ornament:
        tm = _apply_ornament(tm, recipe, spec)
    _apply_knobs(tm, recipe)

    # Every step above mutates geometry after the grammar ran its own repair:
    # blending opens walls, circulation paints new fabric over them, erosion
    # rewrites floor. Re-establish traversability here or the recipe can ship a
    # map in several disconnected pieces, which design lint then reports as an
    # error rather than the compiler simply fixing it.
    tg.connect_components(tm, tg.passable_mask(tm), 'floor', width=1)

    world = tw.World(tm, seed=recipe.seed, name=recipe.name, spec=spec)
    world.artifacts['recipe'] = asdict(recipe)
    world.artifacts['recipe_digest'] = recipe.digest()
    tw.derive_places(world)
    if run_features:
        tw.run_features(world, recipe.features or None, verbose=verbose)
    if validate:
        validate_world(world)
    return world


def _blend_structure(tm, recipe, spec):
    """Carve a second grammar's field into the macro fabric."""
    w, h = tm.w, tm.h
    seed = tf.seed_stream(recipe.seed, 'recipe.structure')
    src = tg.FIELD_SOURCES.get(recipe.structure)
    if src is None:
        return tm
    field = src(w, h, seed, spec).normalized()
    # Open a modest share, and only fabric adjoining existing circulation. An
    # unrestrained blend at high organicity carved out 60% of the remaining
    # walls and erased the macro grammar entirely, leaving an almost uniformly
    # open map that reads as noise rather than as two grammars crossed.
    cut = tg.threshold_for_ratio(field, 0.10 + 0.16 * recipe.organicity)
    carve = field.a > cut
    arr = np.array(tm.base, dtype=object).astype(str)
    solid = arr == 'wall'
    adjoining = tf._dilate(~solid, 2) & solid
    opened = carve & adjoining
    for y, x in zip(*np.nonzero(opened)):
        tm.base[y][x] = 'floor'
    tg.attach(tm, structure=field)
    return tm


def _apply_circulation(tm, recipe, spec):
    """Overlay a circulation system such as a river delta or transit spine."""
    w, h = tm.w, tm.h
    seed = tf.seed_stream(recipe.seed, 'recipe.circulation')
    if recipe.circulation in ('river_delta', 'watershed'):
        hyd = tf.watershed(w, h, seed=seed, sea_level=0.30)
        river = tf._dilate(hyd['river'], 1)
        arr = np.array(tm.base, dtype=object).astype(str)
        for y, x in zip(*np.nonzero(river)):
            if arr[y, x] != 'wall':
                tm.base[y][x] = 'water'
        tg.attach(tm, accumulation=hyd['accumulation'])
    elif recipe.circulation == 'transit':
        sub = dict(spec, template='transit_network')
        other = tg.GRAMMARS['transit_network'](sub, random.Random(seed))
        for y in range(h):
            for x in range(w):
                if other.base[y][x] in ('platform', 'courtyard') and \
                        tm.base[y][x] == 'wall':
                    tm.base[y][x] = other.base[y][x]
    return tm


def _apply_ornament(tm, recipe, spec):
    """Decorative pattern applied to open floor only - never structural."""
    w, h = tm.w, tm.h
    seed = tf.seed_stream(recipe.seed, 'recipe.ornament')
    if recipe.ornament == 'truchet':
        rot = tf.truchet(max(2, w // 10), max(2, h // 10), seed=seed)
        cell = 10
        mask = np.zeros((h, w), dtype=bool)
        for cy in range(rot.shape[0]):
            for cx in range(rot.shape[1]):
                corners = ('NW', 'SE') if rot[cy, cx] == 0 else ('NE', 'SW')
                for corner in corners:
                    tg.stamp_polyline(mask, tg._arc_points(
                        cx * cell, cy * cell, cell, corner), width=1, closed=False)
    elif recipe.ornament == 'mandala':
        r = tf.radial(w, h).normalized()
        a = tf.angular(w, h)
        mask = (np.abs(np.cos(a.a * math.pi * spec.get('order', 8))) > 0.86) & (r.a < 0.9)
    else:
        mask = tf.checker(w, h, 6).a > 0.5
    # Ornament is decoration: it must not repaint most of the floor. Thin it so
    # the pattern reads as inlay over the fabric rather than replacing it.
    rng = np.random.default_rng(seed)
    ys, xs = np.nonzero(mask)
    keep = rng.random(len(xs)) < 0.55
    for y, x in zip(ys[keep], xs[keep]):
        if tm.base[y][x] == 'floor':
            tm.base[y][x] = 'courtyard'
    return tm


def _apply_knobs(tm, recipe):
    """Erosion, verticality and secrets applied to the finished fabric."""
    w, h = tm.w, tm.h
    rng = np.random.default_rng(tf.seed_stream(recipe.seed, 'recipe.knobs'))
    arr = np.array(tm.base, dtype=object).astype(str)
    floor = arr == 'floor'
    ys, xs = np.nonzero(floor)
    if not len(xs):
        return tm

    n_erode = int(len(xs) * recipe.erosion * 0.06)
    for i in rng.choice(len(xs), size=min(n_erode, len(xs)), replace=False):
        tm.base[int(ys[i])][int(xs[i])] = 'rubble'

    n_vert = int(len(xs) * recipe.verticality * 0.02)
    for i in rng.choice(len(xs), size=min(n_vert, len(xs)), replace=False):
        tm.overlay[int(ys[i])][int(xs[i])] = 'stair'

    n_secret = int(len(xs) * recipe.secret_density * 0.01)
    for i in rng.choice(len(xs), size=min(n_secret, len(xs)), replace=False):
        tm.overlay[int(ys[i])][int(xs[i])] = 'hidden'
    return tm


# --------------------------------------------------------------- validation

def validate_world(world):
    """Collect geometry, place, relationship, and field validation findings."""
    _lint_geometry(world)
    _lint_place(world)
    _lint_society(world)
    _lint_simulation(world)
    return world.findings


def _lint_geometry(world):
    walk = world.passable()
    ratio = float(walk.mean())
    if ratio < 0.05:
        world.finding('error', 'geometry.open_ratio',
                      f'only {ratio:.1%} of the map is traversable')
    elif ratio > 0.98:
        world.finding('warn', 'geometry.open_ratio',
                      f'{ratio:.1%} traversable leaves almost no structure')
    _, ncomp = tf.connected_components(walk)
    if ncomp > 1:
        world.finding('error', 'geometry.connectivity',
                      f'{ncomp} disconnected traversable regions')
    degenerate = [p.id for p in world.places.values() if p.area <= 0]
    if degenerate:
        world.finding('error', 'geometry.degenerate_place',
                      f'{len(degenerate)} places have zero area')


def _lint_place(world):
    if not world.places:
        world.finding('error', 'place.empty_catalog', 'no places were derived')
        return
    unfunctioned = [p.id for p in world.places.values() if not p.function]
    if unfunctioned:
        world.finding('warn', 'place.missing_function',
                      f'{len(unfunctioned)} places have no function')
    adjacency = world.topology('physical.adjacency')
    parts = adjacency.connected_parts()
    if len(parts) > 1:
        world.finding('warn', 'place.island_places',
                      f'place graph has {len(parts)} components')
    orphan = [p.id for p in world.places.values()
              if not p.neighbours and p.area > 4]
    if orphan:
        world.finding('warn', 'place.orphan', f'{len(orphan)} sizable places have no neighbour')


def _lint_society(world):
    if world.factions:
        landless = [f['name'] for f in world.factions.values() if not f['territory']]
        if landless:
            world.finding('warn', 'society.landless_faction',
                          f'factions without territory: {landless}')
    rel = world.artifacts.get('relationships', {})
    seen_pairs = set()
    pairs = rel.get('pairs', [])
    if rel and rel.get('count') != len(pairs):
        world.finding('error', 'society.relationship_count', 'relationship count differs from records')
    for pair in pairs:
        a, b = pair.get('a'), pair.get('b')
        key = frozenset((a, b))
        kind, cooperation = pair.get('kind'), pair.get('cooperation')
        valid = (a in world.entities and b in world.entities and a != b
                 and key not in seen_pairs and kind in ('alliance', 'rivalry')
                 and isinstance(cooperation, (int, float)) and math.isfinite(cooperation)
                 and 0 < abs(cooperation) <= 1
                 and (cooperation > 0) == (kind == 'alliance'))
        seen_pairs.add(key)
        if not valid:
            world.finding('error', 'society.relationship_pair', 'invalid or duplicate NPC pair')
            continue
        graph = world.topologies.get(f'society.{kind}')
        other = world.topologies.get('society.rivalry' if kind == 'alliance'
                                     else 'society.alliance')
        if (graph is None or graph.edges.get(a, {}).get(b) != abs(cooperation)
                or graph.edges.get(b, {}).get(a) != abs(cooperation)
                or (other is not None and b in other.edges.get(a, {}))):
            world.finding('error', 'society.relationship_graph', 'NPC pair differs from its graph')
    law = world.artifacts.get('law', {})
    if law:
        for fid, code in law.get('codes', {}).items():
            missing = set(law.get('offences', {})) - set(code.get('severity', {}))
            if missing:
                world.finding('warn', 'society.incomplete_code',
                              f'{code["faction"]} has no severity value for {sorted(missing)}')


def _lint_simulation(world):
    for name, arr in world.fields.items():
        a = np.asarray(arr)
        if not np.isfinite(a).all():
            world.finding('error', 'simulation.non_finite_field',
                          f'field {name} contains NaN or infinity')
        meta = world.field_meta.get(name)
        if meta:
            lo, hi = meta['range']
            if float(a.min()) < lo - 1e-3 or float(a.max()) > hi + 1e-3:
                world.finding('warn', 'simulation.field_out_of_range',
                              f'field {name} spans [{a.min():.3f},{a.max():.3f}], '
                              f'declared [{lo},{hi}]')
    for fid, feat in tw.FEATURES.items():
        if fid not in world.artifacts.get('features_run', []):
            continue
        for art in feat.produces:
            kind, _, key = art.partition('.')
            if kind == 'fields' and key not in world.fields:
                world.finding('warn', 'simulation.undelivered_output',
                              f'{fid} declares {art} but produced no such field')
            elif kind == 'topology' and key and key not in world.topologies:
                if not any(t.startswith(key) for t in world.topologies):
                    world.finding('warn', 'simulation.undelivered_output',
                                  f'{fid} declares {art} but produced no such topology')


# ------------------------------------------------------ 40 the world compiler

@feature('exp.compiler', 40, 'World Compiler', 'Export & Runtime Integration',
         consumes=('places',), produces=('artifacts.manifest',), part='II',
         last=True)
def world_compiler(world):
    """Assemble the immutable build manifest: what ran, what it produced, and
    the hashes that let a consumer tell whether anything changed."""
    manifest = {
        'schema': SCHEMA_VERSION,
        'world': world.summary(),
        'recipe': world.artifacts.get('recipe'),
        'recipe_digest': world.artifacts.get('recipe_digest'),
        'features': [],
        'fields': {},
        'topologies': {},
        'places': len(world.places),
        'warnings': [],
    }
    # include self: the compiler is by definition running when it reports
    ran = set(world.artifacts.get('features_run', [])) | {'exp.compiler'}
    for fid, feat in sorted(tw.FEATURES.items(), key=lambda kv: kv[1].number):
        manifest['features'].append({**feat.to_dict(), 'ran': fid in ran})
    for name, arr in sorted(world.fields.items()):
        a = np.asarray(arr, dtype=np.float32)
        manifest['fields'][name] = {
            'shape': list(a.shape),
            'min': round(float(a.min()), 4), 'max': round(float(a.max()), 4),
            'mean': round(float(a.mean()), 4),
            'hash': hashlib.sha256(a.tobytes()).hexdigest()[:12],
            **(world.field_meta.get(name) or {})}
    for name, top in sorted(world.topologies.items()):
        manifest['topologies'][name] = {'nodes': len(top.nodes),
                                        'edges': top.edge_count(),
                                        'directed': top.directed}
    world.artifacts['manifest'] = manifest
    return manifest


# --------------------------------------------------------------- projections

def find_loops(world, limit=6):
    """Find connections that close cycles outside a spanning tree."""
    adjacency = world.topology('physical.adjacency')
    parent, loops = {}, []

    def find(a):
        while parent.get(a, a) != a:
            parent[a] = parent.get(parent[a], parent[a])
            a = parent[a]
        return a

    for a, nbrs in adjacency.edges.items():
        for b in nbrs:
            if a >= b:
                continue
            ra, rb = find(a), find(b)
            if ra == rb:
                dist = adjacency.shortest_paths(a)
                if b in dist:
                    loops.append({'from': a, 'to': b,
                                  'connection_cost': round(dist[b], 2)})
            else:
                parent[ra] = rb
    loops.sort(key=lambda l: -l['connection_cost'])
    return loops[:limit]


def level_markdown(world):
    """Describe a generated gameplay snapshot with readable location labels."""
    a = world.artifacts
    places = sorted(world.places.values(), key=lambda p: -p.area)
    loops = find_loops(world)
    public = world.topology('circulation.public', create=False)
    service = world.topology('circulation.service', create=False)
    machines = a.get('machines', {}).get('machines', [])
    perception = a.get('perception', [])
    strata = sorted({p.stratum for p in places})
    nm = world.place_name
    family = world.spec.get('template', 'custom map').replace('_', ' ').title()
    public_edges = public.edge_count() if public else 0
    service_edges = service.edge_count() if service else 0

    lines = [
        f"# {world.name}", '',
        '> Generated gameplay snapshot. Location labels include approximate',
        '> map coordinates; JSON exports contain the corresponding identifiers.', '',
        '## Overview', '',
        f"A {family} map with {len(places)} locations, "
        f"{len(strata)} elevation layers, and {len(world.factions)} factions.", '',
        '## Atmosphere', '',
    ]
    pacing = a.get('pacing', {})
    for key, label in (('calmest', 'Calmer location'), ('most_suspenseful', 'More suspenseful location')):
        sample = pacing.get(key, [])
        if sample:
            entry = sample[0] if key == 'calmest' else sample[-1]
            p = world.places[entry['place']]
            light = float(world.field('light', 0.5)[p.cells].mean())
            lighting = 'dim lighting' if light < 0.33 else 'moderate lighting' if light < 0.67 else 'bright lighting'
            lines.append(f"- **{label}:** {nm(p.id)}, with {lighting} and {p.area} tiles.")
    if not pacing:
        lines.append('Atmosphere fields were not included in this build.')
    lines += ['', '## Locations', '',
              f"- **Elevation layers:** {', '.join(str(s) for s in strata) or 'none'}",
              f"- **Districts:** {', '.join(sorted({p.district for p in places if p.district})) or 'none'}",
              '- **Largest locations:**']
    lines += [f"  - {nm(p.id)} — {p.area} tiles" for p in places[:3]]
    lines += ['', '## Routes', '',
              f"- Public connections: {public_edges}",
              f"- Service connections: {service_edges}",
              f"- Sheltered locations: {len([p for p in places if 'private' in p.tags])}", '',
              '## Alternate connections', '']
    if loops:
        for loop in loops[:4]:
            lines.append(f"- {nm(loop['from'])} ↔ {nm(loop['to'])} "
                         f"(connection cost {loop['connection_cost']}).")
    else:
        lines.append('- No additional connections close a cycle in the location graph.')
    lines += ['', '## Sight and sound', '']
    for obs in perception[:3]:
        lines.append(f"- From {nm(obs['observer_place'])}: "
                     f"{obs['visible_cells']} tiles are visible and {obs['audible_cells']} audible; "
                     f"{obs['heard_but_unseen']} are audible beyond the sightline.")
    if not perception:
        lines.append('- No observer samples in this build.')
    lines += ['', '## Room sizes', '']
    if places:
        largest = places[0]
        lines.append(f"- The largest location covers {largest.area} tiles "
                     f"and connects to {len(largest.neighbours)} neighbouring locations.")
        lines.append(f"- Its tags: {', '.join(sorted(largest.tags)) or 'none'}.")
    else:
        lines.append('- No locations in this build.')
    lines += ['', '## Machines', '']
    machine_names = {m['id']: f"{m['kind']} at {nm(m['place'])}" for m in machines}
    for m in machines[:4]:
        needs = ', '.join(machine_names.get(mid, mid) for mid in m['requires']) or 'no upstream machine'
        lines.append(f"- **{m['kind'].title()}** at {nm(m['place'])}: "
                     f"provides {m['output']}; requires {needs}; condition {m['condition']:.0%}.")
    if not machines:
        lines.append('- No machines in this build.')

    lines += ['', '## NPCs and factions', '']
    for e in list(world.entities.values())[:8]:
        fac = world.factions.get(e['faction'], {}).get('name', 'unaligned')
        lines.append(f"- **{e['name']}**, {e['role']}, {fac}; "
                     f"home: {nm(e['home'])}; work: {nm(e['work'])}.")
    relationships = a.get('relationships', {}).get('pairs', [])
    if relationships:
        lines += ['', '### Alliances and rivalries', '']
        for pair in relationships[:4]:
            first, second = (world.entities[eid]['name'] for eid in (pair['a'], pair['b']))
            lines.append(f"- {first} and {second}: {pair['kind']}, "
                         f"cooperation {pair['cooperation']:+.2f} on a −1 to +1 scale.")
    chron = a.get('chronicle', {}).get('prose', [])
    if chron:
        lines += ['', '### Generated background', '']
        lines += [f"- {line}" for line in chron[:8]]
    lines += ['', '## Quest hooks', '']
    for quest in a.get('operations', {}).get('contracts', [])[:4]:
        lines.append(f"- **{quest['kind'][0].upper() + quest['kind'][1:]}** at {nm(quest['target_place'])}.")
    for quest in a.get('investigations', {}).get('quests', [])[:2]:
        lines.append(f"- Follow {len(quest['clue_ids'])} clues at {nm(quest['place'])} "
                     f"to {quest['objective']}.")
    schedule = a.get('director', {}).get('schedule', [])
    if schedule:
        lines += ['', '### Encounter suggestions', '',
                  'Seeded selections from encounter categories:', '']
        for beat in schedule[:4]:
            lines.append(f"- {beat['beat'].title()} at {nm(beat['place'])}: {beat['reason']}.")
    lines += ['', '## Build notes', '',
              f"- Generated construction eras: {', '.join(a.get('build_order', [])) or 'none'}.",
              f"- Hazard flags: {len(a.get('structure', {}).get('failures', []))} unstable locations, "
              f"{len(a.get('health', {}).get('hotspots', []))} environmental hotspots.", '',
              '## Validation', '']
    for f in world.findings[:6]:
        lines.append(f"- [{f['severity']}] {f['rule']}: {f['message']}")
    if not world.findings:
        lines.append('- No design-check findings. Gameplay balance still needs playtesting.')
    return '\n'.join(lines) + '\n'


FIELD_RAMPS = {
    'default': [(12, 16, 30), (40, 80, 140), (90, 170, 200), (240, 240, 210)],
    'heat':    [(10, 8, 20), (120, 30, 40), (220, 110, 30), (255, 240, 180)],
    'cool':    [(8, 12, 24), (30, 70, 120), (70, 150, 190), (230, 245, 250)],
    'threat':  [(10, 10, 14), (70, 20, 30), (180, 50, 40), (255, 200, 120)],
}


def render_field(world, name, ramp='default', tile_px=4, label=None):
    """Render a field as a labelled heatmap."""
    from PIL import Image, ImageDraw
    arr = np.asarray(world.field(name), dtype=np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    norm = (arr - lo) / max(hi - lo, 1e-6)
    stops = FIELD_RAMPS.get(ramp, FIELD_RAMPS['default'])
    xs = np.linspace(0, 1, len(stops))
    rgb = np.zeros(arr.shape + (3,), dtype=np.float32)
    for c in range(3):
        rgb[..., c] = np.interp(norm, xs, [s[c] for s in stops])
    img = Image.fromarray(np.repeat(np.repeat(rgb.astype(np.uint8), tile_px, 0),
                                    tile_px, 1), 'RGB')
    d = ImageDraw.Draw(img)
    d.text((4, 4), label or f"{name}  [{lo:.2f},{hi:.2f}]",
           fill=(255, 255, 255), font=tsy.DEFAULT_FONT)
    return img


def export_world(world, root, style='procedural_dark_tiles', tile_px=6,
                 debug_fields=('light', 'sound_pressure', 'threat', 'privacy',
                               'wear', 'control', 'contamination', 'memory'),
                 text_only=False):
    """Export a snapshot, optionally refreshing only JSON and Markdown files."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    spec = world.spec or {'id': 0}

    ts.export_map_json(world.tilemap, {**spec, 'name': world.name,
                                       'template': spec.get('template', 'recipe')},
                       root / 'tilemap.json')
    if not text_only:
        (root / 'debug').mkdir(exist_ok=True)
        tsy.render_map(world.tilemap, spec, style=style, tile_px=tile_px,
                       label=world.name).save(root / 'render.png')

    ramps = {'light': 'heat', 'sound_pressure': 'cool', 'threat': 'threat',
             'contamination': 'heat', 'wear': 'default', 'control': 'default',
             'privacy': 'cool', 'memory': 'threat'}
    made = []
    for name in (() if text_only else debug_fields):
        if world.has_field(name):
            render_field(world, name, ramps.get(name, 'default'),
                         max(2, tile_px // 2)).save(root / 'debug' / f'{name}.png')
            made.append(name)

    (root / 'places.json').write_text(json.dumps(
        [world.places[p].to_dict() for p in world.place_order], indent=2), encoding='utf-8')
    (root / 'topologies.json').write_text(json.dumps(
        {k: v.to_dict() for k, v in world.topologies.items()}, indent=2), encoding='utf-8')
    (root / 'entities.json').write_text(json.dumps(world.entities, indent=2), encoding='utf-8')
    (root / 'factions.json').write_text(json.dumps(world.factions, indent=2), encoding='utf-8')
    (root / 'artifacts.json').write_text(json.dumps(
        world.artifacts, indent=2, default=str), encoding='utf-8')
    (root / 'events.json').write_text(json.dumps(world.events, indent=2), encoding='utf-8')
    (root / 'findings.json').write_text(json.dumps(world.findings, indent=2), encoding='utf-8')
    (root / 'manifest.json').write_text(json.dumps(
        world.artifacts.get('manifest', {}), indent=2, default=str), encoding='utf-8')
    (root / 'level-spec.md').write_text(level_markdown(world), encoding='utf-8')

    if not text_only:
        np.savez_compressed(root / 'fields.npz',
                            **{k: np.asarray(v, dtype=np.float32)
                               for k, v in world.fields.items()})
    return {'debug_fields': made, 'root': str(root)}


# ------------------------------------------------------- the example recipe

OBSIDIAN_SALT_KEEP = WorldRecipe(
    name='The Obsidian Salt Keep',
    macro='voronoi_kingdoms',
    structure='reaction_diffusion',
    circulation='river_delta',
    ornament='truchet',
    symmetry=0.72,
    organicity=0.84,
    verticality=0.30,
    erosion=0.42,
    secret_density=0.15,
    style='obsidian_gold',
    seed=93741,
    size=128,
)

EXAMPLE_RECIPES = {
    'salt-keep': OBSIDIAN_SALT_KEEP,
    'trench-station': WorldRecipe(
        name='Trench Station', macro='circuit_city', structure='fbm',
        circulation='transit', ornament='checker', symmetry=0.2,
        organicity=0.25, verticality=0.6, erosion=0.2, secret_density=0.25,
        style='neon_circuit', seed=20481, size=128),
    'obsidian-metropolis': WorldRecipe(
        name='Obsidian Metropolis', macro='phyllotaxis_city',
        structure='voronoi', circulation='transit', ornament='mandala',
        symmetry=0.9, organicity=0.15, verticality=0.45, erosion=0.25,
        secret_density=0.1, style='architectural_blueprint', seed=7717, size=128),
    'coral-reliquary': WorldRecipe(
        name='The Coral Reliquary', macro='reaction_diffusion',
        structure='growth', circulation=None, ornament='mandala',
        symmetry=0.6, organicity=0.95, verticality=0.2, erosion=0.55,
        secret_density=0.3, style='parchment_ink', seed=5150, size=128),
}

def replay(recipe, expect=None):
    """Recompile a recipe and compare its digest, event count, places, and fields."""
    world = compile_recipe(recipe)
    manifest = world.artifacts.get('manifest', {})
    report = {
        'recipe_digest': recipe.digest(),
        'events': len(world.events),
        'field_hashes': {k: v['hash'] for k, v in manifest.get('fields', {}).items()},
        'place_order': list(world.place_order),
    }
    if expect is not None:
        report['matches'] = {
            'digest': expect.get('recipe_digest') == report['recipe_digest'],
            'events': expect.get('events') == report['events'],
            'places': expect.get('place_order') == report['place_order'],
            'fields': expect.get('field_hashes') == report['field_hashes'],
        }
        report['identical'] = all(report['matches'].values())
    return world, report


def replay_check(recipe):
    """Compile twice and diff. Returns (ok, report)."""
    _, first = replay(recipe)
    _, second = replay(recipe, expect=first)
    diverged = [k for k, v in second['matches'].items() if not v]
    return second['identical'], {'diverged': diverged, **second}
