"""Coverage and behaviour tests for 22 map families and 40 simulation features.

Roguelike Depth Layers contains 62 distinct components. Run directly:

    python test_goals.py           # coverage + behaviour
    python test_goals.py --full    # also every grammar at size, and all recipes
"""
from __future__ import annotations

import argparse
import copy
import functools
from collections import Counter, deque
import json
import math
import tempfile
from unittest.mock import patch
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import tilegen_system as ts
import tilegen_fields as tf
import tilegen_grammars as tg
import tilegen_grammars2  # noqa: F401
import tilegen_overworld as ow
import tilegen_autotile as ta
import tilegen_world as tw
import tilegen_layers  # noqa: F401
import tilegen_society as society
import tilegen_compiler as tc
import tilegen_style as tsy
import tilegen_runtime_budget as rb
import tilegen_vector as tv
import tilegen_lowend_runtime as rt
import tilegen_socket_ext as se
import tilegen_wfc as wfc
import tilegen_routes as tr
import tilegen_corridors as co
import tilegen_keylock as kl
import tilegen_convchain as cc
import tilegen_backtrack as bt
import tilegen_csp as csp

FAILURES = []

# The 22 map families covered by this suite (21 is the cellular-cave family,
# 22 the climate overworld).
SPECIES = [
    (1, 'Voronoi Kingdoms', 'voronoi_kingdoms'),
    (2, 'Reaction-Diffusion Labyrinths', 'reaction_diffusion'),
    (3, 'Differential-Growth Dungeons', 'differential_growth'),
    (4, 'Truchet Knot Worlds', 'truchet'),
    (5, 'Space-Filling Curve Maps', 'space_filling'),
    (6, 'Quasicrystal Temples', 'quasicrystal'),
    (7, 'Phyllotaxis Cities', 'phyllotaxis_city'),
    (8, 'L-System Root Worlds', 'lsystem_roots'),
    (9, 'DLA Crystal Caverns', 'dla_caverns'),
    (10, 'River-Delta / Watershed Maps', 'watershed'),
    (11, 'Tectonic Worlds', 'tectonic'),
    (12, 'Flow-Field Maps', 'flow_field'),
    (13, 'Contour-Line Architecture', 'contour_terraces'),
    (14, 'Circuit-Board Megacities', 'circuit_city'),
    (15, 'Transit-Network Maps', 'transit_network'),
    (16, 'Non-Euclidean Portal Maps', 'portal_maps'),
    (17, 'Fractal Archipelagos', 'archipelago'),
    (18, 'Generative Mandalas', 'mandala'),
    (19, 'Semantic Silhouette Maps', 'silhouette'),
    (20, 'Hybrid Morph Maps', 'hybrid_morph'),
    (21, 'Cellular Cave Systems', 'ca_caves'),
    (22, 'Climate Overworlds', 'overworld'),
]


def check(name, condition, detail=''):
    print(f"  [{'ok  ' if condition else 'FAIL'}] {name}{(' - ' + detail) if detail else ''}")
    if not condition:
        FAILURES.append(name)
    return condition


# ------------------------------------------------------------ goal coverage

def test_species_coverage():
    print("\n22 map families")
    for num, name, template in SPECIES:
        check(f"{num:02d} {name}", template in tg.GRAMMARS, template)


def test_atlas_coverage():
    print("\n40 simulation features")
    by_number = {f.number: f for f in tw.FEATURES.values()}
    for num in range(21, 61):
        feat = by_number.get(num)
        check(f"{num:02d} {feat.name if feat else '???'}", feat is not None,
              feat.id if feat else 'MISSING')


def test_goal_total():
    print("\ntotals")
    grammars = len([t for _, _, t in SPECIES if t in tg.GRAMMARS])
    feats = len([n for n in range(21, 61)
                 if any(f.number == n for f in tw.FEATURES.values())])
    check('22 map families implemented', grammars == 22, f'{grammars}/22')
    check('40 simulation features 21-60 implemented', feats == 40, f'{feats}/40')
    check('62 distinct components', grammars + feats == 62,
          f'{grammars} map families + {feats} simulation features = {grammars + feats}')
    check('feature numbers are unique and complete',
          sorted(f.number for f in tw.FEATURES.values()) == list(range(21, 61)))


# --------------------------------------------------------------- behaviour

def test_grammars(size=64, seeds=(3, 9, 17, 24)):
    print(f"\ngrammar behaviour ({size}x{size})")
    for num, name, template in SPECIES:
        fn = tg.GRAMMARS[template]
        ok = True
        detail = ''
        for seed in seeds:
            spec = {'id': 700 + num, 'size': size, 'template': template, 'seed': seed}
            tm = fn(spec, random.Random(seed))
            pm = tg.passable_mask(tm)
            _, ncomp = tf.connected_components(pm)
            det = tm.base == fn(spec, random.Random(seed)).base
            if not (ncomp == 1 and 0.03 < pm.mean() < 0.99 and det
                    and tm.w == size and tm.h == size):
                ok = False
                detail = f'seed {seed}: comps={ncomp} open={pm.mean():.0%} det={det}'
        varies = (fn({'id': 1, 'size': size, 'template': template, 'seed': 1},
                     random.Random(1)).base !=
                  fn({'id': 1, 'size': size, 'template': template, 'seed': 2},
                     random.Random(2)).base)
        check(f"{num:02d} {template}", ok and varies,
              detail or ('' if varies else 'ignores seed'))


def test_world_pipeline():
    print("\nworld pipeline")
    spec = {'id': 1, 'size': 72, 'template': 'voronoi_kingdoms', 'seed': 5}
    tm = tg.GRAMMARS['voronoi_kingdoms'](spec, random.Random(5))
    world = tw.World(tm, seed=5, name='pipeline', spec=spec)
    tw.derive_places(world)
    check('places derived', len(world.places) > 5, f'{len(world.places)}')
    check('place ids unique', len({p.id for p in world.places.values()}) == len(world.places))
    check('adjacency graph connected',
          len(world.topology('physical.adjacency').connected_parts()) == 1)

    ran = tw.run_features(world)
    check('all 40 features ran', len(ran) == len(tw.FEATURES),
          f'{len(ran)}/{len(tw.FEATURES)}')
    check('no feature errors', not world.findings,
          '; '.join(f['rule'] for f in world.findings[:3]))
    check('fields published', len(world.fields) > 30, f'{len(world.fields)}')
    check('topologies published', len(world.topologies) > 12, f'{len(world.topologies)}')
    check('entities created', len(world.entities) > 0, f'{len(world.entities)}')
    check('factions created', len(world.factions) > 0, f'{len(world.factions)}')
    check('events emitted', len(world.events) > 0, f'{len(world.events)}')
    return world


def test_topologies_disagree(world):
    """Different systems should produce distinct connection graphs."""
    print("\ntopology independence")
    walk = world.topology('traversal.walk')
    acoustic = world.topology('perception.acoustic')
    rumor = world.topology('society.rumor')
    pub = world.topology('circulation.public')
    svc = world.topology('circulation.service')

    def edgeset(t):
        return {(a, b) for a, nbrs in t.edges.items() for b in nbrs}

    check('acoustic graph differs from walking graph',
          edgeset(acoustic) != edgeset(walk))
    check('rumour graph differs from adjacency',
          edgeset(rumor) != edgeset(world.topology('physical.adjacency')))
    check('public and service circulation are distinct',
          edgeset(pub) != edgeset(svc) and svc.edge_count() > 0)


def small_world(seed=7, population=0):
    world = tw.World(ts.TileMap(8, 4, default_base='floor'), seed=seed)
    for x in (0, 4):
        cells = np.zeros((4, 8), dtype=bool)
        cells[:, x:x + 4] = True
        world.add_place('room', cells, function='hall')
    world.entities = {f'npc_{i}': {'id': f'npc_{i}'} for i in range(population)}
    return world


def test_relationships():
    print("\nalliances, rivalries, and cooperation")
    bad, kinds = [], set()
    for template in ('voronoi_kingdoms', 'space_filling', 'circuit_city'):
        for seed in (4, 17):
            spec = {'id': 1, 'size': 64, 'template': template, 'seed': seed}
            w = tw.World(tg.GRAMMARS[template](spec, random.Random(seed)), seed=seed, spec=spec)
            tw.derive_places(w)
            tw.run_features(w)
            pairs = w.artifacts['relationships']['pairs']
            if not pairs:
                bad.append(f'{template}/{seed}: missing pairs')
            seen = set()
            for pair in pairs:
                a, b, kind, cooperation = (pair[k] for k in ('a', 'b', 'kind', 'cooperation'))
                key = frozenset((a, b))
                if (a == b or a not in w.entities or b not in w.entities or key in seen
                        or kind not in ('alliance', 'rivalry')
                        or not 0 < abs(cooperation) <= 1
                        or (cooperation > 0) != (kind == 'alliance')):
                    bad.append(f'{template}/{seed}: invalid pair')
                seen.add(key)
                kinds.add(kind)
                graph = w.topology(f'society.{kind}')
                if graph.edges[a][b] != abs(cooperation) or graph.edges[b][a] != abs(cooperation):
                    bad.append(f'{template}/{seed}: graph mismatch')
            tc._lint_society(w)
            if any(f['severity'] == 'error' for f in w.findings):
                bad.append(f'{template}/{seed}: lint error')
    check('unique NPC pairs, signed cooperation, and symmetric graphs agree', not bad, '; '.join(bad[:2]))
    check('alliances and rivalries both generated', kinds == {'alliance', 'rivalry'})
    empty = tw.World(ts.TileMap(4, 4))
    check('empty map produces no relationships', society.npc_relationships(empty) == [])
    check('one NPC produces no relationships', society.npc_relationships(small_world(population=1)) == [])
    two = society.npc_relationships(small_world(population=2))
    check('two NPCs produce one unordered pair', len(two) == 1)
    a, b = small_world(population=40), small_world(population=40)
    check('relationship generation is reproducible', society.npc_relationships(a) == society.npc_relationships(b))


def test_quest_hooks(world):
    print("\nquest hooks and encounter selection")
    contracts = world.artifacts['operations']['contracts']
    expected = {'item recovery', 'NPC rescue', 'disable mechanism', 'dungeon exploration'}
    check('adventure objectives cover all four kinds', {c['kind'] for c in contracts} == expected)
    check('mission steps refer to existing locations',
          all(step['place'] in world.places for c in contracts for step in c['steps']))
    check('mission actions vary with their objectives', len({c['steps'][2]['step'] for c in contracts}) == 4)
    hooks = world.artifacts['investigations']
    clues = {c['id']: c for c in hooks['clues']}
    check('investigation hooks link existing clues, places, and NPCs', bool(hooks['quests']) and all(
        q['place'] in world.places and q['npc'] in world.entities and len(q['clue_ids']) == 3
        and all(cid in clues and clues[cid]['place'] == q['place'] and clues[cid]['npc'] == q['npc']
                for cid in q['clue_ids']) for q in hooks['quests']))
    check('clue quality stays in range', all(0 <= c['quality'] <= 1 for c in clues.values()))
    empty = small_world()
    check('no incidents produce no investigation quests', society.investigation_quests(empty) == [])
    empty.artifacts['violence'] = {'incidents': [{'place': empty.place_order[0]}]}
    q = society.investigation_quests(empty)[0]
    check('unpopulated encounter offers exploration', q['npc'] is None and q['objective'] == 'explore the encounter site')
    a, b = small_world(population=4), small_world(population=4)
    for w in (a, b):
        w.artifacts['violence'] = {'incidents': [{'place': w.place_order[0]}]}
        w.artifacts['identity'] = {'disguises': [{'entity': 'npc_0'}]}
    b.set_field('wear', np.ones((4, 8), dtype=np.float32))
    qa, qb = society.investigation_quests(a), society.investigation_quests(b)
    check('quest assignments are seeded independently of clue quality', qa == qb and
          a.artifacts['investigations']['clues'] != b.artifacts['investigations']['clues'])
    check('director handles no encounter candidates', society.world_director(small_world()) == [])
    w = small_world()
    w.places[w.place_order[0]].tags.add('market')
    check('director falls back to available category',
          all(e['beat'] == 'shortage' for e in society.world_director(w)))
    a, b = small_world(), small_world()
    for w in (a, b):
        w.places[w.place_order[0]].tags.add('volatile')
        w.places[w.place_order[1]].tags.add('market')
    threat = np.zeros((4, 8), dtype=np.float32)
    threat[:, :4] = 1
    a.set_field('threat', threat)
    b.set_field('threat', 1 - threat)
    schedule = society.world_director(a)
    check('local threat is not used as a candidate score', schedule == society.world_director(b))
    estimate = 0.5
    categories_ok = True
    for encounter in schedule:
        want_up = encounter['target_tension'] > estimate
        categories_ok &= (encounter['beat'] == 'escalate') == want_up
        estimate += 0.1 if want_up else -0.08
    check('director respects categories when both are available', categories_ok)


def test_text_export(world):
    print("\ntext-only export")
    scratch = HERE.parent / '.local'
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='text-export-', dir=scratch) as directory:
        root = Path(directory)
        # Existing binary outputs must survive a text refresh byte for byte.
        for name in ('render.png', 'fields.npz', 'debug/light.png'):
            path = root / name
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(b'existing binary output')
        with patch.object(tsy, 'render_map', side_effect=AssertionError('unexpected render')), \
             patch.object(tc, 'render_field', side_effect=AssertionError('unexpected heatmap')):
            tc.export_world(world, root, text_only=True)
        check('text refresh preserves images and arrays', all(
            (root / name).read_bytes() == b'existing binary output'
            for name in ('render.png', 'fields.npz', 'debug/light.png')))
        check('text export contains current relationship records',
              json.loads((root / 'artifacts.json').read_text())['relationships'] == world.artifacts['relationships'])
        check('exported pairs resolve to exported NPCs',
              set(json.loads((root / 'entities.json').read_text())) == set(world.entities))
        check('level notes regenerated through the formatter',
              (root / 'level-spec.md').read_text() == tc.level_markdown(world))
        fresh = root / 'fresh'
        tc.export_world(world, fresh, text_only=True)
        check('fresh text export creates no binary outputs', all(p.suffix in ('.json', '.md') for p in fresh.iterdir()))


def test_feature_graph():
    print("\nfeature dependency graph")
    order = tw.resolve_order()
    check('every feature scheduled', len(order) == len(tw.FEATURES),
          f'{len(order)}/{len(tw.FEATURES)}')
    pos = {fid: i for i, fid in enumerate(order)}
    producer = {}
    for fid, feat in tw.FEATURES.items():
        for art in feat.produces:
            producer.setdefault(art, fid)
    violations = []
    for fid, feat in tw.FEATURES.items():
        for art in feat.consumes:
            src = producer.get(art)
            if src and src != fid and pos[src] > pos[fid]:
                violations.append(f'{fid} consumes {art} before {src}')
    check('dependencies respected', not violations, '; '.join(violations[:3]))
    check('world compiler runs last', order[-1] == 'exp.compiler', order[-1])


def test_compiler(full=False):
    print("\nrecipes and compiler")
    recipes = tc.EXAMPLE_RECIPES if full else {'salt-keep': tc.OBSIDIAN_SALT_KEEP}
    for key, recipe in recipes.items():
        r = tc.WorldRecipe(**{**vars(recipe), 'size': 96 if not full else recipe.size})
        t0 = time.time()
        world = tc.compile_recipe(r)
        errors = [f for f in world.findings if f['severity'] == 'error']
        manifest = world.artifacts.get('manifest', {})
        ran = sum(1 for f in manifest.get('features', []) if f['ran'])
        check(f"{key} compiles clean",
              not errors and ran == len(tw.FEATURES),
              f"{time.time() - t0:.1f}s places={len(world.places)} "
              f"features={ran}/{len(tw.FEATURES)} errors={len(errors)}")
        check(f"{key} manifest complete",
              len(manifest.get('fields', {})) > 30 and manifest.get('recipe_digest'),
              f"{len(manifest.get('fields', {}))} fields")
        md = tc.level_markdown(world)
        check(f"{key} level notes cover the snapshot",
              all(h in md for h in ('## Overview', '## Atmosphere', '## Locations',
                                    '## Routes', '## Sight and sound', '## Room sizes',
                                    '## Machines', '## NPCs and factions', '## Quest hooks')),
              f'{len(md)} chars')
        largest = sorted(world.places.values(), key=lambda p: -p.area)[:3]
        listed = md.split('- **Largest locations:**', 1)[1].split('## Routes', 1)[0]
        check(f"{key} largest locations use size order", all(
            listed.index(world.place_name(a.id)) < listed.index(world.place_name(b.id))
            for a, b in zip(largest, largest[1:])))



def test_determinism():
    print("\ndeterminism")
    a = tc.compile_recipe(tc.WorldRecipe(**{**vars(tc.OBSIDIAN_SALT_KEEP), 'size': 64}))
    b = tc.compile_recipe(tc.WorldRecipe(**{**vars(tc.OBSIDIAN_SALT_KEEP), 'size': 64}))
    check('same recipe yields the same tilemap', a.tilemap.base == b.tilemap.base)
    check('same recipe yields the same place ids', a.place_order == b.place_order)
    check('same recipe yields the same digest',
          a.artifacts['recipe_digest'] == b.artifacts['recipe_digest'])
    fa = a.artifacts['manifest']['fields']
    fb = b.artifacts['manifest']['fields']
    differing = [k for k in fa if fa[k]['hash'] != fb.get(k, {}).get('hash')]
    check('all field hashes reproduce', not differing, f'differ: {differing[:3]}')
    check('NPC records reproduce', a.entities == b.entities)
    for key in ('relationships', 'investigations', 'operations', 'chronicle', 'director'):
        check(f'{key} records reproduce', a.artifacts[key] == b.artifacts[key])
    check('event contents reproduce', a.events == b.events)
    check('readable level descriptions reproduce', tc.level_markdown(a) == tc.level_markdown(b))

    # Compare independent recompilations of each example recipe.
    for key, recipe in tc.EXAMPLE_RECIPES.items():
        ok, rep = tc.replay_check(tc.WorldRecipe(**{**vars(recipe), 'size': 72}))
        check(f'{key} replays identically',
              ok, f"events={rep['events']} diverged={rep['diverged'] or 'none'}")


def test_validators_catch_problems():
    """A validator that never fires is not a validator."""
    print("\nvalidator sensitivity")
    spec = {'id': 1, 'size': 48, 'template': 'voronoi_kingdoms', 'seed': 2}
    tm = tg.GRAMMARS['voronoi_kingdoms'](spec, random.Random(2))
    world = tw.World(tm, seed=2, spec=spec)
    tw.derive_places(world)
    # deliberately break connectivity by walling off a strip
    for x in range(world.w):
        world.tilemap.base[world.h // 2][x] = 'wall'
    tc.validate_world(world)
    caught = [f for f in world.findings if f['rule'] == 'geometry.connectivity']
    check('lint detects a severed map', bool(caught),
          caught[0]['message'] if caught else 'not detected')

    w2 = small_world(population=12)
    society.npc_relationships(w2)
    original = copy.deepcopy(w2.artifacts['relationships'])
    mutations = [
        ('self pair', lambda r: r['pairs'][0].update(b=r['pairs'][0]['a'])),
        ('unknown NPC', lambda r: r['pairs'][0].update(b='missing')),
        ('duplicate pair', lambda r: r['pairs'].append(dict(r['pairs'][0]))),
        ('invalid cooperation', lambda r: r['pairs'][0].update(cooperation=2)),
        ('wrong sign', lambda r: r['pairs'][0].update(cooperation=-r['pairs'][0]['cooperation'])),
        ('unknown kind', lambda r: r['pairs'][0].update(kind='unknown')),
    ]
    for name, mutate in mutations:
        w2.artifacts['relationships'] = copy.deepcopy(original)
        mutate(w2.artifacts['relationships'])
        w2.findings = []
        tc._lint_society(w2)
        check(f'lint detects {name}', any(f['rule'] == 'society.relationship_pair' for f in w2.findings))
    w2.artifacts['relationships'] = original
    first = original['pairs'][0]
    w2.topology('society.' + first['kind']).edges[first['a']][first['b']] = 0
    w2.findings = []
    tc._lint_society(w2)
    check('lint detects a mismatched relationship graph',
          any(f['rule'] == 'society.relationship_graph' for f in w2.findings))


def test_backwards_compatibility():
    print("\nbackwards compatibility")
    check('original 35 templates still registered',
          all(t in ts.GENERATORS for t in
              ('hub_spokes', 'spiral', 'grid_rooms', 'micro_matrix')))
    spec = ts.all_specs[0]
    tm = ts.make_layout(spec)
    check('original layout still builds', tm.w == spec['size'])
    img = tsy.render_map(tm, spec, style='procedural_dark_tiles', tile_px=4)
    check('original layout renders through the style registry', img.size[0] > 0)


def test_runtime_budget():
    """Check scheduling reductions and runtime helper behaviour."""
    print("\nlow-end runtime budget")
    spec = {'id': 1, 'size': 96, 'template': 'voronoi_kingdoms', 'seed': 5}
    tm = tg.GRAMMARS['voronoi_kingdoms'](spec, random.Random(5))

    r = rb.simulate(tm, agents=150, ticks=120, seed=5)
    check('throttling cuts pathfinding calls by >80%',
          r['path_reduction'] > 0.8, f"{r['path_reduction']:.1%}")
    check('off-screen culling cuts animation updates',
          r['animation_reduction'] > 0.4, f"{r['animation_reduction']:.1%}")
    check('staggering flattens the per-tick spike',
          r['spike_reduction'] > 0.5, f"{r['spike_reduction']:.1%}")

    ids = [f'a{i}' for i in range(200)]
    staggered = rb.PathScheduler(interval=20, jitter=6, seed=3).load_profile(ids, 60)
    lumped = rb.PathScheduler(interval=20, jitter=0, seed=3).load_profile(ids, 60)
    check('same mean load, lower peak',
          abs(sum(staggered) - sum(lumped)) <= len(ids) and
          max(staggered) < max(lumped) / 2,
          f"peak {max(staggered)} vs {max(lumped)}")

    passable = tg.passable_mask(tm)
    baked = rb.BakedNavigation(passable)
    baked.bake('g', (tm.w // 2, tm.h // 2))
    ys, xs = np.nonzero(passable)
    sample = [(int(xs[i]), int(ys[i])) for i in range(0, len(xs), max(1, len(xs) // 50))]
    routed = sum(1 for p in sample if baked.step('g', p) is not None)
    check('baked navigation routes from anywhere reachable',
          routed >= len(sample) - 1, f'{routed}/{len(sample)}')

    streamer = rb.ChunkStreamer(tm, chunk=32, radius=1, workers=2)
    t0 = time.time()
    wanted = streamer.request_around((tm.w // 2, tm.h // 2))
    request_ms = (time.time() - t0) * 1000
    idle = streamer.wait_idle(5.0)
    ready = streamer.ready()
    streamer.close()
    check('chunk request does not block the caller', request_ms < 50,
          f'{request_ms:.1f}ms')
    check('background workers deliver the chunks',
          idle and len(ready) == len(wanted), f'{len(ready)}/{len(wanted)}')

    scaler = rb.ResolutionScaler(base_tile_px=8, min_tile_px=2, target_ms=16.7)
    for _ in range(40):
        scaler.record(30.0)
    dropped = scaler.tile_px
    for _ in range(80):
        scaler.record(6.0)
    check('resolution drops under load and recovers',
          dropped < 8 and scaler.tile_px == 8, f'{dropped}px then {scaler.tile_px}px')

    # Sample across the whole map: taking the first 300 passable cells picks a
    # band along one edge, so every entity lands in the same ring. The viewport
    # must also be small relative to the map, or the freeze ring (3x the half
    # viewport) covers everything and that tier can never engage - which is
    # correct behaviour on a small map, but tests nothing.
    culler = rb.ViewportCuller(tm.w // 5, tm.h // 6)
    step = max(1, len(xs) // 300)
    spread = np.stack([xs[::step][:300], ys[::step][:300]], axis=1)
    counts = culler.counts((tm.w // 2, tm.h // 2), spread)
    check('viewport LOD partitions entities into all three tiers',
          sum(counts.values()) == len(spread) and counts['active'] > 0 and
          counts['simulated'] > 0 and counts['frozen'] > 0, str(counts))


# ------------------------------------------------- variable-radius sampling

def _rule_gap(points, rmap, rule):
    """Smallest (distance - demanded spacing) over all pairs of points."""
    p = np.asarray(points, dtype=np.float64)
    r = rmap[p[:, 1].astype(int), p[:, 0].astype(int)].astype(np.float64)
    d = np.hypot(p[:, None, 0] - p[None, :, 0], p[:, None, 1] - p[None, :, 1])
    need = {'mean': 0.5 * (r[:, None] + r[None, :]),
            'max': np.maximum(r[:, None], r[None, :]),
            'min': np.minimum(r[:, None], r[None, :])}[rule]
    np.fill_diagonal(d, np.inf)
    return float((d - need).min())


def test_poisson_variable_radius():
    print("\nvariable-radius Poisson-disc sampling")
    seed = tf.seed_stream(21, 'test.poisson_field')

    cases = [(120, 90, 7.5, 0, 24, 3), (64, 64, 5.0, 3, 30, 0), (200, 50, 11.0, 9, 10, 2),
             (80, 80, 3.0, seed, 24, 0)]
    check('scalar radius reproduces poisson_disc bit for bit',
          all(np.array_equal(tf.poisson_disc_field(*a), tf.poisson_disc(*a)) for a in cases))

    w, h = 400, 200
    rmap = np.full((h, w), 4.0, dtype=np.float32)
    rmap[:, w // 2:] = 8.0
    pts = tf.poisson_disc_field(w, h, rmap, seed=seed, k=30)
    check('field sampling returns (n, 2) float32', pts.ndim == 2 and pts.shape[1] == 2
          and pts.dtype == np.float32 and 1500 < len(pts) < 2600, f'{len(pts)} points')
    gap = _rule_gap(pts, rmap, 'mean')
    check("rule 'mean': every pair keeps (r_p + r_q) / 2", gap >= -1e-9, f'min gap {gap:.4f}')
    yy, xx = np.mgrid[0:100, 0:160]
    blocks = (4.0 + 4.0 * (((xx // 20) + (yy // 20)) % 2)).astype(np.float32)
    counts = {}
    for rule in ('max', 'min'):
        sub = tf.poisson_disc_field(160, 100, blocks, seed=seed + 1, rule=rule)
        gap = _rule_gap(sub, blocks, rule)
        counts[rule] = len(sub)
        check(f"rule {rule!r}: every pair keeps its spacing", gap >= -1e-9 and len(sub) > 100,
              f'{len(sub)} points, min gap {gap:.4f}')
    check("rule 'max' packs looser than rule 'min' on a checkerboard of radii",
          counts['max'] < counts['min'], f"{counts['max']} < {counts['min']}")

    # maximality: how many random spots could still take a point
    rng = np.random.default_rng(tf.seed_stream(21, 'test.poisson_probe'))
    probes = rng.uniform([0, 0], [w, h], size=(20000, 2))
    rc = rmap[probes[:, 1].astype(int), probes[:, 0].astype(int)].astype(np.float64)
    rp = rmap[pts[:, 1].astype(int), pts[:, 0].astype(int)].astype(np.float64)
    p64 = pts.astype(np.float64)
    free = 0
    for i in range(0, len(probes), 1000):
        d = np.hypot(probes[i:i + 1000, None, 0] - p64[None, :, 0],
                     probes[i:i + 1000, None, 1] - p64[None, :, 1])
        free += int((d >= 0.5 * (rc[i:i + 1000, None] + rp[None, :])).all(axis=1).sum())
    check('maximal: at most 1% of random spots could still accept a point',
          free <= 0.01 * len(probes), f'{100.0 * free / len(probes):.2f}% free')

    left = int((pts[:, 0] < 180).sum())
    right = int((pts[:, 0] >= 220).sum())
    ratio = left / max(right, 1)
    check('density follows the map (r=4 left, r=8 right, ideal 4)', 3.4 <= ratio <= 4.6,
          f'ratio {ratio:.2f}')

    # mask
    mask = np.ones((h, w), dtype=bool)
    mask[50:150, 100:300] = False
    masked = tf.poisson_disc_field(w, h, rmap, seed=seed, mask=mask)
    on_false = ~mask[masked[:, 1].astype(int), masked[:, 0].astype(int)]
    check('mask: no point lands on a False cell', len(masked) > 500 and not on_false.any(),
          f'{len(masked)} points')
    check('empty mask gives an empty array',
          tf.poisson_disc_field(w, h, rmap, seed=seed, mask=np.zeros((h, w), dtype=bool)).shape == (0, 2))
    lone = np.zeros((h, w), dtype=bool)
    lone[7, 9] = True
    one = tf.poisson_disc_field(w, h, rmap, seed=seed, mask=lone)
    check('a one-pixel mask still yields exactly its site',
          one.shape == (1, 2) and int(one[0, 0]) == 9 and int(one[0, 1]) == 7)

    # determinism and margin
    again = tf.poisson_disc_field(w, h, rmap, seed=seed, k=30)
    other = tf.poisson_disc_field(w, h, rmap, seed=seed + 1, k=30)
    check('same arguments give the same points', np.array_equal(pts, again))
    check('a different seed gives different points',
          other.shape != pts.shape or not np.array_equal(other, pts))
    margined = tf.poisson_disc_field(120, 80, np.full((80, 120), 6.0), seed=seed, margin=10)
    check('margin is respected',
          len(margined) > 20 and bool(((margined >= 10).all()) and (margined[:, 0] < 110).all()
                                       and (margined[:, 1] < 70).all()))

    # edge cases
    def raises(fn):
        try:
            fn()
        except ValueError:
            return True
        return False
    bad = np.full((h, w), 4.0); bad[3, 3] = 0.0
    check('non-positive radius raises ValueError',
          raises(lambda: tf.poisson_disc_field(w, h, 0.0)) and raises(lambda: tf.poisson_disc_field(w, h, bad))
          and raises(lambda: tf.poisson_disc_field(w, h, -2.0)))
    check('bad sizes, shapes and rules raise ValueError',
          raises(lambda: tf.poisson_disc_field(0, 10, 2.0)) and raises(lambda: tf.poisson_disc_field(10, 10, np.ones((3, 3))))
          and raises(lambda: tf.poisson_disc_field(10, 10, 2.0, rule='median'))
          and raises(lambda: tf.poisson_disc_field(10, 10, 2.0, mask=np.ones((3, 3), dtype=bool))))
    check('a margin beyond half the size gives an empty array',
          tf.poisson_disc_field(40, 40, 3.0, margin=21).shape == (0, 2)
          and tf.poisson_disc_field(40, 40, np.full((40, 40), 3.0), margin=25).shape == (0, 2))
    check('a radius larger than the domain yields exactly one point',
          tf.poisson_disc_field(50, 50, 400.0).shape == (1, 2)
          and tf.poisson_disc_field(50, 50, np.full((50, 50), 400.0)).shape == (1, 2))
    check('a Field is accepted as the radius map',
          np.array_equal(tf.poisson_disc_field(60, 40, tf.Field(np.full((40, 60), 5.0)), seed=seed),
                         tf.poisson_disc_field(60, 40, np.full((40, 60), 5.0, dtype=np.float32), seed=seed)))


# ------------------------------------- vector Delaunay, Voronoi, Lloyd, graphs

def _delaunay_report(tri):
    """Exact structural audit of a triangulation: returns a list of problems."""
    X, Y = tri.lattice
    n, T, hull = len(tri.points), tri.tris.tolist(), tri.hull.tolist()
    problems = []

    def orient(a, b, c):
        return (X[b] - X[a]) * (Y[c] - Y[a]) - (Y[b] - Y[a]) * (X[c] - X[a])

    for a, b, c in T:
        if orient(a, b, c) <= 0:
            problems.append('clockwise or flat triangle')
            break
    strict = 0
    for a, b, c in T:
        for d in range(n):
            if d in (a, b, c):
                continue
            adx, ady, bdx, bdy = X[a] - X[d], Y[a] - Y[d], X[b] - X[d], Y[b] - Y[d]
            cdx, cdy = X[c] - X[d], Y[c] - Y[d]
            det = ((adx * adx + ady * ady) * (bdx * cdy - cdx * bdy)
                   + (bdx * bdx + bdy * bdy) * (cdx * ady - adx * cdy)
                   + (cdx * cdx + cdy * cdy) * (adx * bdy - bdx * ady))
            strict += det > 0
    if strict:
        problems.append(f'{strict} sites strictly inside a circumcircle')
    if len(T) != 2 * n - 2 - len(hull):
        problems.append(f'{len(T)} triangles, expected {2 * n - 2 - len(hull)}')
    if len(tri.edges()) != 3 * n - 3 - len(hull):
        problems.append('edge count off')
    for t, (a, b, c) in enumerate(T):
        for i in range(3):
            u, w = (a, b, c)[(i + 1) % 3], (a, b, c)[(i + 2) % 3]
            nb = int(tri.neighbors[t][i])
            if nb >= 0:
                nv = T[nb]
                if not any((nv[(j + 1) % 3], nv[(j + 2) % 3]) == (w, u) and tri.neighbors[nb][j] == t
                           for j in range(3)):
                    problems.append('asymmetric neighbours')
                    return problems
            elif not (u in hull and w in hull):
                problems.append('boundary edge off the hull')
                return problems
    area2 = sum(orient(a, b, c) for a, b, c in T)
    hull_area2 = sum(X[hull[i]] * Y[hull[(i + 1) % len(hull)]] - X[hull[(i + 1) % len(hull)]] * Y[hull[i]]
                     for i in range(len(hull)))
    if area2 != hull_area2:
        problems.append('triangle areas do not tile the hull')
    if len({v for t in T for v in t}) != n:
        problems.append('a site is not a vertex')
    return problems


def _triangle_set(tri):
    return {tuple(sorted(tuple(tri.points[v]) for v in t)) for t in tri.tris.tolist()}


def test_vector_delaunay():
    print("\nvector Delaunay triangulation")
    rng = np.random.default_rng(tf.seed_stream(31, 'test.vector.delaunay'))

    # general position: empty circumcircle, checked with floats independently of the predicates
    worst = 0.0
    for n in (3, 4, 17, 120, 300):
        P = rng.random((n, 2)) * 200.0 + 40.0
        tri = tv.delaunay(P)
        a, b, c = (tri.points[tri.tris[:, i]] for i in range(3))
        cc = tri.circumcenters()
        radius = np.hypot(*(a - cc).T)
        gap = np.hypot(tri.points[None, :, 0] - cc[:, None, 0], tri.points[None, :, 1] - cc[:, None, 1]) \
            - radius[:, None]
        for t, row in enumerate(tri.tris):
            gap[t, row] = np.inf
        worst = min(worst, float(gap.min()))
    check('no site lies strictly inside any circumcircle (float audit)', worst > -1e-7, f'worst {worst:.2e}')

    bad = []
    for n in (3, 5, 40, 250):
        P = rng.random((n, 2)) * 90.0
        tri = tv.delaunay(P)
        problems = _delaunay_report(tri)
        if problems:
            bad.append(f'n={n}: {problems[0]}')
    check('triangles CCW, counts 2n-2-h and 3n-3-h, neighbours symmetric, areas tile the hull',
          not bad, '; '.join(bad))
    tri = tv.delaunay(rng.random((60, 2)))
    h, nv = len(tri.hull), len(tri.points)
    euler = nv - len(tri.edges()) + (len(tri.tris) + 1)
    check('Euler V - E + F = 2 with the outer face', euler == 2, f'{euler}')
    check('canonical output: rows rotated to the smallest index and sorted',
          bool((tri.tris[:, 0] < tri.tris[:, 1]).all() and (tri.tris[:, 0] < tri.tris[:, 2]).all()
               and sorted(map(tuple, tri.tris.tolist())) == list(map(tuple, tri.tris.tolist()))))

    # independence from input order, insertion order and seed
    P = rng.random((80, 2)) * 10.0
    base = tv.delaunay(P)
    same_seed = [tv.delaunay(P, seed_name=name) for name in ('a', ('w', 'b', 3), ('w', 'c'))]
    check('insertion order and seed do not change the result',
          all(np.array_equal(t.tris, base.tris) and np.array_equal(t.neighbors, base.neighbors)
              and np.array_equal(t.hull, base.hull) for t in same_seed))
    perm = rng.permutation(len(P))
    check('permuting the input gives the same triangulation',
          _triangle_set(tv.delaunay(P[perm])) == _triangle_set(base))
    moved = tv.delaunay(P * 37.5 + np.array([1234.5, -987.25]))
    check('scaling and translating leaves the combinatorial triangulation unchanged',
          np.array_equal(moved.tris, base.tris) and np.array_equal(moved.hull, base.hull))

    # degenerate and cocircular inputs
    lattice = np.array([(x, y) for x in range(9) for y in range(7)], dtype=np.float64)
    grid = tv.delaunay(lattice)
    shuffled = [tv.delaunay(lattice, seed_name=('g', i)) for i in range(4)]
    check('regular grid (many cocircular quadruples) is valid and deterministic',
          not _delaunay_report(grid) and len(grid.tris) == 2 * 8 * 6
          and all(np.array_equal(g.tris, grid.tris) for g in shuffled)
          and _triangle_set(tv.delaunay(lattice[rng.permutation(len(lattice))])) == _triangle_set(grid))
    theta = 2 * np.pi * np.arange(24) / 24
    ring = np.stack([np.cos(theta), np.sin(theta)], axis=1) * 500.0
    circ = tv.delaunay(ring)
    check('24 cocircular points: a valid fan, identical for every insertion order',
          not _delaunay_report(circ) and len(circ.tris) == 22
          and all(np.array_equal(tv.delaunay(ring, seed_name=('c', i)).tris, circ.tris) for i in range(4)))
    rows = np.array([(x, y) for y in (0, 1, 3) for x in range(5)], dtype=np.float64)
    check('collinear runs on the hull stay boundary vertices',
          not _delaunay_report(tv.delaunay(rows)) and len(tv.delaunay(rows).hull) == 12)
    random_lattice = []
    for _ in range(25):
        pool = [(x, y) for x in range(6) for y in range(6)]
        pick = rng.choice(len(pool), size=int(rng.integers(5, 30)), replace=False)
        pts = np.array([pool[i] for i in pick], dtype=np.float64)
        t1 = tv.delaunay(pts)
        if not t1.degenerate:
            random_lattice.append(not _delaunay_report(t1) and
                                  _triangle_set(t1) == _triangle_set(tv.delaunay(pts[::-1], seed_name='r')))
    check('25 random lattice subsets are valid and order independent',
          len(random_lattice) >= 20 and all(random_lattice), f'{len(random_lattice)} non-collinear sets')

    line = np.array([(0, 0), (3, 3), (1, 1), (2, 2), (5, 5)], dtype=np.float64)
    flat = tv.delaunay(line)
    check('collinear input: no triangles, hull is the sorted extent',
          flat.degenerate and flat.hull.tolist() == [0, 2, 3, 1, 4] and flat.tris.shape == (0, 3)
          and flat.edges().tolist() == [[0, 2], [2, 3], [1, 3], [1, 4]])
    for n in (0, 1, 2):
        t = tv.delaunay(rng.random((n, 2)))
        check(f'{n} points: no triangles', t.tris.shape == (0, 3) and len(t.points) == n)
    dup = np.array([(0, 0), (1, 0), (0, 1), (1, 0), (0, 0), (1, 1), (1 + 4e-10, 1e-10)], dtype=np.float64)
    td = tv.delaunay(dup)
    check('duplicates within 1e-9 merge into their first occurrence',
          len(td.points) == 4 and td.index_map.tolist() == [0, 1, 2, 1, 0, 3, 1] and not _delaunay_report(td))
    check('non-finite or mis-shaped input raises ValueError',
          all(_raises_value_error(fn) for fn in (lambda: tv.delaunay([[0.0, float('nan')]]),
                                                 lambda: tv.delaunay(np.zeros((3, 3))))))

    t0 = time.time()
    big = tv.delaunay(np.random.default_rng(5).random((5000, 2)) * 1000.0, seed_name='speed')
    elapsed = time.time() - t0
    check('5000 points triangulate within the 5 s budget',
          len(big.tris) > 9000 and elapsed < 5.0, f'{elapsed:.2f}s')


def _raises_value_error(fn):
    try:
        fn()
    except ValueError:
        return True
    return False


def _cell_area(poly):
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float((x * np.roll(y, -1) - np.roll(x, -1) * y).sum())


def _in_convex(poly, p, slack=1e-9):
    nxt = np.roll(poly, -1, axis=0)
    cross = (nxt[:, 0] - poly[:, 0]) * (p[1] - poly[:, 1]) - (nxt[:, 1] - poly[:, 1]) * (p[0] - poly[:, 0])
    return bool((cross >= -slack).all())


def _quantisation_energy(sites, box, n=240):
    xs = box[0] + (np.arange(n) + 0.5) * (box[2] - box[0]) / n
    ys = box[1] + (np.arange(n) + 0.5) * (box[3] - box[1]) / n
    gx, gy = np.meshgrid(xs, ys)
    pix = np.stack([gx.ravel(), gy.ravel()], axis=1)
    d2 = ((pix[:, None, :] - sites[None, :, :]) ** 2).sum(axis=2).min(axis=1)
    return float(d2.mean() * (box[2] - box[0]) * (box[3] - box[1]))


def test_vector_voronoi_lloyd():
    print("\nvector Voronoi cells and Lloyd relaxation")
    rng = np.random.default_rng(tf.seed_stream(32, 'test.vector.voronoi'))
    box = (0.0, 0.0, 64.0, 48.0)
    area = 64.0 * 48.0

    P = rng.random((45, 2)) * np.array([64.0, 48.0])
    tri = tv.delaunay(P)
    cells = tv.voronoi_cells(tri, box)
    total = sum(_cell_area(c) for c in cells if len(c))
    check('cells partition the box (areas sum to the box area)', abs(total - area) <= 1e-9 * area,
          f'{total:.9f} vs {area}')
    check('every cell is convex, counter-clockwise, and holds its own site',
          all(len(c) >= 3 and _cell_area(c) > 0 and _in_convex(c, tri.points[i]) for i, c in enumerate(cells)))
    probes = rng.random((1000, 2)) * np.array([64.0, 48.0])
    wrong = 0
    for p in probes:
        nearest = int(np.argmin(((tri.points - p) ** 2).sum(axis=1)))
        wrong += not _in_convex(cells[nearest], p)
    check('1000 probes: the nearest site owns the containing cell', wrong == 0, f'{wrong} wrong')

    # agreement with the raster Voronoi on integer pixel coordinates
    sites = tf.poisson_disc(64, 64, 9, seed=tf.seed_stream(32, 'test.vector.raster'))
    raster = tf.voronoi(64, 64, sites)
    vtri = tv.delaunay(sites.astype(np.float64))
    vcells = tv.voronoi_cells(vtri, (-0.5, -0.5, 63.5, 63.5))
    yy, xx = np.mgrid[0:64, 0:64]
    label = -np.ones((64, 64), dtype=np.int32)
    for k, c in enumerate(vcells):
        inside = np.ones((64, 64), dtype=bool)
        for (ax, ay), (bx, by) in zip(c, np.roll(c, -1, axis=0)):
            inside &= ((bx - ax) * (yy - ay) - (by - ay) * (xx - ax)) >= 0
        label[inside & (label < 0)] = k
    clear = raster.border_tension.a > 2e-6
    check('matches the raster tf.voronoi on a 64x64 grid away from borders',
          bool((label[clear] == raster.cell[clear]).all()) and not (label < 0).any(),
          f'{int((label[clear] != raster.cell[clear]).sum())} of {int(clear.sum())} differ')

    # edge cases
    one = tv.voronoi_cells(tv.delaunay([[10.0, 10.0]]), box)
    check('one site owns the whole box', len(one) == 1 and abs(_cell_area(one[0]) - area) < 1e-9)
    line = tv.delaunay([[8.0, 24.0], [32.0, 24.0], [56.0, 24.0]])
    strips = tv.voronoi_cells(line, box)
    check('collinear sites get strips between bisectors',
          [round(_cell_area(c), 6) for c in strips] == [20.0 * 48, 24.0 * 48, 20.0 * 48]
          and abs(strips[0][:, 0].max() - 20.0) < 1e-12)
    outside = tv.delaunay([[10.0, 10.0], [50.0, 10.0], [30.0, 40.0], [30.0, 500.0], [-400.0, 20.0]])
    far = tv.voronoi_cells(outside, box)
    check('sites outside the bounds take part; the cells still partition the box',
          len(far[3]) == 0 and len(far[4]) == 0 and abs(sum(_cell_area(c) for c in far if len(c)) - area) < 1e-9)
    near = tv.voronoi_cells(tv.delaunay([[10.0, 10.0], [30.0, 10.0], [20.0, 70.0]]), box)
    check('a site just outside the box owns the part of it nearest to it',
          len(near[2]) >= 3 and abs(sum(_cell_area(c) for c in near if len(c)) - area) < 1e-9)
    check('zero-area bounds raise ValueError',
          _raises_value_error(lambda: tv.voronoi_cells(tri, (0, 0, 0, 5)))
          and _raises_value_error(lambda: tv.lloyd_relax_vector(P, (1, 1, 1, 9))))

    # Lloyd relaxation
    big = (0.0, 0.0, 100.0, 100.0)
    cur = rng.random((30, 2)) * 100.0
    energies = [_quantisation_energy(cur, big)]
    for _ in range(6):
        cur = tv.lloyd_relax_vector(cur, big, iterations=1)
        energies.append(_quantisation_energy(cur, big))
    check('quantisation energy never increases and drops by 2x',
          all(b <= a * (1 + 1e-4) for a, b in zip(energies, energies[1:])) and energies[-1] < 0.6 * energies[0],
          ' -> '.join(f'{e:.0f}' for e in energies[::2]))
    lattice = np.array([((i + 0.5) * 20.0, (j + 0.5) * 20.0) for i in range(5) for j in range(5)])
    check('a lattice of cell centres is a fixed point',
          float(np.abs(tv.lloyd_relax_vector(lattice, big, iterations=3) - lattice).max()) < 1e-9)
    quad = tv.lloyd_relax_vector(np.array([[20.0, 35.0], [80.0, 35.0], [20.0, 65.0], [80.0, 65.0]]), big, 3)
    check('a symmetric 2x2 configuration stays symmetric',
          np.allclose(np.sort(quad[:, 0]), np.sort(100.0 - quad[:, 0]), atol=1e-9)
          and np.allclose(np.sort(quad[:, 1]), np.sort(100.0 - quad[:, 1]), atol=1e-9), str(quad.round(3).tolist()))
    scatter = rng.random((40, 2)) * 100.0
    plain = tv.lloyd_relax_vector(scatter, big, iterations=12)
    left = np.zeros((100, 100))
    left[:, :50] = 1.0
    left[:, 50:] = 0.002
    pulled = tv.lloyd_relax_vector(scatter, big, iterations=12, density=left)
    check('density concentrated on the left pulls the mean x of the sites left',
          pulled[:, 0].mean() < plain[:, 0].mean() - 2.0 and (pulled[:, 0] < 50).sum() > (plain[:, 0] < 50).sum(),
          f'{plain[:, 0].mean():.1f} -> {pulled[:, 0].mean():.1f}')
    flat = tv.lloyd_relax_vector(scatter, big, iterations=1, density=np.full((400, 400), 3.0))
    exact = tv.lloyd_relax_vector(scatter, big, iterations=1)
    check('constant density reproduces the unweighted result',
          float(np.abs(flat - exact).max()) < 0.35, f'max shift {float(np.abs(flat - exact).max()):.3f}')
    dups = np.vstack([scatter[:5], scatter[:2]])
    out = tv.lloyd_relax_vector(dups, big, iterations=2)
    check('merged duplicates relax together; sites stay inside the bounds',
          out.shape == (7, 2) and np.array_equal(out[5], out[0]) and np.array_equal(out[6], out[1])
          and bool(((out >= 0) & (out <= 100)).all()))
    check('clamps sites that start outside the bounds',
          bool(((tv.lloyd_relax_vector([[-50.0, 20.0], [150.0, 80.0], [50.0, 50.0]], big, 2) >= 0).all())))
    check('zero iterations return the input; empty input is fine',
          np.array_equal(tv.lloyd_relax_vector(scatter, big, 0), scatter)
          and tv.lloyd_relax_vector(np.zeros((0, 2)), big).shape == (0, 2))


def test_vector_proximity_graphs():
    print("\nproximity graphs")
    rng = np.random.default_rng(tf.seed_stream(33, 'test.vector.graphs'))

    def pairs(a):
        return {tuple(e) for e in a.tolist()}

    def brute(tri):
        X, Y = tri.lattice
        n = len(X)
        gabriel, rng_edges = set(), set()
        for i in range(n):
            for j in range(i + 1, n):
                length = (X[i] - X[j]) ** 2 + (Y[i] - Y[j]) ** 2
                g = r = True
                for k in range(n):
                    if k == i or k == j:
                        continue
                    a = (X[i] - X[k]) ** 2 + (Y[i] - Y[k]) ** 2
                    b = (X[j] - X[k]) ** 2 + (Y[j] - Y[k]) ** 2
                    g &= a + b > length
                    r &= not (a < length and b < length)
                if g:
                    gabriel.add((i, j))
                if r:
                    rng_edges.add((i, j))
        return gabriel, rng_edges

    chain, sizes = [], []
    for n in (50, 120, 300, 500):
        tri = tv.delaunay(rng.random((n, 2)) * 100.0)
        g = {k: pairs(v) for k, v in tv.proximity_graphs(tri).items()}
        chain.append(g['mst'] <= g['rng'] <= g['gabriel'] <= g['delaunay'] and len(g['mst']) == n - 1)
        sizes.append(len(g['rng']) / n)
    check('MST <= RNG <= Gabriel <= Delaunay on random sets of 50-500 points; MST has n-1 edges',
          all(chain), f'RNG edges per site {sizes[-1]:.2f}')

    exact = []
    for trial in range(40):
        kind = trial % 4
        if kind == 0:
            P = rng.random((int(rng.integers(5, 60)), 2)) * 50.0
        elif kind == 1:
            pool = [(x, y) for x in range(7) for y in range(7)]
            P = np.array([pool[i] for i in rng.choice(49, size=int(rng.integers(5, 35)), replace=False)], dtype=float)
        elif kind == 2:
            m = int(rng.integers(4, 9)) * 2
            th = 2 * np.pi * np.arange(m) / m
            P = np.concatenate([np.stack([np.cos(th), np.sin(th)], 1), 2.0 * np.stack([np.cos(th), np.sin(th)], 1)])
        else:
            P = np.array([(x, y) for y in (0, 1, 3) for x in range(int(rng.integers(2, 6)))], dtype=float)
        tri = tv.delaunay(P)
        if tri.degenerate:
            continue
        g = {k: pairs(v) for k, v in tv.proximity_graphs(tri).items()}
        gabriel, rng_edges = brute(tri)
        exact.append(g['gabriel'] == gabriel and g['rng'] == rng_edges
                     and g['mst'] <= g['rng'] <= g['gabriel'] <= g['delaunay'])
    check('Gabriel and RNG equal brute force on random, lattice, cocircular and row inputs',
          len(exact) >= 35 and all(exact), f'{len(exact)} sets')

    # the lune can hold a site that neighbours neither endpoint, so testing only the
    # Delaunay neighbours of p and q keeps edges the definition rejects (seed index 2)
    P = np.random.default_rng(tf.seed_stream(33, 'test.vector.lune', 2)).random((300, 2))
    tri = tv.delaunay(P)
    D = ((P[:, None, :] - P[None, :, :]) ** 2).sum(axis=2)
    keep = np.zeros((300, 300), dtype=bool)
    for i in range(300):
        lune = (D[i][None, :] < D[i][:, None]) & (D < D[i][:, None])
        keep[i] = ~lune.any(axis=1)
    keep &= ~np.eye(300, dtype=bool)
    expect = {(i, j) for i in range(300) for j in range(i + 1, 300) if keep[i, j] and keep[j, i]}
    graphs = {k: pairs(v) for k, v in tv.proximity_graphs(tri).items()}
    nb = [set(v) for v in tri.vertex_neighbors()]
    shortcut = {(i, j) for (i, j) in graphs['gabriel']
                if not any(D[i, r] < D[i, j] and D[j, r] < D[i, j] for r in (nb[i] | nb[j]) - {i, j})}
    check('RNG of 300 points equals the all-pairs definition', graphs['rng'] == expect, f'{len(expect)} edges')
    check('the neighbours-only shortcut would have kept extra edges here', shortcut > expect,
          f'{len(shortcut) - len(expect)} extra')

    mst_ok = []
    for n in (12, 40, 80):
        P = rng.random((n, 2)) * 30.0
        tri = tv.delaunay(P)
        edges = tv.proximity_graphs(tri)['mst']
        total = float(np.hypot(*(tri.points[edges[:, 0]] - tri.points[edges[:, 1]]).T).sum())
        dist = np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1])
        seen, best, prim = {0}, dist[0].copy(), 0.0
        for _ in range(n - 1):
            best[list(seen)] = np.inf
            j = int(best.argmin())
            prim += float(best[j])
            seen.add(j)
            best = np.minimum(best, dist[j])
        mst_ok.append(abs(total - prim) <= 1e-6 * prim)
    check('MST length equals Prim on the complete graph (n = 12, 40, 80)', all(mst_ok))

    line = tv.proximity_graphs(tv.delaunay([[0.0, 0.0], [2.0, 2.0], [1.0, 1.0], [5.0, 5.0]]))
    check('collinear sites: every graph is the chain', all(pairs(v) == {(0, 2), (1, 2), (1, 3)} for v in line.values()))
    none = tv.proximity_graphs(tv.delaunay([[1.0, 2.0]]))
    check('a single site has no edges', all(v.shape == (0, 2) for v in none.values()))
    g = tv.proximity_graphs(tv.delaunay(np.array([(x, y) for x in range(4) for y in range(4)], dtype=float)))
    check('on a grid the diagonals fall out of Gabriel and RNG, the lattice edges stay',
          len(g['gabriel']) == 24 and len(g['rng']) == 24 and len(g['mst']) == 15, str({k: len(v) for k, v in g.items()}))


# ------------------------------------------------------ Prim-order joins

def _joined(mask, strategy, width=1):
    """Run connect_components on a bool mask; return (tilemap, carved cells, components)."""
    h, w = mask.shape
    tm = ts.TileMap(w, h, default_base='wall')
    for y, x in zip(*np.nonzero(mask)):
        tm.base[y][x] = 'floor'
    before = int(tg.passable_mask(tm).sum())
    tg.connect_components(tm, tg.passable_mask(tm), 'floor', width=width, min_component=1,
                          **({'strategy': strategy} if strategy else {}))
    after = tg.passable_mask(tm)
    return tm, int(after.sum()) - before, tf.connected_components(after)[1]


def test_prim_join():
    print("\nPrim-order component joins")
    fixture = np.zeros((16, 30), dtype=bool)
    fixture[0:10, 0:10] = True          # main region
    fixture[0:5, 13:16] = True          # 3 cells of wall away from it
    fixture[6:10, 19:22] = True         # near the second region, far from the main one
    _, star_cells, star_n = _joined(fixture, 'star')
    _, prim_cells, prim_n = _joined(fixture, 'prim')
    check('chain fixture: prim hops through the middle region (8 cells), star runs both home (12)',
          (star_cells, prim_cells, star_n, prim_n) == (12, 8, 1, 1),
          f'star {star_cells}, prim {prim_cells}')
    plan = tg.prim_join_plan(tf.connected_components(fixture)[0], 1)
    check('prim_join_plan lists the closest join first and ends at the farthest region',
          len(plan) == 2 and plan[0][0] in range(13, 16) and plan[1][0] in range(19, 22), str(plan))

    blobs = []
    for seed in range(1, 11):
        f = tf.fbm(96, 64, 6.0, 4, seed=seed).a
        mask = f > np.quantile(f, 0.62)
        if tf.connected_components(mask)[1] >= 4:
            blobs.append(mask)
    star_total = prim_total = 0
    wins = single = 0
    for mask in blobs:
        _, a, na = _joined(mask, 'star')
        tm, b, nb = _joined(mask, 'prim')
        star_total += a
        prim_total += b
        wins += b <= a
        single += na == 1 and nb == 1
    check('prim joins every region into one 4-connected component', blobs and single == len(blobs),
          f'{single}/{len(blobs)} maps')
    check('prim never carves more than star on noise blobs and saves at least 40% overall',
          wins == len(blobs) and prim_total <= 0.6 * star_total,
          f'star {star_total} -> prim {prim_total} cells over {len(blobs)} maps')
    tm1, _, _ = _joined(blobs[0], 'prim')
    tm2, _, _ = _joined(blobs[0], 'prim')
    check('prim is deterministic', tm1.base == tm2.base)
    tm_default, _, _ = _joined(blobs[0], None)
    tm_star, _, _ = _joined(blobs[0], 'star')
    check("the default strategy is 'star' and unchanged", tm_default.base == tm_star.base)
    check('an unknown strategy raises ValueError',
          _raises_value_error(lambda: _joined(blobs[0], 'mst')))

    _, thin, _ = _joined(blobs[1], 'prim', width=1)
    _, wide, n_wide = _joined(blobs[1], 'prim', width=3)
    check('wide corridors still join everything, and carve more cells', n_wide == 1 and wide > thin,
          f'{thin} -> {wide} cells')
    check('a mask that is already one region needs no corridor',
          _joined(np.pad(np.ones((5, 5), dtype=bool), 3), 'prim')[1:] == (0, 1))

    # archipelago bridges hop between islands when asked
    saved, same = 0, True
    for seed in range(3, 13):
        counts = {}
        for strategy in ('star', 'prim'):
            spec = {'id': 5, 'size': 96, 'template': 'archipelago', 'seed': seed, 'sea_level': 0.5,
                    'scale': 3.4, 'min_island': 8, 'bridge_strategy': strategy}
            amap = tg.GRAMMARS['archipelago'](spec, random.Random(seed))
            counts[strategy] = (int((np.array(amap.base, dtype=object) == 'bridge').sum()),
                                tf.connected_components(tg.passable_mask(amap))[1])
        same &= counts['prim'][1] == 1 and counts['prim'][0] <= counts['star'][0]
        saved += counts['star'][0] - counts['prim'][0]
    check("archipelago bridge_strategy='prim': one component, never more bridge tiles than star",
          same and saved > 20, f'{saved} bridge tiles saved over 10 seeds')


# --------------------------------------------------------- cellular caves

def _ca_reference(walls, passes=5):
    """Plain-loop neighbour-count rule (synchronous), the oracle for tf.ca_smooth."""
    cur = [[bool(v) for v in row] for row in walls]
    h, w = len(cur), len(cur[0])
    for _ in range(passes):
        nxt = [row[:] for row in cur]
        for y in range(h):
            for x in range(w):
                n = 0
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dx or dy:
                            yy, xx = y + dy, x + dx
                            n += cur[yy][xx] if (0 <= yy < h and 0 <= xx < w) else 1
                if n > 4:
                    nxt[y][x] = True
                elif n < 4:
                    nxt[y][x] = False
        cur = nxt
    return np.array(cur)


def test_ca_caves():
    print("\ncellular cave family")
    rng = np.random.default_rng(tf.seed_stream(41, 'test.ca'))

    corner = tf.neighbour_count(np.zeros((4, 5), dtype=bool))
    check('outside the map counts as wall: a corner of open floor sees 5, an edge 3, the middle 0',
          corner[0, 0] == 5 and corner[0, 2] == 3 and corner[2, 2] == 0
          and int(tf.neighbour_count(np.zeros((4, 5), dtype=bool), outside=False).max()) == 0)
    walls = rng.random((18, 23)) < 0.47
    check('ca_smooth equals the plain-loop rule (wall > 4, floor < 4, unchanged at 4)',
          all(np.array_equal(tf.ca_smooth(walls, passes), _ca_reference(walls, passes)) for passes in (1, 2, 5)))
    lone = np.ones((7, 7), dtype=bool)
    lone[3, 3] = False
    check('an isolated floor cell fills in; a solid wall border never opens',
          bool(tf.ca_smooth(lone, 1).all()) and bool(tf.ca_smooth(np.ones((5, 5), dtype=bool), 3).all()))
    diag = np.zeros((6, 6), dtype=bool)
    diag[1, 1] = diag[2, 2] = True
    diag[4:6, 0:3] = True
    kept = tf.remove_small_regions(diag, 3)
    check('remove_small_regions is 4-connected: diagonal specks are separate regions',
          not kept[1, 1] and not kept[2, 2] and kept[4:6, 0:3].all() and tf.remove_small_regions(diag, 1).sum() == diag.sum())

    border_ok, ratios, comps, det, varies = [], [], [], [], []
    for (w, h) in ((128, 72), (64, 64), (96, 60)):
        for seed in (3, 11, 29):
            spec = {'id': 1, 'size': max(w, h), 'width': w, 'height': h, 'template': 'ca_caves', 'seed': seed}
            tm = tg.GRAMMARS['ca_caves'](spec, random.Random(seed))
            again = tg.GRAMMARS['ca_caves'](spec, random.Random(seed))
            other = tg.GRAMMARS['ca_caves'](dict(spec, seed=seed + 1), random.Random(seed + 1))
            pm = tg.passable_mask(tm)
            border = np.zeros((h, w), dtype=bool)
            border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
            border_ok.append(not pm[border].any() and (tm.w, tm.h) == (w, h))
            ratios.append(float(pm.mean()))
            comps.append(tf.connected_components(pm)[1])
            det.append(tm.base == again.base and tm.overlay == again.overlay)
            varies.append(tm.base != other.base)
    check('border cells are walls and the map has the requested size', all(border_ok))
    check('open ratio lands within 0.04 of the 0.5 target', all(abs(r - 0.5) <= 0.04 for r in ratios),
          f'{min(ratios):.3f}..{max(ratios):.3f}')
    check('one 4-connected component after the joins', all(c == 1 for c in comps))
    check('deterministic per seed, different between seeds', all(det) and all(varies))
    targets = []
    for target in (0.35, 0.65):
        spec = {'id': 1, 'size': 96, 'template': 'ca_caves', 'seed': 7, 'open_ratio': target}
        pm = tg.passable_mask(tg.GRAMMARS['ca_caves'](spec, random.Random(7)))
        targets.append((target, round(float(pm.mean()), 3)))
    check('the initial fill is tuned to other targets too', all(abs(t - got) <= 0.05 for t, got in targets),
          str(targets))

    # region cleanup on the raw automaton (before the joins)
    smallest = []
    for seed in range(1, 7):
        u = np.random.default_rng(tf.seed_stream(seed, 'geo.ca_caves')).random((72, 128))
        opened = tilegen_grammars2.ca_cave_mask(u)
        open_sizes = np.bincount(tf.connected_components(opened)[0].ravel())[1:]
        wall_sizes = np.bincount(tf.connected_components(~opened)[0].ravel())[1:]
        smallest.append((int(open_sizes.min()), int(wall_sizes.min())))
    check('every room and every wall mass has at least 50 cells before joining',
          all(min(pair) >= 50 for pair in smallest), f'smallest (room, wall) {min(smallest)}')
    u = np.random.default_rng(tf.seed_stream(5, 'geo.ca_caves')).random((72, 128))
    f45 = tilegen_grammars2.ca_cave_mask(u, fill=0.45)
    f50 = tilegen_grammars2.ca_cave_mask(u, fill=0.5)
    check('an explicit fill skips the tuning and is deterministic',
          np.array_equal(f45, tilegen_grammars2.ca_cave_mask(u, fill=0.45))
          and abs(float(f45.mean()) - float(f50.mean())) > 0.05)

    # Prim-ordered joins carve far fewer cells than the star topology on the same rooms
    star_cells = prim_cells = 0
    per_seed, room_counts = [], []
    for seed in range(1, 9):
        spec = {'id': 1, 'size': 128, 'width': 128, 'height': 72, 'template': 'ca_caves', 'seed': seed,
                'fill': 0.55, 'min_room': 12, 'min_wall_region': 12}
        u = np.random.default_rng(tf.seed_stream(seed, 'geo.ca_caves')).random((72, 128))
        before = tilegen_grammars2.ca_cave_mask(u, fill=0.55, min_wall=12, min_room=12)
        room_counts.append(tf.connected_components(before)[1])
        a = int(tg.passable_mask(tg.GRAMMARS['ca_caves'](dict(spec, join='star'), random.Random(seed))).sum()) - int(before.sum())
        b = int(tg.passable_mask(tg.GRAMMARS['ca_caves'](dict(spec, join='prim'), random.Random(seed))).sum()) - int(before.sum())
        per_seed.append(b <= a)
        star_cells += a
        prim_cells += b
    check('prim joins carve at most as many cells as star on every cave and at most half overall',
          min(room_counts) >= 3 and all(per_seed) and prim_cells <= 0.5 * star_cells,
          f'{star_cells} -> {prim_cells} carved cells over rooms {min(room_counts)}-{max(room_counts)}')

    # hybrid source and recipe integration
    hybrid = tg.gen_hybrid_morph({'id': 2, 'size': 64, 'template': 'hybrid_morph', 'seed': 4,
                                  'macro': 'ca_caves', 'structure': 'fbm', 'op': 'mul'}, random.Random(4))
    pm = tg.passable_mask(hybrid)
    check("'ca_caves' works as a Hybrid Morph source", tf.connected_components(pm)[1] == 1 and 0.05 < pm.mean() < 0.98,
          f'open {pm.mean():.0%}')
    world = tc.compile_recipe(tc.WorldRecipe(macro='ca_caves', size=64, seed=2024, name='cave test'))
    errors = [f for f in world.findings if f['severity'] == 'error']
    check('a recipe with macro ca_caves compiles with all 40 features',
          not errors and len(world.artifacts['features_run']) == len(tw.FEATURES),
          f'{len(world.places)} places, {len(errors)} errors')
    blended = tc.compile_recipe(tc.WorldRecipe(macro='voronoi_kingdoms', structure='ca_caves', size=64, seed=5))
    check('ca_caves can be blended in as a recipe structure',
          not [f for f in blended.findings if f['severity'] == 'error'])


# ------------------------------------------------ socket WFC: D4 variants

@functools.lru_cache(maxsize=1)
def _shipped_library():
    """The module library of the 100 shipped layouts, as the legacy runtime builds it."""
    records = rt.load_layout_catalog(HERE / 'tilegen_output')
    modules, compat = rt.build_module_library(records)
    return modules, compat


def test_socket_d4():
    print("\nsocket WFC: D4 module variants")
    modules, compat = _shipped_library()
    check('shipped library: 100 modules', len(modules) == 100)

    chiral = [[(3 * x + y * y + (x * y) % 3) % 7 for x in range(5)] for y in range(5)]
    forms = {tuple(map(tuple, se.transform_grid(chiral, q, m))) for q, m in se.ORIENTATIONS}
    closed = all(tuple(map(tuple, se.transform_grid(se.transform_grid(chiral, q1, m1), q2, m2))) in forms
                 for q1, m1 in se.ORIENTATIONS for q2, m2 in se.ORIENTATIONS)
    check('the eight orientations of a chiral grid are distinct and closed under composition',
          len(forms) == 8 and closed)
    check('four quarter turns and a double mirror are the identity',
          se.transform_grid(chiral, 4, False) == chiral
          and se.transform_grid(se.transform_grid(chiral, 0, True), 0, True) == chiral)
    check('a clockwise quarter turn sends the top-left cell to the top-right',
          se.transform_grid([[1, 2], [3, 4]], 1) == [[3, 1], [4, 2]])

    same = all(se.module_sockets(m['largest_component_mask'])[k] == m[k]
               for m in modules for k in ('side_masks', 'side_slots', 'side_anchor_map', 'side_metrics'))
    check('recomputing sockets from the stored masks reproduces all 400 side masks', same)
    check('compute_compat reproduces the legacy adjacency bit for bit', se.compute_compat(modules) == compat)

    variants, vcompat = se.expand_d4(modules)
    check('100 modules expand to 656 distinct oriented variants', len(variants) == 656, f'{len(variants)}')
    per_source = Counter(v['source_state'] for v in variants)
    check('33 modules are self-symmetric (fewer than 8 variants)',
          sum(1 for c in per_source.values() if c < 8) == 33 and set(per_source) == set(range(100)))
    before = {tuple(m['side_masks'][d] for d in 'NESW') for m in modules}
    after = {tuple(v['side_masks'][d] for d in 'NESW') for v in variants}
    check('distinct (N, E, S, W) socket signatures grow from 36 to 144', (len(before), len(after)) == (36, 144),
          f'{len(before)} -> {len(after)}')
    identity = [v for v in variants if v['orientation'] == 'r0']
    check('the identity orientation is each module unchanged and comes first',
          len(identity) == 100 and all(v['base'] == modules[v['source_state']]['base']
                                       and v['side_masks'] == modules[v['source_state']]['side_masks']
                                       and v['source_render'] == modules[v['source_state']]['source_render']
                                       for v in identity)
          and all(variants[i]['orientation'] == 'r0' for i in
                  [min(j for j, v in enumerate(variants) if v['source_state'] == s) for s in range(100)]))
    check('state indices are consecutive and non-identity variants drop the unrotated source image',
          [v['state_index'] for v in variants] == list(range(656))
          and all(v['source_render'] == '' for v in variants if v['orientation'] != 'r0'))
    full, _ = se.expand_d4(modules, dedupe=False)
    check('without dedupe every module gives exactly eight states', len(full) == 800)

    # sockets follow the geometry: side costs are carried round by the orientation
    def moved(side, quarter, mirrored):
        if mirrored:
            side = {'N': 'N', 'S': 'S', 'E': 'W', 'W': 'E'}[side]
        for _ in range(quarter):
            side = {'N': 'E', 'E': 'S', 'S': 'W', 'W': 'N'}[side]
        return side
    ok = True
    for v in full:
        quarter, mirrored = int(v['orientation'][1:]) // 90, v['orientation'][0] == 'm'
        src = modules[v['source_state']]
        for side in 'NESW':
            old = sorted(e['cost'] for e in src['side_metrics'][side])
            new = sorted(e['cost'] for e in v['side_metrics'][moved(side, quarter, mirrored)])
            ok &= old == new
    check('side depth costs are carried round by every orientation (800 modules x 4 sides)', ok)
    check('variant compat degrees are recomputed',
          all(v['compat_degree'] == sum(rt.bit_count(vcompat[d][v['state_index']]) for d in 'NESW') for v in variants))
    rot_only, _ = se.expand_d4(modules, orientations=se.ORIENTATIONS[:4])
    check('a subset of orientations is honoured and an empty one is refused',
          len(rot_only) < 656 and {v['orientation'] for v in rot_only} == {'r0', 'r90', 'r180', 'r270'}
          and _raises_value_error(lambda: se.expand_d4(modules, orientations=())))

    # the unchanged legacy solver and assembler work on the bigger library
    sizes_ok, valid, built = [], [], []
    for grid, seed in ((8, 3), (10, 4), (12, 5)):
        sol = rt.solve_socket_wfc(variants, vcompat, grid, grid, seed=seed, max_restarts=32)
        sizes_ok.append(sol['success'] and sol['attempts'] == 1)
        valid.append(sol['success'] and se.check_placements(vcompat, grid, grid, sol['placements']))
        if sol['success']:
            tmap, _, connectors = rt.assemble_world(variants, sol['placements'], grid, grid, 2)
            built.append((tmap.w, tmap.h) == (grid * 16 + (grid - 1) * 2,) * 2 and len(connectors) == 2 * grid * (grid - 1))
    check('the legacy solver fills 8x8, 10x10 and 12x12 grids on attempt 1', all(sizes_ok), str(sizes_ok))
    check('every neighbouring pair is allowed and the legacy assembler builds the worlds', all(valid) and all(built))
    sol = rt.solve_socket_wfc(variants, vcompat, 6, 6, seed=11, max_restarts=24)
    sources = {variants[s]['source_state'] for s in sol['placements']}
    oriented = len(set(sol['placements']))
    check('orientations hide repeats: more distinct tiles than distinct source layouts',
          oriented >= len(sources), f'{oriented} oriented states from {len(sources)} layouts')


# ------------------------------------------------ socket WFC: solver hooks

def test_socket_hooks():
    print("\nsocket WFC: solver hooks")
    modules, compat = _shipped_library()

    same = []
    for preset in rt.WORLD_PRESETS:
        legacy = rt.solve_socket_wfc(modules, compat, preset['grid_w'], preset['grid_h'], seed=preset['seed'],
                                     unique_states=preset['unique_states'], max_restarts=preset['max_restarts'])
        ext = se.solve_preset(modules, compat, preset)
        same.append((legacy['success'], legacy['attempts'], legacy['placements'])
                    == (ext['success'], ext['attempts'], ext['placements']))
    for seed in range(12):
        for grid_w, grid_h, unique in ((6, 6, False), (5, 7, False), (4, 4, True)):
            a = rt.solve_socket_wfc(modules, compat, grid_w, grid_h, seed=seed, unique_states=unique, max_restarts=40)
            b = se.solve_socket_wfc_ext(modules, compat, grid_w, grid_h, seed=seed, unique_states=unique,
                                        max_restarts=40)
            same.append((a['success'], a['attempts'], a['placements']) == (b['success'], b['attempts'], b['placements']))
    wbias = [rt.solve_socket_wfc(modules, compat, 6, 6, seed=s, max_restarts=30)['placements']
             == se.solve_socket_wfc_ext(modules, compat, 6, 6, seed=s, max_restarts=30,
                                        weight=se.default_weight)['placements'] for s in range(6)]
    check('with no hook the extended solver makes the legacy draws (presets and 36 seeded runs)',
          all(same), f'{sum(same)}/{len(same)} identical')
    check('weight=default_weight reproduces the legacy choice exactly', all(wbias))

    # initial domains
    sparse = se.states_where(modules, lambda m: m['open_ratio'] < 0.1)
    border = se.border_cells(8, 8)
    run = se.solve_socket_wfc_ext(modules, compat, 8, 8, seed=23, domains=se.ban_states(modules, border, sparse))
    check('a banned set stays off the border (24 near-empty modules, 28 border cells)',
          run['success'] and not sparse >> run['placements'][0] & 1
          and all(not sparse >> run['placements'][y * 8 + x] & 1 for x, y in border)
          and se.check_placements(compat, 8, 8, run['placements']))
    keepers = se.states_where(modules, lambda m: m['theme'] in ('dungeon', 'fortress', 'castle'))
    centre = [(x, y) for y in range(2, 6) for x in range(2, 6)]
    rule = se.allow_states(centre, keepers)
    as_dict = se.solve_socket_wfc_ext(modules, compat, 8, 8, seed=23, domains=rule)
    as_list = se.solve_socket_wfc_ext(modules, compat, 8, 8, seed=23,
                                      domains=[rule.get((i % 8, i // 8), se.all_states(modules)) for i in range(64)])
    as_call = se.solve_socket_wfc_ext(modules, compat, 8, 8, seed=23, domains=lambda x, y: rule.get((x, y)))
    check('a region restricted to a theme set holds only those states, however the rule is given',
          as_dict['success'] and all(keepers >> as_dict['placements'][y * 8 + x] & 1 for x, y in centre)
          and as_dict['placements'] == as_list['placements'] == as_call['placements'])
    merged = se.merge_domains(se.allow_states([(0, 0)], 0b111), se.allow_states([(0, 0), (1, 1)], 0b110))
    check('merge_domains intersects rules that name the same cell', merged == {(0, 0): 0b110, (1, 1): 0b110})
    dead = se.solve_socket_wfc_ext(modules, compat, 4, 4, seed=1, domains=se.allow_states([(1, 1)], 0), max_restarts=3)
    check('an emptied domain fails every attempt with reason domains',
          not dead['success'] and dead['failures'] == ['domains'] * 3)

    # pinned cells
    start = min((i for i, m in enumerate(modules) if m['open_ratio'] > 0.2), key=lambda i: modules[i]['open_ratio'])
    boss = max(range(len(modules)), key=lambda i: modules[i]['open_ratio'])
    pinned = se.solve_socket_wfc_ext(modules, compat, 8, 8, seed=23, fixed={(0, 0): start, (4, 4): boss})
    as_tuples = se.solve_socket_wfc_ext(modules, compat, 8, 8, seed=23, fixed=[(0, 0, start), (4, 4, boss)])
    check('pinned cells keep their module and the grid stays valid',
          pinned['success'] and pinned['placements'][0] == start and pinned['placements'][4 * 8 + 4] == boss
          and se.check_placements(compat, 8, 8, pinned['placements']) and pinned['placements'] == as_tuples['placements'])
    clash = next(s for s in range(len(modules)) if not compat['E'][boss] >> s & 1)
    bad = se.solve_socket_wfc_ext(modules, compat, 6, 6, seed=2, fixed={(2, 2): boss, (3, 2): clash}, max_restarts=3)
    check('two incompatible neighbouring pins fail with reason propagation',
          not bad['success'] and set(bad['failures']) == {'propagation'} and len(bad['failures']) == 3)
    outside = se.solve_socket_wfc_ext(modules, compat, 4, 4, seed=2, fixed={(1, 1): 5}, max_restarts=2,
                                      domains=se.allow_states([(1, 1)], 1 << 6))
    check('a pin outside its restricted domain fails with reason fixed',
          not outside['success'] and outside['failures'] == ['fixed', 'fixed'])
    check('bad cells, states and rules raise ValueError', all(_raises_value_error(f) for f in (
        lambda: se.solve_socket_wfc_ext(modules, compat, 4, 4, fixed={(9, 0): 1}),
        lambda: se.solve_socket_wfc_ext(modules, compat, 4, 4, fixed={(0, 0): 500}),
        lambda: se.solve_socket_wfc_ext(modules, compat, 4, 4, fixed=[(0, 0, 1), (0, 0, 2)]),
        lambda: se.solve_socket_wfc_ext(modules, compat, 4, 4, domains=[1, 2, 3]),
        lambda: se.solve_socket_wfc_ext(modules, compat, 4, 4, domains={(4, 4): 1}))))

    # weights and priorities
    caves = se.states_where(modules, lambda m: m['theme'] in ('cave', 'caves'))

    def cave_lover(x, y, state, module, used):
        return se.default_weight(x, y, state, module, used) * (50 if caves >> state & 1 else 1)
    plain = biased = 0
    for seed in range(8):
        plain += sum(caves >> s & 1 for s in se.solve_socket_wfc_ext(modules, compat, 8, 8, seed=seed)['placements'])
        got = se.solve_socket_wfc_ext(modules, compat, 8, 8, seed=seed, weight=cave_lover)
        biased += sum(caves >> s & 1 for s in got['placements'])
        assert se.check_placements(compat, 8, 8, got['placements'])
    check('a weight hook steers the choice (cave modules 8x8, 8 seeds)', biased > 4 * plain, f'{plain} -> {biased}')
    again = se.solve_socket_wfc_ext(modules, compat, 8, 8, seed=3, weight=cave_lover)
    check('hooked solves are deterministic', again['placements'] ==
          se.solve_socket_wfc_ext(modules, compat, 8, 8, seed=3, weight=cave_lover)['placements'])
    hubs = {i for i, m in enumerate(modules) if m['template'] == 'hub_spokes'}
    top = se.solve_socket_wfc_ext(modules, compat, 1, 1, seed=4, priority=lambda x, y, s, m: int(s in hubs))
    check('only the highest-priority states are eligible', top['success'] and top['placements'][0] in hubs)
    zero = se.solve_socket_wfc_ext(modules, compat, 4, 4, seed=4, weight=lambda *a: 0.0)
    check('an all-zero weight falls back to a uniform choice instead of failing', zero['success'])

    # uniqueness by source layout across orientations, and the contradiction callback
    variants, vcompat = se.expand_d4(modules)
    unique = se.solve_socket_wfc_ext(variants, vcompat, 10, 10, seed=37, unique_states=True, max_restarts=128,
                                     groups=se.source_groups(variants))
    check('10x10 unique-by-source on the 656-state library uses each of the 100 layouts once',
          unique['success'] and sorted(variants[s]['source_state'] for s in unique['placements']) == list(range(100))
          and se.check_placements(vcompat, 10, 10, unique['placements']))
    twin = [i for i, v in enumerate(variants) if v['source_state'] == variants[5]['source_state']][:2]
    clash = se.solve_socket_wfc_ext(variants, vcompat, 4, 4, seed=1, unique_states=True, max_restarts=2,
                                    groups=se.source_groups(variants), fixed={(0, 0): twin[0], (3, 3): twin[1]})
    check('two pins from one group fail a unique run with reason unique',
          len(twin) == 2 and not clash['success'] and clash['failures'] == ['unique', 'unique'])
    seen = []
    result = se.solve_preset(modules, compat, rt.WORLD_PRESETS[2], on_contradiction=lambda a, r: seen.append((a, r)))
    check('on_contradiction fires once per failed attempt, in order',
          result['success'] and len(seen) == result['attempts'] - 1 == len(result['failures'])
          and [a for a, _ in seen] == list(range(len(seen))) and [r for _, r in seen] == result['failures'],
          f"{len(seen)} failures: {result['failures']}")

    # reroll only what a pin invalidates
    first = se.solve_socket_wfc_ext(modules, compat, 8, 8, seed=23)['placements']
    swap = next(s for s in range(len(modules)) if s != first[27] and compat['E'][s] >> first[28] & 1
                and compat['W'][s] >> first[26] & 1 and compat['S'][s] >> first[35] & 1
                and compat['N'][s] >> first[19] & 1)
    easy = se.reroll_after_pin(modules, compat, 8, 8, first, {(3, 3): swap}, seed=5)
    full = se.solve_socket_wfc_ext(modules, compat, 8, 8, seed=5, fixed={(3, 3): swap})
    check('a pin that fits redoes only its own cell; a full re-solve rewrites the grid',
          easy['success'] and easy['changed'] == [27] and easy['placements'][27] == swap and easy['rings'] == 0
          and sum(a != b for a, b in zip(full['placements'], first)) > 40)
    stiff = next(s for s in range(len(modules)) if not compat['E'][s] >> first[28] & 1)
    hard = se.reroll_after_pin(modules, compat, 8, 8, first, {(3, 3): stiff}, seed=5)
    check('a pin that clashes with a neighbour rerolls a few cells and stays valid',
          hard['success'] and hard['placements'][27] == stiff and 1 < len(hard['changed']) < 16
          and se.check_placements(compat, 8, 8, hard['placements']), f"{len(hard['changed'])} cells changed")
    check('reroll is deterministic and rejects a previous of the wrong size',
          easy['placements'] == se.reroll_after_pin(modules, compat, 8, 8, first, {(3, 3): swap}, seed=5)['placements']
          and _raises_value_error(lambda: se.reroll_after_pin(modules, compat, 8, 8, first[:10], {(3, 3): swap})))
    uniq = se.solve_socket_wfc_ext(modules, compat, 4, 4, seed=3, unique_states=True, max_restarts=50)['placements']
    grown = se.reroll_after_pin(modules, compat, 4, 4, uniq, {(0, 0): uniq[15]}, seed=1, unique_states=True,
                                max_restarts=50)
    check('when a kept cell clashes (unique state taken twice) the rerolled region grows a ring',
          grown['success'] and grown['rings'] >= 1 and grown['placements'][0] == uniq[15]
          and len(set(grown['placements'])) == 16 and se.check_placements(compat, 4, 4, grown['placements']),
          f"{grown['rings']} ring(s), {len(grown['changed'])} cells changed")
    impossible = se.reroll_after_pin(modules, compat, 8, 8, first, {(3, 3): swap}, seed=5, max_restarts=2,
                                     domains=se.allow_states([(3, 3)], 1 << ((swap + 1) % len(modules))))
    check('a pin the domains forbid returns a failed result, not an exception', not impossible['success'])


# ------------------------------------------------- socket WFC: count quotas

def test_socket_quota():
    print("\nsocket WFC: count quotas")
    modules, compat = _shipped_library()

    # tri-state counting on hand-made domains: set = {state 0}
    doms = [0b01, 0b11, 0b10, 0b11, 0b01]
    q = se.Quota(0b01, 'at_most', 3)
    check('tally counts yes (domain inside the set), maybe (partly) and no', q.tally(doms) == (2, 2))
    full = list(doms)
    changed = deque()
    ok = se.Quota(0b01, 'at_most', 2).apply(full, changed)
    check('at_most at its limit bans the set from every maybe cell',
          ok and full == [0b01, 0b10, 0b10, 0b10, 0b01] and sorted(changed) == [1, 3])
    full, changed = list(doms), deque()
    check('at_most over its limit is a contradiction', not se.Quota(0b01, 'at_most', 1).apply(full, changed))
    full, changed = list(doms), deque()
    ok = se.Quota(0b01, 'at_least', 4).apply(full, changed)
    check('at_least with just enough room selects the set in every maybe cell',
          ok and full == [0b01, 0b01, 0b10, 0b01, 0b01] and sorted(changed) == [1, 3])
    check('at_least with too little room is a contradiction',
          not se.Quota(0b01, 'at_least', 5).apply(list(doms), deque()))
    full, changed = list(doms), deque()
    check('exactly 2 here is already met: the maybe cells lose the set',
          se.Quota(0b01, 'exactly', 2).apply(full, changed) and full == [0b01, 0b10, 0b10, 0b10, 0b01])
    scoped = se.Quota(0b01, 'at_most', 0, cells=[(0, 0), (1, 0)])
    check('a cell list limits the count to those cells', scoped.tally(doms, [0, 1]) == (1, 1)
          and not scoped.apply(list(doms), deque(), [0, 1]))
    check('bad ops, negative counts and cells outside the grid raise ValueError', all(_raises_value_error(f) for f in (
        lambda: se.Quota(1, 'about', 3), lambda: se.Quota(1, 'at_most', -1),
        lambda: se.solve_socket_wfc_ext(modules, compat, 3, 3, quotas=[se.Quota(1, 'at_most', 1, cells=[(5, 5)])]))))

    # theme caps on the two presets that may repeat modules
    micro = lambda m: m['theme'] == 'micro'
    sparse = lambda m: m['open_ratio'] < 0.1
    base = {}
    for preset in rt.WORLD_PRESETS:
        legacy = rt.solve_socket_wfc(modules, compat, preset['grid_w'], preset['grid_h'], seed=preset['seed'],
                                     unique_states=preset['unique_states'], max_restarts=preset['max_restarts'])
        base[preset['name']] = (sum(micro(modules[s]) for s in legacy['placements']),
                                sum(sparse(modules[s]) for s in legacy['placements']))
    check('baseline: legacy worlds are 14-20% micro and 16-31% near-empty modules',
          base == {'tiny_fast': (5, 11), 'balanced': (11, 10), 'all100_unique': (20, 24)}, str(base))
    capped = []
    for preset in rt.WORLD_PRESETS[:2]:
        cells = preset['grid_w'] * preset['grid_h']
        limit = cells // 10
        got = se.solve_preset(modules, compat, preset, quotas=[
            se.quota_where(modules, micro, 'at_most', limit, name='micro'),
            se.quota_where(modules, sparse, 'at_most', limit, name='sparse')])
        pl = got['placements']
        capped.append(got['success'] and sum(micro(modules[s]) for s in pl) <= limit
                      and sum(sparse(modules[s]) for s in pl) <= limit
                      and se.check_placements(compat, preset['grid_w'], preset['grid_h'], pl))
    check('micro and near-empty modules capped at 10% of the cells on tiny_fast and balanced', all(capped))

    # exactly one hub / boss, and a floor on a theme
    boss = max(range(len(modules)), key=lambda i: modules[i]['open_ratio'])
    counts, valid = [], []
    for seed in range(8):
        got = se.solve_socket_wfc_ext(modules, compat, 8, 8, seed=seed, max_restarts=40,
                                      quotas=[se.Quota([boss], 'exactly', 1, name='boss')])
        counts.append(got['placements'].count(boss) if got['success'] else -1)
        valid.append(got['success'] and se.check_placements(compat, 8, 8, got['placements']))
    check('exactly one boss module in every 8x8 world (8 seeds)', counts == [1] * 8 and all(valid), str(counts))
    caves = se.states_where(modules, lambda m: m['theme'] in ('cave', 'caves'))
    got = se.solve_socket_wfc_ext(modules, compat, 8, 8, seed=3, quotas=[se.Quota(caves, 'at_least', 20)])
    check('at_least 20 cave cells is met', got['success'] and sum(caves >> s & 1 for s in got['placements']) >= 20)
    again = se.solve_socket_wfc_ext(modules, compat, 8, 8, seed=3, quotas=[se.Quota(caves, 'at_least', 20)])
    check('quota solves are deterministic', again['placements'] == got['placements'])

    centre = [(x, y) for y in range(2, 6) for x in range(2, 6)]
    sparse_mask = se.states_where(modules, sparse)
    local = se.solve_socket_wfc_ext(modules, compat, 8, 8, seed=23,
                                    quotas=[se.Quota(sparse_mask, 'at_most', 0, cells=centre)])
    check('a quota limited to a region keeps the set out of it only',
          local['success'] and not any(sparse_mask >> local['placements'][y * 8 + x] & 1 for x, y in centre))

    # pins and infeasible quotas
    pinned = se.solve_socket_wfc_ext(modules, compat, 6, 6, seed=2, fixed={(0, 0): boss}, max_restarts=20,
                                     quotas=[se.Quota([boss], 'exactly', 1)])
    check('a pin counts toward the quota: the other cells may not repeat the boss',
          pinned['success'] and pinned['placements'].count(boss) == 1 and pinned['placements'][0] == boss)
    two = se.solve_socket_wfc_ext(modules, compat, 6, 6, seed=2, max_restarts=3,
                                  fixed={(0, 0): boss, (5, 5): boss}, quotas=[se.Quota([boss], 'exactly', 1)])
    check('two pins against exactly 1 fail with reason quota',
          not two['success'] and two['failures'] == ['quota'] * 3)
    everything = se.solve_socket_wfc_ext(modules, compat, 4, 4, seed=1, max_restarts=2,
                                         quotas=[se.Quota((1 << len(modules)) - 1, 'at_most', 3)])
    check('capping the set of all states below the cell count is a contradiction',
          not everything['success'] and everything['failures'] == ['quota', 'quota'])
    short = se.solve_socket_wfc_ext(modules, compat, 3, 3, seed=1, max_restarts=2, quotas=[se.Quota([boss], 'at_least', 10)])
    check('asking for more cells than the grid has fails fast', not short['success'] and short['failures'] == ['quota'] * 2)
    capped_unique = se.solve_socket_wfc_ext(modules, compat, 10, 10, seed=37, unique_states=True, max_restarts=4,
                                            quotas=[se.quota_where(modules, micro, 'at_most', 10)])
    check('a theme cap cannot coexist with using every module once (the all100_unique preset)',
          not capped_unique['success'])
    free = se.solve_socket_wfc_ext(modules, compat, 6, 6, seed=2)
    none = se.solve_socket_wfc_ext(modules, compat, 6, 6, seed=2, quotas=[])
    check('an empty quota list changes nothing', free['placements'] == none['placements'])


# ------------------------------------------------ overlapping-model WFC

def _window_set(exemplar, n, periodic, symmetry):
    """Brute-force oracle: every n x n window of the exemplar (wrapping when
    periodic) under the symmetry group, as byte strings."""
    ex = np.asarray(exemplar)
    rows, cols = ex.shape
    found = set()
    ys = range(rows) if periodic else range(rows - n + 1)
    xs = range(cols) if periodic else range(cols - n + 1)
    for y in ys:
        for x in xs:
            window = np.array([[ex[(y + j) % rows, (x + i) % cols] for i in range(n)] for j in range(n)])
            variants = [window]
            if symmetry >= 2 and symmetry != 4:
                variants.append(np.fliplr(window))
            if symmetry in (4, 8):
                variants += [np.rot90(window, k) for k in (1, 2, 3)]
                if symmetry == 8:
                    variants += [np.rot90(np.fliplr(window), k) for k in (1, 2, 3)] + [np.fliplr(window)]
            found.update(np.ascontiguousarray(v).tobytes() for v in variants)
    return found


def _bad_windows(out, n, periodic, allowed):
    """Count the n x n windows of `out` (wrapping when periodic) that are not in `allowed`."""
    rows, cols = out.shape
    bad = 0
    ys = range(rows) if periodic else range(rows - n + 1)
    xs = range(cols) if periodic else range(cols - n + 1)
    for y in ys:
        for x in xs:
            window = np.array([[out[(y + j) % rows, (x + i) % cols] for i in range(n)] for j in range(n)])
            bad += window.tobytes() not in allowed
    return bad


def _agree(a, b, dx, dy):
    """Brute-force: pattern b at offset (dx, dy) from pattern a agrees on the overlap."""
    n = a.shape[0]
    for y in range(n):
        for x in range(n):
            by, bx = y - dy, x - dx
            if 0 <= by < n and 0 <= bx < n and a[y, x] != b[by, bx]:
                return False
    return True


class _ZeroNoise:
    """Stand-in stream: no noise, so entropy ties fall to the lowest cell index."""

    def random(self, size=None):
        return 0.0 if size is None else np.zeros(size)


def test_overlapping_wfc():
    print("\noverlapping-model wave function collapse")
    gen = np.random.default_rng(tf.seed_stream(51, 'test.wfc'))
    rooms = np.array([[1, 1, 1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 1, 0, 0, 1], [1, 0, 0, 0, 1, 0, 0, 1],
                      [1, 0, 0, 0, 0, 0, 0, 1], [1, 1, 1, 0, 1, 1, 1, 1], [1, 0, 0, 0, 1, 0, 0, 1],
                      [1, 0, 0, 0, 0, 0, 0, 1], [1, 1, 1, 1, 1, 1, 1, 1]])
    blobs = gen.integers(0, 2, size=(9, 9))
    swirl = np.array([[(x + 2 * y + (x * y) // 3) % 4 for x in range(7)] for y in range(7)])

    # (1) window membership, exhaustively, across modes
    cases, bad = 0, []
    for name, ex in (('rooms', rooms), ('blobs', blobs), ('swirl', swirl)):
        for n in (2, 3):
            for periodic_in in (True, False):
                for periodic_out in (False, True):
                    for symmetry in (1, 2, 4, 8):
                        out, stats = wfc.overlapping_wfc(ex, 14, 11, n=n, periodic_in=periodic_in,
                                                         periodic_out=periodic_out, symmetry=symmetry,
                                                         seed=cases, max_tries=20)
                        cases += 1
                        if out is None:
                            continue
                        allowed = _window_set(ex, n, periodic_in, symmetry)
                        if out.shape != (11, 14) or _bad_windows(out, n, periodic_out, allowed):
                            bad.append(f'{name} n={n} pin={periodic_in} pout={periodic_out} sym={symmetry}')
    check('every N x N window of the output belongs to the exemplar pattern set (96 configurations)',
          cases == 96 and not bad, '; '.join(bad[:2]) or f'{cases} runs')

    # (2), (3) degenerate exemplars
    flat = wfc.overlapping_wfc(np.full((4, 4), 7), 9, 6, n=3, seed=1)[0]
    check('a single-valued exemplar gives a uniform output of the right shape',
          flat is not None and flat.shape == (6, 9) and bool((flat == 7).all()))
    stripes = np.tile(np.array([0, 1]), (6, 3))
    rows_ok = []
    for seed in range(5):
        out, _ = wfc.overlapping_wfc(stripes, 12, 9, n=3, periodic_in=True, periodic_out=False, seed=seed)
        alternating = bool((out[:, 1:] != out[:, :-1]).all())
        rows_ok.append(out is not None and alternating and bool((out == out[0]).all()))
    check('column stripes: every row alternates and adjacent rows are identical', all(rows_ok))
    board = (np.indices((8, 8)).sum(axis=0) % 2)
    out, _ = wfc.overlapping_wfc(board, 13, 10, n=3, seed=4)
    check('a checkerboard exemplar gives a checkerboard',
          out is not None and bool(((out[:, 1:] != out[:, :-1]).all()) and ((out[1:] != out[:-1]).all())))

    # (4) arc consistency after every propagation, white box, against brute-force adjacency
    model = wfc.PatternModel(blobs, 2, True, 1)
    solver = wfc.WaveSolver(model, 10, 8, periodic=False, rng=np.random.default_rng(5))
    solver.initialize()
    ac_ok, steps = True, 0
    support = {(d, a): [b for b in range(model.count) if _agree(model.patterns[a], model.patterns[b], dx, dy)]
               for d, (dx, dy) in enumerate(wfc.DIRECTIONS) for a in range(model.count)}
    status = 'collapsed'
    while status == 'collapsed':
        wave = solver.wave()
        for y in range(8):
            for x in range(10):
                for pid in np.flatnonzero(wave[y, x]):
                    for d, (dx, dy) in enumerate(wfc.DIRECTIONS):
                        if 0 <= x + dx < 10 and 0 <= y + dy < 8:
                            ac_ok &= bool(wave[y + dy, x + dx][support[(d, int(pid))]].any())
        status = solver.step()
        steps += 1
    check('after the initial pass and every propagation each allowed pattern has support in every neighbour',
          ac_ok and steps > 5 and status == 'done', f'{steps} steps')
    check('compat tables equal the brute-force overlap test',
          all(sorted(i for i in range(model.count) if model.compat[d][a] >> i & 1) == support[(d, a)]
              for d in range(4) for a in range(model.count)))
    model = wfc.PatternModel(rooms, 3, False, 1)

    # (5) the collapsed cell has minimal entropy (zero noise: first index on ties)
    solver = wfc.WaveSolver(model, 8, 7, periodic=False, rng=_ZeroNoise())
    solver.initialize()
    weights = model.weights.astype(float)
    picks_ok, checked = True, 0
    while True:
        wave = solver.wave()
        entropy = {}
        for y in range(7):
            for x in range(8):
                allowed = weights[wave[y, x]]
                if len(allowed) > 1:
                    entropy[y * 8 + x] = math.log(allowed.sum()) - float((allowed * np.log(allowed)).sum() / allowed.sum())
        status = solver.step()
        if status != 'collapsed':
            break
        best = min(entropy.values())
        picks_ok &= solver.last[0] == min(c for c, e in entropy.items() if abs(e - best) < 1e-12)
        checked += 1
    check('every collapse takes the undecided cell of least entropy', picks_ok and checked > 5, f'{checked} collapses')

    # (6) weights: A twice as frequent as B, no constraints between them (1x1 windows)
    freq = sum(int(wfc.overlapping_wfc(np.array([[0, 0, 1]]), 1, 1, n=1, seed=i)[0][0, 0] == 0)
               for i in range(2000)) / 2000
    sigma = math.sqrt((2 / 3) * (1 / 3) / 2000)
    check('pattern frequencies follow the weights (2:1 over 2000 runs, within 3 sigma)',
          abs(freq - 2 / 3) <= 3 * sigma, f'{freq:.3f} vs 0.667')

    # (7) an unsatisfiable ground fails every attempt at once
    one = np.array([[0, 1, 0], [1, 1, 0], [0, 0, 1]])
    out, stats = wfc.overlapping_wfc(one, 9, 9, n=3, periodic_in=False, ground=True, max_tries=7)
    check('an impossible ground returns None after exactly max_tries quick attempts',
          out is None and stats['tries'] == 7 and stats['restarts'] == 7 and stats['collapses'] == 0
          and stats['failures'] == ['initial'] * 7)

    # ground that works: sky above, the last row of the exemplar at the bottom of the output
    land = np.zeros((8, 8), dtype=int)
    land[6:, :] = 1
    land[5, 2] = 2
    land[4, 2] = 2
    holds = []
    model = wfc.PatternModel(land, 3, True, 1)
    g_ids = set(model.ground)
    for seed in range(4):
        out, _ = wfc.overlapping_wfc(land, 14, 12, n=3, periodic_in=True, ground=True, seed=seed, max_tries=20)
        ok = out is not None and bool((out[-1] == 1).all())
        if ok:
            keyed = {p.tobytes(): i for i, p in enumerate(model.patterns)}
            ids = [[keyed[out[y:y + 3, x:x + 3].tobytes()] for x in range(12)] for y in range(10)]
            ok = all(i in g_ids for i in ids[-1]) and not any(i in g_ids for row in ids[:-1] for i in row)
        holds.append(ok)
    check('ground: the last window row uses ground patterns only, and nowhere else does',
          all(holds), f'{sum(holds)}/4')

    # (8) determinism
    first = wfc.overlapping_wfc(blobs, 20, 20, n=3, seed=11)
    check('same seed, identical grid and stats', np.array_equal(first[0], wfc.overlapping_wfc(blobs, 20, 20, n=3, seed=11)[0])
          and first[1] == wfc.overlapping_wfc(blobs, 20, 20, n=3, seed=11)[1])
    free2 = [wfc.overlapping_wfc(blobs, 20, 20, n=2, seed=s)[0] for s in (1, 2, 3)]
    check('different seeds give different grids for a non-trivial exemplar',
          not np.array_equal(free2[0], free2[1]) and not np.array_equal(free2[1], free2[2]))

    # (9) speed
    t0 = time.time()
    fast, stats = wfc.overlapping_wfc(gen.integers(0, 3, size=(8, 8)), 32, 32, n=3, seed=2)
    t_random = time.time() - t0
    t0 = time.time()
    free, stats2 = wfc.overlapping_wfc(gen.integers(0, 2, size=(8, 8)), 32, 32, n=2, seed=2, max_tries=20)
    t_free = time.time() - t0
    t0 = time.time()
    rich, stats3 = wfc.overlapping_wfc(rooms, 32, 32, n=3, symmetry=8, seed=2, max_tries=20)
    t_rich = time.time() - t0
    check('32x32 from an 8x8 exemplar in under 2 s: N = 3 random, N = 2 with 960 free collapses, rooms with D4 symmetry',
          fast is not None and free is not None and rich is not None and max(t_random, t_free, t_rich) < 2.0,
          f'{t_random:.2f}s / {t_free:.2f}s ({stats2["collapses"]} collapses) / {t_rich:.2f}s')

    # contradictions restart on the next stream, and the hook sees each one
    seen = []
    hard = np.array([[1, 1, 2, 2, 0, 2], [1, 1, 2, 1, 0, 0], [2, 1, 1, 1, 2, 1],
                     [0, 2, 0, 0, 2, 1], [0, 0, 1, 2, 1, 2], [0, 0, 2, 2, 0, 0]])
    out, stats = wfc.overlapping_wfc(hard, 20, 20, n=2, periodic_in=False, seed=3, max_tries=30,
                                     on_contradiction=lambda a, r: seen.append((a, r)))
    check('a contradiction restarts on the next stream and fires on_contradiction in order',
          out is not None and stats['restarts'] == len(seen) == len(stats['failures']) == 2
          and stats['tries'] == 3 and seen == [(0, 'propagation'), (1, 'propagation')]
          and not _bad_windows(out, 2, False, _window_set(hard, 2, False, 1)),
          f"{stats['restarts']} restarts, {stats['collapses']} collapses")
    out, stats = wfc.overlapping_wfc(hard, 20, 20, n=3, periodic_in=False, seed=3, max_tries=5)
    check('when every attempt contradicts the result is None after max_tries',
          out is None and stats['tries'] == 5 and stats['restarts'] == 5)

    # quotas, through the shared csp core
    marked = np.zeros((8, 8), dtype=int)
    marked[3, 3] = 2
    marked[5, 6] = 2
    counts = []
    for seed in range(5):
        out, _ = wfc.overlapping_wfc(marked, 16, 16, n=2, periodic_in=True, periodic_out=True, seed=seed,
                                     quotas=[lambda m: wfc.tile_quota(m, {2}, 'exactly', 1, name='door')])
        counts.append(None if out is None else int((out == 2).sum()))
    check('exactly one door tile in a periodic output, from a callable quota', counts == [1] * 5, str(counts))
    out, _ = wfc.overlapping_wfc(marked, 16, 16, n=2, periodic_in=True, periodic_out=True, seed=2,
                                 quotas=[lambda m: wfc.tile_quota(m, {2}, 'at_least', 5)])
    check('at_least 5 marked cells is met', out is not None and int((out == 2).sum()) >= 5)
    out, stats = wfc.overlapping_wfc(marked, 16, 16, n=2, periodic_in=True, periodic_out=True, seed=2, max_tries=3,
                                     quotas=[lambda m: wfc.tile_quota(m, {2}, 'at_least', 300)])
    check('an impossible quota fails fast with reason initial', out is None and stats['failures'] == ['initial'] * 3)

    # errors and shapes
    check('N larger than the exemplar, output smaller than N, bad symmetry, bad n and float tiles raise ValueError',
          all(_raises_value_error(f) for f in (
              lambda: wfc.overlapping_wfc(np.zeros((2, 2), dtype=int), 8, 8, n=3),
              lambda: wfc.overlapping_wfc(rooms, 2, 8, n=3),
              lambda: wfc.overlapping_wfc(rooms, 8, 8, n=3, symmetry=3),
              lambda: wfc.overlapping_wfc(rooms, 8, 8, n=6),
              lambda: wfc.overlapping_wfc(rooms.astype(float) + 0.5, 8, 8, n=3),
              lambda: wfc.overlapping_wfc(np.zeros(5, dtype=int), 8, 8, n=1))))
    sym = [wfc.PatternModel(swirl, 3, True, s).count for s in (1, 2, 4, 8)]
    check('more symmetry never means fewer patterns', sym[0] <= sym[1] <= sym[3] and sym[0] <= sym[2] <= sym[3],
          str(sym))
    tiny, stats = wfc.overlapping_wfc(rooms, 3, 3, n=3, periodic_in=False, seed=1, max_tries=20)
    check('an output exactly one window wide is one window of the exemplar',
          tiny is not None and tiny.tobytes() in _window_set(rooms, 3, False, 1))


# ------------------------------------------------------ inertia routes

def _tilemap_from_rows(rows, **tiles):
    """TileMap from text rows: '.' floor, '#' wall, 'r' rubble, 'o' floor with a teleport."""
    tm = ts.TileMap(len(rows[0]), len(rows), default_base='wall')
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            tm.base[y][x] = {'.': 'floor', '#': 'wall', 'r': 'rubble', 'o': 'floor'}[ch]
            if ch == 'o':
                tm.overlay[y][x] = 'teleport'
    return tm


def _route_valid(tm, profile, path):
    passable = rt.build_passable_grid(tm, profile)
    portals = set(rt.find_teleport_nodes(tm, passable))
    for (ax, ay), (bx, by) in zip(path, path[1:]):
        step = abs(ax - bx) + abs(ay - by) == 1 and passable[by][bx]
        if not (step or ((ax, ay) in portals and (bx, by) in portals)):
            return False
    return all(passable[y][x] for x, y in path)


def test_inertia_routes():
    print("\ninertia routes")
    rows = ['.......', '...#...', '#......', '..##...', '##.##..', '.......', '.......']
    tm = _tilemap_from_rows(rows)
    dist, prev, _ = rt.shortest_paths(tm, 'npc', (0, 0))
    legacy = rt.reconstruct_path(prev, (0, 0), (6, 6))
    inertia = tr.inertia_path(tm, 'npc', (0, 0), (6, 6))
    check('hand-made 7x7 map: the legacy route has 7 turns, the inertia walk 1, at the same cost',
          (tr.count_turns(legacy), tr.count_turns(inertia)) == (7, 1)
          and tr.path_cost(tm, 'npc', inertia) == tr.path_cost(tm, 'npc', legacy) == dist[6][6] == 12
          and _route_valid(tm, 'npc', inertia), f'turns {tr.count_turns(legacy)} -> {tr.count_turns(inertia)}')
    check('the walk is deterministic', inertia == tr.inertia_path(tm, 'npc', (0, 0), (6, 6)))

    band = _tilemap_from_rows(['.....r.....'] * 3 + ['.....#.....'] + ['...........'])
    d2, _, _ = rt.shortest_paths(band, 'npc', (0, 1))
    around = tr.inertia_path(band, 'npc', (0, 1), (10, 1))
    check('weighted terrain: the route pays the cheapest toll, rubble costs 2',
          tr.path_cost(band, 'npc', around) == d2[1][10] and _route_valid(band, 'npc', around))
    portal = _tilemap_from_rows(['.o........o..'])
    jump = tr.inertia_path(portal, 'npc', (0, 0), (12, 0))
    pd, _, _ = rt.shortest_paths(portal, 'npc', (0, 0))
    check('a teleport jump is taken when it is cheaper, and costs 1',
          jump == [[0, 0], [1, 0], [10, 0], [11, 0], [12, 0]] and tr.path_cost(portal, 'npc', jump) == pd[0][12] == 4)
    cut = _tilemap_from_rows(['...#...'])
    check('an unreachable goal gives an empty route; start == goal gives the start; a wall goal gives nothing',
          tr.inertia_path(cut, 'npc', (0, 0), (6, 0)) == [] and tr.inertia_path(cut, 'npc', (1, 0), (1, 0)) == [[1, 0]]
          and tr.inertia_path(cut, 'npc', (0, 0), (3, 0)) == [])
    dist_field, _ = tr.cost_to_goal(tm, 'npc', (6, 6))
    check('the cost-to-goal field matches the legacy Dijkstra from the start; walls are out of reach',
          dist_field[0][0] == dist[6][6] and dist_field[6][6] == 0 and dist_field[1][3] == tr.INF)

    records = rt.load_layout_catalog(HERE / 'tilegen_output')
    summary = {}
    for profile in ('npc', 'drone'):
        old_turns = new_turns = routes = better = worse = 0
        problems = []
        for rec in records:
            tmap = rec['tilemap']
            pair = rt.farthest_pair(tmap, profile)
            if pair is None or not pair['success']:
                continue
            new = tr.farthest_pair_inertia(tmap, profile)
            same = all(new[k] == pair[k] for k in pair if k not in ('path', 'path_steps'))
            if not (same and new['path'][0] == pair['path'][0] and new['path'][-1] == pair['path'][-1]
                    and _route_valid(tmap, profile, new['path'])
                    and tr.path_cost(tmap, profile, new['path']) == pair['path_cost']):
                problems.append(rec['id'])
            if pair['path_steps'] >= 12:
                a, b = tr.count_turns(pair['path']), tr.count_turns(new['path'])
                old_turns += a
                new_turns += b
                routes += 1
                better += b < a
                worse += b > a
        summary[profile] = (routes, old_turns / routes, new_turns / routes, better, worse, problems)
    check('on all 100 layouts the inertia route has the same endpoints, validity and path cost for npc and drone',
          all(not v[5] for v in summary.values()), str({k: v[5] for k, v in summary.items()}))
    check('routes of 12+ steps average at least 40% fewer turns (83 npc routes: 16.3 -> 8.3)',
          all(v[2] <= 0.6 * v[1] for v in summary.values()) and summary['npc'][0] >= 80,
          '; '.join(f'{k}: {v[0]} routes, {v[1]:.1f} -> {v[2]:.1f} turns, better {v[3]} worse {v[4]}'
                    for k, v in summary.items()))
    check('inertia is better on most routes and worse on very few',
          all(v[3] >= 0.55 * v[0] and v[4] <= 0.08 * v[0] for v in summary.values()))


# --------------------------------------------- dead ends and corridor loops

def _mask_from_rows(rows):
    return np.array([[ch == '#' for ch in row] for row in rows])


def _maze(offset, loopiness, size=61):
    tm = ts.TileMap(size + 3 + offset, size + 3 + offset, default_base='wall')
    tm.carve_maze(offset, offset + 1, size, size, base='floor', loopiness=loopiness)
    return tm


def test_dead_end_retraction():
    print("\ndead-end retraction and corridor loops")
    path = _mask_from_rows(['######'])
    ring = _mask_from_rows(['###', '#.#', '###'])
    plus = _mask_from_rows(['.#.', '###', '.#.'])
    block = _mask_from_rows(['##', '##'])
    metrics = {name: co.corridor_metrics(m) for name, m in
               (('path', path), ('ring', ring), ('plus', plus), ('block', block), ('empty', np.zeros((3, 3), dtype=bool)))}
    check('metrics: a straight path has 2 dead ends and no loop',
          (metrics['path']['vertices'], metrics['path']['edges'], metrics['path']['dead_ends'],
           metrics['path']['cycle_rank']) == (6, 5, 2, 0))
    check('metrics: a ring has 8 cells, 8 links, one loop and no dead end',
          (metrics['ring']['vertices'], metrics['ring']['edges'], metrics['ring']['dead_ends'],
           metrics['ring']['cycle_rank']) == (8, 8, 0, 1))
    check('metrics: a plus has 4 dead ends and a junction; a 2x2 block is one loop; empty is zeros',
          (metrics['plus']['dead_ends'], metrics['plus']['junctions'], metrics['plus']['cycle_rank']) == (4, 1, 0)
          and metrics['block']['cycle_rank'] == 1 and set(metrics['empty'].values()) == {0})
    two = _mask_from_rows(['##.##'])
    check('metrics: components are counted and the rank is per component',
          co.corridor_metrics(two)['components'] == 2 and co.corridor_metrics(two)['cycle_rank'] == 0)
    lone = _mask_from_rows(['#.#'])
    check('an isolated cell is a dead end to retract but not counted as a dead end in the metrics',
          co.dead_end_cells(lone).tolist() == [[True, False, True]] and co.corridor_metrics(lone)['dead_ends'] == 0
          and co.corridor_metrics(lone)['isolated'] == 2)

    tail = _mask_from_rows(['.....', '.###.', '.#.#.', '.###.', '..#..', '..#..', '..#..'])
    cut = co.retract_dead_ends(tail)
    check('a tail hanging off a ring retracts back to the ring', cut.tolist() == ring_with_margin(5, 7).tolist())
    check('a path or a tree retracts to nothing; keep protects cells and one round peels one layer',
          not co.retract_dead_ends(path).any() and not co.retract_dead_ends(plus).any()
          and co.retract_dead_ends(path, keep=_mask_from_rows(['#.....'])).sum() == 1
          and co.retract_dead_ends(path, max_passes=1).sum() == 4)
    check('retraction is idempotent and never adds cells',
          np.array_equal(co.retract_dead_ends(cut), cut) and not bool((cut & ~tail).any()))

    # invariants on random masks
    gen = np.random.default_rng(tf.seed_stream(61, 'test.corridors'))
    ok_rank = ok_clean = ok_subset = True
    for _ in range(40):
        m = gen.random((14, 17)) < gen.uniform(0.35, 0.7)
        keep = gen.random(m.shape) < 0.05
        out = co.retract_dead_ends(m, keep)
        ok_rank &= co.corridor_metrics(m)['cycle_rank'] == co.corridor_metrics(out)['cycle_rank'] \
            or bool((keep & m).any())
        ok_clean &= not co.dead_end_cells(out, keep).any()
        ok_subset &= not (out & ~m).any() and bool(((keep & m) <= out).all())
        free = co.retract_dead_ends(m)
        ok_rank &= co.corridor_metrics(free)['cycle_rank'] == co.corridor_metrics(m)['cycle_rank']
    check('random masks: cycle rank is unchanged, no unprotected dead end is left, the result is a subset',
          ok_rank and ok_clean and ok_subset)

    # carve_maze corridors: the intake measurements
    rows = []
    for loopiness in (0.08, 0.10, 0.18):
        removed = opened = 0
        rank_same = clean = connected = True
        dead = 0
        for offset in range(1, 11):
            m = tg.passable_mask(_maze(offset, loopiness))
            before, out = co.corridor_metrics(m), co.retract_dead_ends(m)
            after = co.corridor_metrics(out)
            removed += before['vertices'] - after['vertices']
            opened += before['vertices']
            dead += before['dead_ends']
            rank_same &= before['cycle_rank'] == after['cycle_rank']
            clean &= after['dead_ends'] == 0
            connected &= after['components'] == 1
        rows.append((loopiness, 100.0 * removed / opened, dead / 10.0, rank_same and clean and connected))
    check('carve_maze: retraction removes ~27% / 25% / 17% of open cells at loopiness 0.08 / 0.10 / 0.18, keeps one component',
          all(r[3] for r in rows) and abs(rows[0][1] - 27.5) < 4 and abs(rows[1][1] - 24.9) < 4
          and abs(rows[2][1] - 16.7) < 4 and rows[0][1] > rows[1][1] > rows[2][1],
          ', '.join(f'{r[0]}: {r[1]:.1f}% ({r[2]:.0f} dead ends/maze)' for r in rows))
    perfect = tg.passable_mask(_maze(2, 0.0))
    check('a perfect maze (a tree) retracts away completely', not co.retract_dead_ends(perfect).any())

    # insert cycles
    tree = tg.passable_mask(_maze(1, 0.0))
    grown, opened = co.insert_cycles(tree, 12, seed=5)
    before, after = co.corridor_metrics(tree), co.corridor_metrics(grown)
    n, s_, w_, e_ = co._neighbours(tree)
    between = all(((n[y, x] and s_[y, x]) or (w_[y, x] and e_[y, x])) and not tree[y, x] for x, y in opened)
    touching = any(max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1 for i, a in enumerate(opened) for b in opened[:i])
    blocks_before = int((tree[:-1, :-1] & tree[1:, :-1] & tree[:-1, 1:] & tree[1:, 1:]).sum())
    blocks_after = int((grown[:-1, :-1] & grown[1:, :-1] & grown[:-1, 1:] & grown[1:, 1:]).sum())
    check('insert_cycles opens exactly the requested loops: rank +12, still one component, walls between two corridors',
          len(opened) == 12 and after['cycle_rank'] - before['cycle_rank'] == 12 and after['components'] == before['components']
          and between and not touching and blocks_after == blocks_before == 0,
          f"rank {before['cycle_rank']} -> {after['cycle_rank']}")
    again, opened2 = co.insert_cycles(tree, 12, seed=5)
    other, opened3 = co.insert_cycles(tree, 12, seed=6)
    check('insert_cycles is deterministic per named stream and differs between seeds',
          opened == opened2 and np.array_equal(grown, again) and opened != opened3)
    same, none = co.insert_cycles(tree, 0)
    flood, many = co.insert_cycles(tree, 10 ** 6, seed=1)
    check('zero loops change nothing; asking for too many opens every independent candidate and stops',
          np.array_equal(same, tree) and not none and 12 < len(many) < 10 ** 6
          and co.corridor_metrics(flood)['cycle_rank'] == len(many)
          and _raises_value_error(lambda: co.insert_cycles(tree, -1)))
    walled = np.zeros((5, 5), dtype=bool)
    walled[2, 0] = walled[2, 4] = True
    check('a wall between two different components is not a candidate (that would merge, not loop)',
          not co.insert_cycles(walled, 5)[1])

    # Nystrom's recipe: loops first, then retract: a maze with loops and no dead ends
    braided, _ = co.insert_cycles(tree, 12, seed=2)
    braided_cut = co.retract_dead_ends(braided)
    mb = co.corridor_metrics(braided_cut)
    check('insert cycles then retract dead ends: no dead ends, 12 loops kept, one component',
          mb['dead_ends'] == 0 and mb['cycle_rank'] == 12 and mb['components'] == 1 and braided_cut.sum() < braided.sum(),
          f"{int(braided.sum())} -> {int(braided_cut.sum())} cells")

    # TileMap wrappers and the compiled-world lint
    tm = _maze(1, 0.1)
    before_pass = tg.passable_mask(tm)
    for y, x in zip(*np.nonzero(before_pass)):
        tm.overlay[y][x] = 'locked'
    removed = co.retract_tilemap(tm)
    after_pass = tg.passable_mask(tm)
    gone = before_pass & ~after_pass
    check('retract_tilemap turns the retracted cells into walls and clears only their overlays',
          removed == int(gone.sum()) > 100 and co.corridor_metrics(after_pass)['dead_ends'] == 0
          and all(tm.base[y][x] == 'wall' and tm.overlay[y][x] == '' for y, x in zip(*np.nonzero(gone)))
          and all(tm.overlay[y][x] == 'locked' for y, x in zip(*np.nonzero(after_pass))))
    opened = co.insert_cycles_tilemap(tm, 4, seed=1)
    check('insert_cycles_tilemap opens floor cells', len(opened) == 4 and all(tm.base[y][x] == 'floor' for x, y in opened))
    world = tw.World(tm, seed=1, name='corridor lint')
    metrics = co.lint_corridors(world)
    check('lint_corridors reports metrics without touching the findings unless asked',
          metrics['cycle_rank'] >= 4 and not world.findings)
    co.lint_corridors(world, add_findings=True)
    check('with add_findings it records one geometry.corridors finding carrying the numbers',
          len(world.findings) == 1 and world.findings[0]['rule'] == 'geometry.corridors'
          and world.findings[0]['cycle_rank'] == metrics['cycle_rank'] and world.findings[0]['severity'] == 'info')
    check('masks must be 2D', _raises_value_error(lambda: co.retract_dead_ends(np.zeros(5, dtype=bool))))


def ring_with_margin(w, h):
    """The 3x3 ring of the tail fixture, placed in a w x h field of walls."""
    out = np.zeros((h, w), dtype=bool)
    out[1, 1:4] = out[3, 1:4] = True
    out[2, 1] = out[2, 3] = True
    return out


# ---------------------------------------------------- lock-and-key gating

def _graph(edges, extra_nodes=()):
    adj = {}
    for a, b in edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    for n in extra_nodes:
        adj.setdefault(n, set())
    return {k: sorted(v) for k, v in adj.items()}


def _random_graph(gen, n, extra_edges):
    edges = []
    for i in range(1, n):
        parent = i - 1 if gen.random() < 0.6 else int(gen.integers(0, i))
        edges.append((parent, i))
    for _ in range(extra_edges):
        a, b = (int(v) for v in gen.integers(0, n, size=2))
        if a != b:
            edges.append((a, b))
    return _graph(edges)


def _reachable(adj, s, removed=()):
    removed = set(removed)
    seen, stack = {s}, [s]
    while stack:
        v = stack.pop()
        for u in adj[v]:
            if u not in seen and u not in removed:
                seen.add(u)
                stack.append(u)
    return seen


def test_key_lock_gating():
    print("\nlock-and-key gating")
    gen = np.random.default_rng(tf.seed_stream(71, 'test.keylock'))

    cuts_ok = seps_ok = True
    for trial in range(40):
        adj = _random_graph(gen, int(gen.integers(6, 28)), int(gen.integers(0, 5)))
        nodes = sorted(adj)
        brute_cuts = {v for v in nodes if len(_reachable(adj, nodes[0] if v != nodes[0] else nodes[1], [v])) < len(nodes) - 1}
        cuts_ok &= kl.articulation_points(adj, root=nodes[0]) == {v for v in nodes if v != nodes[0] and
                                                                   len(_reachable(adj, nodes[0], [v])) < len(nodes) - 1} \
            | ({nodes[0]} if len(_reachable(adj, nodes[1], [nodes[0]])) < len(nodes) - 1 else set())
        for _ in range(4):
            s_, t_ = (int(v) for v in gen.choice(len(nodes), size=2, replace=False))
            s_, t_ = nodes[s_], nodes[t_]
            expect = [v for v in nodes if v not in (s_, t_) and t_ not in _reachable(adj, s_, [v])]
            got = kl.separators(adj, s_, t_)
            dist = kl.bfs_distances(adj, [s_])
            seps_ok &= set(got) == set(expect) and got == sorted(got, key=lambda v: (dist[v], v))
    check('articulation_points equals brute force removal on 40 random graphs', cuts_ok)
    check('separators(s, t) equals brute force and comes ordered by distance from s (160 pairs)', seps_ok)

    line = _graph([(i, i + 1) for i in range(9)])
    check('far_apart on a path finds its two ends; on a tree it is the diameter',
          kl.far_apart(line) == (0, 9) or kl.far_apart(line) == (9, 0))
    tree_ok = True
    for _ in range(20):
        tree = _random_graph(gen, int(gen.integers(8, 30)), 0)
        a, b = kl.far_apart(tree)
        diameter = max(max(kl.bfs_distances(tree, [v]).values()) for v in tree)
        tree_ok &= kl.bfs_distances(tree, [a])[b] == diameter
    check('far_apart is exact on 20 random trees', tree_ok)
    check('shortest_path and bfs_distances agree', all(
        len(kl.shortest_path(line, 0, v)) - 1 == kl.bfs_distances(line, [0])[v] for v in range(10))
          and kl.shortest_path(line, 0, 9, blocked=[5]) == [])

    # a route that is deliberately not the shortest
    open_mask = np.ones((14, 14), dtype=bool)
    grid = kl.grid_graph(open_mask)
    lengths, valid = [], True
    for seed in range(12):
        path = kl.temperature_path(grid, (0, 0), (13, 13), tf.rng_for(seed, 'test.path'), 2.0)
        valid &= (path[0], path[-1]) == ((0, 0), (13, 13)) and len(set(path)) == len(path) \
            and all(b in grid[a] for a, b in zip(path, path[1:]))
        lengths.append(len(path) - 1)
    check('temperature paths are simple, valid, and longer than the shortest (26 steps) on average',
          valid and min(lengths) >= 26 and sum(lengths) / len(lengths) > 30, f'mean {sum(lengths) / len(lengths):.1f}')
    check('temperature 0 is a shortest path; same seed gives the same path; other seeds differ',
          len(kl.temperature_path(grid, (0, 0), (13, 13), tf.rng_for(1, 'x'), 0)) - 1 == 26
          and kl.temperature_path(grid, (0, 0), (13, 13), tf.rng_for(3, 'test.path'), 2.0)
          == kl.temperature_path(grid, (0, 0), (13, 13), tf.rng_for(3, 'test.path'), 2.0)
          and len({tuple(kl.temperature_path(grid, (0, 0), (13, 13), tf.rng_for(s, 'test.path'), 2.0))
                   for s in range(5)}) > 1)
    split = _graph([(0, 1), (2, 3)])
    check('an unreachable goal gives an empty path', kl.temperature_path(split, 0, 3, tf.rng_for(1, 'x')) == [])

    # the solvability replay on hand-made graphs: a spine 0..9 with branches A = 100 (off 2) and B = 101 (off 5)
    A, B = 100, 101
    spine = _graph([(i, i + 1) for i in range(9)] + [(2, A), (5, B)])
    ok = kl.check_solvable(spine, 0, 9, {3: 1, 6: 2}, {1: A, 2: B})
    check('keys on branches before their locks: solvable, picked up in order, reach grows',
          ok['solvable'] and ok['collected'] == [1, 2] and ok['reach'] == [4, 8, 12] and ok['stuck'] == [],
          str(ok['reach']))
    both = kl.check_solvable(spine, 0, 9, {3: 1, 6: 2}, {1: A, 2: A})
    behind = kl.check_solvable(spine, 0, 9, {3: 1, 6: 2}, {1: B, 2: A})
    own = kl.check_solvable(spine, 0, 9, {3: 1}, {1: 4})
    deadlock = kl.check_solvable(spine, 0, 9, {3: 1, 6: 2}, {1: 7, 2: B})
    check('a key behind its own lock, behind an earlier lock, or two keys each behind the other lock is unsolvable',
          both['solvable'] and not behind['solvable'] and behind['stuck'] == [3] and behind['collected'] == [2]
          and not own['solvable'] and own['collected'] == []
          and not deadlock['solvable'] and deadlock['stuck'] == [3, 6] and deadlock['collected'] == [])
    copies = kl.check_solvable(spine, 0, 9, {3: 1}, {1: [B, A]})
    locked_goal = kl.check_solvable(spine, 0, 9, {9: 1}, {1: A})
    locked_start = kl.check_solvable(spine, 0, 9, {0: 1}, {1: A})
    check('a key with several copies, a locked goal and a locked start',
          copies['solvable'] and locked_goal['solvable'] and locked_goal['collected'] == [1] and not locked_start['solvable'])

    # planning on a spine with a side branch at every fifth cell: keys must come in order
    edges = [(i, i + 1) for i in range(40)]
    for i in range(5, 40, 5):
        edges += [(i, 1000 + 10 * i), (1000 + 10 * i, 1001 + 10 * i), (1001 + 10 * i, 1002 + 10 * i)]
    comb = _graph(edges)
    plan = kl.plan_gates(comb, locks=3, seed=2, start=0, goal=40)
    order_ok = [g['node'] for g in plan['locks']] == sorted(g['node'] for g in plan['locks'])
    dist0 = kl.bfs_distances(comb, [0])
    each_ok = all(g['node'] in kl.separators(comb, 0, 40) and g['key'] in _reachable(comb, 0, [g['node']])
                  and 40 not in _reachable(comb, 0, [g['node']]) for g in plan['locks'])
    check('spine with branches: 3 locks on separators, each key before its lock, picked up in order',
          len(plan['locks']) == 3 and plan['solvable'] and order_ok and each_ok
          and plan['check']['collected'] == [1, 2, 3] and plan['reason'] == '', str(plan['check']['reach']))
    keys = [g['key'] for g in plan['locks']]
    check('keys sit on side branches, apart from the route and from each other',
          len(set(keys)) == 3 and all(k >= 1000 for k in keys))
    again = kl.plan_gates(comb, locks=3, seed=2, start=0, goal=40)
    other = kl.plan_gates(comb, locks=3, seed=9, start=0, goal=40)
    check('plans are deterministic per seed and the seed moves the keys',
          again == plan and [g['key'] for g in other['locks']] != keys)
    moved = [dict(g) for g in plan['locks']]
    broken = kl.check_solvable(comb, 0, 40, {g['node']: g['id'] for g in moved},
                               {1: moved[0]['key'], 2: moved[1]['key'], 3: 40 - 1})
    check('moving a key behind its lock makes the validator reject the layout',
          plan['solvable'] and not broken['solvable'])

    random_ok = []
    for trial in range(30):
        adj = _random_graph(gen, int(gen.integers(10, 60)), int(gen.integers(0, 4)))
        got = kl.plan_gates(adj, locks=int(gen.integers(1, 4)), seed=trial)
        locks = {g['node']: g['id'] for g in got['locks']}
        random_ok.append(got['solvable'] and kl.check_solvable(adj, got['start'], got['goal'], locks,
                                                              {g['id']: g['key'] for g in got['locks']})['solvable']
                         and all(got['goal'] not in _reachable(adj, got['start'], [n]) for n in locks))
    check('30 random graphs: every plan is solvable and every lock really blocks the goal', all(random_ok))

    # degradation
    ring = _graph([(i, (i + 1) % 12) for i in range(12)])
    rp = kl.plan_gates(ring, locks=3, seed=1)
    check('a ring has no chokepoint: no locks, trivially solvable, the reason says so',
          rp['locks'] == [] and rp['solvable'] and 'chokepoint' in rp['reason'])
    check('empty, single-node and disconnected graphs degrade without errors',
          kl.plan_gates({}, 2)['reason'] == 'no nodes' and kl.plan_gates({0: []}, 2)['locks'] == []
          and not kl.plan_gates(split, 1, start=0, goal=3)['solvable'] and kl.plan_gates(spine, 0)['locks'] == []
          and _raises_value_error(lambda: kl.far_apart({})))

    # raster maps
    maze = tg.passable_mask(_maze(1, 0.04))
    plan = kl.plan_on_mask(maze, 3, seed=4)
    graph = kl.grid_graph(maze)
    cells_ok = all(g['node'] in kl.separators(graph, plan['start'], plan['goal'])
                   and g['key'] in _reachable(graph, plan['start'], [g['node']]) for g in plan['locks'])
    check('a braided maze: three locks on one-cell corridors, keys before them, solvable',
          len(plan['locks']) == 3 and plan['solvable'] and cells_ok and all(len(graph[g['node']]) == 2 for g in plan['locks']),
          f"{len(plan['chokepoints'])} chokepoints, reach {plan['check']['reach']}")
    tm = _maze(1, 0.04)
    before_base = [row[:] for row in tm.base]
    kl.apply_plan(tm, plan)
    check('apply_plan marks locks and keys as overlays and leaves the tiles alone',
          tm.base == before_base and all(tm.overlay[g['node'][1]][g['node'][0]] == 'locked'
                                         and tm.overlay[g['key'][1]][g['key'][0]] == 'key' for g in plan['locks']))
    open_cells = np.ones((12, 12), dtype=bool)
    check('raster plans are deterministic per seed; on open ground the seed draws a different route',
          plan == kl.plan_on_mask(maze, 3, seed=4)
          and kl.plan_on_mask(open_cells, 0, seed=1)['path'] != kl.plan_on_mask(open_cells, 0, seed=2)['path'])

    # the world feature is opt-in
    n_before = len(tw.FEATURES)
    check('importing the module registers nothing: still the 40 depth features',
          n_before == 40 and kl.FEATURE_ID not in tw.FEATURES
          and sorted(f.number for f in tw.FEATURES.values()) == list(range(21, 61)))
    try:
        check('register() adds nar.gates as feature 61 and is idempotent',
              kl.register() == 'nar.gates' == kl.register() and len(tw.FEATURES) == 41 and tw.FEATURES['nar.gates'].number == 61)
        spec = {'id': 3, 'size': 64, 'template': 'silhouette', 'seed': 7, 'locks': 3}
        smap = tg.GRAMMARS['silhouette'](spec, random.Random(7))
        world = tw.World(smap, seed=7, name='gates', spec=spec)
        tw.derive_places(world)
        ran = tw.run_features(world, ['nar.gates'])
        gates = world.artifacts.get('gates', {})
        check('the feature publishes artifacts.gates with places, an unlock order and a solvable verdict',
              ran == ['nar.gates'] and not world.findings and gates['solvable'] and len(gates['locks']) == 3
              and sorted(gates['unlock_order']) == [1, 2, 3] and gates['start']['cell'] != gates['goal']['cell']
              and all(g['lock']['place'] is None or g['lock']['place'] in world.places for g in gates['locks']),
              f"route {gates['route_length']}, {gates['chokepoints']} chokepoints")
        recipe = tc.WorldRecipe(macro='portal_maps', size=64, seed=11, features=('nar.gates', 'exp.compiler'))
        compiled = tc.compile_recipe(recipe)
        check('a recipe can run it alongside the compiler, and the manifest lists it',
              not [f for f in compiled.findings if f['severity'] == 'error'] and 'gates' in compiled.artifacts
              and any(f['id'] == 'nar.gates' and f['ran'] for f in compiled.artifacts['manifest']['features']))
        flat = tw.World(ts.TileMap(12, 12, default_base='floor'), seed=1, name='open', spec={'locks': 2})
        flat_record = kl.key_lock_gating(flat)
        check('an open map has no chokepoint: the feature degrades to an empty solvable record',
              flat_record['locks'] == [] and flat_record['solvable'] and flat_record['reason'] != '' and not flat.findings)
        with patch.object(kl, 'plan_on_mask', return_value={'start': (1, 1), 'goal': (2, 2), 'path': [(1, 1), (2, 2)],
                                                            'chokepoints': [], 'locks': [], 'solvable': False,
                                                            'check': None, 'reason': 'forced'}):
            bad = tw.World(ts.TileMap(8, 8, default_base='floor'), seed=1, name='bad', spec={})
            kl.key_lock_gating(bad)
        check('an unsolvable plan is reported as an error finding',
              [f['rule'] for f in bad.findings] == ['mission.unsolvable'])
    finally:
        kl.unregister()
    check('unregister() restores the 40-feature registry',
          len(tw.FEATURES) == 40 and sorted(f.number for f in tw.FEATURES.values()) == list(range(21, 61)))


# ------------------------------------------------------------ contour tracing

def _boundary_edge_count(mask):
    """Unit edges between differing 4-neighbours, the array padded with background."""
    padded = np.pad(mask, 1)
    return int((padded[1:, :] != padded[:-1, :]).sum() + (padded[:, 1:] != padded[:, :-1]).sum())


def _enclosed_background(mask):
    """Number of 8-connected background components that do not touch the array border."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    count = 0
    for y0, x0 in zip(*np.nonzero(~mask)):
        if seen[y0, x0]:
            continue
        seen[y0, x0] = True
        stack, open_to_border = [(y0, x0)], False
        while stack:
            y, x = stack.pop()
            open_to_border |= y in (0, h - 1) or x in (0, w - 1)
            for ny in range(max(0, y - 1), min(h, y + 2)):
                for nx in range(max(0, x - 1), min(w, x + 2)):
                    if not mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
        count += not open_to_border
    return count


def _ring_edges(ring):
    pts = [tuple(p) for p in np.asarray(ring).tolist()]
    return [tuple(sorted((pts[i], pts[(i + 1) % len(pts)]))) for i in range(len(pts))]


def _ring_length(ring):
    ring = np.asarray(ring)
    return float(np.hypot(*(np.roll(ring, -1, axis=0) - ring).T).sum())


def test_trace_rings():
    print("\ncontour tracing")
    area = tv.signed_area

    # ---- mode 'edges': named shapes
    one = tv.trace_rings(_mask_from_rows(['...', '.#.', '...']))
    check('a single pixel is one ring of 4 vertices, a unit square around its sample',
          len(one) == 1 and one[0].tolist() == [[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5]])
    block = tv.trace_rings(_mask_from_rows(['....', '.##.', '.##.', '....']))
    plus = tv.trace_rings(_mask_from_rows(['.#.', '###', '.#.']))
    check('a 2 x 2 block is one ring with 8 boundary edges, a plus shape one with 12',
          [len(block), len(block[0]), area(block[0])] == [1, 8, 4.0]
          and [len(plus), len(plus[0]), area(plus[0])] == [1, 12, 5.0])
    diagonal = _mask_from_rows(['#.', '.#'])
    check('pixels touching only at a corner are two rings (4-connected foreground)',
          len(tv.trace_rings(diagonal)) == 2 and len(tv.trace_rings(diagonal, simple=False)) == 2)
    full = tv.trace_rings(np.ones((3, 4), dtype=bool))
    check('a full mask is one ring along the array edge',
          len(full) == 1 and len(full[0]) == 14 and area(full[0]) == 12.0
          and full[0].min() == -0.5 and full[0][:, 0].max() == 3.5 and full[0][:, 1].max() == 2.5)
    check('an empty mask, an all-below field and a zero-size array give no ring',
          tv.trace_rings(np.zeros((4, 4), dtype=bool)) == [] and tv.trace_rings(np.zeros((4, 4))) == []
          and tv.trace_rings(np.zeros((4, 4)), mode='interp') == [] and tv.trace_rings(np.zeros((0, 3))) == [])
    pinch = _mask_from_rows(['###', '#.#', '##.'])
    whole, split = tv.trace_rings(pinch, simple=False), tv.trace_rings(pinch)
    check('a boundary that touches itself at a corner is one walk, or splits there into a ring and a hole',
          len(whole) == 1 and len(whole[0]) == len({tuple(p) for p in whole[0].tolist()}) + 1
          and sorted(area(r) for r in split) == [-1.0, 8.0])
    check('level equal to a data value counts as inside',
          [area(r) for r in tv.trace_rings(np.array([[0.0, 0.5, 1.0]]), level=0.5)] == [2.0])
    noise = tf.Field(np.random.default_rng(3).random((20, 30)), 'noise')
    check('a Field gives the same rings as its array',
          all(np.array_equal(a, b) for a, b in zip(tv.trace_rings(noise, 0.4), tv.trace_rings(noise.a, 0.4)))
          and len(tv.trace_rings(noise, 0.4)) == len(tv.trace_rings(noise.a, 0.4)) > 0)
    bad = np.zeros((3, 3))
    bad[1, 1] = np.nan
    check('non-finite values, a non-2D source and unknown options raise ValueError',
          all(_raises_value_error(fn) for fn in (
              lambda: tv.trace_rings(bad), lambda: tv.trace_rings(bad, mode='interp'),
              lambda: tv.trace_rings(np.full((3, 3), np.inf)), lambda: tv.trace_rings(np.zeros(5)),
              lambda: tv.trace_rings(np.zeros((3, 3)), mode='bevel'),
              lambda: tv.trace_rings(np.zeros((3, 3)), saddle='high'))))

    # ---- mode 'edges': oracles on random masks
    rng = np.random.default_rng(tf.seed_stream(41, 'test.vector.rings'))
    fails, pinched, nested = Counter(), 0, 0
    for _ in range(200):
        h, w = (int(v) for v in rng.integers(2, 17, 2))
        mask = rng.random((h, w)) < rng.uniform(0.2, 0.6)
        walks, rings = tv.trace_rings(mask, simple=False), tv.trace_rings(mask)
        fg, bg = tf.connected_components(mask)[1], _enclosed_background(mask)
        fails['ring count'] += len(walks) != fg + bg
        pinched += len(rings) != len(walks)
        for group in (walks, rings):
            fails['area sum'] += abs(sum(area(r) for r in group) - mask.sum()) > 1e-9
            fails['perimeter'] += abs(sum(_ring_length(r) for r in group) - _boundary_edge_count(mask)) > 1e-9
        edges = [e for r in rings for e in _ring_edges(r)]
        fails['simple'] += any(len({tuple(p) for p in r.tolist()}) != len(r) for r in rings)
        fails['shared edge'] += len(edges) != len(set(edges))
        fails['same edges'] += sorted(edges) != sorted(e for r in walks for e in _ring_edges(r))
        firsts = [tuple(r[0]) for r in rings]
        fails['canonical'] += firsts != sorted(firsts) or any(tuple(r[0]) != min(map(tuple, r.tolist())) for r in rings)
        fails['deterministic'] += any(not np.array_equal(a, b) for a, b in zip(rings, tv.trace_rings(mask.copy())))
        labels = tf.connected_components(mask)[0]
        sizes = np.bincount(labels.ravel())
        tree = tv.ring_tree(rings)
        holes = [i for i, r in enumerate(rings) if area(r) < 0]
        nested += len(holes)
        fails['tree holes'] += sorted(i for v in tree.values() for i in v) != holes
        for outer, inner in tree.items():
            p, q = rings[outer][0], rings[outer][1]
            pixel = (p + q) / 2.0 + 0.5 * np.array([-(q - p)[1], (q - p)[0]])      # foreground side of the first edge
            comp = labels[int(round(pixel[1])), int(round(pixel[0]))]
            fails['tree area'] += comp == 0 or abs(area(rings[outer]) + sum(area(rings[i]) for i in inner) - sizes[comp]) > 1e-9
    check('area sum = True pixels and perimeter = differing-neighbour edges on 200 random masks',
          not fails['area sum'] and not fails['perimeter'], 'both for the walks and for the simple rings')
    check("ring count = 4-connected foreground + enclosed 8-connected background (walks, simple=False)",
          not fails['ring count'])
    check('simple rings: no repeated vertex, no shared edge, same edges as the walks',
          not fails['simple'] and not fails['shared edge'] and not fails['same edges'], f'{pinched} masks needed a split')
    check('rings start at their smallest vertex, are sorted by it, and repeat exactly',
          not fails['canonical'] and not fails['deterministic'])
    check('ring_tree: every hole has an outer, and outer area + hole areas = its component size',
          not fails['tree holes'] and not fails['tree area'], f'{nested} holes')

    donut = np.ones((11, 11), dtype=bool)
    donut[2:9, 2:9] = False
    donut[4:7, 4:7] = True
    donut[5, 5] = False
    rings = tv.trace_rings(donut)
    check("ring_tree nests a donut, its island and the island's own hole",
          [area(r) for r in rings] == [121.0, -49.0, 9.0, -1.0] and tv.ring_tree(rings) == {0: [1], 2: [3]})

    # ---- large array
    start = time.time()
    field = tf.perlin(1024, 1024, scale=40.0, seed=7)
    mask = field.a > tg.threshold_for_ratio(field, 0.45)
    big = tv.trace_rings(mask)
    elapsed = time.time() - start
    check('a 1024 x 1024 mask traces within the 10 s budget, area = pixel count',
          big and abs(sum(area(r) for r in big) - mask.sum()) < 1e-6 and elapsed < 10.0,
          f'{len(big)} rings, {sum(len(r) for r in big)} vertices, {elapsed:.2f}s')

    # ---- mode 'interp'
    yy, xx = np.mgrid[0:100, 0:100].astype(float)
    disc = tv.trace_rings(-((xx - 49.5) ** 2 + (yy - 49.5) ** 2), level=-400.0, mode='interp')
    ring = disc[0]
    radius = np.hypot(ring[:, 0] - 49.5, ring[:, 1] - 49.5)
    check('disc of radius 20 on a 100 x 100 grid: one outer ring, length and area within 0.5%, vertices within 0.1',
          len(disc) == 1 and area(ring) > 0 and abs(_ring_length(ring) / (2 * np.pi * 20) - 1) < 0.005
          and abs(area(ring) / (np.pi * 400) - 1) < 0.005 and np.abs(radius - 20).max() < 0.1,
          f'length {_ring_length(ring):.2f}, area {area(ring):.1f}, worst {np.abs(radius - 20).max():.4f}')
    ramp = tv.trace_rings(xx, level=30.25, mode='interp')[0]
    on_line = (ramp[:, 0] < 31.0) & (ramp[:, 1] >= 0.0) & (ramp[:, 1] <= 99.0)
    check('a linear ramp at level c gives the straight polyline x = c to 1e-9',
          on_line.sum() == 100 and np.abs(ramp[on_line, 0] - 30.25).max() < 1e-9)
    bump = np.exp(-(((xx - 49.5) / 18.0) ** 2 + ((yy - 40.0) / 11.0) ** 2) / 2.0)
    found = [tv.trace_rings(bump, level=lv, mode='interp') for lv in (0.2, 0.4, 0.6, 0.8)]
    spans = [((f[0][:, 0] - 49.5) / 18.0) ** 2 + ((f[0][:, 1] - 40.0) / 11.0) ** 2 for f in found]
    check('rings of a unimodal bump nest: a higher level lies strictly inside the lower one',
          all(len(f) == 1 for f in found) and all(hi.max() < lo.min() for lo, hi in zip(spans, spans[1:])))

    sad = np.array([[1.0, 0.0], [0.0, 1.0]])
    check("an ambiguous cell is joined by 'center' when its average reaches the level, kept apart by 'low'",
          (len(tv.trace_rings(sad, 0.5, 'interp')), len(tv.trace_rings(sad, 0.5, 'interp', saddle='low'))) == (1, 2)
          and len(tv.trace_rings(np.array([[1.0, 0.4], [0.4, 1.0]]), 0.5, 'interp')) == 1
          and len(tv.trace_rings(np.array([[1.0, 0.4], [0.4, 1.0]]), 0.5, 'interp', saddle='low')) == 2)
    check("an ambiguous cell whose average falls short of the level stays apart under 'center'",
          len(tv.trace_rings(sad, 0.6, 'interp')) == 2)
    ones = np.ones((4, 5))
    padded, flat = (tv.trace_rings(ones, 0.5, 'interp', pad=p)[0] for p in (True, False))
    check('pad=True closes rings half a sample beyond the border, pad=False on the border samples',
          (padded[:, 0].min(), padded[:, 0].max(), padded[:, 1].min()) == (-0.5, 4.5, -0.5)
          and (flat[:, 0].min(), flat[:, 0].max(), flat[:, 1].max()) == (0.0, 4.0, 3.0) and area(flat) == 12.0)

    cross, degenerate = Counter(), 0
    for _ in range(120):
        h, w = (int(v) for v in rng.integers(2, 15, 2))
        values = rng.random((h, w))
        level = float(rng.uniform(0.2, 0.8))
        rings = tv.trace_rings(values, level, 'interp')
        degenerate += any(area(r) == 0 for r in rings)
        cross['vertices'] += sum(len(r) for r in rings) != _boundary_edge_count(values >= level)
        for ring in rings:
            for x, y in ring.tolist():
                if x in (-0.5, w - 0.5) or y in (-0.5, h - 0.5):
                    continue                                    # on the closing margin
                if y == round(y):
                    j = int(np.floor(x))
                    value = values[int(y), j] * (1 - (x - j)) + values[int(y), j + 1] * (x - j)
                else:
                    i = int(np.floor(y))
                    value = values[i, int(x)] * (1 - (y - i)) + values[i + 1, int(x)] * (y - i)
                cross['level'] += abs(value - level) > 1e-9
        for saddle in ('center', 'low'):
            for pad in (True, False):
                tv.trace_rings(values, level, 'interp', pad=pad, saddle=saddle)      # every ring closes
    check('interp: each vertex sits where the interpolated field equals the level, one per straddling edge',
          not cross['level'] and not cross['vertices'] and not degenerate, '120 random fields')


# -------------------------------------------------- scattered-data (RBF) fields

def test_rbf_field():
    print("\nscattered-data fields")
    rng = np.random.default_rng(tf.seed_stream(43, 'test.vector.rbf'))
    W, H = 40, 32
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    centers = np.unique(rng.integers(0, [W, H], (14, 2)), axis=0).astype(float)
    cols, rows = centers[:, 0].astype(int), centers[:, 1].astype(int)
    values = rng.normal(size=len(centers)) * 3.0
    kernels = tv.RBF_KERNELS

    def eps(kernel):
        return 0.08 if kernel in ('gaussian', 'multiquadric', 'inverse_multiquadric') else 1.0

    def field(kernel, v=values, c=centers, **kw):
        return tv.rbf_field(c, v, W, H, kernel=kernel, epsilon=eps(kernel), **kw)

    check('six kernels are offered', set(kernels) == {'thin_plate', 'cubic', 'linear', 'gaussian', 'multiquadric',
                                                      'inverse_multiquadric'})
    made = {k: field(k) for k in kernels}
    check('the result is a float32 Field of shape (h, w) and a repeat call gives the same numbers',
          all(isinstance(f, tf.Field) and f.a.dtype == np.float32 and f.a.shape == (H, W)
              and np.array_equal(f.a, field(k).a) for k, f in made.items()))
    worst = max(float(np.abs(f.a[rows, cols] - values).max() / np.ptp(values)) for f in made.values())
    check('every kernel interpolates its centers to 1e-6 relative (smoothing 0)', worst < 1e-6, f'worst {worst:.1e}')
    check('a well-conditioned fit carries no warning in its expr',
          all('warning' not in f.expr for f in made.values()) and made['gaussian'].expr.startswith('rbf(gaussian, n=14'),
          made['gaussian'].expr)

    def reference(kernel, c, f, shape, smoothing):
        """The defining formulas solved as one dense system with a plane term (centers not collinear)."""
        def phi(r):
            if kernel == 'thin_plate':
                with np.errstate(divide='ignore', invalid='ignore'):
                    return np.where(r > 0, r * r * np.log(np.where(r > 0, r, 1.0)), 0.0)
            return {'cubic': lambda: r ** 3, 'linear': lambda: r, 'gaussian': lambda: np.exp(-(shape * r) ** 2),
                    'multiquadric': lambda: np.sqrt(1 + (shape * r) ** 2),
                    'inverse_multiquadric': lambda: 1 / np.sqrt(1 + (shape * r) ** 2)}[kernel]()
        m = len(c)
        poly = np.column_stack([np.ones(m), c])
        dense = np.block([[phi(np.hypot(c[:, None, 0] - c[None, :, 0], c[:, None, 1] - c[None, :, 1]))
                           + smoothing * np.eye(m), poly], [poly.T, np.zeros((3, 3))]])
        sol = np.linalg.solve(dense, np.concatenate([f, np.zeros(3)]))
        return phi(np.hypot(xx[..., None] - c[:, 0], yy[..., None] - c[:, 1])) @ sol[:m] + sol[m] + sol[m + 1] * xx + sol[m + 2] * yy

    few, few_values = centers[:8], values[:8]
    gaps = {}
    for k in kernels:
        for lam in (0.0, 0.7):
            shape = 0.15 if k in ('gaussian', 'multiquadric', 'inverse_multiquadric') else 1.0
            ours = tv.rbf_field(few, few_values, W, H, kernel=k, epsilon=shape, smoothing=lam).a
            gaps[(k, lam)] = float(np.abs(ours - reference(k, few, few_values, shape, lam)).max() / np.ptp(few_values))
    check('all six kernels match a dense solve of their defining formulas, with and without smoothing',
          max(gaps.values()) < 1e-4, f'worst {max(gaps.values()):.1e} at {max(gaps, key=gaps.get)}')
    plane_values = 3.0 + 0.7 * centers[:, 0] - 1.3 * centers[:, 1]
    plane = 3.0 + 0.7 * xx - 1.3 * yy
    worst = max(float(np.abs(field(k, plane_values).a - plane).max()) for k in kernels)
    check('data from a plane is reproduced on the whole grid to 1e-5 (degree 1, every kernel)', worst < 1e-5,
          f'worst {worst:.1e}')
    shift = np.array([37.5, -12.25])
    moved = {k: tv.rbf_field(centers + shift, values, W, H, kernel=k, epsilon=eps(k),
                             bounds=(shift[0], shift[1], shift[0] + W - 1, shift[1] + H - 1)) for k in kernels}
    worst = max(float(np.abs(moved[k].a - made[k].a).max() / np.ptp(values)) for k in kernels)
    check('moving centers and bounds together leaves the field unchanged (translation equivariance)',
          worst < 1e-6, f'worst {worst:.1e}')
    worst = 0.0
    for s in (0.5, 3.0):
        scaled = tv.rbf_field(centers * s, values, W, H, bounds=(0, 0, s * (W - 1), s * (H - 1)))
        worst = max(worst, float(np.abs(scaled.a - made['thin_plate'].a).max() / np.ptp(values)))
    check('thin plate: scaling the coordinates changes nothing (the extra term is absorbed by the plane)',
          worst < 1e-5, f'worst {worst:.1e}')

    left = np.unique(rng.integers(0, [W // 2 - 1, H], (7, 2)), axis=0).astype(float)
    mirrored = np.vstack([left, left * [-1.0, 1.0] + [W - 1.0, 0.0]])
    mirrored_values = np.tile(rng.normal(size=len(left)), 2)
    worst = max(float(np.abs(f.a - f.a[:, ::-1]).max()) for f in
                (field(k, mirrored_values, mirrored) for k in ('thin_plate', 'cubic', 'gaussian')))
    check('mirror-symmetric centers and values give a mirror-symmetric field', worst < 1e-6, f'worst {worst:.1e}')

    noisy = 0.05 * centers[:, 0] + rng.normal(size=len(centers))
    residual = [float(np.linalg.norm(field('thin_plate', noisy, smoothing=s).a[rows, cols] - noisy))
                for s in (0.0, 1e-3, 1e-1, 1.0, 10.0, 1e3, 1e6)]
    check('the residual at the centers never decreases as smoothing grows',
          all(b >= a - 1e-6 for a, b in zip(residual, residual[1:])) and residual[-1] > 100 * residual[0] + 1.0,
          ' '.join(f'{r:.3g}' for r in residual))
    fit = np.linalg.lstsq(np.column_stack([np.ones(len(centers)), centers]), noisy, rcond=None)[0]
    worst = float(np.abs(field('thin_plate', noisy, smoothing=1e12).a - (fit[0] + fit[1] * xx + fit[2] * yy)).max())
    check('smoothing 1e12 tends to the least-squares plane (1e-4)', worst < 1e-4, f'worst {worst:.1e}')
    worst = max(float(np.abs(field(k, np.full(len(centers), 4.5)).a - 4.5).max()) for k in kernels)
    check('constant data gives a constant field', worst < 1e-6, f'worst {worst:.1e}')

    spiky = tv.rbf_field(centers, values, W, H, kernel='gaussian', epsilon=50.0, degree=-1)
    elsewhere = np.ones((H, W), dtype=bool)
    elsewhere[rows, cols] = False
    check('a gaussian with a huge epsilon gives identity weights: the data at the centers, zero between',
          np.abs(spiky.a[rows, cols] - values).max() < 1e-6 and np.abs(spiky.a[elsewhere]).max() < 1e-12)

    # ---- options and data
    check('conditionally positive definite kernels need their polynomial degree',
          all(_raises_value_error(lambda k=k, d=d: field(k, degree=d))
              for k, d in (('thin_plate', 0), ('thin_plate', -1), ('cubic', 0), ('linear', -1), ('multiquadric', -1)))
          and not any(_raises_value_error(lambda k=k, d=d: field(k, degree=d))
                      for k, d in (('thin_plate', 1), ('linear', 0), ('multiquadric', 0), ('gaussian', -1),
                                   ('inverse_multiquadric', -1))))
    check('unknown kernels and out-of-range options raise ValueError',
          all(_raises_value_error(fn) for fn in (
              lambda: field('bessel'), lambda: field('thin_plate', degree=2), lambda: field('thin_plate', smoothing=-1.0),
              lambda: tv.rbf_field(centers, values, W, H, kernel='gaussian', epsilon=0.0),
              lambda: tv.rbf_field(centers, values, 0, H), lambda: tv.rbf_field(centers, values, W, H, bounds=(0, 0, 1)),
              lambda: tv.rbf_field(centers, values, W, H, bounds=(0, 0, np.inf, 1)))))
    try:
        tv.rbf_field(np.vstack([centers, centers[3]]), np.append(values, 1.0), W, H)
        named = ''
    except ValueError as exc:
        named = str(exc)
    check('duplicate centers raise ValueError naming the pair', f'centers 3 and {len(centers)} ' in named, named)
    nan_values = values.copy()
    nan_values[2] = np.nan
    far = centers.copy()
    far[0, 0] = np.inf
    many = np.stack([np.arange(tv.RBF_MAX_CENTERS + 1.0), np.zeros(tv.RBF_MAX_CENTERS + 1)], axis=1)
    check('NaN values, infinite centers, wrong shapes and more than 1500 centers raise ValueError',
          tv.RBF_MAX_CENTERS == 1500 and all(_raises_value_error(fn) for fn in (
              lambda: tv.rbf_field(centers, nan_values, W, H), lambda: tv.rbf_field(far, values, W, H),
              lambda: tv.rbf_field(centers, values[:-1], W, H), lambda: tv.rbf_field(centers[:, :1], values, W, H),
              lambda: tv.rbf_field(many, np.zeros(len(many)), 4, 4))))

    none = tv.rbf_field(np.zeros((0, 2)), np.zeros(0), 5, 3)
    check('no centers: a constant 0 field, or a list of them for vector values',
          none.a.shape == (3, 5) and not none.a.any() and tv.rbf_field([], [], 5, 3).a.shape == (3, 5)
          and len(tv.rbf_field(np.zeros((0, 2)), np.zeros((0, 2)), 5, 3)) == 2)
    one = tv.rbf_field([[3.0, 4.0]], [7.0], 10, 10)
    two = tv.rbf_field([[3.0, 4.0], [4.0, 4.0]], [7.0, 2.0], 10, 10)
    tilted = tv.rbf_field([[3.0, 4.0], [6.0, 8.0]], [7.0, 2.0], 12, 12, kernel='cubic')
    gx, gy = np.meshgrid(np.arange(12.0), np.arange(12.0))
    ramp = 7.0 + (2.0 - 7.0) * ((gx - 3.0) * 3.0 + (gy - 4.0) * 4.0) / 25.0
    check('one center: that value everywhere; two centers: the ramp along the line through them',
          np.allclose(one.a, 7.0) and 'warning' in one.expr
          and np.allclose(two.a[4, :5], [22.0, 17.0, 12.0, 7.0, 2.0], atol=1e-5) and 'ramp' in two.expr
          and np.abs(tilted.a - ramp).max() < 1e-5, two.expr)
    line = np.array([[2.0, 3.0], [5.0, 7.0], [8.0, 11.0], [11.0, 15.0]])         # collinear, direction (3, 4)
    line_values = np.array([1.0, 2.0, 0.0, 3.0])
    on_line = {k: tv.rbf_field(line, line_values, 20, 20, kernel=k, epsilon=0.2) for k in kernels}
    check('collinear centers: finite fields that still interpolate; degree 1 -> 0, or a ramp for thin_plate and cubic',
          all(np.isfinite(f.a).all() and 'warning' in f.expr
              and np.abs(f.a[line[:, 1].astype(int), line[:, 0].astype(int)] - line_values).max() < 1e-5
              for f in on_line.values())
          and 'degree=0' in on_line['linear'].expr and 'degree 1 -> 0' in on_line['gaussian'].expr
          and 'ramp' in on_line['thin_plate'].expr and 'ramp' in on_line['cubic'].expr)
    crowd = np.unique(rng.integers(0, [W, H], (50, 2)), axis=0).astype(float)
    smooth = tv.rbf_field(crowd, rng.normal(size=len(crowd)), W, H, kernel='gaussian', epsilon=0.02)
    check('an ill-conditioned system gets the ridge and says so in its expr; the field stays finite',
          'ridge 1e-10 added' in smooth.expr and np.isfinite(smooth.a).all(), smooth.expr)

    # ---- vectors, bounds, chunks
    pair = tv.rbf_field(centers, np.stack([values, 2.0 * values + 1.0], axis=1), W, H)
    check('(n, d) values give a list of d Fields, each equal to its own scalar fit',
          isinstance(pair, list) and len(pair) == 2 and np.array_equal(pair[0].a, made['thin_plate'].a)
          and np.abs(pair[1].a - (2.0 * made['thin_plate'].a + 1.0)).max() < 1e-4
          and len(tv.rbf_field(centers, values[:, None], W, H)) == 1)
    explicit = tv.rbf_field(centers, values, W, H, bounds=(0, 0, W - 1, H - 1))
    ramp_values = 2.0 + 0.5 * centers[:, 0] - 0.25 * centers[:, 1]
    cropped = tv.rbf_field(centers, ramp_values, 5, 4, bounds=(10.0, 20.0, 50.0, 60.0))
    gx, gy = np.meshgrid(np.linspace(10.0, 50.0, 5), np.linspace(20.0, 60.0, 4))
    check('bounds = (0, 0, w - 1, h - 1) is the default; other bounds place the first and last pixel there',
          np.array_equal(explicit.a, made['thin_plate'].a)
          and np.abs(cropped.a - (2.0 + 0.5 * gx - 0.25 * gy)).max() < 1e-5
          and tv.rbf_field(centers, ramp_values, 1, 1, bounds=(10.0, 20.0, 50.0, 60.0)).a[0, 0]
          == np.float32(2.0 + 0.5 * 10.0 - 0.25 * 20.0))
    with patch.object(tv, '_RBF_CHUNK', 37):
        chunked = {k: field(k) for k in ('thin_plate', 'gaussian')}
    check('the block size of the evaluation does not change the numbers',
          all(np.array_equal(chunked[k].a, made[k].a) for k in chunked))

    pins = np.array([[10.0, 8.0], [30.0, 22.0], [4.0, 26.0], [34.0, 4.0], [20.0, 16.0]])
    pinned = tv.rbf_field(pins, [1.0, -1.0, 0.0, 0.0, 0.0], W, H)
    hills = pinned.a > tg.threshold_for_ratio(pinned, 0.15)
    rings = tv.trace_rings(hills)
    near = min(rings, key=lambda r: np.hypot(*(r.mean(axis=0) - pins[0])))
    check('pins -> ratio threshold -> rings: the high ground is a region around the mountain pin',
          abs(hills.mean() - 0.15) < 0.01 and hills[8, 10] and not hills[22, 30]
          and abs(sum(tv.signed_area(r) for r in rings) - hills.sum()) < 1e-9 and tv.signed_area(near) > 0,
          f'{len(rings)} rings, {hills.mean():.3f} of the map')


# ------------------------------------------- random-order frontier regions

def _frontier_reference(w, h, pts, seed, mask=None, fifo=False):
    """The fill restated on (x, y) tuples with plain list surgery: an open list,
    a pop at index int(u * len), the oldest element moved into the vacated slot.
    `fifo` pops the oldest instead (a breadth-first search). Returns (label,
    depth) dicts."""
    allowed = np.ones((h, w), dtype=bool) if mask is None else mask
    label, depth, open_list = {}, {}, []
    for i, (px, py) in enumerate(pts):
        x = min(max(int(math.floor(px + 0.5)), 0), w - 1)
        y = min(max(int(math.floor(py + 0.5)), 0), h - 1)
        if (x, y) not in label:
            label[(x, y)], depth[(x, y)] = i, 0
            open_list.append((x, y))
    draws = np.random.default_rng(seed).random(int(allowed.sum())).tolist()
    for k in range(len(draws) + 1):
        if not open_list:
            break
        span = len(open_list)
        j = 0 if fifo else min(int(draws[k] * span), span - 1)
        cur = open_list[j]
        open_list[j] = open_list[0]
        del open_list[0]
        for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
            nx, ny = cur[0] + dx, cur[1] + dy
            if 0 <= nx < w and 0 <= ny < h and allowed[ny, nx] and (nx, ny) not in label:
                label[(nx, ny)], depth[(nx, ny)] = label[cur], depth[cur] + 1
                open_list.append((nx, ny))
    return label, depth


def _as_arrays(label, depth, w, h):
    cell = np.full((h, w), -1, dtype=np.int32)
    steps = np.full((h, w), np.inf)
    for (x, y), v in label.items():
        cell[y, x], steps[y, x] = v, depth[(x, y)]
    return cell, steps


def _seam_cells(cell):
    """How many 4-neighbour pairs lie in different regions."""
    return int((cell[:, :-1] != cell[:, 1:]).sum() + (cell[:-1, :] != cell[1:, :]).sum())


def _bfs_distance(open_mask, seeds):
    """Shortest 4-connected path length from the seed pixels through open cells."""
    h, w = open_mask.shape
    dist = np.full((h, w), np.inf)
    queue = deque()
    for x, y in seeds:
        dist[y, x] = 0
        queue.append((x, y))
    while queue:
        x, y = queue.popleft()
        for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and open_mask[ny, nx] and dist[ny, nx] == np.inf:
                dist[ny, nx] = dist[y, x] + 1
                queue.append((nx, ny))
    return dist


def test_random_frontier():
    print("\nrandom-order frontier regions")
    cases = [(64, 64, 12, 1), (96, 96, 19, 3), (80, 50, 10, 7), (33, 70, 9, 4)]
    fills = []
    for w, h, radius, seed in cases:
        sites = tf.poisson_disc(w, h, radius, seed=seed, margin=1)
        fills.append((w, h, sites, seed, tf.random_frontier_regions(w, h, sites, seed=seed)))

    check('without a mask every pixel is claimed by a region that exists',
          all((r.cell >= 0).all() and r.cell.max() < len(s) and r.cell.dtype == np.int32
              and r.cell.shape == (h, w) for w, h, s, _, r in fills))
    check('every site owns its own pixel at depth 0',
          all(r.cell[int(np.floor(y + 0.5)), int(np.floor(x + 0.5))] == i
              and r.depth.a[int(np.floor(y + 0.5)), int(np.floor(x + 0.5))] == 0
              for w, h, s, _, r in fills for i, (x, y) in enumerate(s)))
    check('every region is one 4-connected piece (the 4-connectivity rule holds)',
          all(tf.connected_components(r.cell == i)[1] == 1 for w, h, s, _, r in fills for i in range(len(s))))

    ok = tight = True
    ratios = []
    for w, h, s, _, r in fills:
        ix = np.floor(s[:, 0] + 0.5).astype(int)
        iy = np.floor(s[:, 1] + 0.5).astype(int)
        yy, xx = np.mgrid[0:h, 0:w]
        manhattan = np.abs(xx - ix[r.cell]) + np.abs(yy - iy[r.cell])
        ok &= bool((r.depth.a >= manhattan).all()) and bool((r.depth.a == np.round(r.depth.a)).all())
        ratios.append(float(r.depth.a.mean() / manhattan.mean()))
        # the fill tree: a cell at depth d > 0 has a neighbour of its region at depth d - 1
        parent = np.zeros((h, w), dtype=bool)
        for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
            sh_c = np.full((h, w), -2, dtype=np.int64)
            sh_d = np.full((h, w), -2.0)
            ys0, ys1 = max(0, dy), h + min(0, dy)
            xs0, xs1 = max(0, dx), w + min(0, dx)
            sh_c[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx] = r.cell[ys0:ys1, xs0:xs1]
            sh_d[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx] = r.depth.a[ys0:ys1, xs0:xs1]
            parent |= (sh_c == r.cell) & (sh_d == r.depth.a - 1)
        tight &= bool((parent | (r.depth.a == 0)).all())
    check('depth is a whole number of steps, at least the Manhattan distance to the site', ok)
    check('every cell below the sites hangs off a neighbour one step shallower in its own region', tight)

    same = []
    for w, h, s, seed, r in fills:
        label, depth = _frontier_reference(w, h, s, seed)
        cell, steps = _as_arrays(label, depth, w, h)
        same.append(np.array_equal(cell, r.cell) and np.array_equal(steps, r.depth.a))
    check('regions and depths equal an independent list-surgery restatement of the fill (4 grids)', all(same))
    rng = np.random.default_rng(tf.seed_stream(41, 'test.frontier.mask'))
    masked = []
    for k in range(6):
        w, h = int(rng.integers(8, 40)), int(rng.integers(8, 40))
        blocked = tf.ca_smooth(rng.random((h, w)) < 0.42, 2) & (rng.random((h, w)) < 0.9)
        free = np.argwhere(~blocked)
        if len(free) < 6:
            continue
        pick = free[rng.choice(len(free), size=min(4, len(free)), replace=False)]
        pts = pick[:, ::-1].astype(float) + rng.uniform(-0.4, 0.4, size=pick.shape)
        pts = np.clip(pts, 0, [w - 1, h - 1])
        # sites must round onto open cells for the call to be legal
        keep = [i for i, (x, y) in enumerate(pts)
                if not blocked[min(max(int(np.floor(y + 0.5)), 0), h - 1),
                               min(max(int(np.floor(x + 0.5)), 0), w - 1)]]
        pts = pts[keep]
        got = tf.random_frontier_regions(w, h, pts, seed=k, mask=~blocked)
        label, depth = _frontier_reference(w, h, pts, k, mask=~blocked)
        cell, steps = _as_arrays(label, depth, w, h)
        masked.append(np.array_equal(got.cell, cell) and np.array_equal(got.depth.a, steps)
                      and bool((got.cell[blocked] == -1).all()) and bool(np.isinf(got.depth.a[blocked]).all()))
    check('with a blocking mask the fill still equals the restatement, and never enters a blocked cell',
          len(masked) >= 4 and all(masked), f'{len(masked)} random masks')

    # exact cases
    strip = tf.random_frontier_regions(40, 1, [(7, 0)], seed=5)
    check('a 1 x 40 strip has one path, so depth is |x - 7| whatever the order',
          (strip.cell == 0).all() and np.array_equal(strip.depth.a[0], np.abs(np.arange(40) - 7)))
    two = tf.random_frontier_regions(30, 1, [(4, 0), (22, 0)], seed=9)
    cuts = np.flatnonzero(np.diff(two.cell[0]) != 0)
    check('two sites on a strip split it into two intervals, the cut between them',
          len(cuts) == 1 and 4 <= cuts[0] < 22 and (two.cell[0, :cuts[0] + 1] == 0).all()
          and (two.cell[0, cuts[0] + 1:] == 1).all())
    one = tf.random_frontier_regions(21, 15, [(10, 7)], seed=3)
    manhattan = np.abs(np.arange(21)[None, :] - 10) + np.abs(np.arange(15)[:, None] - 7)
    fifo_label, fifo_depth = _frontier_reference(21, 15, [(10, 7)], 3, fifo=True)
    _, fifo_steps = _as_arrays(fifo_label, fifo_depth, 21, 15)
    check('one site claims the grid; a first-in-first-out list would give the Manhattan distance exactly',
          (one.cell == 0).all() and np.array_equal(fifo_steps, manhattan) and (one.depth.a >= manhattan).all())
    check('the random pop is what makes depth noisy: mean depth is at least 3% above Manhattan',
          float(one.depth.a.mean()) > 1.03 * float(manhattan.mean()),
          f'{one.depth.a.mean():.2f} vs {manhattan.mean():.2f}')
    check('on every test grid the mean depth exceeds the Manhattan mean', all(g > 1.0 for g in ratios),
          ', '.join(f'{g:.2f}' for g in ratios))

    # borders against exact Voronoi
    longer = []
    for w, h, s, _, r in fills:
        vor = tf.voronoi(w, h, s)
        longer.append(int(r.border_mask().sum()) / int(vor.border_mask().sum()))
    check('seams are jagged: borders are longer than the Voronoi bisectors on every grid (+5% or more)',
          min(longer) > 1.05, ', '.join(f'{x:.2f}' for x in longer))
    differ = [float((r.cell != tf.voronoi(w, h, s).cell).mean()) for w, h, s, _, r in fills]
    check('and the partition differs from Voronoi on 5% to 40% of the pixels',
          all(0.05 < d < 0.40 for d in differ), ', '.join(f'{d:.2f}' for d in differ))
    w, h, s, _, r = fills[1]
    graph = {i: set() for i in range(len(s))}
    for (a, b) in r.adjacency():
        graph[a].add(b)
        graph[b].add(a)
    seen, todo = {0}, [0]
    while todo:
        for nb in graph[todo.pop()]:
            if nb not in seen:
                seen.add(nb)
                todo.append(nb)
    check('adjacency() and border_mask() work on the result like a VoronoiResult',
          len(seen) == len(s) and r.border_mask().any() and not r.border_mask(1).all())
    c = r.centroids()
    xs = np.bincount(r.cell.ravel(), weights=np.mgrid[0:h, 0:w][1].ravel(), minlength=len(s))
    cnt = np.bincount(r.cell.ravel(), minlength=len(s))
    whole = tf.random_frontier_regions(9, 5, [(0, 0)], seed=1).centroids()
    check('centroids are the mean pixel of each region (and (4, 2) for one region on a 9 x 5 grid)',
          np.allclose(c[:, 0], xs / cnt) and np.allclose(whole, [[4.0, 2.0]]))

    # the distance-field recipe: seeds from a mask, stop cells as the mask
    w, h = 41, 29
    walls = np.zeros((h, w), dtype=bool)
    walls[:, 20] = True
    walls[10:19, 20] = False                       # a doorway in the wall
    walls[0:5, 30:35] = True
    seeds = np.zeros((h, w), dtype=bool)
    seeds[3, 3] = seeds[25, 5] = True
    near = tf.random_frontier_regions(w, h, np.argwhere(seeds)[:, ::-1], seed=8, mask=~walls)
    truth = _bfs_distance(~walls, [(3, 3), (5, 25)])
    check('the distance recipe: 0 on the seeds, never below the shortest path, +inf only on the stop cells',
          near.depth.a[3, 3] == 0 and near.depth.a[25, 5] == 0 and (near.depth.a[~walls] >= 0).all()
          and np.isfinite(near.depth.a[~walls]).all() and np.isinf(near.depth.a[walls]).all()
          and bool((near.depth.a[~walls] >= truth[~walls]).all()))
    sealed = walls.copy()
    sealed[:, 20] = True
    cut = tf.random_frontier_regions(w, h, [(3, 3)], seed=8, mask=~sealed)
    check('a pocket the fill cannot reach stays at region -1 and depth +inf',
          (cut.cell[:, :20] == 0).sum() == (~sealed[:, :20]).sum() and (cut.cell[:, 21:] == -1).all()
          and np.isinf(cut.depth.a[:, 21:]).all())

    # determinism and the argument contract
    w, h, s, seed, r = fills[0]
    again = tf.random_frontier_regions(w, h, s, seed=seed)
    other = tf.random_frontier_regions(w, h, s, seed=seed + 1)
    check('equal arguments give equal regions; another seed gives other borders',
          np.array_equal(r.cell, again.cell) and not np.array_equal(r.cell, other.cell))
    dup = tf.random_frontier_regions(10, 10, [(3.2, 3.1), (2.9, 3.4), (7, 7)], seed=1)
    check('two sites rounding to one pixel: the lower index keeps it, the other region is empty (NaN centroid)',
          dup.cell[3, 3] == 0 and not (dup.cell == 1).any() and np.isnan(dup.centroids()[1]).all()
          and (dup.cell >= 0).all())
    edge = tf.random_frontier_regions(8, 6, [(-3.0, 2.0), (50.0, 9.0)], seed=1)
    check('sites outside the grid are clipped onto its edge',
          edge.cell[2, 0] == 0 and edge.cell[5, 7] == 1)
    nothing = tf.random_frontier_regions(7, 5, np.zeros((0, 2)), seed=1)
    check('no sites: nothing is claimed', (nothing.cell == -1).all() and np.isinf(nothing.depth.a).all())
    check('bad sizes, masks, sites and blocked sites raise ValueError', all(_raises_value_error(f) for f in (
        lambda: tf.random_frontier_regions(0, 5, [(0, 0)]),
        lambda: tf.random_frontier_regions(5, 5, [(0, 0)], mask=np.ones((4, 5), dtype=bool)),
        lambda: tf.random_frontier_regions(5, 5, [(float('nan'), 0)]),
        lambda: tf.random_frontier_regions(5, 5, [(2, 2)], mask=np.arange(25).reshape(5, 5) != 12))))

    # the tectonic option
    print("\ntectonic worlds: plate_shape")
    seeds = (3, 9, 17)
    builds = {}
    for shape in ('voronoi', 'frontier'):
        for sd in seeds:
            spec = {'id': 711, 'size': 64, 'template': 'tectonic', 'seed': sd, 'plate_shape': shape}
            builds[shape, sd] = tg.GRAMMARS['tectonic'](spec, random.Random(sd))
    plain = {sd: tg.GRAMMARS['tectonic']({'id': 711, 'size': 64, 'template': 'tectonic', 'seed': sd},
                                         random.Random(sd)) for sd in seeds}
    exact = []
    for sd in seeds:
        sites = tf.poisson_disc(64, 64, 12, seed=tg.grammar_seed({'id': 711, 'seed': sd}, 'env.tectonic'), margin=1)
        exact.append(np.array_equal(plain[sd].fields['plate'], tf.voronoi(64, 64, sites, relax=2).cell))
    check("the default is 'voronoi': no plate_shape and plate_shape='voronoi' give the same map, "
          "whose plates are the exact Voronoi cells of the Poisson-disc sites",
          all(plain[sd].base == builds['voronoi', sd].base and plain[sd].overlay == builds['voronoi', sd].overlay
              and np.array_equal(plain[sd].fields['plate'], builds['voronoi', sd].fields['plate'])
              and plain[sd].plates == builds['voronoi', sd].plates for sd in seeds) and all(exact))
    frontier_ok = True
    for sd in seeds:
        tm = builds['frontier', sd]
        plate = tm.fields['plate']
        count = tm.plates['count']
        frontier_ok &= (sorted(np.unique(plate).tolist()) == list(range(count)) and count >= 4
                        and all(tf.connected_components(plate == i)[1] == 1 for i in range(count))
                        and tf.connected_components(tg.passable_mask(tm))[1] == 1
                        and (tm.w, tm.h) == (64, 64))
    check('frontier plates: every plate exists and is one 4-connected piece; the map is one component',
          frontier_ok)
    check('frontier maps are deterministic per seed and differ from the Voronoi maps',
          all(builds['frontier', sd].base == tg.GRAMMARS['tectonic'](
              {'id': 711, 'size': 64, 'template': 'tectonic', 'seed': sd, 'plate_shape': 'frontier'},
              random.Random(sd)).base and builds['frontier', sd].base != builds['voronoi', sd].base
              for sd in seeds))
    check('the fill has its own stream: plate motions equal the Voronoi ones when the plate count is equal',
          all(builds['frontier', sd].plates['motion'] == builds['voronoi', sd].plates['motion']
              for sd in seeds if builds['frontier', sd].plates['count'] == builds['voronoi', sd].plates['count']))
    seams = {shape: sum(_seam_cells(builds[shape, sd].fields['plate']) for sd in seeds)
             for shape in ('voronoi', 'frontier')}
    check('plate seams are longer with the frontier fill (jagged), on the finished maps',
          seams['frontier'] > 1.04 * seams['voronoi'], f"{seams['frontier']} vs {seams['voronoi']} seam cells")
    stressed = all((builds['frontier', sd].fields['tectonic_stress'] > 0).any() for sd in seeds)
    kinds = set()
    for sd in seeds:
        kinds |= {t for row in builds['frontier', sd].base for t in row}
    check('boundaries are still typed: stress is painted, and mountains, rifts and land all appear',
          stressed and {'wall', 'abyss', 'floor'} <= kinds, str(sorted(kinds)))
    wide = tg.GRAMMARS['tectonic']({'id': 712, 'width': 90, 'height': 52, 'template': 'tectonic', 'seed': 4,
                                    'plate_shape': 'frontier', 'plate_size': 11, 'relax': 0}, random.Random(4))
    check('a non-square map with relax 0 and a smaller plate_size also works',
          (wide.w, wide.h) == (90, 52) and tf.connected_components(tg.passable_mask(wide))[1] == 1
          and wide.plates['count'] > 8)
    check("an unknown plate_shape raises ValueError",
          _raises_value_error(lambda: tg.GRAMMARS['tectonic'](
              {'id': 713, 'size': 32, 'template': 'tectonic', 'seed': 1, 'plate_shape': 'hexagon'}, random.Random(1))))


# ------------------------------------------------------------- ConvChain

def _d4_variants(window):
    """The eight dihedral variants of a square window (tuples of rows), each counted
    even when equal: mirror or not, then 0 to 3 quarter turns."""
    out = []
    for mirror in (False, True):
        cur = [row[::-1] for row in window] if mirror else [list(row) for row in window]
        for _ in range(4):
            out.append(tuple(tuple(row) for row in cur))
            cur = [list(row) for row in zip(*cur[::-1])]
    return out


def _window_counts(sample, n, periodic, symmetry):
    """pattern code -> number of occurrences, by plain loops. A pattern's code is
    sum(bit << (dy * n + dx))."""
    h, w = len(sample), len(sample[0])
    picks = {1: [0], 2: [0, 4], 4: [0, 1, 2, 3], 8: list(range(8))}[symmetry]
    counts = Counter()
    for y in range(h if periodic else h - n + 1):
        for x in range(w if periodic else w - n + 1):
            win = [[int(sample[(y + dy) % h][(x + dx) % w]) for dx in range(n)] for dy in range(n)]
            variants = _d4_variants(win)
            for i in picks:
                counts[sum(variants[i][dy][dx] << (dy * n + dx) for dy in range(n) for dx in range(n))] += 1
    return counts


def _score_loops(grid, counts, floor, n, periodic):
    """log score of a grid, by plain loops over its windows."""
    h, w = grid.shape
    total = 0.0
    for y in range(h if periodic else h - n + 1):
        for x in range(w if periodic else w - n + 1):
            code = sum(int(grid[(y + dy) % h, (x + dx) % w]) << (dy * n + dx)
                       for dy in range(n) for dx in range(n))
            total += math.log(counts.get(code, floor))
    return total


def _reference_chain(counts, floor, n, w, h, T, sweeps, periodic, seed, fill, free=None, init=None):
    """The sampler restated without any vectorisation: the same draws in the same
    order, but every cell flipped on its own, its ratio taken from two whole-grid
    scores. Classes come from `cc.ring_colours`."""
    rng = tf.rng_for(seed, 'convchain')
    free = np.ones((h, w), dtype=bool) if free is None else free
    grid = ((rng.random((h, w)) < fill) & free) if init is None else np.array(init, dtype=bool)
    cx, nx = cc.ring_colours(w, n, periodic)
    cy, ny = cc.ring_colours(h, n, periodic)
    members = {k: [] for k in range(nx * ny)}
    for y in range(h):
        for x in range(w):
            if free[y, x]:
                members[cy[y] * nx + cx[x]].append((x, y))
    for _ in range(sweeps):
        for k in rng.permutation(nx * ny):
            if not members[k]:
                continue
            draws = rng.random(len(members[k]))
            for (x, y), u in zip(members[k], draws):
                before = _score_loops(grid, counts, floor, n, periodic)
                grid[y, x] ^= True
                ratio = _score_loops(grid, counts, floor, n, periodic) - before
                if not u < math.exp(min(ratio * (1.0 / T), 0.0)):
                    grid[y, x] ^= True
    return grid


def _exact_moments(counts, floor, n, w, h, periodic, T):
    """E[open cells] and E[log score] of the target distribution on a tiny grid,
    by enumerating every state: pi proportional to exp(score / T)."""
    states = np.arange(1 << (w * h), dtype=np.int64)
    bits = ((states[:, None] >> np.arange(w * h)) & 1).astype(bool).reshape(-1, h, w)
    codes = 0
    for dy in range(n):
        for dx in range(n):
            part = np.roll(bits, (-dy, -dx), axis=(1, 2)) if periodic else bits[:, dy:dy + h - n + 1, dx:dx + w - n + 1]
            codes = codes + (part.astype(np.int64) << (dy * n + dx))
    table = np.full(1 << (n * n), math.log(floor))
    for code, c in counts.items():
        table[code] = math.log(c)
    score = table[codes].sum(axis=(1, 2))
    p = np.exp(score / T - (score / T).max())
    p /= p.sum()
    return float((p * bits.sum(axis=(1, 2))).sum()), float((p * score).sum())


def test_convchain():
    print("\nConvChain: Metropolis synthesis from a bilevel sample")
    rng = np.random.default_rng(tf.seed_stream(52, 'test.convchain'))

    # the weights are window counts with the dihedral symmetries, cut by the WFC's PatternModel
    ok = True
    for h, w, n, periodic, symmetry in ((7, 6, 2, True, 8), (9, 9, 3, True, 8), (9, 7, 3, False, 8),
                                        (8, 8, 3, True, 4), (8, 9, 2, False, 2), (6, 7, 4, True, 1)):
        sample = rng.random((h, w)) < 0.45
        model = cc.ConvChainModel(sample, n, symmetry, periodic)
        counts = _window_counts(sample.tolist(), n, periodic, symmetry)
        ok &= (dict(zip(model.keys.tolist(), model.weights.tolist())) == {k: float(v) for k, v in counts.items()}
               and bool((np.diff(model.keys) > 0).all()) and isinstance(model.patterns, wfc.PatternModel)
               and model.weights.sum() == sum(counts.values()))
    check('weights equal plain-loop window counts, every symmetry subgroup, wrapped or not (6 samples)', ok)
    lone = cc.ConvChainModel(np.array([[1, 1], [1, 0]], dtype=bool), 2, 1, False, floor=0.25)
    check('a pattern the sample lacks weighs the floor; a seen one weighs its count',
          np.allclose(lone.log_weight_of([0b0111, 0b1111, 0]), [0.0, math.log(0.25), math.log(0.25)]),
          'code 0b0111 is the sample window: bits are row-major, top-left lowest')
    sample = rng.random((10, 12)) < 0.4
    model = cc.ConvChainModel(sample, 3)
    check('every window of the sample is known to its own model (coverage 1.0), noise is not',
          model.coverage(sample, periodic=True) == 1.0 and model.coverage(rng.random((30, 30)) < 0.5) < 0.9)

    # scores and single-flip ratios against plain loops
    ok = ratio_ok = True
    worst = 0.0
    for h, w, n, periodic in ((9, 7, 2, False), (9, 7, 3, True), (6, 6, 4, False), (8, 11, 3, False), (5, 5, 5, True)):
        sample = rng.random((8, 9)) < 0.5
        model = cc.ConvChainModel(sample, n, 8, True, 0.1)
        counts = _window_counts(sample.tolist(), n, True, 8)
        grid = rng.random((h, w)) < 0.5
        ok &= abs(model.log_weight(grid, periodic) - _score_loops(grid, counts, 0.1, n, periodic)) < 1e-9
        cells = [(x, y) for y in range(h) for x in range(w)]
        got = model.flip_log_ratio(grid, cells, periodic)
        base = _score_loops(grid, counts, 0.1, n, periodic)
        for (x, y), g in zip(cells, got):
            flipped = grid.copy()
            flipped[y, x] ^= True
            worst = max(worst, abs(g - (_score_loops(flipped, counts, 0.1, n, periodic) - base)))
    check('log_weight equals a plain-loop sum of log counts (5 grids, wrapped and not)', ok)
    check('flipping a cell changes the score by the product of its N x N covering windows: ratios match '
          'two full scores for every cell', worst < 1e-9, f'worst {worst:.1e}')

    # colour classes: two cells of one class share no window
    clash = 0
    for w, h, n, periodic in ((7, 5, 3, True), (10, 6, 3, True), (9, 9, 3, True), (5, 5, 5, True), (11, 4, 4, True),
                              (13, 8, 2, True), (9, 7, 3, False), (6, 12, 4, False)):
        cx, nx = cc.ring_colours(w, n, periodic)
        cy, ny = cc.ring_colours(h, n, periodic)
        ids = cy[:, None] * nx + cx[None, :]
        cells = [(x, y) for y in range(h) for x in range(w)]
        for i, (x1, y1) in enumerate(cells):
            for x2, y2 in cells[i + 1:]:
                if ids[y1, x1] != ids[y2, x2]:
                    continue
                dx, dy = abs(x1 - x2), abs(y1 - y2)
                if periodic:
                    dx, dy = min(dx, w - dx), min(dy, h - dy)
                clash += dx < n and dy < n
    check('colour classes: no two cells of a class are closer than N in both axes (8 sizes, torus and plane)',
          clash == 0, f'{clash} clashes')
    check('a torus whose side is not a multiple of N uses unequal blocks (5 -> 3 colours for N = 2)',
          cc.ring_colours(5, 2, True)[1] == 3 and cc.ring_colours(6, 2, True)[1] == 2
          and cc.ring_colours(5, 2, False)[1] == 2)

    # the vectorised sweeps equal a one-cell-at-a-time restatement with the same draws
    same = []
    for w, h, n, periodic, T, free_rim in ((9, 7, 3, False, 1.0, False), (9, 7, 3, True, 0.7, False),
                                           (8, 6, 2, True, 1.0, False), (11, 5, 4, True, 1.5, False),
                                           (10, 9, 3, False, 0.8, True), (7, 7, 2, True, 1.0, True)):
        sample = rng.random((10, 10)) < 0.4
        model = cc.ConvChainModel(sample, n, 8, True, 0.1)
        counts = _window_counts(sample.tolist(), n, True, 8)
        free = None
        if free_rim:
            free = np.ones((h, w), dtype=bool)
            free[0, :] = free[-1, :] = free[:, 0] = free[:, -1] = False
        got, stats = model.sample(w, h, T, 5, periodic, seed=11, free=free, fill=0.4)
        want = _reference_chain(counts, 0.1, n, w, h, T, 5, periodic, 11, 0.4, free)
        same.append(np.array_equal(got, want) and len(stats['flips']) == 5)
    check('the colour-class sweeps give exactly the grid of a one-flip-at-a-time reference with the same draws '
          '(6 setups: torus, plane, fixed rim, N = 2..4)', all(same), str(same))
    start, _ = cc.ConvChainModel(sample, 3).sample(12, 9, sweeps=0, seed=5)
    check('draw order: with no sweeps the grid is the fair-coin fill of the named stream convchain',
          np.array_equal(start, tf.rng_for(5, 'convchain').random((9, 12)) < 0.5))

    # stationary distribution, against exact enumeration on tiny grids
    zs = []
    for (w, h, n, periodic, symmetry, T) in ((4, 4, 2, True, 4, 0.5), (5, 3, 2, True, 4, 1.0), (4, 4, 2, False, 8, 0.7)):
        sample = np.array([[1, 1, 0, 1, 0, 0], [1, 0, 0, 1, 0, 1], [0, 0, 1, 1, 0, 1],
                           [0, 1, 1, 0, 0, 1], [1, 1, 0, 0, 1, 1], [1, 0, 0, 1, 1, 0]], dtype=bool)
        model = cc.ConvChainModel(sample, n, symmetry, True, 0.1)
        counts = _window_counts(sample.tolist(), n, True, symmetry)
        exact_open, exact_score = _exact_moments(counts, 0.1, n, w, h, periodic, T)
        runs = []
        for seed in range(10):
            seen = []
            model.sample(w, h, T, 260, periodic, seed=seed,
                         on_sweep=lambda k, g: seen.append((g.sum(), model.log_weight(g, periodic))) if k > 40 else None)
            runs.append(np.mean(seen, axis=0))
        runs = np.array(runs)
        mean, err = runs.mean(axis=0), runs.std(axis=0, ddof=1) / math.sqrt(len(runs))
        zs.append((((mean[0] - exact_open) / err[0]), ((mean[1] - exact_score) / err[1])))
    check('the chain samples exp(score / T): mean open cells and mean score agree with exact enumeration '
          '(4x4 torus T=0.5, 5x3 torus T=1, 4x4 plane T=0.7; within 4 standard errors)',
          all(abs(z) < 4 for pair in zs for z in pair),
          ', '.join(f'{a:+.1f}/{b:+.1f}' for a, b in zs) + ' sigma')

    # N = 1 has a closed form: each cell is independent, P(open) = w1^(1/T) / (w0^(1/T) + w1^(1/T))
    sample = np.arange(100).reshape(10, 10) % 10 < 7                    # 70 open, 30 closed
    cases = []
    for floor, T in ((0.1, 1.0), (0.1, 0.5), (0.1, 2.0)):
        model = cc.ConvChainModel(sample, 1, 8, True, floor)
        w1, w0 = 70 * 8.0, 30 * 8.0
        want = w1 ** (1 / T) / (w0 ** (1 / T) + w1 ** (1 / T))
        seen = []
        model.sample(30, 30, T, 60, seed=4, on_sweep=lambda k, g: seen.append(g.mean()) if k > 20 else None)
        cases.append(bool(abs(np.mean(seen) - want) < 0.03))
    allopen = cc.ConvChainModel(np.ones((6, 6), dtype=bool), 1, 8, True, floor=200.0)
    seen = []
    allopen.sample(30, 30, 1.0, 60, seed=4, on_sweep=lambda k, g: seen.append(g.mean()) if k > 20 else None)
    cases.append(bool(abs(np.mean(seen) - 288.0 / (288.0 + 200.0)) < 0.03))
    check('N = 1: the open share equals the closed form at T = 1, 0.5, 2, and the floor weighs unseen patterns',
          all(cases), str(cases))

    # free cells, start states, errors
    sample = rng.random((12, 12)) < 0.4
    model = cc.ConvChainModel(sample, 3)
    pinned = np.zeros((14, 15), dtype=bool)
    pinned[3:6, 4:9] = True
    free = np.ones((14, 15), dtype=bool)
    free[3:6, 4:9] = False
    free[0, :] = free[-1, :] = free[:, 0] = free[:, -1] = False
    got, stats = model.sample(15, 14, 1.0, 20, seed=3, init=pinned, free=free)
    check('cells outside `free` keep their `init` value; the rim stays closed; free cells move',
          (got[3:6, 4:9]).all() and not got[0].any() and not got[-1].any() and not got[:, 0].any()
          and not got[:, -1].any() and sum(stats['flips']) > 0)
    frozen, stats = model.sample(15, 14, 1.0, 5, seed=3, init=pinned, free=np.zeros((14, 15), dtype=bool))
    check('with no free cell nothing flips and `init` comes back', np.array_equal(frozen, pinned)
          and stats['flips'] == [0] * 5)
    again, _ = model.sample(15, 14, 1.0, 20, seed=3, init=pinned, free=free)
    other, _ = model.sample(15, 14, 1.0, 20, seed=4, init=pinned, free=free)
    check('equal arguments give equal grids; another seed another grid',
          np.array_equal(got, again) and not np.array_equal(got, other))
    check('the model is built once and reused: convchain() equals model.sample()',
          np.array_equal(cc.convchain(sample, 15, 14, n=3, sweeps=12, seed=9)[0],
                         cc.ConvChainModel(sample, 3).sample(15, 14, sweeps=12, seed=9)[0]))
    check('bad samples, options and shapes raise ValueError', all(_raises_value_error(f) for f in (
        lambda: cc.ConvChainModel(np.arange(12).reshape(3, 4)),
        lambda: cc.ConvChainModel(np.zeros(5, dtype=bool)),
        lambda: cc.ConvChainModel(np.zeros((6, 6), dtype=bool), floor=0.0),
        lambda: cc.ConvChainModel(np.zeros((2, 2), dtype=bool), 3),
        lambda: model.sample(10, 10, temperature=0.0),
        lambda: model.sample(10, 10, sweeps=-1),
        lambda: model.sample(2, 10),
        lambda: model.sample(10, 10, fill=1.5),
        lambda: model.sample(10, 10, init=np.zeros((9, 10), dtype=bool)),
        lambda: model.sample(10, 10, free=np.zeros((10, 9), dtype=bool)),
        lambda: model.flip_log_ratio(np.zeros((6, 6), dtype=bool), [(6, 0)]),
        lambda: model.window_codes(np.zeros((2, 8), dtype=bool)))))

    # the shipped layouts as samples: what carries over is local structure
    records = {rec['id']: rec for rec in rt.load_layout_catalog(HERE / 'tilegen_output')}
    rows = []
    for rid in (1, 3, 5, 7, 9):
        sample = tg.passable_mask(records[rid]['tilemap'])
        model = cc.ConvChainModel(sample, 3)
        before, _ = model.sample(48, 48, sweeps=0, seed=2, fill=float(sample.mean()))
        after, stats = model.sample(48, 48, sweeps=30, seed=2, fill=float(sample.mean()))
        rows.append((rid, model.coverage(before), model.coverage(after), model.log_weight(before) / 2116,
                     model.log_weight(after) / 2116, stats['flips'][0], stats['flips'][-1]))
    check('layouts 1, 3, 5, 7, 9 (N = 3, T = 1): 30 sweeps turn noise into windows the layout contains '
          '(at least 90% known, from at most 60%)',
          all(r[2] >= 0.90 and r[1] <= 0.60 for r in rows),
          ' '.join(f'{r[0]}:{r[1]:.2f}->{r[2]:.2f}' for r in rows))
    check('the mean log weight per window rises by at least 3 and the flips die down',
          all(r[4] - r[3] >= 3.0 and r[6] < 0.25 * r[5] for r in rows),
          ' '.join(f'{r[3]:.1f}->{r[4]:.1f}' for r in rows))
    rec = records[7]
    tm = cc.convchain_tilemap(rec['tilemap'], 48, 40, sweeps=30, seed=6)
    as_mask = cc.convchain_tilemap(tg.passable_mask(rec['tilemap']), 48, 40, sweeps=30, seed=6)
    open_cells = tg.passable_mask(tm)
    check('convchain_tilemap: a wall/floor TileMap of the asked size, one 4-connected region, sealed rim',
          (tm.w, tm.h) == (48, 40) and {t for row in tm.base for t in row} <= {'wall', 'floor'}
          and tf.connected_components(open_cells)[1] == 1 and 0.10 < open_cells.mean() < 0.90
          and not (open_cells[0].any() or open_cells[-1].any() or open_cells[:, 0].any() or open_cells[:, -1].any()))
    check('a TileMap sample and its passable mask give the same map; the same seed repeats it',
          tm.base == as_mask.base and tm.base == cc.convchain_tilemap(rec['tilemap'], 48, 40, sweeps=30, seed=6).base
          and tm.base != cc.convchain_tilemap(rec['tilemap'], 48, 40, sweeps=30, seed=7).base)
    sparse = records[1]['tilemap']                                       # 19% open
    thin = tg.passable_mask(cc.convchain_tilemap(sparse, 40, 40, sweeps=0, seed=3, rim=False)).mean()
    half = tg.passable_mask(cc.convchain_tilemap(sparse, 40, 40, sweeps=0, seed=3, rim=False, fill=0.5)).mean()
    check("with no sweeps convchain_tilemap shows the start state: the sample's own open share by default "
          "(under 20% for a 19% layout), or `fill` (0.5 -> about half, 0 -> all wall, 1 -> all floor)",
          thin < 0.20 and half > 0.40
          and not tg.passable_mask(cc.convchain_tilemap(sparse, 20, 20, sweeps=0, rim=False, fill=0.0)).any()
          and tg.passable_mask(cc.convchain_tilemap(sparse, 20, 20, sweeps=0, rim=False, fill=1.0)).all(),
          f'{thin:.2f} by default, {half:.2f} with fill 0.5')
    check("`join` accepts 'star' and a bad value is refused",
          tf.connected_components(tg.passable_mask(
              cc.convchain_tilemap(rec['tilemap'], 40, 40, sweeps=10, seed=1, join='star')))[1] == 1
          and _raises_value_error(lambda: cc.convchain_tilemap(rec['tilemap'], 40, 40, sweeps=4, join='ring')))


# ------------------------------------------------------- diagonal kisses

def _kisses_loops(m):
    """Top-left corners of the checkerboard 2 x 2 blocks, by plain loops."""
    h, w = m.shape
    return {(x, y) for y in range(h - 1) for x in range(w - 1)
            if (m[y, x] and m[y + 1, x + 1] and not m[y, x + 1] and not m[y + 1, x])
            or (m[y, x + 1] and m[y + 1, x] and not m[y, x] and not m[y + 1, x + 1])}


def _components8(m):
    """8-connected labelling of a bool mask by plain flood fill: (labels, count)."""
    h, w = m.shape
    labels = np.zeros((h, w), dtype=int)
    count = 0
    for y in range(h):
        for x in range(w):
            if m[y, x] and not labels[y, x]:
                count += 1
                labels[y, x] = count
                stack = [(y, x)]
                while stack:
                    cy, cx = stack.pop()
                    for ny in range(max(0, cy - 1), min(h, cy + 2)):
                        for nx in range(max(0, cx - 1), min(w, cx + 2)):
                            if m[ny, nx] and not labels[ny, nx]:
                                labels[ny, nx] = count
                                stack.append((ny, nx))
    return labels, count


def _open_neighbours(m, x, y):
    h, w = m.shape
    return sum(bool(m[y + dy, x + dx]) for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0))
               if 0 <= x + dx < w and 0 <= y + dy < h)


def _cave_xml(seed, w=64, h=64):
    """The cave of MarkovJunior's models/Cave.xml: 43.5% of cells D (wall), then
    A -> D with 5 to 8 wall neighbours and D -> A with 6 to 8 open neighbours,
    both together, until nothing changes. Returns the open (A) mask."""
    wall = np.random.default_rng(seed).random((h, w)) < 0.435
    for _ in range(200):
        wall_n = tf.neighbour_count(wall, outside=False)
        open_n = tf.neighbour_count(~wall, outside=False)
        new = wall.copy()
        new[~wall & (wall_n >= 5)] = True
        new[wall & (open_n >= 6)] = False
        if (new == wall).all():
            break
        wall = new
    return ~wall


def test_kiss_cleanup():
    print("\ndiagonal kisses")
    patterns = [[[(i >> (2 * r + c)) & 1 == 1 for c in range(2)] for r in range(2)] for i in range(16)]
    flagged = [i for i, p in enumerate(patterns) if co.count_kisses(np.array(p)) == 1]
    check('of the 16 possible 2 x 2 blocks exactly the two checkerboards are kisses',
          flagged == [6, 9] and patterns[6] == [[False, True], [True, False]]
          and patterns[9] == [[True, False], [False, True]] and sum(co.count_kisses(np.array(p)) for p in patterns) == 2,
          f'blocks {flagged}')
    rng = np.random.default_rng(tf.seed_stream(71, 'test.kisses'))
    scans = []
    for _ in range(30):
        m = rng.random((int(rng.integers(2, 20)), int(rng.integers(2, 20)))) < rng.uniform(0.2, 0.8)
        xs, ys = co.kiss_blocks(m)
        scans.append(set(zip(xs.tolist(), ys.tolist())) == _kisses_loops(m) and len(xs) == co.count_kisses(m)
                     and list(zip(ys, xs)) == sorted(zip(ys, xs)))
    check('kiss_blocks equals a plain-loop scan on 30 random masks, in row-major order', all(scans))
    check('masks without a 2 x 2 block, or of one colour, or as integers, behave',
          co.count_kisses(np.ones((1, 7), dtype=bool)) == 0 and co.count_kisses(np.zeros((6, 6), dtype=bool)) == 0
          and co.count_kisses(np.ones((6, 6), dtype=bool)) == 0
          and co.count_kisses(np.array([[1, 0], [0, 1]])) == 1 and co.count_kisses(np.zeros((0, 4), dtype=bool)) == 0)

    # exact cases
    staircases = []
    for n in (2, 5, 9):
        for line in (np.eye(n, dtype=bool), np.eye(n, dtype=bool)[:, ::-1]):
            out, opened = co.open_diagonal_kisses(line, seed=n)
            staircases.append(co.count_kisses(line) == n - 1 and len(opened) == n - 1 and int(out.sum()) == 2 * n - 1
                              and tf.connected_components(out)[1] == 1 and co.count_kisses(out) == 0
                              and not (line & ~out).any())
    check('a diagonal of n cells has n - 1 kisses; clearing them opens exactly n - 1 cells, a staircase of '
          '2n - 1 cells in one 4-connected piece (n = 2, 5, 9, both slants)', all(staircases), str(staircases))
    lone = _mask_from_rows(['....', '.#..', '..#.', '....'])
    picks = Counter()
    for seed in range(120):
        out, opened = co.open_diagonal_kisses(lone, seed)
        picks[tuple(opened)] += 1
    check('one kiss: exactly one of its two blocking cells opens, and chance picks each about half the time',
          set(picks) == {((2, 1),), ((1, 2),)} and all(40 <= v <= 80 for v in picks.values()), str(dict(picks)))
    cascade = _mask_from_rows(['.#..', '#...', '..#.', '....'])
    sizes, clean = Counter(), True
    for seed in range(60):
        out, opened = co.open_diagonal_kisses(cascade, seed)
        sizes[len(opened)] += 1
        clean &= co.count_kisses(out) == 0
    check('opening a blocker can make a new kiss, which is scanned for and cleared (1 or 2 cells opened)',
          clean and set(sizes) == {1, 2} and min(sizes.values()) >= 10, str(dict(sizes)))
    shared = _mask_from_rows(['.#..', '#.#.', '....'])
    sizes, clean = Counter(), True
    for seed in range(60):
        out, opened = co.open_diagonal_kisses(shared, seed)
        sizes[len(opened)] += 1
        clean &= co.count_kisses(out) == 0
    check('two kisses sharing a blocker: when it is picked the second kiss is cleared too (1 or 2 cells opened)',
          clean and set(sizes) == {1, 2}, str(dict(sizes)))

    # invariants on random masks
    ok_none = ok_sub = ok_deg = ok_comp = ok_merge = True
    reduced = 0
    for _ in range(40):
        h, w = int(rng.integers(5, 22)), int(rng.integers(5, 22))
        m = rng.random((h, w)) < rng.uniform(0.3, 0.7)
        snapshot = m.copy()
        out, opened = co.open_diagonal_kisses(m, seed=int(rng.integers(1000)))
        labels8, n8 = _components8(m)
        labels4, n4 = tf.connected_components(out)
        ok_none &= co.count_kisses(out) == 0 and np.array_equal(m, snapshot)
        ok_sub &= bool((m <= out).all()) and int(out.sum()) - int(m.sum()) == len(opened) == len(set(opened)) \
            and all(not m[y, x] and out[y, x] for x, y in opened)
        ok_deg &= all(_open_neighbours(out, x, y) >= 2 for x, y in opened)
        ok_comp &= tf.connected_components(out)[1] <= n8 <= tf.connected_components(m)[1]
        ok_merge &= all(len(set(labels4[labels8 == k].tolist())) == 1 for k in range(1, n8 + 1))
        reduced += tf.connected_components(out)[1] < tf.connected_components(m)[1]
    check('random masks: no kiss is left, the input is untouched, only closed cells open and each had two open neighbours',
          ok_none and ok_sub and ok_deg)
    check('components only merge: 4-components after <= 8-components before <= 4-components before, and each '
          '8-connected region lands inside one 4-connected region', ok_comp and ok_merge and reduced > 20,
          f'{reduced} of 40 masks lost components')
    busy = rng.random((40, 40)) < 0.5
    one, two = co.open_diagonal_kisses(busy, 3)[0], co.open_diagonal_kisses(busy, 3)[0]
    other = co.open_diagonal_kisses(busy, 4)[0]
    check('equal seeds give equal results; another seed another one', np.array_equal(one, two)
          and not np.array_equal(one, other), f'{co.count_kisses(busy)} kisses')

    # cells that must stay closed
    pair = _mask_from_rows(['....', '.#..', '..#.', '....'])
    keep_one = np.zeros_like(pair)
    keep_one[1, 2] = True
    keep_both = keep_one.copy()
    keep_both[2, 1] = True
    only = {co.open_diagonal_kisses(pair, s, keep_one)[1][0] for s in range(30)}
    out, opened = co.open_diagonal_kisses(pair, 1, keep_both)
    check('a kiss with one kept blocker opens the other whatever the seed; with both kept it stays',
          only == {(1, 2)} and not opened and np.array_equal(out, pair) and co.count_kisses(out) == 1)
    rim = np.zeros((20, 20), dtype=bool)
    rim[0, :] = rim[-1, :] = rim[:, 0] = rim[:, -1] = True
    wild = rng.random((20, 20)) < 0.5
    held, opened = co.open_diagonal_kisses(wild, 2, rim)
    check('a kept rim is never opened (other cells are), and every kiss left has both blockers kept',
          not (held & ~wild & rim).any() and len(opened) > 10 and all(not rim[y, x] for x, y in opened)
          and all(rim[by, bx] for x, y in zip(*co.kiss_blocks(held))
                  for bx, by in ((x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)) if not held[by, bx]))
    check('bad shapes raise ValueError', all(_raises_value_error(f) for f in (
        lambda: co.kiss_blocks(np.zeros((3, 3, 3), dtype=bool)),
        lambda: co.open_diagonal_kisses(np.zeros((6, 6), dtype=bool), 0, np.zeros((5, 6), dtype=bool)))))

    # the cave of MarkovJunior's Cave.xml: the intake measurement
    rows = []
    for seed in range(60):
        m = _cave_xml(seed)
        out, opened = co.open_diagonal_kisses(m, seed)
        labels, parts4 = tf.connected_components(m)
        share = lambda mask: float(np.bincount(tf.connected_components(mask)[0].ravel())[1:].max() / mask.sum())
        rows.append((parts4, _components8(m)[1], tf.connected_components(out)[1], share(m), share(out),
                     co.count_kisses(m), co.count_kisses(out)))
    r = np.array(rows, dtype=float)
    check('Cave.xml caves (64x64, 60 seeds): 26.7 four-connected but 17.2 eight-connected regions per cave; '
          'clearing the kisses leaves exactly the eight-connected count',
          23 < r[:, 0].mean() < 30 and 15 < r[:, 1].mean() < 20 and (r[:, 2] <= r[:, 1]).all()
          and (r[:, 2] == r[:, 1]).mean() >= 0.95 and (r[:, 6] == 0).all() and (r[:, 5] > 0).all(),
          f'{r[:, 0].mean():.1f} four, {r[:, 1].mean():.1f} eight, {r[:, 2].mean():.1f} after')
    check('and the largest region holds more of the open cells afterwards (0.605 -> 0.653)',
          r[:, 4].mean() - r[:, 3].mean() > 0.02, f'{r[:, 3].mean():.3f} -> {r[:, 4].mean():.3f}')

    # TileMaps: only walls open, other closed tiles are left alone
    h, w = 5, 12
    floor = np.zeros((h, w), dtype=bool)
    lava = np.zeros((h, w), dtype=bool)
    abyss = np.zeros((h, w), dtype=bool)
    floor[0, 0] = floor[1, 1] = True            # kiss A: two walls block it
    floor[0, 4] = floor[1, 5] = True            # kiss B: a lava cell and a wall block it
    lava[0, 5] = True
    floor[0, 8] = floor[1, 9] = True            # kiss C: a lava cell and an abyss cell block it
    lava[0, 9] = True
    abyss[1, 8] = True
    tm = tg.build(w, h, [(floor, tg.FLOOR), (lava, tg.LAVA), (abyss, tg.ABYSS)], default=tg.WALL)
    tm.overlay[1][4] = 'hidden'
    before = [row[:] for row in tm.base]
    opened = co.open_kisses_tilemap(tm, seed=2)
    changed = {(x, y) for y in range(h) for x in range(w) if tm.base[y][x] != before[y][x]}
    check('open_kisses_tilemap opens only wall cells (a kiss between lava and abyss stays) and drops their overlay',
          changed == set(opened) and len(opened) == 2 and (4, 1) in opened
          and all(before[y][x] == tg.WALL and tm.base[y][x] == tg.FLOOR for x, y in opened)
          and tm.overlay[1][4] == '' and tm.base[0][5] == tg.LAVA and tm.base[1][8] == tg.ABYSS
          and len({(1, 0), (0, 1)} & set(opened)) == 1 and co.count_kisses(tg.passable_mask(tm)) == 1)
    tm2 = tg.build(w, h, [(floor, tg.FLOOR), (lava, tg.LAVA), (abyss, tg.ABYSS)], default=tg.WALL)
    opened2 = co.open_kisses_tilemap(tm2, seed=2, tile=tg.COURT, openable=(tg.WALL, tg.LAVA))
    check('`openable` widens what may open, `tile` sets what it becomes, and the abyss still blocks',
          (9, 0) in opened2 and tm2.base[0][9] == tg.COURT and tm2.base[1][8] == tg.ABYSS
          and co.count_kisses(tg.passable_mask(tm2)) == 0)

    outcome = []
    for name, seed in (('voronoi_kingdoms', 3), ('contour_terraces', 3), ('mandala', 9)):
        tm = tg.GRAMMARS[name]({'id': 731, 'size': 64, 'template': name, 'seed': seed}, random.Random(seed))
        base = [row[:] for row in tm.base]
        mask = tg.passable_mask(tm)
        kisses = co.count_kisses(mask)
        opened = co.open_kisses_tilemap(tm, seed)
        after = tg.passable_mask(tm)
        outcome.append(kisses > 30 and co.count_kisses(after) == 0
                       and tf.connected_components(after)[1] == tf.connected_components(mask)[1] == 1
                       and {(x, y) for y in range(64) for x in range(64) if tm.base[y][x] != base[y][x]} == set(opened)
                       and all(base[y][x] == tg.WALL for x, y in opened) and len(opened) <= kisses + 5)
    check('on the voronoi_kingdoms, contour_terraces and mandala maps (73, 108 and 45 kisses) the cleanup removes them all '
          'and keeps one component', all(outcome), str(outcome))
    tm = tg.GRAMMARS['tectonic']({'id': 732, 'size': 64, 'template': 'tectonic', 'seed': 3}, random.Random(3))
    co.open_kisses_tilemap(tm, 3)
    left = tg.passable_mask(tm)
    xs, ys = co.kiss_blocks(left)
    check('tectonic rifts (abyss) block kisses the cleanup may not open: those remain, and every one of them is blocked by non-wall tiles',
          len(xs) > 0 and all(tm.base[by][bx] != tg.WALL for x, y in zip(xs, ys)
                              for bx, by in ((x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)) if not left[by, bx]))


# ---------------------------------------------- backtracking with banning

_DIRS4 = [('E', 1, 0), ('W', -1, 0), ('S', 0, 1), ('N', 0, -1)]


def _chess(w, h, jokers=False, masked=()):
    """A torus of w x h cells where state 0 may only touch 1 and 1 only 0 (state 2, with
    `jokers`, touches everything). Cells in `masked` can only hold the joker, all
    others only 0 and 1. Returns (grid, domains, propagated without a contradiction)."""
    allowed = {0: 0b110, 1: 0b101, 2: 0b111} if jokers else {0: 0b10, 1: 0b01}
    compat = {key: [allowed[s] for s in sorted(allowed)] for key in 'ESWN'}
    grid = csp.BitGrid(compat, w, h, _DIRS4, periodic=True)
    doms = [0b011 if jokers else 0b11] * (w * h)
    for x, y in masked:
        doms[y * w + x] = 0b100
    return grid, doms, grid.propagate(doms, deque(range(w * h)))


class _KeepOne:
    """A rule that state 1 must stay possible in every cell: the idea of DeBroglie's
    DontBanOneConstraint, which makes every choice but state 1 contradict."""

    def apply(self, doms, changed, indices=None):
        return all(d & 0b10 for d in doms)


class _Queens:
    """N queens as a problem for `bt.search`, to show the loop needs nothing of grids or sockets: one cell per row,
    the columns a queen may still take as its domain, a placed queen (a one-column domain) removing the columns and
    diagonals it attacks from every other row."""

    def __init__(self, n, rng):
        self.n, self.rng, self.doms = n, rng, [(1 << n) - 1] * n

    def settle(self):
        doms, placed = self.doms, set()
        while True:
            queens = [(row, d.bit_length() - 1) for row, d in enumerate(doms)
                      if d and not d & (d - 1) and row not in placed]
            if not queens:
                return all(doms)
            for row, col in queens:
                placed.add(row)
                for other in range(self.n):
                    gap = abs(other - row)
                    if gap:
                        doms[other] &= ~((1 << col) | (1 << col - gap if col >= gap else 0) | (1 << col + gap))
            if not all(doms):
                return False

    def pick(self):
        open_rows = [(d.bit_count(), row) for row, d in enumerate(self.doms) if d & (d - 1)]
        return min(open_rows)[1] if open_rows else None

    def choose(self, row):
        cols = list(csp.iter_bits(self.doms[row]))
        return cols[int(self.rng.random() * len(cols))]

    def snapshot(self):
        return list(self.doms)

    def restore(self, saved):
        self.doms[:] = saved

    def decide(self, row, col):
        self.doms[row] = 1 << col
        return self.settle()

    def forbid(self, row, col):
        self.doms[row] &= ~(1 << col)
        return bool(self.doms[row]) and self.settle()


class _DeadEnds:
    """`levels` cells of `width` values in which every complete assignment is a dead end, found by its last decision:
    nothing is solvable, so the complete search visits every one of the width ** levels leaves. No draws: the first
    undecided cell takes its lowest value, which makes the whole search, and the patience schedule, hand-countable."""

    def __init__(self, levels, width):
        self.doms = [(1 << width) - 1] * levels

    def settle(self):
        return all(self.doms) and any(d & (d - 1) for d in self.doms)

    def pick(self):
        return next((cell for cell, d in enumerate(self.doms) if d & (d - 1)), None)

    def choose(self, cell):
        return (self.doms[cell] & -self.doms[cell]).bit_length() - 1

    def snapshot(self):
        return list(self.doms)

    def restore(self, saved):
        self.doms[:] = saved

    def decide(self, cell, value):
        self.doms[cell] = 1 << value
        return self.settle()

    def forbid(self, cell, value):
        self.doms[cell] &= ~(1 << value)
        return self.settle()


class _Picky(_DeadEnds):
    """Two cells of two values where only (1, 1) is no dead end, and the second cell has nothing to try (`choose` answers
    None) once the first is 0: the loop must treat that as a contradiction of the decision before."""

    def __init__(self):
        super().__init__(2, 2)

    def settle(self):
        return all(self.doms) and (any(d & (d - 1) for d in self.doms) or self.doms == [0b10, 0b10])

    def choose(self, cell):
        return None if cell == 1 and self.doms[0] == 0b01 else super().choose(cell)


def _random_compat(n_states, density, rng):
    """Random socket-style compatibility: j may sit east of i iff i may sit west of j."""
    compat = {d: [0] * n_states for d in 'NESW'}
    for a in range(n_states):
        for b in range(n_states):
            if rng.random() < density:
                compat['E'][a] |= 1 << b
                compat['W'][b] |= 1 << a
            if rng.random() < density:
                compat['S'][a] |= 1 << b
                compat['N'][b] |= 1 << a
    modules = [{'compat_degree': max(1, sum(compat[d][i].bit_count() for d in 'NESW'))} for i in range(n_states)]
    return modules, compat


def _planted(n_states, w, h, density, seed, unique=True):
    """A compatibility table at `density` that contains a hidden solution, so
    every instance is satisfiable: (modules, compat)."""
    rng = random.Random(seed)
    if unique:
        arrangement = list(range(n_states))
        rng.shuffle(arrangement)
    else:
        arrangement = [rng.randrange(n_states) for _ in range(w * h)]
    east = [[rng.random() < density for _ in range(n_states)] for _ in range(n_states)]
    south = [[rng.random() < density for _ in range(n_states)] for _ in range(n_states)]
    for y in range(h):
        for x in range(w):
            if x + 1 < w:
                east[arrangement[y * w + x]][arrangement[y * w + x + 1]] = True
            if y + 1 < h:
                south[arrangement[y * w + x]][arrangement[(y + 1) * w + x]] = True
    compat = {d: [0] * n_states for d in 'NESW'}
    for a in range(n_states):
        for b in range(n_states):
            if east[a][b]:
                compat['E'][a] |= 1 << b
                compat['W'][b] |= 1 << a
            if south[a][b]:
                compat['S'][a] |= 1 << b
                compat['N'][b] |= 1 << a
    modules = [{'compat_degree': sum(compat[d][i].bit_count() for d in 'NESW')} for i in range(n_states)]
    return modules, compat


def _satisfiable(compat, w, h, n_states, unique=False, exactly=None, domains=None, pins=None, groups=None, quotas=()):
    """Brute force over every assignment: does one respect the compatibility, with every state (or, with `groups`,
    every group of states) used at most once (`unique`), state 0 used exactly `exactly` times, the initial `domains`
    (a mask per cell), the `pins` ({cell index: state}) and the `Quota` objects?"""
    cells = w * h
    assign = np.indices((n_states,) * cells).reshape(cells, -1)
    ok = np.ones(assign.shape[1], dtype=bool)
    table = {d: np.array([[compat[d][a] >> b & 1 for b in range(n_states)] for a in range(n_states)], dtype=bool)
             for d in 'ES'}
    for y in range(h):
        for x in range(w):
            if x + 1 < w:
                ok &= table['E'][assign[y * w + x], assign[y * w + x + 1]]
            if y + 1 < h:
                ok &= table['S'][assign[y * w + x], assign[(y + 1) * w + x]]
    if unique:
        group_of = np.array(groups if groups is not None else range(n_states))
        for i in range(cells):
            for j in range(i + 1, cells):
                ok &= group_of[assign[i]] != group_of[assign[j]]
    if exactly is not None:
        ok &= (assign == 0).sum(axis=0) == exactly
    for cell, mask in enumerate(domains or ()):
        ok &= np.array([mask >> state & 1 for state in range(n_states)], dtype=bool)[assign[cell]]
    for cell, state in (pins or {}).items():
        ok &= assign[cell] == state
    for quota in quotas:
        count = np.isin(assign, [state for state in range(n_states) if quota.mask >> state & 1]).sum(axis=0)
        ok &= {'at_most': count <= quota.count, 'at_least': count >= quota.count,
               'exactly': count == quota.count}[quota.op]
    return bool(ok.any())


def _respects(placements, compat, w, h, domains, pins, groups, quotas, unique):
    """Does a finished arrangement obey every constraint, checked cell by cell?"""
    ok = se.check_placements(compat, w, h, placements)
    ok = ok and all(domains[i] >> state & 1 for i, state in enumerate(placements))
    ok = ok and all(placements[cell] == state for cell, state in pins.items())
    if unique:
        ids = [groups[state] if groups is not None else state for state in placements]
        ok = ok and len(set(ids)) == len(ids)
    for quota in quotas:
        count = sum(quota.mask >> state & 1 for state in placements)
        ok = ok and {'at_most': count <= quota.count, 'at_least': count >= quota.count,
                     'exactly': count == quota.count}[quota.op]
    return ok


def _all_different_reference(masks, group_of):
    """The all-different forward check restated on sets, to a fixpoint: a cell whose states all lie in one group is
    decided to it; two cells decided to a group, or an empty domain, is a contradiction (None); a decided group leaves
    every other cell. Returns the narrowed masks."""
    doms = [{s for s in range(len(group_of)) if m >> s & 1} for m in masks]
    while True:
        decided = {}
        for i, d in enumerate(doms):
            if not d:
                return None
            groups = {group_of[s] for s in d}
            if len(groups) == 1:
                group = next(iter(groups))
                if group in decided:
                    return None
                decided[group] = i
        changed = False
        for group, owner in decided.items():
            for i, d in enumerate(doms):
                if i != owner:
                    kept = {s for s in d if group_of[s] != group}
                    if kept != d:
                        if not kept:
                            return None
                        doms[i], changed = kept, True
        if not changed:
            return [sum(1 << s for s in d) for d in doms]


def test_backtracking():
    print("\nbacktracking with banning")

    # DeBroglie's chessboard: a torus of two mutually exclusive states
    solved_even, proofs, budgets, patient = [], [], [], []
    for w, h in ((10, 10), (4, 6), (8, 2)):
        grid, doms, ok = _chess(w, h)
        got = bt.search(bt.GridProblem(grid, doms, random.Random(w * h)))
        board = [[doms[y * w + x] for x in range(w)] for y in range(h)]
        solved_even.append(ok and got['status'] == 'solved' and all(
            board[y][x] in (1, 2) and board[y][x] != board[y][(x + 1) % w] and board[y][x] != board[(y + 1) % h][x]
            for y in range(h) for x in range(w)))
    for w, h in ((11, 11), (5, 4), (3, 3), (9, 6)):
        grid, doms, ok = _chess(w, h)
        got = bt.search(bt.GridProblem(grid, doms, random.Random(1)))
        proofs.append(ok and got['status'] == 'unsat' and got['backtracks'] >= 1)
        grid, doms, ok = _chess(w, h)
        budgets.append(bt.search(bt.GridProblem(grid, doms, random.Random(1)), bt.Backtrack(0))['status'] == 'budget')
        grid, doms, ok = _chess(w, h)
        calm = bt.search(bt.GridProblem(grid, doms, random.Random(1)), bt.Backtrack(patience=True))
        patient.append((calm['status'], calm['backjumps']))
    check('a chessboard on an even torus (10x10, 4x6, 8x2) is solved: every cell differs from its four wrapped neighbours',
          all(solved_even), str(solved_even))
    check('an odd periodic chessboard (11x11, 5x4, 3x3, 9x6) is proved impossible: the search runs out of decisions '
          'to undo, which no number of restarts could show', all(proofs), str(proofs))
    check('with no backtracking allowed the first contradiction is a restart (status budget); with the patience option '
          'the proof still stands when no backjump was needed (status unsat, 0 backjumps)',
          all(budgets) and set(patient) == {('unsat', 0)}, str(patient))
    masked = [(x, 10) for x in range(11)] + [(10, y) for y in range(10)]
    grid, doms, ok = _chess(11, 11, jokers=True, masked=masked)
    got = bt.search(bt.GridProblem(grid, doms, random.Random(2)))
    board = [[doms[y * 11 + x] for x in range(11)] for y in range(11)]
    check('a mask makes the odd chessboard solvable: with the last row and column held by a joker state the 10x10 '
          'block is a proper chessboard',
          ok and got['status'] == 'solved' and all(board[10][x] == 4 for x in range(11))
          and all(board[y][10] == 4 for y in range(10)) and all(
              board[y][x] != board[y][x + 1] and board[y][x] != board[y + 1][x]
              for y in range(9) for x in range(9)) and all(board[y][9] != board[y + 1][9] for y in range(9)))

    # a rule that forces backtracking: every choice but state 1 contradicts
    compat = {key: [0b1111] * 4 for key in 'ESWN'}
    grid = csp.BitGrid(compat, 5, 5, _DIRS4)
    doms = [0b1111] * 25
    problem = bt.GridProblem(grid, doms, random.Random(3), rules=[_KeepOne()])
    got = bt.search(problem)
    check('a constraint that forbids losing state 1 forces backtracking: every cell ends at state 1 after at most 75 bans '
          '(each ban is a distinct (cell, state) of 25 x 3)',
          got['status'] == 'solved' and all(d == 0b10 for d in doms) and 0 < got['backtracks'] <= 75,
          f"{got['backtracks']} backtracks, {got['decisions']} decisions")
    wins = 0
    for attempt in range(300):
        doms = [0b1111] * 25
        grid = csp.BitGrid(compat, 5, 5, _DIRS4)
        wins += bt.search(bt.GridProblem(grid, doms, random.Random(1000 + attempt), rules=[_KeepOne()]),
                          bt.Backtrack(0))['status'] == 'solved'
    check('restarts alone (budget 0) never get there: 300 attempts need all 25 draws to be state 1, chance 4^-25',
          wins == 0)

    # pigeonhole: n + 1 cells that must all differ, n states. A node whose cells share k free states costs B(k)
    # backtracks: the k - 1 decisions on the cell it splits are each undone once, and each of the k branches (the last
    # forced by the bans) is a node with k - 1 free states, so B(k) = (k - 1) + k * B(k - 1) with B(1) = 0, i.e.
    # B(k) = k! - 1. The complete search therefore makes exactly n! - 1 whatever it draws: none wasted, none skipped
    def pigeonhole(n, **options):
        return se.solve_socket_wfc_ext([{'compat_degree': 4}] * n, {d: [(1 << n) - 1] * n for d in 'NESW'}, n + 1, 1,
                                       unique_states=True, **options)
    sizes = [n for n in (3, 4, 5, 6) for _ in (0, 1)]
    rows = [pigeonhole(n, seed=i, max_restarts=3, backtrack=bt.Backtrack(max_backtracks=None))
            for i, n in enumerate(sizes)]
    check('n + 1 cells that must differ with n states (n = 3 to 6, two seeds each): the complete search proves it '
          'impossible in one attempt with exactly n! - 1 backtracks',
          all(r['failures'] == ['unsat'] and r['attempts'] == 1 and r['backtracks'] == math.factorial(n) - 1
              for r, n in zip(rows, sizes)), str([r['backtracks'] for r in rows]))
    exact = pigeonhole(5, seed=0, max_restarts=2, backtrack=119)
    short = pigeonhole(5, seed=0, max_restarts=2, backtrack=118)
    check('a budget of exactly the 119 backtracks of that proof still ends it as unsat; one less ends every attempt as '
          'budget, after 118 backtracks each',
          exact['failures'] == ['unsat'] and exact['backtracks'] == 119
          and short['failures'] == ['budget', 'budget'] and short['backtracks'] == 236)
    eager = pigeonhole(6, seed=0, max_restarts=3, backtrack=bt.Backtrack(max_backtracks=None, patience=True))
    check('with patience the same search skips a part of the tree and gives up (exhausted) rather than prove anything: '
          'with no budget at all every attempt still ends, far short of the 719 backtracks of the proof, and the '
          'restarts are used',
          eager['failures'] == ['exhausted'] * 3 and eager['attempts'] == 3 and eager['backjumps'] >= 3
          and eager['backtracks'] < 3 * 100, f"{eager['backtracks']} backtracks, {eager['backjumps']} jumps")

    # a problem that is no grid: N queens through the same loop
    def queens_ok(board):
        return len(set(board)) == len(board) and all(
            abs(board[a] - board[b]) != b - a for a in range(len(board)) for b in range(a + 1, len(board)))
    boards = {n: [] for n in (1, 2, 3, 4, 6, 8, 10)}
    for n in boards:
        for seed in range(12):
            queens = _Queens(n, random.Random(seed))
            got = bt.search(queens)
            boards[n].append((got['status'], tuple(d.bit_length() - 1 for d in queens.doms)))
    check('N queens as a custom problem: n = 2 and 3 are proved impossible, n = 1, 4, 6, 8 and 10 give valid boards '
          '(12 seeds each), and the 12 seeds find both 4-queens solutions',
          all(status == 'unsat' for n in (2, 3) for status, _ in boards[n])
          and all(status == 'solved' and queens_ok(board) for n in (1, 4, 6, 8, 10) for status, board in boards[n])
          and {board for _, board in boards[4]} == {(1, 3, 0, 2), (2, 0, 3, 1)},
          str(sorted({board for _, board in boards[4]})))

    # a tree in which every leaf is a dead end: there is nothing to find, so the work is countable by hand
    dead = [(levels, width, bt.search(_DeadEnds(levels, width), bt.Backtrack(None)))
            for levels, width in ((2, 2), (3, 2), (4, 2), (2, 3), (3, 3), (4, 3), (5, 2))]
    check('a tree whose every leaf is a dead end: the complete search ends as unsat after exactly width ** levels - 1 '
          'backtracks (one per leaf but the last), no backjump, decisions as deep as the tree, and every decision made '
          'was undone once',
          all(got['status'] == 'unsat' and got['backtracks'] == width ** levels - 1 and got['backjumps'] == 0
              and got['max_depth'] == levels and got['decisions'] == got['backtracks'] for levels, width, got in dead),
          str([got['backtracks'] for _, _, got in dead]))
    picky = _Picky()
    way_out = bt.search(picky, bt.Backtrack(None))
    check('a cell with nothing to try is a contradiction of the decision before it: the first cell is banned from 0 '
          'at once (no decision is made on the empty-handed cell) and (1, 1), the one way out, is found after 2 '
          'backtracks and 2 decisions',
          (way_out['status'], way_out['backtracks'], way_out['decisions']) == ('solved', 2, 2)
          and picky.doms == [0b10, 0b10], str(way_out))
    hand = bt.search(_DeadEnds(4, 2), bt.Backtrack(None, patience=True))
    shallow = bt.search(_DeadEnds(3, 3), bt.Backtrack(None, patience=True))
    deep = bt.search(_DeadEnds(6, 2), bt.Backtrack(None, patience=True))
    check('patience on 4 levels of 2 values, counted by hand: the 11th contradiction finds level 0 expired and jumps back '
          'over the 2 decisions standing, 7 more backtracks empty the stack: exhausted, not unsat (the jump skipped part '
          'of the tree), after 17 backtracks and 1 backjump (19 decisions: the 17 undone one by one and the 2 the jump '
          'undid)',
          (hand['status'], hand['backtracks'], hand['backjumps'], hand['decisions']) == ('exhausted', 17, 1, 19),
          str(hand))
    check('patience gives up when the schedule asks to jump back past the first decision: 3 levels of 3 values, the first '
          'jump would go 4 back from depth 3 (exhausted after 10 backtracks, no jump made); 6 levels of 2, the second '
          'level does (exhausted after a single jump, far short of the 63 backtracks of the proof)',
          (shallow['status'], shallow['backtracks'], shallow['backjumps']) == ('exhausted', 10, 0)
          and deep['status'] == 'exhausted' and deep['backjumps'] == 1 and deep['backtracks'] < 63, f'{shallow} {deep}')

    # completeness: the search agrees with exhaustive enumeration on random small instances
    rng = random.Random(77)
    rows = []
    for k in range(40):
        modules, comp = _random_compat(4, rng.choice((0.3, 0.35, 0.4, 0.45)), rng)
        res = se.solve_socket_wfc_ext(modules, comp, 3, 3, seed=k, max_restarts=5,
                                      backtrack=bt.Backtrack(max_backtracks=None))
        sat = _satisfiable(comp, 3, 3, 4)
        valid = res['success'] and se.check_placements(comp, 3, 3, res['placements'])
        rows.append((sat, res['success'] == sat and (valid if sat else res['failures'] == ['unsat']
                                                       and res['attempts'] == 1)))
    check('3x3 grids of 4 states (40 random tables): solved exactly when a solution exists, and an unsatisfiable table is '
          'reported as unsat after one attempt', all(ok for _, ok in rows) and 8 <= sum(s for s, _ in rows) <= 32,
          f'{sum(s for s, _ in rows)} satisfiable, {sum(not s for s, _ in rows)} not')
    rows = []
    for k in range(30):
        modules, comp = _random_compat(6, rng.choice((0.35, 0.45, 0.55)), rng)
        res = se.solve_socket_wfc_ext(modules, comp, 3, 2, seed=k, unique_states=True, max_restarts=5,
                                      backtrack=bt.Backtrack(max_backtracks=None))
        sat = _satisfiable(comp, 3, 2, 6, unique=True)
        valid = res['success'] and se.check_placements(comp, 3, 2, res['placements']) \
            and len(set(res['placements'])) == 6
        rows.append((sat, res['success'] == sat and (valid if sat else res['failures'] == ['unsat'])))
    check('3x2 grids with six states each used once (30 random tables): solved exactly when an all-different solution '
          'exists', all(ok for _, ok in rows) and 4 <= sum(s for s, _ in rows) <= 26,
          f'{sum(s for s, _ in rows)} satisfiable, {sum(not s for s, _ in rows)} not')
    rows = []
    for k in range(30):
        modules, comp = _random_compat(4, rng.choice((0.3, 0.4, 0.5)), rng)
        quota = csp.Quota([0], 'exactly', 2)
        res = se.solve_socket_wfc_ext(modules, comp, 3, 3, seed=k, max_restarts=5, quotas=[quota],
                                      backtrack=bt.Backtrack(max_backtracks=None))
        sat = _satisfiable(comp, 3, 3, 4, exactly=2)
        valid = res['success'] and se.check_placements(comp, 3, 3, res['placements']) \
            and res['placements'].count(0) == 2
        rows.append((sat, res['success'] == sat and (valid if sat else res['failures'] == ['unsat'])))
    check('3x3 grids with a quota of exactly two cells of state 0 (30 random tables): solved exactly when one exists',
          all(ok for _, ok in rows) and 4 <= sum(s for s, _ in rows) <= 26,
          f'{sum(s for s, _ in rows)} satisfiable, {sum(not s for s, _ in rows)} not')
    rows = []
    for k in range(60):
        w, h, n = rng.choice(((3, 3, 3), (4, 2, 4), (3, 2, 4), (2, 3, 5), (2, 2, 5), (2, 2, 4)))
        modules, comp = _random_compat(n, rng.choice((0.5, 0.6, 0.7)), rng)
        full = (1 << n) - 1
        domains = [rng.randrange(1, full + 1) if rng.random() < 0.3 else full for _ in range(w * h)]
        pins = {rng.randrange(w * h): rng.randrange(n)} if rng.random() < 0.4 else {}
        quotas = ([csp.Quota(rng.randrange(1, full), rng.choice(csp.QUOTA_OPS), rng.randint(0, 3))]
                  if rng.random() < 0.5 else [])
        unique = w * h <= n and rng.random() < 0.6
        groups = [0, 1, 2, 3, 3] if n == 5 and rng.random() < 0.5 else None
        res = se.solve_socket_wfc_ext(modules, comp, w, h, seed=k, max_restarts=3, domains=domains,
                                      fixed={(c % w, c // w): state for c, state in pins.items()}, quotas=quotas,
                                      unique_states=unique, groups=groups, backtrack=bt.Backtrack(max_backtracks=None))
        sat = _satisfiable(comp, w, h, n, unique=unique, domains=domains, pins=pins, groups=groups, quotas=quotas)
        rows.append((sat, res['success'] == sat and (not sat or _respects(
            res['placements'], comp, w, h, domains, pins, groups, quotas, unique))))
    check('60 random small tables with every hook at once (initial domains, pins, a quota, all-different over groups of '
          'states): solved exactly when a solution exists, and every solution obeys all of them',
          all(ok for _, ok in rows) and 15 <= sum(s for s, _ in rows) <= 45,
          f'{sum(s for s, _ in rows)} satisfiable, {sum(not s for s, _ in rows)} not')
    proof = se.solve_socket_wfc_ext([{'compat_degree': 4}] * 3, {d: [0b111] * 3 for d in 'NESW'}, 2, 2, seed=1,
                                    unique_states=True, max_restarts=6, backtrack=True)
    legacy = se.solve_socket_wfc_ext([{'compat_degree': 4}] * 3, {d: [0b111] * 3 for d in 'NESW'}, 2, 2, seed=1,
                                     unique_states=True, max_restarts=6)
    check('four cells that must all differ with only three states: backtracking proves it (unsat, one attempt); '
          'the restart loop burns every restart',
          not proof['success'] and proof['failures'] == ['unsat'] and proof['attempts'] == 1
          and not legacy['success'] and len(legacy['failures']) == 6 and legacy['attempts'] == 6)

    # the shipped library
    modules, compat = _shipped_library()
    same = []
    for preset in rt.WORLD_PRESETS[:2]:
        owner = se.solve_preset(modules, compat, preset)
        mine = se.solve_preset(modules, compat, preset, backtrack=True)
        same.append(owner['attempts'] == mine['attempts'] == 1 and owner['placements'] == mine['placements']
                    and mine['backtracks'] == 0 and 'backtracks' not in owner)
    check('presets tiny_fast and balanced (no contradiction): backtrack=True makes the owner solver\'s draws, placements '
          'and attempt count, with 0 backtracks', all(same), str(same))
    preset = rt.WORLD_PRESETS[2]
    owner = se.solve_preset(modules, compat, preset)
    mine = se.solve_preset(modules, compat, preset, backtrack=True)
    check('preset all100_unique: the owner needs restarts, backtracking succeeds on attempt 1 with a valid 10x10 world '
          'that uses each of the 100 layouts once',
          owner['attempts'] > 1 and mine['success'] and mine['attempts'] == 1
          and se.check_placements(compat, 10, 10, mine['placements']) and sorted(mine['placements']) == list(range(100)),
          f"owner {owner['attempts']} attempts, mine {mine['attempts']} ({mine['backtracks']} backtracks)")
    sweep = []
    for seed in range(8):
        owner = rt.solve_socket_wfc(modules, compat, 10, 10, seed=seed, unique_states=True, max_restarts=128)
        mine = se.solve_socket_wfc_ext(modules, compat, 10, 10, seed=seed, unique_states=True, max_restarts=128,
                                       backtrack=True)
        sweep.append((owner['attempts'], mine['attempts'], mine['backtracks'],
                      mine['success'] and se.check_placements(compat, 10, 10, mine['placements'])
                      and sorted(mine['placements']) == list(range(100))))
    check('eight seeds of the real 10x10 all-different solve: the owner restarts on four of them (up to 3 attempts), '
          'backtracking solves every one in a single attempt with a valid world, three of them after 2 to 33 backtracks',
          all(att == 1 and ok for _, att, _, ok in sweep) and sum(own > 1 for own, _, _, _ in sweep) >= 4
          and max(own for own, _, _, _ in sweep) >= 3 and sum(back > 0 for _, _, back, _ in sweep) >= 3,
          f"owner {[r[0] for r in sweep]}, backtracks {[r[2] for r in sweep]}")
    base = [rt.solve_socket_wfc(modules, compat, 6, 6, seed=s, max_restarts=30) for s in range(12)]
    mine = [se.solve_socket_wfc_ext(modules, compat, 6, 6, seed=s, max_restarts=30, backtrack=True) for s in range(12)]
    check('twelve 6x6 solves: identical placements to the legacy solver, seed for seed',
          all(a['placements'] == b['placements'] and a['attempts'] == b['attempts'] for a, b in zip(base, mine)))
    flavour = lambda x, y, state, module, used: 1.0 + (state * 7 + x + 2 * y) % 5          # not the default weight
    prefer_open = lambda x, y, state, module: int(module['open_ratio'] > 0.3)
    rows = []
    for seed in range(4):
        plain = se.solve_socket_wfc_ext(modules, compat, 6, 6, seed=seed, max_restarts=30)['placements']
        for hooks in (dict(weight=flavour), dict(priority=prefer_open), dict(weight=flavour, priority=prefer_open)):
            owner = se.solve_socket_wfc_ext(modules, compat, 6, 6, seed=seed, max_restarts=30, **hooks)
            mine = se.solve_socket_wfc_ext(modules, compat, 6, 6, seed=seed, max_restarts=30, backtrack=True, **hooks)
            rows.append((owner['attempts'] == 1 and owner['placements'] == mine['placements']
                         and mine['backtracks'] == 0, owner['placements'] != plain))
    check('a weight hook, a priority hook and both (four 6x6 seeds each): backtracking makes the owner solver\'s draws '
          'and placements with the same hooks, and the hooks do change the world',
          all(same for same, _ in rows) and all(moved for _, moved in rows), str(rows))
    lure_compat = {d: [0b1111, 0b1111, 0b1111, 0b0000] for d in 'NESW'}   # 0-2 mix freely; 3 allows nothing beside it
    lure_modules = [{'compat_degree': 4}] * 4
    lure = lambda x, y, state, module: 1 if state == 3 else 0
    stuck = se.solve_socket_wfc_ext(lure_modules, lure_compat, 3, 3, seed=1, max_restarts=20, priority=lure)
    freed = se.solve_socket_wfc_ext(lure_modules, lure_compat, 3, 3, seed=1, max_restarts=20, priority=lure,
                                    backtrack=True)
    check('a priority is a preference, not a wall: with a top-ranked state that always contradicts, the restart loop '
          'picks it first every time and fails all 20 attempts, backtracking bans it cell by cell (9 backtracks on 3x3) '
          'and ends with a valid world without it',
          not stuck['success'] and stuck['attempts'] == 20 and freed['success'] and freed['attempts'] == 1
          and freed['backtracks'] == 9 and 3 not in freed['placements']
          and se.check_placements(lure_compat, 3, 3, freed['placements']))

    # a budget of zero is the restart loop, draw for draw, even through restarts; backtracking needs none
    rows = []
    for seed in range(8):
        modules2, comp = _planted(60, 8, 8, 0.10, seed, unique=False)
        owner = se.solve_socket_wfc_ext(modules2, comp, 8, 8, seed=seed, max_restarts=64)
        zero = se.solve_socket_wfc_ext(modules2, comp, 8, 8, seed=seed, max_restarts=64, backtrack=0)
        full = se.solve_socket_wfc_ext(modules2, comp, 8, 8, seed=seed, max_restarts=64, backtrack=True)
        rows.append(((owner['success'], owner['attempts'], owner['placements'])
                     == (zero['success'], zero['attempts'], zero['placements']),
                     owner['attempts'] if owner['success'] else 99, full['attempts'],
                     full['success'] and se.check_placements(comp, 8, 8, full['placements'])))
    check('eight sparse planted tables (60 states, 8x8, 10% compatible): backtrack=0 equals the restart loop result for '
          'result, including the table the owner never solves', all(r[0] for r in rows), str([r[1] for r in rows]))
    check('and with backtracking every one is solved on the first attempt; the owner needed 3 to 11 restarts on most',
          all(r[2] == 1 and r[3] for r in rows) and sum(r[1] >= 3 for r in rows) >= 6)

    # planted solutions: 100 states, 10x10, every state once
    rows = []
    for density, seed in ((0.5, 0), (0.5, 1), (0.4, 0), (0.4, 1), (0.3, 0)):
        modules3, comp = _planted(100, 10, 10, density, seed)
        owner = se.solve_socket_wfc_ext(modules3, comp, 10, 10, seed=seed, unique_states=True, max_restarts=8)
        mine = se.solve_socket_wfc_ext(modules3, comp, 10, 10, seed=seed, unique_states=True, max_restarts=8,
                                       backtrack=True)
        rows.append((owner['success'], mine['success'] and mine['attempts'] == 1
                     and se.check_placements(comp, 10, 10, mine['placements']) and len(set(mine['placements'])) == 100,
                     mine['backtracks']))
    check('planted 10x10 tables of 100 states used once (densities 0.5, 0.4, 0.3): backtracking finds a valid '
          'all-different arrangement on the first attempt where 8 restarts do not',
          all(r[1] for r in rows) and sum(not r[0] for r in rows) >= 4, str([(r[0], r[2]) for r in rows]))

    # hooks and budgets
    modules3, comp = _planted(100, 10, 10, 0.4, 0)
    capped = se.solve_socket_wfc_ext(modules3, comp, 10, 10, seed=0, unique_states=True, max_restarts=3, backtrack=5)
    seen = []
    se.solve_socket_wfc_ext(modules3, comp, 10, 10, seed=0, unique_states=True, max_restarts=3, backtrack=5,
                            on_contradiction=lambda a, r: seen.append((a, r)))
    check('a budget of 5 ends each attempt after exactly 5 backtracks (reason budget), is reported per attempt and then '
          'restarts', not capped['success'] and capped['failures'] == ['budget'] * 3 and capped['attempts'] == 3
          and capped['backtracks'] == 15 and seen == [(0, 'budget'), (1, 'budget'), (2, 'budget')])
    hard_modules, hard_comp = _planted(100, 10, 10, 0.3, 1)
    jumpy = se.solve_socket_wfc_ext(hard_modules, hard_comp, 10, 10, seed=1, unique_states=True, max_restarts=1,
                                    backtrack=bt.Backtrack(40, patience=True))
    check('with patience a backjump counts against the budget like a backtrack: a budget of 40 on a hard table ends the '
          'attempt after 40 contradictions, at least one of them a backjump',
          jumpy['failures'] == ['budget'] and jumpy['backtracks'] + jumpy['backjumps'] == 40 and jumpy['backjumps'] >= 1,
          f"{jumpy['backtracks']} backtracks + {jumpy['backjumps']} backjumps")
    peak = [0]

    def watch(x, y, state, module, used):
        peak[0] = max(peak[0], max(used.values(), default=0))
        return se.default_weight(x, y, state, module, used)
    watched = se.solve_socket_wfc_ext(modules3, comp, 10, 10, seed=0, unique_states=True, max_restarts=3,
                                      backtrack=True, weight=watch)
    check('usage counts follow the undone decisions: with every state used once, no count the weight hook is shown '
          'exceeds 1 (a run with 13 backtracks)',
          watched['success'] and watched['backtracks'] > 0 and peak[0] == 1,
          f"{watched['backtracks']} backtracks, peak count {peak[0]}")
    used = Counter()
    socket_problem = se._SocketProblem(se._socket_grid(comp, 10, 10), [(1 << 100) - 1] * 100, random.Random(0),
                                       [bt.AllDifferent(list(range(100)))], [None], modules3, 10, used, None, None)
    ran = bt.search(socket_problem)
    check('usage counts are exactly the decisions still standing: after a run with backtracks they add up to the '
          'decisions made minus the decisions undone, one use for each state (none lost on an undo, none counted twice)',
          ran['status'] == 'solved' and ran['backtracks'] > 0 and sum(used.values()) == ran['decisions'] - ran['backtracks']
          and set(used.values()) == {1}, f"{ran['decisions']} decisions, {ran['backtracks']} undone, {sum(used.values())} counted")
    modules, compat = _shipped_library()
    caves = se.states_where(modules, lambda m: m['theme'] in ('cave', 'caves'))
    sparse = se.states_where(modules, lambda m: m['open_ratio'] < 0.1)
    start = min((i for i, m in enumerate(modules) if m['open_ratio'] > 0.2), key=lambda i: modules[i]['open_ratio'])
    boss = max(range(len(modules)), key=lambda i: modules[i]['open_ratio'])
    hooks = dict(quotas=[se.Quota(caves, 'at_most', 4, name='caves')], fixed={(0, 0): start, (4, 4): boss},
                 domains=se.ban_states(modules, se.border_cells(8, 8), sparse), weight=se.default_weight)
    run = se.solve_socket_wfc_ext(modules, compat, 8, 8, seed=9, backtrack=True, **hooks)
    again = se.solve_socket_wfc_ext(modules, compat, 8, 8, seed=9, backtrack=True, **hooks)
    check('pins, banned domains, a quota and a weight hook all hold under backtracking, and the run is deterministic',
          run['success'] and run['placements'] == again['placements'] and run['placements'][0] == start
          and run['placements'][4 * 8 + 4] == boss and sum(caves >> s & 1 for s in run['placements']) <= 4
          and all(not sparse >> run['placements'][y * 8 + x] & 1 for x, y in se.border_cells(8, 8))
          and se.check_placements(compat, 8, 8, run['placements']))
    first = se.solve_socket_wfc_ext(modules, compat, 8, 8, seed=23, backtrack=True)['placements']
    swap = next(s for s in range(len(modules)) if s != first[27] and compat['E'][s] >> first[28] & 1
                and compat['W'][s] >> first[26] & 1 and compat['S'][s] >> first[35] & 1
                and compat['N'][s] >> first[19] & 1)
    rerolled = se.reroll_after_pin(modules, compat, 8, 8, first, {(3, 3): swap}, seed=5, backtrack=True)
    check('reroll_after_pin passes the hook through', rerolled['success'] and rerolled['changed'] == [27]
          and 'backtracks' in rerolled)
    clash = next(s for s in range(len(modules)) if not compat['E'][boss] >> s & 1)
    setups = [
        se.solve_socket_wfc_ext(modules, compat, 4, 4, seed=1, max_restarts=3, backtrack=True,
                                domains=se.allow_states([(1, 1)], 0))['failures'],
        se.solve_socket_wfc_ext(modules, compat, 6, 6, seed=2, max_restarts=3, backtrack=True,
                                fixed={(2, 2): boss, (3, 2): clash})['failures'],
        se.solve_socket_wfc_ext(modules, compat, 4, 4, seed=2, max_restarts=3, backtrack=True, unique_states=True,
                                fixed={(0, 0): 7, (3, 3): 7})['failures'],
        se.solve_socket_wfc_ext(modules, compat, 4, 4, seed=2, max_restarts=3, backtrack=True, unique_states=True,
                                domains=se.allow_states([(0, 0), (1, 0)], 1 << 5))['failures']]
    check('failures before the search keep their legacy reasons: emptied domain, clashing pins, one state pinned twice '
          'and, found by the all-different rule, one state forced into two cells',
          setups == [['domains'] * 3, ['propagation'] * 3, ['unique'] * 3, ['unique'] * 3], str(setups))
    check('backtrack=False and None are the legacy solve: no extra result keys, same placements',
          all(set(se.solve_socket_wfc_ext(modules, compat, 5, 5, seed=3, backtrack=b)) ==
              {'success', 'attempts', 'placements', 'failures'} for b in (None, False))
          and se.solve_socket_wfc_ext(modules, compat, 5, 5, seed=3, backtrack=False)['placements'] ==
          rt.solve_socket_wfc(modules, compat, 5, 5, seed=3)['placements'])
    check('bad options raise ValueError', all(_raises_value_error(f) for f in (
        lambda: bt.Backtrack(-1), lambda: bt.Backtrack(2.5), lambda: bt.Backtrack('5'),
        lambda: bt.Backtrack(float('inf')), lambda: bt.Backtrack(float('nan')), lambda: bt.Backtrack.coerce('yes'),
        lambda: se.solve_socket_wfc_ext(modules, compat, 3, 3, backtrack='yes'))))

    # the all-different rule: groups of states, a cell decided once its domain lies in one group
    rule = bt.AllDifferent([0, 0, 1, 1, 2])
    doms = [0b00011, 0b00101, 0b10010, 0b11100]
    changed = deque()
    ok = rule.apply(doms, changed, None)
    check('AllDifferent: a cell inside one group (here {0, 1}) keeps its own states and removes the group from the other '
          'cells; cells it narrows are reported',
          ok and doms == [0b00011, 0b00100, 0b10000, 0b11100] and sorted(changed) == [1, 2], f'{doms} {list(changed)}')
    twice = [0b00001, 0b00010, 0b11100]
    free = [0b00110, 0b11001, 0b11111]
    check('AllDifferent: two cells inside one group, or an empty domain, are a contradiction; with no decided cell '
          'nothing changes', not rule.apply(twice, deque(), None) and not rule.apply([0b00011, 0], deque(), None)
          and rule.apply(free, deque(), None) and free == [0b00110, 0b11001, 0b11111])
    rng_rule = random.Random(5)
    agree, contradictions, narrowed = [], 0, 0
    for _ in range(600):
        n = rng_rule.randint(2, 7)
        group_of = [rng_rule.randrange(max(1, n - rng_rule.randint(0, 3))) for _ in range(n)]
        masks = [rng_rule.randrange(1, 1 << n) for _ in range(rng_rule.randint(2, 6))]
        rule, doms = bt.AllDifferent(group_of), list(masks)
        while True:
            changed = deque()
            if not rule.apply(doms, changed, None):
                doms = None
                break
            if not changed:
                break
        want = _all_different_reference(masks, group_of)
        agree.append(doms == want)
        contradictions += want is None
        narrowed += want is not None and want != masks
    check('AllDifferent to a fixpoint equals a set-based restatement of the all-different forward check on 600 random '
          'cases (2 to 7 states in 1 to 7 groups, 2 to 6 cells)', all(agree) and contradictions > 100 and narrowed > 50,
          f'{contradictions} contradictions, {narrowed} narrowed')
    variants, vcompat = se.expand_d4(modules)
    grouped = se.solve_socket_wfc_ext(variants, vcompat, 10, 10, seed=37, unique_states=True, max_restarts=8,
                                      groups=se.source_groups(variants), backtrack=True)
    check('with D4 variants (656 states) and one-layout-once groups, backtracking solves 10x10 on attempt 1 and uses '
          'each of the 100 layouts once', grouped['success'] and grouped['attempts'] == 1
          and se.check_placements(vcompat, 10, 10, grouped['placements'])
          and sorted(variants[s]['source_state'] for s in grouped['placements']) == list(range(100)))

    # the ready-made problem on its own
    grid, doms, ok = _chess(4, 4)
    problem = bt.GridProblem(grid, doms, random.Random(5))
    fresh = problem.snapshot()
    bt.search(problem)
    only = doms[0].bit_length() - 1
    done = (problem.pick(), problem.forbid(0, only), doms[0] == 1 << only)
    problem.restore(fresh)
    check('GridProblem on its own: nothing is left to pick once every cell is decided, the only state of a cell cannot be '
          'forbidden, and restore puts a snapshot back',
          done == (None, False, True) and doms == fresh and problem.pick() is not None)
    drawn = bt.GridProblem(grid, [0b1011], random.Random(9))
    spread = Counter(drawn.choose(0) for _ in range(600))
    check('GridProblem.choose draws every state of the domain, uniformly: states 0, 1 and 3 of 0b1011 about 200 times '
          'each in 600 draws, state 2 never', set(spread) == {0, 1, 3} and all(150 <= n <= 250 for n in spread.values()),
          str(sorted(spread.items())))

    # the patience backjump
    pat = bt._Patience()
    for _ in range(20):
        pat.chose()
    amounts = []

    def contradiction():
        """One contradiction: ask, undo that many decisions, redo them."""
        amount = pat.backjump()
        for _ in range(amount):
            pat.undone()
        for _ in range(amount):
            pat.chose()
        amounts.append(amount)
    for _ in range(10 + 1 + 6 + 1 + 2 + 1 + 6 + 1):
        contradiction()
    check('patience schedule at depth 20: 10 ordinary backtracks, a jump of 4; 6 more, a jump of 8; 2 more, a jump of 4; '
          '6 more, a jump of 16 (levels wait 10, 20, 40 undone decisions and jump 4, 8, 16 levels back)',
          amounts == [1] * 10 + [4] + [1] * 6 + [8] + [1] * 2 + [4] + [1] * 6 + [16], str(amounts))
    pat = bt._Patience()
    for _ in range(12):
        pat.chose()
    for _ in range(3):
        pat.backjump()
        pat.undone()
        pat.chose()
    pat.chose()                                   # a new greatest depth forgets the schedule
    after = []
    for _ in range(11):
        after.append(pat.backjump())
        pat.undone()
        pat.chose()
    check('reaching a new greatest depth resets the schedule: the next 10 contradictions are ordinary again',
          after[:10] == [1] * 10 and after[10] == 4, str(after))
    pat = bt._Patience()
    for _ in range(20):
        pat.chose()
    for _ in range(12):
        pat.undone()                              # the depth falls to 8, below the 16 a first jump would aim for
    below = [pat.backjump(), pat.backjump()]
    check('a patience level whose target lies above the standing depth asks for an ordinary backtrack, never for no undo '
          'at all', below == [1, 1], str(below))
    modules3, comp = _planted(100, 10, 10, 0.3, 1)
    calm = se.solve_socket_wfc_ext(modules3, comp, 10, 10, seed=1, unique_states=True, max_restarts=8, backtrack=True)
    eager = se.solve_socket_wfc_ext(modules3, comp, 10, 10, seed=1, unique_states=True, max_restarts=8,
                                    backtrack=bt.Backtrack(patience=True))
    again = se.solve_socket_wfc_ext(modules3, comp, 10, 10, seed=1, unique_states=True, max_restarts=8,
                                    backtrack=bt.Backtrack(patience=True))
    check('with patience a hard table (0.3 compatible, 784 plain backtracks) is still solved validly, with backjumps, and '
          'the run is deterministic',
          eager['success'] and se.check_placements(comp, 10, 10, eager['placements'])
          and len(set(eager['placements'])) == 100 and eager['backjumps'] > 0
          and eager['placements'] == again['placements'] and calm['success'],
          f"plain {calm['backtracks']} backtracks; patience {eager['backtracks']} backtracks + {eager['backjumps']} jumps")


def test_blueprint_export():
    print("\nlevel blueprint export")
    import tilegen_blueprint as tb
    import tilegen_keylock as kl
    kl.register()
    try:
        world = tc.compile_recipe(tc.WorldRecipe(macro='ca_caves', size=48, seed=3, name='bp test'))
    finally:
        kl.unregister()
    bp = tb.to_blueprint(world)
    check('blueprint is structurally valid', tb.check_blueprint(bp) == [], str(tb.check_blueprint(bp)[:2]))
    check('grid dimensions match world', (bp['gridDimensions']['width'], bp['gridDimensions']['height']) == (world.w, world.h))
    ok = world.passable()
    check('tile rows round-trip passability', all(
        (bp['tiles'][y][x] != '0') == bool(ok[y][x]) for y in range(world.h) for x in range(world.w)))
    check('bases on walkable tiles', bp['tiles'][bp['playerBase']['gz']][bp['playerBase']['gx']] != '0'
          and all(bp['tiles'][b['gz']][b['gx']] != '0' for b in bp['enemyBases']))
    check('guards map to entities', len(bp['badGuys']) > 0 and {g['id'] for g in bp['badGuys']}
          <= {k for k, v in world.entities.items() if v['role'] == 'guard'})
    check('zones mirror places', len(bp['zones']) == min(200, len(world.place_order)))
    check('locks carry the gate plan', len(bp['locks']) == len(world.artifacts.get('gates', {}).get('locks', [])))
    with tempfile.TemporaryDirectory(prefix='blueprint-') as d:
        path = Path(d) / 'level.blueprint.json'
        written = tb.write_blueprint(world, path)
        back, errs = tb.load_blueprint(path)
        check('file round-trips every field', not errs and back == json.loads(json.dumps(written)))
        check('export is deterministic', json.dumps(tb.to_blueprint(world), sort_keys=True) == json.dumps(bp, sort_keys=True))
    bad = json.loads(json.dumps(bp))
    bad['tiles'][0] = bad['tiles'][0][:-1] + '9'
    check('reader rejects malformed tiles', tb.check_blueprint(bad) == ['tiles rows malformed'])
    bad = json.loads(json.dumps(bp)); bad['playerBase'] = {'gx': -1, 'gz': 0}
    check('reader rejects out-of-bounds base', 'playerBase' in tb.check_blueprint(bad))


def test_bot_playtest():
    print("\nbot playtest")
    import tilegen_playtest as pt
    import tilegen_keylock as kl
    kl.register()
    try:
        world = tc.compile_recipe(tc.WorldRecipe(macro='ca_caves', size=48, seed=3, name='pt test'))
    finally:
        kl.unregister()
    r = pt.playtest(world)
    ok = world.passable()
    check('bot finishes a solvable gated map', r['solved'] and r['locks'] > 0 and r['keys'] == r['locks'], str(r['reason']))
    check('trail is a 4-connected walk on walkable cells', all(
        ok[y][x] for x, y in r['trail']) and all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
                                                  for a, b in zip(r['trail'], r['trail'][1:])))
    check('bot never crosses a lock before holding its key', all(
        r['trail'].index(tuple(g['key']['cell'])) < r['trail'].index(tuple(g['lock']['cell']))
        for g in world.artifacts['gates']['locks'] if tuple(g['lock']['cell']) in r['trail']))
    check('walk is no shorter than the lock-free shortest path', r['steps'] >= r['shortest'] > 0 and r['detour'] >= 1.0)
    check('fun score is bounded and deterministic', 0 < r['fun'] <= 100 and pt.playtest(world) == r)
    bad = tc.compile_recipe(tc.WorldRecipe(macro='ca_caves', size=48, seed=3, name='pt bad'))
    sx, sy = r['trail'][0]
    wall = tuple(int(v[0]) for v in np.nonzero(~bad.passable()))[::-1]      # a key nobody can reach
    bad.artifacts['gates'] = {'start': {'cell': [sx, sy]}, 'goal': {'cell': list(r['trail'][-1])}, 'locks': [
        {'lock': {'cell': [sx + dx, sy + dy]}, 'key': {'cell': list(wall)}}
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))]}
    rb = pt.lint_playtest(bad)
    check('lint reports a bot that cannot finish', not rb['solved'] and any(
        f['rule'] == 'playtest.unsolved' for f in bad.findings), str(bad.findings[-1:]))


def test_erosion():
    print("\nfield erosion")
    h = tf.fbm(96, 96, 3.0, 6, 0.5, 2.0, 7).normalized()
    th = tf.thermal_erode(h, talus=0.01, iterations=30)
    check('thermal creep conserves mass', abs(float(th.a.sum()) - float(h.a.sum())) < 0.1)
    grad = lambda f: float(np.abs(np.diff(f.a, axis=0)).max())
    check('thermal creep lowers the steepest step', grad(th) < grad(h), f"{grad(h):.3f} -> {grad(th):.3f}")
    se = tf.stream_power_erode(h, iterations=14)
    filled = tf.fill_depressions(h.a, 2e-3).a  # the stream-power default fill_eps
    acc = tf.flow_accumulation(filled)[0].a
    ch = acc >= np.quantile(acc, 0.98)
    drop = lambda f: float((filled - f.a)[ch].mean() - (filled - f.a)[~ch].mean())
    check('stream power incises channels more than interfluves', drop(se) > 0, f"extra channel drop {drop(se):.4f}")
    check('erosion never raises above the filled surface', bool((se.a <= filled + 1e-6).all()))
    check('erosion is deterministic', bool((tf.eroded(h).a == tf.eroded(h).a).all()))
    check('stream power incises channels >0.01 deeper than interfluves (was 0.006)', drop(se) > 0.01, f"{drop(se):.4f}")
    fl2 = tf.fill_depressions(h.a, 2e-3).a
    check('epsilon-slope fill leaves no flat step', bool((np.abs(np.diff(fl2, axis=0)) > 0).mean() > 0.99))
    ef, lk = tf.eroded(h, lakes=True)
    check('lakes=True returns a basin mask beside the field', lk.dtype == bool and lk.shape == h.shape and lk.any())
    check('eroded field is cached per input', tf.eroded(h) is tf.eroded(h) and tf.eroded(h, cache=False) is not tf.eroded(h))
    cm = tf.channel_map(h)
    check('channel_map in [0,1], trunks reach 1, hillslopes 0', cm.min() >= 0 and cm.max() <= 1 and cm.max() > 0.99 and (cm == 0).mean() > 0.3)
    lm, soft = tf.smooth_lakes(lk)
    edge = lambda m: int((m[1:] != m[:-1]).sum() + (m[:, 1:] != m[:, :-1]).sum())
    check('smooth_lakes rounds the shore (fewer boundary steps)', edge(lm) < edge(lk), f"{edge(lk)} -> {edge(lm)}")
    plain, rich = tf.relief_rgb(ef, lk, channels=0.0), tf.relief_rgb(ef, lk, channels=1.0)
    check('relief_rgb: uint8 RGB at upscale, deterministic', plain.dtype == np.uint8 and plain.shape == (384, 384, 3) and (rich == tf.relief_rgb(ef, lk, channels=1.0)).all())
    land = ~np.kron(lm, np.ones((4, 4), bool))
    trunk = land & np.kron(cm > 0.8, np.ones((4, 4), bool))
    check('channels>0 darkens trunk channels', trunk.sum() > 100 and rich[trunk].mean() < plain[trunk].mean() - 10, f"{plain[trunk].mean():.1f} -> {rich[trunk].mean():.1f}")
    import tempfile
    from PIL import Image
    with tempfile.TemporaryDirectory(prefix='png16-') as d:
        p = Path(d) / 'hm.png'
        Image.fromarray((h.a * 65535).astype(np.uint16)).save(p)
        back = tf.field_from_png16(p)
        check('png16 heightmap loads as a [0,1] field', back.shape == h.shape and float(np.abs(back.a - h.a).max()) < 2e-4)


def test_edit_overlay():
    print("\nedit overlay")
    recipe = lambda seed: tc.WorldRecipe(macro='ca_caves', size=40, seed=seed, name='edit')
    snap = lambda m: json.dumps([m.base, m.overlay])
    base = tc.compile_recipe(recipe(3), run_features=False, validate=False).tilemap
    live = tc.compile_recipe(recipe(3), run_features=False, validate=False).tilemap
    ov = tw.EditOverlay(live.w, live.h)
    ov.paint(live, 5, 5, base='lava'); ov.paint(live, 6, 5, overlay='torch'); ov.paint(live, 5, 5, overlay='sign')
    ov.paint(live, 9, 9, base='lava'); ov.paint(live, 9, 9, base=base.get(9, 9)[0])
    check('edits are sparse and merge per cell', sorted(ov.edits) == ['5,5', '6,5'], str(sorted(ov.edits)))
    check('reverting a cell drops its edit', '9,9' not in ov.edits)
    again = tc.compile_recipe(recipe(3), run_features=False, validate=False).tilemap
    rep = ov.apply(again)
    check('same recipe: edits re-apply onto the regenerated base exactly', snap(again) == snap(live) and rep['stale'] == [], str(rep))
    back = tw.EditOverlay.from_json(ov.to_json())
    again2 = tc.compile_recipe(recipe(3), run_features=False, validate=False).tilemap
    back.apply(again2)
    check('JSON round-trip applies identically', back.edits == ov.edits and snap(again2) == snap(live))
    check('diff() recovers the same overlay', tw.EditOverlay.diff(base, live).edits.keys() == ov.edits.keys()
          and snap(tw.EditOverlay.diff(base, live).apply(tc.compile_recipe(recipe(3), run_features=False, validate=False).tilemap) and live) == snap(live))
    other = tc.compile_recipe(recipe(4), run_features=False, validate=False).tilemap
    rep = ov.apply(other, skip_stale=True)
    check('different seed: moved base flagged stale and skipped', rep['stale'] and rep['applied'] + len(rep['stale']) >= len(ov.edits)
          and all(other.get(*map(int, k.split(',')))[0] != 'lava' or k not in rep['stale'] for k in ov.edits), str(rep))
    check('out-of-bounds edits are reported not applied', tw.EditOverlay(4, 4, {'99,99': {'was': ['a', ''], 'base': 'b', 'overlay': None}}).apply(other)['oob'] == ['99,99'])
    check('overlay is deterministic', ov.to_json() == tw.EditOverlay.from_json(ov.to_json()).to_json())


def test_tile_atlas():
    print("\nautotile atlas handoff")
    with tempfile.TemporaryDirectory(prefix='tiles-') as td:
        apath, spath = ta.save_tile_atlas(ta.make_edge_atlas('dungeon', 16, seed=5), td, 'dungeon')
        at = tsy.load_tile_atlas(apath, spath)
        meta = json.loads(apath.read_text(encoding='utf-8'))
        check('atlas loads 16 tiles of the declared size', len(at['tiles']) == 16 and all(
            t.size == (at['tile'], at['tile']) for t in at['tiles'].values()), str(at['tile']))
        bad = json.loads(json.dumps(meta)); bad['autotile']['scheme'] = 'blob-47'
        p = Path(td) / 'a.json'; p.write_text(json.dumps(bad))
        try:
            tsy.load_tile_atlas(p, spath); ok = False
        except ValueError:
            ok = True
        check('unsupported scheme is refused', ok)
    g = np.zeros((5, 5), bool); g[1:4, 1:4] = True
    m = tsy.edge_masks(g)
    check('edge masks: centre 15, edge N-open, corner, void -1',
          m[2][2] == 15 and m[1][2] == 2 + 4 + 8 and m[1][1] == 2 + 4 and m[3][1] == 1 + 2 and m[0][0] == -1, str(m[1:4, 1:4].tolist()))
    one = np.zeros((3, 3), bool); one[1, 1] = True
    check('isolated cell is mask 0, border counts as open', tsy.edge_masks(one)[1][1] == 0 and tsy.edge_masks(np.ones((1, 1), bool))[0][0] == 0)
    tm = tc.compile_recipe(tc.WorldRecipe(macro='ca_caves', size=24, seed=3, name='tiles'), run_features=False, validate=False).tilemap
    img = tsy.render_tiles(tm, at)
    check('render is one tile per cell', img.size == (24 * at['tile'], 24 * at['tile']))
    check('render is deterministic', tsy.render_tiles(tm, at).tobytes() == img.tobytes())
    sty = tsy.render_tiles(tm, at, mode='styled')
    proc = tsy.render_map(tm, {}, tile_px=at['tile'], show_label=False)
    check('styled tiles keep procedural colour (moss, walls) at the same size', sty.size == proc.size and
          float(np.abs(np.asarray(sty, float).mean((0, 1)) - np.asarray(proc, float).mean((0, 1))).max()) < 12)
    check('styled tiles add atlas relief', sty.tobytes() != proc.tobytes() and tsy.render_tiles(tm, at, mode='styled').tobytes() == sty.tobytes())
    try:
        tsy.render_tiles(tm, at, mode='x'); ok = False
    except ValueError:
        ok = True
    check('unknown render_tiles mode is refused', ok)


def test_overworld():
    print("\nclimate overworld")
    spec = {'id': 1, 'size': 96, 'seed': 7}
    a, b = ow.build_overworld(spec), ow.build_overworld(spec)
    layers = ('height', 'biome', 'river', 'lake', 'temperature', 'moisture', 'cover')
    check('every layer, site and route is deterministic',
          all(np.array_equal(getattr(a, k), getattr(b, k)) for k in layers)
          and a.settlements == b.settlements and a.routes == b.routes)
    check('the seed changes the world', not np.array_equal(a.biome, ow.build_overworld(dict(spec, seed=8)).biome))
    land = ~a.sea
    check('land share lands on the target ratio', abs(float(land.mean()) - 0.46) < 0.03, f'{land.mean():.3f}')
    arch = ow.build_overworld({'size': 96, 'seed': 7, 'shape': 'archipelago'})
    _, isles = tf.connected_components(~arch.sea)
    check('archipelago: many islands and less land', isles >= 5 and (~arch.sea).mean() < land.mean(), f'{isles} islands')
    for bad in ({'shape': 'donut'}, {'climate': 'lukewarm'}, {'wind': 'up'}):
        try:
            ow.build_overworld(dict(spec, size=32, **bad)); ok = False
        except ValueError:
            ok = True
        check(f"unknown {next(iter(bad))} is refused", ok)

    third = a.shape[0] // 3
    north, south = a.temperature[:third][land[:third]], a.temperature[-third:][land[-third:]]
    check('temperate: the northern third is colder than the southern third',
          north.size and south.size and north.mean() < south.mean() - 0.15, f'{north.mean():.2f} vs {south.mean():.2f}')
    ramp = np.tile(np.linspace(0.0, 1.0, 50), (30, 1))
    lapse = ow.temperature(ramp, 0.2, 5, 'temperate', lapse=0.45, jitter=0.0)
    check('altitude cools: temperature falls along a rising slope',
          bool((np.diff(lapse[:, 12:], axis=1) <= 0).all()) and bool((np.diff(lapse[-1, 12:]) < 0).all()))
    flat = ow.temperature(np.zeros((40, 10)), 0.5, 5, 'pole_to_pole', jitter=0.0)
    check('pole_to_pole: the equator runs across the middle', flat[20].mean() > flat[1].mean() + 0.8 and flat[20].mean() > flat[-2].mean() + 0.8)
    xs = np.arange(60)
    ridge = np.tile(0.1 + 0.85 * np.exp(-((xs - 30) / 4.0) ** 2), (40, 1))
    sea = np.tile((xs < 6) | (xs >= 54), (40, 1))
    pw, pe = ow.precipitation(ridge, sea, 'west'), ow.precipitation(ridge, sea, 'east')
    check('rain shadow: the lee of a range is much drier than its windward face',
          pw[:, 36:47].mean() < 0.6 * pw[:, 15:26].mean(), f'{pw[:, 15:26].mean():.3f} vs {pw[:, 36:47].mean():.3f}')
    check('an east wind mirrors the shadow', pe[:, 15:26].mean() < 0.6 * pe[:, 36:47].mean())

    labels, n = tf.connected_components(a.river)
    mouth = tf.dilate(a.sea | a.lake, 1)
    check('every river reaches the sea or a lake', n > 0 and all(mouth[labels == i].any() for i in range(1, n + 1)), f'{n} rivers')
    wet = a.river | a.sea | a.lake
    pad = np.pad(wet, 1)
    joins = pad[:-2, 1:-1].astype(int) + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:]
    ok = all(wet.flat[int(a.receiver[i])] or joins.flat[i] >= 2 for i in np.flatnonzero(a.river))
    check('rivers follow their drainage downstream (or fill a diagonal step)', ok)
    check('lakes sit in basins on land', a.lake.any() and not (a.lake & a.sea).any())

    land_ids = set(np.unique(a.biome[land]).tolist()) - set(ow.WATER_BIOMES)
    check('at least eight land biomes on a temperate continent', len(land_ids) >= 8, str(sorted(ow.BIOMES[i].name for i in land_ids)))
    snow, desert = a.biome == ow.BIOME['snow'], a.biome == ow.BIOME['desert']
    check('snow lies only where it is cold', not snow.any() or float(a.temperature[snow].max()) < 0.17)
    check('deserts are dry', not desert.any() or float(a.moisture[desert].max()) < 0.2)
    tm = ow.to_tilemap(a)
    base = np.array(tm.base, dtype=object)
    peaks = np.isin(a.biome, (ow.BIOME['alpine'], ow.BIOME['snow']))
    check('peaks are walls or carved passes; walls cover at least 2.5% of the map',
          bool(np.isin(base[peaks], ('wall', 'platform')).all()) and (base == 'wall').mean() >= 0.025,
          f"{(base == 'wall').mean():.3f}")
    wet = np.isin(a.biome, ow.WATER_BIOMES)
    check('water biomes are water tiles (bridges where roads cross rivers)',
          bool(np.isin(base[wet], ('water', 'bridge')).all()) and bool((base[a.sea | a.lake] == 'water').all()))

    sites, size = a.settlements, 96
    tiers = [s['tier'] for s in sites]
    check('one city, then towns, villages and hamlets', tiers.count(3) == 1 and {0, 1, 2} <= set(tiers), str(tiers))
    spaced = all(math.hypot(p['x'] - q['x'], p['y'] - q['y']) + 1e-9 >=
                 max(ow.SPACING[q['tier']] * size, p['radius'] + q['radius'] + ow.SPACING[0] * size * 0.6)
                 for i, p in enumerate(sites) for q in sites[i + 1:])
    check('central-place spacing holds between every pair', spaced)
    dry = all(not (a.sea[s['y'], s['x']] or a.lake[s['y'], s['x']] or a.river[s['y'], s['x']] or peaks[s['y'], s['x']])
              for s in sites)
    check('settlements stand on dry, open ground', dry)
    check('names are unique', len({s['name'] for s in sites}) == len(sites))
    d_sea = tf.distance_to_mask(a.sea).a
    check('settings are port, river, hill or market; ports sit by the sea',
          {s['setting'] for s in sites} <= {'port', 'river', 'hill', 'market'}
          and all(d_sea[s['y'], s['x']] <= 3.0 for s in sites if s['setting'] == 'port'))

    walk = all(abs(x0 - x1) + abs(y0 - y1) == 1 for r in a.routes for (x0, y0), (x1, y1) in zip(r['cells'], r['cells'][1:]))
    check('every route is a 4-connected walk', a.routes and walk, f'{len(a.routes)} routes')
    check('no route crosses sea or lake', not any(a.sea[y, x] or a.lake[y, x] for r in a.routes for x, y in r['cells']))
    net = np.zeros(a.shape, dtype=bool)
    for r in a.routes:
        for x, y in r['cells']:
            net[y, x] = True
    for s_ in sites:
        for x, y in ow._footprint(a, s_):
            net[y, x] = True
    nl, _ = tf.connected_components(net)
    ll, _ = tf.connected_components(~(a.sea | a.lake))
    cap = sites[0]
    same = [s_ for s_ in sites if ll[s_['y'], s_['x']] == ll[cap['y'], cap['x']]]
    check("one road network joins every settlement on the capital's landmass",
          len({nl[s_['y'], s_['x']] for s_ in same}) == 1, f'{len(same)} settlements')
    road_on_river = {(x, y) for r in a.routes for x, y in r['cells'] if a.river[y, x]}
    check('bridges exactly where roads cross rivers', {(int(x), int(y)) for y, x in zip(*np.nonzero(base == 'bridge'))} == road_on_river)
    check('trunks are shared: routes carve fewer new cells than they cover',
          sum(r['new_cells'] for r in a.routes) < sum(r['length'] for r in a.routes))
    check('each settlement is marked on the overlay', all(tm.overlay[s_['y']][s_['x']] == 'district' for s_ in sites))

    rec_ = ow.overworld_record(tm)
    check('record round-trips through JSON', json.loads(json.dumps(rec_)) == json.loads(json.dumps(rec_, sort_keys=True))
          and rec_['schema'] == 'tilegen.overworld/1' and len(rec_['settlements']) == len(sites))
    img = ow.render_overworld(tm, upscale=3, legend=False)
    check('render: a picture at the upscale, deterministic', img.size == (288, 288)
          and ow.render_overworld(tm, upscale=3, legend=False).tobytes() == img.tobytes())
    plain = copy.copy(tm)
    plain.field_palettes = None
    sat = lambda m, st: tsy.render_map(m, spec, style=st, tile_px=3, show_label=False).tobytes()
    check('satellite style honours the biome palette; neon ignores it',
          sat(tm, 'satellite_topographic') != sat(plain, 'satellite_topographic')
          and sat(tm, 'neon_circuit') == sat(plain, 'neon_circuit'))
    world = tc.compile_recipe(tc.WorldRecipe(macro='overworld', size=64, seed=4, name='overworld test'))
    errors = [f for f in world.findings if f['severity'] == 'error']
    check('the world compiler runs every feature on an overworld', not errors and len(world.places) > 5,
          f"places={len(world.places)} errors={[f['rule'] for f in errors][:3]}")
    hy = tg.gen_hybrid_morph({'id': 1, 'size': 48, 'seed': 5, 'macro': 'overworld', 'structure': 'fbm',
                              'op': 'mul'}, random.Random(5))
    check('overworld moisture is a Hybrid Morph source', tf.connected_components(tg.passable_mask(hy))[1] == 1)


def test_autotile():
    print("\ngenerated autotiles")
    at = ta.make_edge_atlas('dungeon', 32, seed=3)
    tiles = {m: np.asarray(t) for m, t in at['tiles'].items()}
    check('16 tiles of the requested size', sorted(tiles) == list(range(16)) and all(t.shape == (32, 32, 4) for t in tiles.values()))
    same = ta.make_edge_atlas('dungeon', 32, seed=3)
    check('deterministic per seed; seed and skin change it',
          all(np.array_equal(tiles[m], np.asarray(same['tiles'][m])) for m in range(16))
          and not np.array_equal(tiles[15], np.asarray(ta.make_edge_atlas('dungeon', 32, seed=4)['tiles'][15]))
          and not np.array_equal(tiles[15], np.asarray(ta.make_edge_atlas('cave', 32, seed=3)['tiles'][15])))
    q = slice(12, 20)
    seam = all((not (m & ta.N) or np.array_equal(tiles[m][0, q], tiles[15][0, q]))
               and (not (m & ta.S) or np.array_equal(tiles[m][-1, q], tiles[15][-1, q]))
               and (not (m & ta.W) or np.array_equal(tiles[m][q, 0], tiles[15][q, 0]))
               and (not (m & ta.E) or np.array_equal(tiles[m][q, -1], tiles[15][q, -1])) for m in range(16))
    check('joined sides show the shared ground texture, so neighbours meet seamlessly', seam)
    ground, rim, accent = ta.PALETTES['dungeon']['classic']
    rim_rgb = {ta._rgb(c) for c in rim + accent}
    cleared = all((m & ta.N) or tuple(tiles[m][0, 16, :3]) in rim_rgb for m in range(16))
    check('every cleared side carries the rim', cleared)
    allowed = {ta._rgb(c) for c in ground + rim + accent}
    check('tiles are palette-limited (ten colours at most)',
          all({tuple(px) for px in t[..., :3].reshape(-1, 3)} <= allowed for t in tiles.values()))
    pn = ta.periodic_noise(32, 4, 9)
    jump = float(np.abs(pn[:, -1] - pn[:, 0]).max())
    check('textures are tile-periodic: the wrap seam is no rougher than the interior',
          jump <= float(np.abs(np.diff(pn, axis=1)).max()) + 1e-9)
    check('47 canonical blob masks out of 256', len(ta.BLOB_MASKS) == 47 and ta.canonical_blob(ta.NE) == 0
          and ta.canonical_blob(ta.N | ta.E | ta.NE) == ta.N | ta.E | ta.NE)
    g = np.zeros((5, 5), bool); g[1:4, 1:4] = True
    bm = ta.blob_masks(g)
    check('blob masks: full centre 255, corner sees two sides and their diagonal, void -1',
          bm[2, 2] == 255 and bm[1, 1] == ta.E | ta.S | ta.SE and bm[0, 0] == -1)
    blob = ta.make_blob_atlas('dungeon', 32, seed=3)
    full = {m: m | (ta.NE if m & 3 == 3 else 0) | (ta.SE if m & 6 == 6 else 0) | (ta.SW if m & 12 == 12 else 0)
            | (ta.NW if m & 9 == 9 else 0) for m in range(16)}
    check('blob tiles without diagonal gaps equal the Wang tiles; a gap adds a notch',
          all(np.array_equal(np.asarray(blob['tiles'][full[m]]), tiles[m]) for m in range(16))
          and not np.array_equal(np.asarray(blob['tiles'][15]), tiles[15]))
    with tempfile.TemporaryDirectory(prefix='autotile-') as td:
        back = tsy.load_tile_atlas(*ta.save_tile_atlas(at, td, 'dungeon'))
        check('save -> load_tile_atlas round-trips every pixel',
              all(np.array_equal(np.asarray(back['tiles'][m]), tiles[m]) for m in range(16)) and back['tile'] == 32)
    tm = tg.GRAMMARS['ca_caves']({'id': 1, 'size': 24, 'seed': 5}, random.Random(5))
    rw = ta.render_autotile(tm, 'cave', tile_px=8)
    rbl = ta.render_autotile(tm, 'cave', tile_px=8, scheme='blob')
    check('render_autotile: one tile per cell; blob corners differ from edge tiles', rw.size == (192, 192)
          and rw.tobytes() != rbl.tobytes() and ta.render_autotile(tm, 'cave', tile_px=8).tobytes() == rw.tobytes())
    refused = 0
    for kw in ({'scheme': 'hex'}, {'skin': 'lava'}, {'palette': 'neon'}):
        try:
            ta.render_autotile(tm, **{'skin': 'cave', **kw}); 
        except ValueError:
            refused += 1
    check('unknown scheme, skin or palette is refused', refused == 3)
    check('every skin ships four palettes', all(len(v) == 4 for v in ta.PALETTES.values()) and set(ta.PALETTES) == set(ta.SKINS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--full', action='store_true')
    args = ap.parse_args()
    t0 = time.time()

    test_species_coverage()
    test_atlas_coverage()
    test_goal_total()
    test_grammars(size=64 if not args.full else 96)
    world = test_world_pipeline()
    test_topologies_disagree(world)
    test_feature_graph()
    test_relationships()
    test_quest_hooks(world)
    test_text_export(world)
    test_compiler(args.full)
    test_determinism()
    test_validators_catch_problems()
    test_runtime_budget()
    test_backwards_compatibility()
    test_poisson_variable_radius()
    test_vector_delaunay()
    test_vector_voronoi_lloyd()
    test_vector_proximity_graphs()
    test_prim_join()
    test_ca_caves()
    test_socket_d4()
    test_socket_hooks()
    test_socket_quota()
    test_overlapping_wfc()
    test_inertia_routes()
    test_dead_end_retraction()
    test_key_lock_gating()
    test_trace_rings()
    test_rbf_field()
    test_random_frontier()
    test_convchain()
    test_kiss_cleanup()
    test_backtracking()
    test_blueprint_export()
    test_bot_playtest()
    test_erosion()
    test_edit_overlay()
    test_tile_atlas()
    test_overworld()
    test_autotile()

    print('\n' + '-' * 60)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): {', '.join(FAILURES[:8])}")
        return 1
    print(f"22 map families and 40 simulation features checked in {time.time() - t0:.1f}s")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
