"""Coverage and behaviour tests for 20 map families and 40 simulation features.

Roguelike Depth Layers contains 60 distinct components. Run directly:

    python test_goals.py           # coverage + behaviour
    python test_goals.py --full    # also every grammar at size, and all recipes
"""
from __future__ import annotations

import argparse
import copy
import json
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
import tilegen_world as tw
import tilegen_layers  # noqa: F401
import tilegen_society as society
import tilegen_compiler as tc
import tilegen_style as tsy
import tilegen_runtime_budget as rb

FAILURES = []

# The 20 map families covered by this suite.
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
]


def check(name, condition, detail=''):
    print(f"  [{'ok  ' if condition else 'FAIL'}] {name}{(' - ' + detail) if detail else ''}")
    if not condition:
        FAILURES.append(name)
    return condition


# ------------------------------------------------------------ goal coverage

def test_species_coverage():
    print("\n20 map families")
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
    check('20 map families implemented', grammars == 20, f'{grammars}/20')
    check('40 simulation features 21-60 implemented', feats == 40, f'{feats}/40')
    check('60 distinct components', grammars + feats == 60,
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

    print('\n' + '-' * 60)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): {', '.join(FAILURES[:8])}")
        return 1
    print(f"20 map families and 40 simulation features checked in {time.time() - t0:.1f}s")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
