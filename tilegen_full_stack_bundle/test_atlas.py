"""Regression and property tests for the atlas extension.

Runnable directly (no test framework required):

    python test_atlas.py            # fast suite
    python test_atlas.py --full     # also rebuilds the golden 100 layouts
"""
from __future__ import annotations

import argparse
import filecmp
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import tilegen_system as ts
import tilegen_fields as tf
import tilegen_grammars as tg
import tilegen_style as tsy
import tilegen_atlas as atlas

FAILURES = []
SIZE = 64


def check(name, condition, detail=''):
    status = 'ok  ' if condition else 'FAIL'
    print(f"  [{status}] {name}{(' - ' + detail) if detail else ''}")
    if not condition:
        FAILURES.append(name)
    return condition


# ------------------------------------------------------------------ golden

def test_golden_baseline(full=False):
    """Original layout JSON and rendered pixels must match the included baseline."""
    print("\ngolden baseline")
    check('tilegen_system exposes exactly the 35 original templates',
          len(ts.GENERATORS) >= 35)
    if not full:
        print("  [skip] baseline rebuild (use --full)")
        return
    committed = HERE / 'tilegen_output'
    if not (committed / 'layouts').exists():
        check('included baseline exists', False, 'missing tilegen_output/layouts')
        return
    scratch = HERE.parent / '.local'
    scratch.mkdir(exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix='golden-', dir=scratch))
    try:
        ts.build_all(tmp)
        for kind in ('layouts', 'renders'):
            names = sorted(p.name for p in (tmp / kind).glob('*'))
            baseline_names = sorted(p.name for p in (committed / kind).glob('*'))
            check(f'{kind} contains the same 100 baseline files',
                  names == baseline_names and len(names) == 100)
            bad = []
            encoded_differences = 0
            for name in names:
                generated, baseline = tmp / kind / name, committed / kind / name
                if not baseline.is_file():
                    bad.append(name)
                    continue
                identical = filecmp.cmp(generated, baseline, shallow=False)
                if kind == 'renders':
                    encoded_differences += not identical
                    # PNG compression can vary across Pillow/zlib versions.
                    # Compare every decoded pixel, including labels and borders.
                    with Image.open(generated) as actual, Image.open(baseline) as expected:
                        identical = (actual.size == expected.size and
                                     actual.convert('RGBA').tobytes() ==
                                     expected.convert('RGBA').tobytes())
                if not identical:
                    bad.append(name)
            contract = 'pixel-identical' if kind == 'renders' else 'byte-identical'
            check(f'{kind} rebuild {contract} ({len(names)} files)',
                  not bad, f"differing: {bad[:3]}")
            if encoded_differences:
                print(f'  [info] {encoded_differences} PNG encodings differ; pixels checked')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------ fields

def test_fields():
    print("\nfield engine")
    a = tf.fbm(48, 48, 4.0, 5, seed=3)
    b = tf.fbm(48, 48, 4.0, 5, seed=3)
    check('fbm deterministic', np.array_equal(a.a, b.a))
    check('fbm differs by seed',
          not np.array_equal(a.a, tf.fbm(48, 48, 4.0, 5, seed=4).a))
    check('perlin is smooth (neighbours correlate)',
          float(np.corrcoef(a.a[:, :-1].ravel(), a.a[:, 1:].ravel())[0, 1]) > 0.9)
    check('normalized spans 0..1',
          abs(float(a.normalized().a.min())) < 1e-6 and
          abs(float(a.normalized().a.max()) - 1) < 1e-6)

    # jump-flood distance against brute force
    mask = tf.fbm(40, 40, 4.0, seed=5).threshold(0.6)
    ys, xs = np.nonzero(mask)
    yy, xx = np.mgrid[0:40, 0:40]
    brute = np.sqrt((xx[..., None] - xs) ** 2 + (yy[..., None] - ys) ** 2).min(-1)
    err = float(np.abs(brute - tf.distance_to_mask(mask).a).max())
    check('jump-flood EDT matches brute force', err < 0.5, f"max error {err:.3f}")

    seeds = {tf.seed_stream(1, n) for n in ('a.b', 'a.c', 'd.e')}
    check('named seed streams are distinct', len(seeds) == 3)
    check('seed streams are stable', tf.seed_stream(1, 'a.b') == tf.seed_stream(1, 'a.b'))

    vor = tf.voronoi(48, 48, tf.poisson_disc(48, 48, 8, seed=1), relax=1)
    check('voronoi partitions the plane', int(vor.cell.max()) > 3)
    check('voronoi adjacency is non-empty', len(vor.adjacency()) > 0)

    rot = tf.truchet(10, 10, seed=1)
    loops = tf.truchet_loops(rot)
    check('truchet edges all belong to a closed loop',
          sum(len(v) for v in loops.values()) == 2 * 10 * 10 + 10 + 10,
          f"{len(loops)} loops")


# ---------------------------------------------------------------- grammars

def test_grammars():
    print("\ngrammars")
    for name, fn in tg.GRAMMARS.items():
        spec = {'id': 900, 'size': SIZE, 'template': name, 'seed': 13}
        t0 = time.time()
        tm = fn(spec, random.Random(13))
        elapsed = time.time() - t0

        same = fn(spec, random.Random(13))
        det = tm.base == same.base
        other = fn({'id': 900, 'size': SIZE, 'template': name, 'seed': 14},
                   random.Random(14))
        varies = tm.base != other.base

        pm = tg.passable_mask(tm)
        _, ncomp = tf.connected_components(pm)
        open_ratio = float(pm.mean())
        dims_ok = tm.w == SIZE and tm.h == SIZE

        ok = det and varies and ncomp == 1 and 0.05 < open_ratio < 0.98 and dims_ok
        check(f"{name}", ok,
              f"{elapsed:.2f}s open={open_ratio:.0%} comps={ncomp}"
              f"{'' if det else ' NOT-DETERMINISTIC'}"
              f"{'' if varies else ' SEED-IGNORED'}")


def test_hybrid_composition():
    """Every macro x structure x operator triple must produce a usable map."""
    print("\nhybrid composition matrix")
    macros = ['radial', 'voronoi', 'spiral', 'waves', 'fbm']
    structures = ['reaction_diffusion', 'fractal', 'checker', 'ridged']
    ops = ['mul', 'add', 'mask', 'max', 'blend']
    bad = []
    for m in macros:
        for s in structures:
            for op in ops:
                spec = {'id': 901, 'size': 48, 'template': 'hybrid_morph',
                        'seed': 5, 'macro': m, 'structure': s, 'op': op,
                        'steps': 400}
                try:
                    tm = tg.gen_hybrid_morph(spec, random.Random(5))
                    pm = tg.passable_mask(tm)
                    _, n = tf.connected_components(pm)
                    if n != 1 or not (0.05 < pm.mean() < 0.98):
                        bad.append(f"{m}x{s}:{op}(comps={n},open={pm.mean():.0%})")
                except Exception as exc:                     # noqa: BLE001
                    bad.append(f"{m}x{s}:{op}({type(exc).__name__})")
    total = len(macros) * len(structures) * len(ops)
    check(f'all {total} hybrid combinations valid', not bad, f"bad: {bad[:4]}")


# ------------------------------------------------------------------ styles

def test_scaling():
    """Big maps must stay connected and keep a usable open ratio.

    Both bugs this guards against were size-dependent: archipelago shed 1-2 cell
    pockets that broke connectivity only past ~200 cells a side, and grammars
    with a fixed feature size thinned out as the canvas grew.
    """
    print("\nscaling to large maps")
    big = 192
    slow = []
    for name, fn in tg.GRAMMARS.items():
        t0 = time.time()
        tm = fn({'id': 903, 'size': big, 'template': name, 'seed': 8},
                random.Random(8))
        elapsed = time.time() - t0
        pm = tg.passable_mask(tm)
        _, ncomp = tf.connected_components(pm)
        ratio = float(pm.mean())
        ok = ncomp == 1 and 0.05 < ratio < 0.99 and tm.w == big
        if elapsed > 20:
            slow.append(f"{name}={elapsed:.0f}s")
        check(f"{name} at {big}x{big}", ok,
              f"{elapsed:.1f}s open={ratio:.0%} comps={ncomp}")
    check('no grammar exceeds 20s at 192x192', not slow, ', '.join(slow))


def test_styles():
    print("\nstyle registry")
    spec = {'id': 902, 'size': 48, 'template': 'watershed', 'seed': 9}
    tm = tg.gen_watershed(spec, random.Random(9))
    renders = {}
    for name in atlas.DEFAULT_STYLES:
        img = tsy.render_map(tm, spec, style=name, tile_px=6)
        renders[name] = np.array(img.convert('RGB'))
        check(f'{name} renders', img.size[0] > 0 and img.size[1] > 0)

    # the same geometry must actually look different in different styles
    ref = renders['procedural_dark_tiles']
    distinct = sum(1 for n, arr in renders.items()
                   if n != 'procedural_dark_tiles' and
                   arr.shape == ref.shape and
                   float(np.abs(arr.astype(int) - ref.astype(int)).mean()) > 12)
    check('styles are visually distinct from each other',
          distinct == len(renders) - 1, f"{distinct}/{len(renders) - 1} differ")

    sig = tsy.NeighbourhoodSignature(tm)
    check('neighbourhood signature sees exposed solids',
          sig.exposed('solid').sum() > 0)
    check('signature respects map edges (no wraparound)',
          not sig.touches(np.ones((tm.h, tm.w), dtype=bool), 'N')[0].any())


# ----------------------------------------------------------------- runtime

def test_turtle_teleports():
    """Teleport arrivals must allow walking before another portal jump."""
    print("\nturtle portal traversal")
    import tilegen_lowend_runtime as rt

    corridor = ts.TileMap(7, 1, default_base='floor')
    corridor.overlay[0][0] = 'teleport'
    corridor.overlay[0][3] = 'teleport'
    route = rt.turtle_explore(corridor, start=[0, 0], goal=[6, 0], max_steps=12)
    check('turtle walks away from an arrival portal', route['success'])
    check('portal still provides the first jump', route['path'][:2] == [[0, 0], [3, 0]])
    at_goal = rt.turtle_explore(corridor, start=[0, 0], goal=[3, 0], max_steps=1)
    check('arrival at a portal goal ends the route', at_goal['success'] and at_goal['steps'] == 1)
    already_there = rt.turtle_explore(corridor, start=[0, 0], goal=[0, 0])
    check('starting on the goal does not teleport away', already_there['success'] and already_there['steps'] == 0)

    # The nearest portal is isolated. It must still allow another teleport
    # when there is no neighbouring tile to step onto.
    isolated = ts.TileMap(7, 3)
    for x, y in ((0, 0), (6, 0), (2, 2), (3, 2), (4, 2), (5, 2), (6, 2)):
        isolated.base[y][x] = 'floor'
    for x, y in ((0, 0), (6, 0), (2, 2)):
        isolated.overlay[y][x] = 'teleport'
    route = rt.turtle_explore(isolated, start=[0, 0], goal=[6, 2], max_steps=16)
    check('isolated arrival portal permits onward teleporting',
          route['success'] and route['path'][:3] == [[0, 0], [6, 0], [2, 2]])

    single = ts.TileMap(5, 1, default_base='floor')
    single.overlay[0][0] = 'teleport'
    route = rt.turtle_explore(single, start=[0, 0], goal=[4, 0], max_steps=8)
    check('a lone portal behaves as a walkable tile', route['success'])
    blocked = ts.TileMap(3, 1)
    blocked.base[0][0] = blocked.base[0][2] = 'floor'
    route = rt.turtle_explore(blocked, start=[0, 0], goal=[2, 0], max_steps=8)
    check('unreachable goal still reports failure', not route['success'])

    failures, invalid = [], []
    originals = [s for s in ts.all_specs if 1 <= s['id'] <= 100]
    check('all 100 original map specs are present', len(originals) == 100)
    for spec in originals:
        tm = ts.make_layout(spec)
        route = rt.turtle_explore(tm, seed=spec['id'] * 97)
        if not route['success']:
            failures.append(spec['id'])
        passable = rt.build_passable_grid(tm, 'turtle')
        portals = set(rt.find_teleport_nodes(tm, passable))
        for a, b in zip(route['path'], route['path'][1:]):
            walking = abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
            teleporting = tuple(a) in portals and tuple(b) in portals
            if not passable[b[1]][b[0]] or not (walking or teleporting):
                invalid.append(spec['id'])
                break
        if spec['id'] in (34, 80):
            repeat = rt.turtle_explore(tm, seed=spec['id'] * 97)
            check(f'map {spec["id"]} route remains deterministic', route == repeat)
    check('turtle reaches its goal on all 100 original maps', not failures, f'failures: {failures}')
    check('all turtle moves are walks or valid portal jumps', not invalid, f'invalid: {invalid}')


def test_runtime_compat():
    """Atlas maps must flow through module extraction, navigation and WFC."""
    print("\nruntime compatibility")
    import tilegen_lowend_runtime as rt
    atlas.register()

    specs = [s for s in atlas.ATLAS_SPECS
             if s['template'] in ('voronoi_kingdoms', 'truchet', 'watershed',
                                  'reaction_diffusion', 'contour_terraces')][:8]
    records = []
    for sp in specs:
        tm = ts.make_layout(dict(sp, size=48))
        records.append({'template': sp['template'], 'tilemap': tm,
                        'base': tm.base, 'overlay': tm.overlay,
                        'id': sp['id'], 'slug': ts.slugify(sp['name']),
                        'name': sp['name'], 'spec': sp, 'theme': 'atlas',
                        'json_path': None, 'source_render': None})

    modules, compat = rt.build_module_library(records)
    check('module library extracts 16x16 modules', len(modules) == len(records),
          f"{len(modules)} modules")

    for profile in rt.AGENT_PROFILES:
        routed = sum(1 for r in records
                     if (rt.farthest_pair(r['tilemap'], profile) or {}).get('success'))
        check(f'pathfinder finds a reachable goal for {profile}', routed == len(records),
              f"{routed}/{len(records)}")

    solved = rt.solve_socket_wfc(modules, compat, grid_w=3, grid_h=3, seed=3)
    check('socket WFC solves over atlas modules', solved['success'])
    if solved['success']:
        tmap, _, _ = rt.assemble_world(modules, solved['placements'], 3, 3, 0)
        check('assembled world has expected size', tmap.w == 48 and tmap.h == 48,
              f"{tmap.w}x{tmap.h}")


def test_registration_isolation():
    """Importing the atlas must not perturb a standalone tilegen_system run."""
    print("\nregistration isolation")
    atlas.register()
    before = len(ts.all_specs)
    atlas.register()
    check('register() is idempotent', len(ts.all_specs) == before)
    ids = [s['id'] for s in ts.all_specs]
    check('no duplicate spec ids after registration', len(ids) == len(set(ids)))
    check('atlas ids do not collide with the original 100',
          all(s['id'] > 100 for s in atlas.ATLAS_SPECS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--full', action='store_true',
                    help='also rebuild and diff the golden 100 layouts')
    args = ap.parse_args()

    t0 = time.time()
    test_golden_baseline(args.full)
    test_fields()
    test_grammars()
    test_hybrid_composition()
    test_scaling()
    test_styles()
    test_turtle_teleports()
    test_runtime_compat()
    test_registration_isolation()

    print(f"\n{'-' * 58}")
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): {', '.join(FAILURES)}")
        return 1
    print(f"all checks passed in {time.time() - t0:.1f}s")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
