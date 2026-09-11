"""The remaining mathematical grammars: features 05, 06, 08, 11, 14, 15, 16, 18, 19.

Same contract as tilegen_grammars: synthesise curves, fields or graphs first,
interpret to tiles last. Importing this module extends GRAMMARS in place.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

import tilegen_fields as tf
import tilegen_grammars as tg
from tilegen_grammars import (WALL, FLOOR, WATER, LAVA, ABYSS, COURT, RUBBLE,
                              BRIDGE, PLATFORM, dims, grammar_seed, build,
                              attach, passable_mask, connect_components,
                              stamp_polyline, carve_l, size_scale,
                              threshold_for_ratio)


# ------------------------------------------------- 05 Space-Filling Curves

def hilbert_points(order):
    """Hilbert curve cell sequence for a 2**order square."""
    n = 1 << order
    pts = []
    for d in range(n * n):
        rx = ry = 0
        t = d
        x = y = 0
        s = 1
        while s < n:
            rx = 1 & (t // 2)
            ry = 1 & (t ^ rx)
            # rotate the quadrant into place
            if ry == 0:
                if rx == 1:
                    x, y = s - 1 - x, s - 1 - y
                x, y = y, x
            x += s * rx
            y += s * ry
            t //= 4
            s *= 2
        pts.append((x, y))
    return pts


def peano_points(order):
    """Peano curve on a 3**order square, by digit-wise reflection."""
    n = 3 ** order
    pts = []
    for d in range(n * n):
        x = y = 0
        t = d
        digits = []
        for _ in range(order):
            digits.append(t % 9)
            t //= 9
        digits.reverse()
        flip = False
        for k, dig in enumerate(digits):
            scale = 3 ** (order - 1 - k)
            col, row = divmod(dig, 3)
            if col % 2 == 1:
                row = 2 - row
            if flip:
                col = 2 - col
            x += col * scale
            y += row * scale
            if col % 2 == 1:
                flip = not flip
        pts.append((x, y))
    return pts


def _turtle(axiom, rules, iterations, angle_deg, step, start=(0.0, 0.0),
            heading=0.0, max_len=200000):
    """Expand an L-system and walk it, returning the polyline."""
    s = axiom
    for _ in range(iterations):
        s = ''.join(rules.get(c, c) for c in s)
        if len(s) > max_len:
            break
    x, y = start
    a = math.radians(heading)
    da = math.radians(angle_deg)
    pts = [(x, y)]
    stack = []
    for c in s:
        if c in 'ABFG':
            x += math.cos(a) * step
            y += math.sin(a) * step
            pts.append((x, y))
        elif c == '+':
            a += da
        elif c == '-':
            a -= da
        elif c == '[':
            stack.append((x, y, a))
        elif c == ']':
            if stack:
                x, y, a = stack.pop()
                pts.append((x, y))
    return pts


def moore_points(order, step=1.0):
    """Moore curve: a closed variant of the Hilbert curve.

    Built from its own L-system. Assembling it by reflecting four Hilbert
    quadrants does not work - the reflections break the curve's continuity at
    the joins and it degenerates into disconnected combs.
    """
    return _turtle('LFL+F+LFL',
                   {'L': '-RF+LFL+FR-', 'R': '+LF-RFR-FL+'},
                   order, 90.0, step)


def gosper_points(order, step=1.0):
    return _turtle('A', {'A': 'A-B--B+A++AA+B-', 'B': '+A-BB--B-A++A+B'},
                   order, 60.0, step)


def sierpinski_points(order, step=1.0):
    return _turtle('A', {'A': 'B-A-B', 'B': 'A+B+A'}, order, 60.0, step)


CURVES = ('hilbert', 'moore', 'peano', 'gosper', 'sierpinski')


def gen_space_filling(spec, rng):
    """Hilbert, Moore, Peano, Gosper and Sierpinski curves as corridors.

    Locality survives across scale: two points close on the curve are close in
    space, so progress feels physically near yet topologically distant - archives,
    bureaucratic complexes, ritual pilgrimages.
    """
    w, h = dims(spec)
    seed = grammar_seed(spec, 'geo.space_filling')
    kind = spec.get('curve', 'hilbert')
    margin = spec.get('margin', 3)
    span = min(w, h) - margin * 2

    if kind == 'hilbert':
        order = spec.get('order', max(2, int(math.log2(max(span // 4, 4)))))
        pts = hilbert_points(order)
    elif kind == 'moore':
        # the Moore L-system emits a far denser curve per order than the
        # iterative Hilbert construction, so it needs one order less to leave
        # walls between its corridors rather than flooding the map
        order = spec.get('order', max(2, int(math.log2(max(span // 8, 4)))))
        pts = moore_points(order)
    elif kind == 'peano':
        order = spec.get('order', max(2, int(math.log(max(span // 4, 3), 3))))
        pts = peano_points(order)
    else:
        order = spec.get('order', 3 if kind == 'gosper' else 5)
        raw = gosper_points(order) if kind == 'gosper' else sierpinski_points(order)
        pts = raw

    arr = np.array(pts, dtype=float)
    arr -= arr.min(axis=0)
    scale = span / max(arr.max(), 1e-6)
    arr = arr * scale + margin

    # The curve itself is exact mathematics and has no randomness, so seed
    # variation comes from its placement: one of the 8 dihedral orientations.
    # Without this every seed returns a byte-identical map.
    variant = spec.get('orientation', tf.seed_stream(seed, 'orientation') % 8)
    cx, cy = arr[:, 0].mean(), arr[:, 1].mean()
    if variant & 1:
        arr[:, 0] = 2 * cx - arr[:, 0]
    if variant & 2:
        arr[:, 1] = 2 * cy - arr[:, 1]
    if variant & 4:
        arr = np.stack([arr[:, 1] - cy + cx, arr[:, 0] - cx + cy], axis=1)
    if spec.get('reverse', bool(tf.seed_stream(seed, 'reverse') % 2)):
        arr = arr[::-1].copy()

    corridor = stamp_polyline(np.zeros((h, w), dtype=bool), arr,
                              width=spec.get('corridor_width', 2), closed=False)

    # widen selected segments into chambers; rooms are annotations on the curve,
    # never modifications of its identity
    chambers = np.zeros((h, w), dtype=bool)
    every = max(2, spec.get('chamber_every', max(6, len(arr) // 24)))
    r = spec.get('chamber_radius', 2)
    for i in range(0, len(arr), every):
        x, y = int(arr[i][0]), int(arr[i][1])
        chambers[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1] = True

    layers = [(corridor, FLOOR), (chambers & tf._dilate(corridor, r), COURT)]
    tm = build(w, h, layers, default=WALL)
    _seal_rim(tm, w, h)
    connect_components(tm, passable_mask(tm), FLOOR, width=1)
    attach(tm, curve_position=_curve_position_field(w, h, arr))
    tm.curve = arr
    tm.curve_kind = kind
    return tm


def _curve_position_field(w, h, pts):
    """Normalised position along the curve - distance in *curve* space, which is
    what makes space-filling maps feel far apart while looking near."""
    field = np.zeros((h, w), dtype=np.float32)
    for i, (x, y) in enumerate(pts):
        xi, yi = int(x), int(y)
        if 0 <= xi < w and 0 <= yi < h:
            field[yi, xi] = i / max(len(pts) - 1, 1)
    return tf.Field(field, 'curve_position')


def _seal_rim(tm, w, h, tile=WALL):
    for x in range(w):
        tm.base[0][x] = tile
        tm.base[h - 1][x] = tile
    for y in range(h):
        tm.base[y][0] = tile
        tm.base[y][w - 1] = tile


# ------------------------------------------------ 06 Quasicrystal Temples

def quasicrystal_field(w, h, symmetry=5, frequency=9.0, phase=0.0):
    """Sum of plane waves at equal angular spacing.

    N waves at pi/N spacing interfere into a pattern with N-fold rotational
    symmetry and no translational period: alien-feeling without being random.
    """
    nx, ny = tf.coords(w, h)
    total = np.zeros((h, w), dtype=np.float32)
    for k in range(symmetry):
        theta = math.pi * k / symmetry
        total += np.cos((nx * math.cos(theta) + ny * math.sin(theta))
                        * frequency + phase)
    return tf.Field(total / symmetry, f'quasicrystal({symmetry},{frequency})')


def gen_quasicrystal(spec, rng):
    """Penrose-flavoured non-periodic temples: plazas and chambers that never
    repeat, laid out on a rigorously ordered lattice."""
    w, h = dims(spec)
    seed = grammar_seed(spec, 'geo.quasicrystal')
    sym = spec.get('symmetry', 5)
    q = quasicrystal_field(w, h, sym, spec.get('frequency', 22.0),
                           spec.get('phase', (seed % 628) / 100.0)).normalized()

    t = threshold_for_ratio(q, spec.get('open_ratio', 0.42))
    plaza = q.a > t
    # the interference maxima are the natural chamber centres
    chamber = q.a > threshold_for_ratio(q, spec.get('chamber_ratio', 0.10))
    seam = q.band(t - 0.03, t + 0.03)

    layers = [(plaza, FLOOR), (chamber, COURT), (seam & plaza, PLATFORM)]
    tm = build(w, h, layers, default=WALL)
    _seal_rim(tm, w, h)
    connect_components(tm, passable_mask(tm), FLOOR, width=1)
    attach(tm, quasicrystal=q, sacred_intensity=q)
    tm.symmetry_order = sym
    return tm


# ------------------------------------------------- 08 L-System Root Worlds

LSYSTEMS = {
    'root':    ('F', {'F': 'F[+F]F[-F]F'}, 22.0),
    'vascular': ('F', {'F': 'FF+[+F-F-F]-[-F+F+F]'}, 24.0),
    'lightning': ('F', {'F': 'F[+F]F[-F][F]'}, 28.0),
    'bush':    ('F', {'F': 'FF-[-F+F+F]+[+F-F-F]'}, 20.0),
    'weed':    ('X', {'X': 'F+[[X]-X]-F[-FX]+X', 'F': 'FF'}, 25.0),
}


def _lsystem_segments(axiom, rules, iterations, angle, step, start, heading):
    """Walk an L-system recording (a, b, depth) segments so branch order is
    known - depth is what makes trunk, corridor and twig different things."""
    s = axiom
    for _ in range(iterations):
        s = ''.join(rules.get(c, c) for c in s)
        if len(s) > 250000:
            break
    x, y = start
    a = math.radians(heading)
    da = math.radians(angle)
    depth = 0
    stack = []
    segments = []
    for c in s:
        if c in 'FGAB':
            nx = x + math.cos(a) * step
            ny = y + math.sin(a) * step
            segments.append(((x, y), (nx, ny), depth))
            x, y = nx, ny
        elif c == '+':
            a += da
        elif c == '-':
            a -= da
        elif c == '[':
            stack.append((x, y, a, depth))
            depth += 1
        elif c == ']':
            if stack:
                x, y, a, depth = stack.pop()
    return segments


def gen_lsystem_roots(spec, rng):
    """Recursive branching where thickness encodes hierarchy: trunk becomes
    highway, branch becomes corridor, twig becomes secret room."""
    w, h = dims(spec)
    seed = grammar_seed(spec, 'geo.lsystem')
    kind = spec.get('system', 'root')
    axiom, rules, angle = LSYSTEMS.get(kind, LSYSTEMS['root'])
    iters = spec.get('iterations', 4)
    step = spec.get('step', max(2.0, min(w, h) / 14.0))
    segs = _lsystem_segments(axiom, rules, iters, spec.get('angle', angle), step,
                             (w / 2.0, h - 3.0), spec.get('heading', -90.0))
    if not segs:
        return build(w, h, [], default=WALL)

    arr = np.array([[s[0][0], s[0][1]] for s in segs] +
                   [[segs[-1][1][0], segs[-1][1][1]]], dtype=float)
    lo = arr.min(axis=0)
    span_xy = np.maximum(arr.max(axis=0) - lo, 1e-6)
    # Fit each axis separately. A root system is far taller than it is wide, and
    # scaling both axes by the same factor collapses it into a vertical stripe
    # with the whole canvas empty either side. Distortion is capped so the
    # branching still reads as a plant rather than a smear.
    avail = np.array([w - 6.0, h - 6.0])
    per_axis = avail / span_xy
    cap = spec.get('aspect_cap', 3.0)
    lo_s, hi_s = per_axis.min(), per_axis.max()
    if hi_s > lo_s * cap:
        per_axis = np.clip(per_axis, lo_s, lo_s * cap)
    scale = per_axis
    off = np.array([w / 2.0, h / 2.0]) - ((arr.max(axis=0) + lo) / 2.0) * scale

    max_depth = max(s[2] for s in segs)
    trunk = np.zeros((h, w), dtype=bool)
    branch = np.zeros((h, w), dtype=bool)
    twig = np.zeros((h, w), dtype=bool)
    twig_tips = []
    for (a, b, d) in segs:
        pa = np.array(a) * scale + off
        pb = np.array(b) * scale + off
        rel = d / max(max_depth, 1)
        if rel < 0.25:
            stamp_polyline(trunk, [pa, pb], width=int(round(spec.get('trunk_width', 5) * size_scale(w, h))), closed=False)
        elif rel < 0.6:
            stamp_polyline(branch, [pa, pb], width=int(round(spec.get('branch_width', 3) * size_scale(w, h))), closed=False)
        else:
            stamp_polyline(twig, [pa, pb], width=max(1, int(round(1.6 * size_scale(w, h)))), closed=False)
            twig_tips.append(pb)

    secret = np.zeros((h, w), dtype=bool)
    srng = np.random.default_rng(seed)
    for p in twig_tips:
        if srng.random() < spec.get('secret_rate', 0.30):
            x, y = int(p[0]), int(p[1])
            r = spec.get('secret_radius', 2)
            secret[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1] = True

    layers = [(twig, FLOOR), (branch, FLOOR), (trunk, PLATFORM), (secret, COURT)]
    tm = build(w, h, layers, default=WALL)
    _seal_rim(tm, w, h)
    for y, x in zip(*np.nonzero(secret & ~tf._erode(secret, 1))):
        tm.overlay[y][x] = 'hidden'
    connect_components(tm, passable_mask(tm), FLOOR, width=1)
    attach(tm, hierarchy=tf.distance_to_mask(trunk).normalized().invert(),
           age=tf.distance_to_mask(twig).normalized())
    tm.branch_depth = max_depth
    return tm


# ----------------------------------------------------- 11 Tectonic Worlds

def gen_tectonic(spec, rng):
    """Plates with motion vectors. Convergent boundaries raise mountains,
    divergent ones open rifts, transform ones shear into faults."""
    w, h = dims(spec)
    seed = grammar_seed(spec, 'env.tectonic')
    nrng = np.random.default_rng(seed)
    sites = tf.poisson_disc(w, h, spec.get('plate_size', max(10, min(w, h) // 5)),
                            seed=seed, margin=1)
    vor = tf.voronoi(w, h, sites, relax=spec.get('relax', 2))
    nplates = int(vor.cell.max()) + 1
    motion = nrng.normal(0, 1, size=(nplates, 2))
    motion /= np.maximum(np.linalg.norm(motion, axis=1, keepdims=True), 1e-6)

    # classify every shared boundary by the plates' relative motion along the
    # boundary normal: closing, opening, or sliding
    convergent = np.zeros((h, w), dtype=bool)
    divergent = np.zeros((h, w), dtype=bool)
    transform = np.zeros((h, w), dtype=bool)
    stress = np.zeros((h, w), dtype=np.float32)
    for (a, b), shared in vor.adjacency().items():
        seam = tg.border_between(vor.cell, a, b) | tg.border_between(vor.cell, b, a)
        if not seam.any():
            continue
        normal = sites[b] - sites[a]
        nrm = np.linalg.norm(normal)
        if nrm < 1e-6:
            continue
        normal = normal / nrm
        closing = float(np.dot(motion[a] - motion[b], normal))
        shear = abs(float(motion[a][0] * normal[1] - motion[a][1] * normal[0]))
        if closing > 0.35:
            convergent |= seam
            stress[seam] = closing
        elif closing < -0.35:
            divergent |= seam
            stress[seam] = -closing
        elif shear > 0.4:
            transform |= seam
            stress[seam] = shear

    base = tf.fbm(w, h, 3.0, 5, seed=seed + 11).normalized()
    ridge = tf.distance_to_mask(convergent).normalized().invert() ** 2
    trench = tf.distance_to_mask(divergent).normalized().invert() ** 2
    elevation = (base * 0.45 + ridge * 0.55 - trench * 0.40).normalized()

    mountain = tf._dilate(convergent, spec.get('range_width', 2))
    rift = tf._dilate(divergent, spec.get('rift_width', 1))
    fault = transform
    sea = elevation.a < spec.get('sea_level', 0.30)
    volcano = convergent & (tf.fbm(w, h, 6.0, 3, seed=seed + 5).a > 0.72)

    layers = [
        (~sea, FLOOR),
        (elevation.a > 0.62, COURT),
        (fault, RUBBLE),
        (mountain, WALL),
        (rift, ABYSS),
        (sea, WATER),
        (volcano, LAVA),
    ]
    tm = build(w, h, layers, default=WATER)

    # passes through the ranges, or the continents are unreachable
    prng = np.random.default_rng(seed + 3)
    ys, xs = np.nonzero(mountain)
    if len(xs):
        for i in prng.choice(len(xs), size=min(spec.get('passes', 10), len(xs)),
                             replace=False):
            tm.set(int(xs[i]), int(ys[i]), FLOOR, 'door')
    land = passable_mask(tm, (FLOOR, COURT, RUBBLE, BRIDGE, PLATFORM))
    connect_components(tm, land, BRIDGE, width=1, seal_tile=WALL)
    # Rifts are carved as impassable void, so a lake can end up walled off from
    # everything even once the landmasses themselves are joined. Repair the full
    # traversable set too, not just the dry part.
    connect_components(tm, passable_mask(tm), FLOOR, width=1, seal_tile=WALL)
    attach(tm, elevation=elevation, plate=vor.cell,
           tectonic_stress=tf.Field(stress, 'tectonic_stress').normalized(),
           slope=elevation.slope().normalized())
    tm.plates = {'count': nplates, 'motion': motion.tolist()}
    return tm


# ----------------------------------------------- 14 Circuit-Board Megacities

def gen_circuit_city(spec, rng):
    """Orthogonal traces, pads, chips and vias. Buses become transit spines and
    vias become vertical connections between decks."""
    w, h = dims(spec)
    seed = grammar_seed(spec, 'geo.circuit')
    nrng = np.random.default_rng(seed)
    pitch = spec.get('pitch', 6)

    trace = np.zeros((h, w), dtype=bool)
    bus = np.zeros((h, w), dtype=bool)
    pad = np.zeros((h, w), dtype=bool)
    chip = np.zeros((h, w), dtype=bool)

    # power/ground buses on a coarse pitch
    for x in range(pitch * 2, w - 2, pitch * 3):
        bus[2:h - 2, max(0, x - 1):x + 2] = True
    for y in range(pitch * 2, h - 2, pitch * 3):
        bus[max(0, y - 1):y + 2, 2:w - 2] = True

    # chips: rectangular components with pads on their edges
    ncomp = spec.get('components', max(4, (w * h) // 900))
    placed = []
    for _ in range(ncomp * 3):
        if len(placed) >= ncomp:
            break
        cw = int(nrng.integers(4, max(5, pitch * 2)))
        ch = int(nrng.integers(4, max(5, pitch * 2)))
        x = int(nrng.integers(3, max(4, w - cw - 3)))
        y = int(nrng.integers(3, max(4, h - ch - 3)))
        if any(abs(x - px) < pw + 3 and abs(y - py) < ph + 3
               for px, py, pw, ph in placed):
            continue
        placed.append((x, y, cw, ch))
        chip[y:y + ch, x:x + cw] = True
        for i in range(x, x + cw, 2):
            pad[max(0, y - 1), i] = True
            pad[min(h - 1, y + ch), i] = True
        for j in range(y, y + ch, 2):
            pad[j, max(0, x - 1)] = True
            pad[j, min(w - 1, x + cw)] = True

    # route Manhattan traces between pads, snapped to the routing pitch
    pads = list(zip(*np.nonzero(pad)))
    nrng.shuffle(pads)
    vias = []
    for i in range(0, len(pads) - 1, 2):
        (y0, x0), (y1, x1) = pads[i], pads[i + 1]
        mid = int(round(((x0 + x1) // 2) / pitch) * pitch)
        mid = max(1, min(w - 2, mid))
        _hline(trace, y0, x0, mid)
        _vline(trace, mid, y0, y1)
        _hline(trace, y1, mid, x1)
        if abs(x0 - x1) > pitch and abs(y0 - y1) > pitch:
            vias.append((mid, y1))
    trace &= ~chip

    layers = [(trace, FLOOR), (bus & ~chip, PLATFORM), (pad & ~chip, COURT),
              (chip, WALL)]
    tm = build(w, h, layers, default=WALL)
    _seal_rim(tm, w, h)
    for (vx, vy) in vias[:spec.get('max_vias', 40)]:
        if tm.inb(vx, vy):
            tm.set(vx, vy, FLOOR, 'elevator')
    connect_components(tm, passable_mask(tm), FLOOR, width=1)
    attach(tm, signal=tf.distance_to_mask(bus).normalized().invert())
    tm.vias = vias
    return tm


def _hline(mask, y, x0, x1):
    if 0 <= y < mask.shape[0]:
        mask[y, min(x0, x1):max(x0, x1) + 1] = True


def _vline(mask, x, y0, y1):
    if 0 <= x < mask.shape[1]:
        mask[min(y0, y1):max(y0, y1) + 1, x] = True


# ------------------------------------------------- 15 Transit-Network Maps

def gen_transit_network(spec, rng):
    """Designed as a network first, then embedded: lines, stations, interchanges
    and express runs, with stations inflated into playable districts."""
    w, h = dims(spec)
    seed = grammar_seed(spec, 'top.transit')
    nrng = np.random.default_rng(seed)
    nstations = spec.get('stations', max(8, (w * h) // 700))
    sites = tf.poisson_disc(w, h, max(6, min(w, h) // 9), seed=seed, margin=5)
    if len(sites) > nstations:
        sites = sites[nrng.choice(len(sites), nstations, replace=False)]
    stations = [(float(x), float(y)) for x, y in sites]

    nlines = spec.get('lines', max(3, len(stations) // 5))
    lines = []
    for li in range(nlines):
        k = int(nrng.integers(3, max(4, len(stations) // 2)))
        idx = list(nrng.choice(len(stations), size=min(k, len(stations)),
                               replace=False))
        # order the line's stops so it sweeps rather than zig-zags
        origin = np.array(stations[idx[0]])
        idx.sort(key=lambda i: math.atan2(stations[i][1] - origin[1],
                                          stations[i][0] - origin[0]))
        lines.append(idx)

    corridor = np.zeros((h, w), dtype=bool)
    express = np.zeros((h, w), dtype=bool)
    for li, idx in enumerate(lines):
        wide = li < spec.get('express_lines', 1)
        target = express if wide else corridor
        for a, b in zip(idx, idx[1:]):
            ax, ay = stations[a]
            bx, by = stations[b]
            # transit lines run orthogonally, like a schematic
            _hline(target, int(ay), int(ax), int(bx))
            _vline(target, int(bx), int(ay), int(by))
    corridor = tf._dilate(corridor, spec.get('line_width', 1))
    express = tf._dilate(express, spec.get('express_width', 2))

    station_mask = np.zeros((h, w), dtype=bool)
    interchange = np.zeros((h, w), dtype=bool)
    use = [0] * len(stations)
    for idx in lines:
        for i in idx:
            use[i] += 1
    for i, (x, y) in enumerate(stations):
        r = spec.get('station_radius', 3) + (2 if use[i] > 1 else 0)
        xi, yi = int(x), int(y)
        (interchange if use[i] > 1 else station_mask)[
            max(0, yi - r):yi + r + 1, max(0, xi - r):xi + r + 1] = True

    layers = [(corridor, FLOOR), (express, PLATFORM),
              (station_mask, COURT), (interchange, COURT)]
    tm = build(w, h, layers, default=WALL)
    _seal_rim(tm, w, h)
    for i, (x, y) in enumerate(stations):
        if tm.inb(int(x), int(y)):
            tm.set(int(x), int(y), COURT, 'district' if use[i] > 1 else 'door')
    connect_components(tm, passable_mask(tm), FLOOR, width=1)
    attach(tm, transit_access=tf.distance_to_mask(
        station_mask | interchange).normalized().invert())
    tm.transit = {'stations': stations, 'lines': lines, 'usage': use}
    return tm


# ---------------------------------------------- 16 Non-Euclidean Portal Maps

def gen_portal_maps(spec, rng):
    """A separate topology graph: walking east out of one room emerges north of
    an unrelated one. Portal pairs are marked with matching glyphs."""
    w, h = dims(spec)
    seed = grammar_seed(spec, 'top.portal')
    nrng = np.random.default_rng(seed)
    ncells = spec.get('chambers', max(6, (w * h) // 700))

    sites = tf.poisson_disc(w, h, max(8, min(w, h) // 6), seed=seed, margin=6)
    if len(sites) > ncells:
        sites = sites[nrng.choice(len(sites), ncells, replace=False)]

    rooms = []
    room_mask = np.zeros((h, w), dtype=bool)
    for (x, y) in sites:
        rw = int(nrng.integers(5, max(6, min(w, h) // 7)))
        rh = int(nrng.integers(5, max(6, min(w, h) // 7)))
        x0 = int(np.clip(x - rw // 2, 1, w - rw - 2))
        y0 = int(np.clip(y - rh // 2, 1, h - rh - 2))
        rooms.append((x0, y0, rw, rh))
        room_mask[y0:y0 + rh, x0:x0 + rw] = True

    tm = build(w, h, [(room_mask, FLOOR)], default=WALL)
    _seal_rim(tm, w, h)

    # rooms are deliberately NOT joined in the plane; the portal graph is the
    # only circulation, so the walkable topology exceeds the visible footprint
    portals = []
    order = list(range(len(rooms)))
    nrng.shuffle(order)
    glyphs = ['teleport', 'rotate', 'shift', 'elevator']
    for i in range(len(order) - 1):
        a, b = order[i], order[i + 1]
        pa = _room_edge_point(rooms[a], nrng)
        pb = _room_edge_point(rooms[b], nrng)
        g = glyphs[i % len(glyphs)]
        tm.set(pa[0], pa[1], FLOOR, g)
        tm.set(pb[0], pb[1], FLOOR, g)
        portals.append({'a': list(pa), 'b': list(pb), 'glyph': g,
                        'rooms': [a, b]})
    if spec.get('loop', True) and len(order) > 2:
        a, b = order[-1], order[0]
        pa = _room_edge_point(rooms[a], nrng)
        pb = _room_edge_point(rooms[b], nrng)
        tm.set(pa[0], pa[1], FLOOR, 'teleport')
        tm.set(pb[0], pb[1], FLOOR, 'teleport')
        portals.append({'a': list(pa), 'b': list(pb), 'glyph': 'teleport',
                        'rooms': [a, b]})

    # the plane itself must still be walkable for tools that ignore portals
    if spec.get('physical_fallback', True):
        connect_components(tm, passable_mask(tm), FLOOR, width=1)
    attach(tm, room=_room_id_field(w, h, rooms))
    tm.portals = portals
    tm.rooms = rooms
    return tm


def _room_edge_point(room, nrng):
    x0, y0, rw, rh = room
    side = int(nrng.integers(4))
    if side == 0:
        return (int(nrng.integers(x0, x0 + rw)), y0)
    if side == 1:
        return (int(nrng.integers(x0, x0 + rw)), y0 + rh - 1)
    if side == 2:
        return (x0, int(nrng.integers(y0, y0 + rh)))
    return (x0 + rw - 1, int(nrng.integers(y0, y0 + rh)))


def _room_id_field(w, h, rooms):
    f = np.full((h, w), -1.0, dtype=np.float32)
    for i, (x0, y0, rw, rh) in enumerate(rooms):
        f[y0:y0 + rh, x0:x0 + rw] = i
    return tf.Field(f, 'room_id')


# ----------------------------------------------------------- 18 Mandalas

def gen_mandala(spec, rng):
    """Radial subdivision, rotational symmetry groups, star polygons and nested
    rosettes - with selected symmetries deliberately broken to give progression."""
    w, h = dims(spec)
    seed = grammar_seed(spec, 'geo.mandala')
    nrng = np.random.default_rng(seed)
    order = spec.get('order', 8)
    rings = spec.get('rings', 5)

    r = tf.radial(w, h).normalized()
    theta = tf.angular(w, h)
    cx, cy = w / 2.0, h / 2.0

    petal = np.cos(theta.a * 2 * math.pi * order) * 0.5 + 0.5
    star = np.abs(np.cos(theta.a * math.pi * order)) ** spec.get('sharpness', 0.6)
    ringf = np.cos(r.a * math.pi * rings * 2) * 0.5 + 0.5

    composite = tf.Field(petal * 0.45 + ringf * 0.35 + star * 0.20,
                         'mandala').normalized()
    open_cells = (composite.a > threshold_for_ratio(
        composite, spec.get('open_ratio', 0.44))) & (r.a < 0.98)
    sanctum = r.a < spec.get('sanctum', 0.12)
    ambulatory = composite.band(0.46, 0.56) & (r.a < 0.95)

    # a deliberate break in the symmetry group: one wedge is profaned
    if spec.get('break_symmetry', True):
        wedge = int(nrng.integers(order))
        lo = wedge / order
        hi = (wedge + 1) / order
        broken = ((theta.a >= lo) & (theta.a < hi) & (r.a > 0.35) & (r.a < 0.95))
        # perturb the wedge rather than replacing it: the break should read as
        # damage to the symmetry group, not as a missing slice of the building
        open_cells = np.where(broken,
                              tf.fbm(w, h, 7.0, 3, seed=seed + 7).a > 0.52,
                              open_cells)
    else:
        broken = np.zeros((h, w), dtype=bool)

    layers = [(open_cells, FLOOR), (ambulatory, COURT), (sanctum, PLATFORM),
              (broken & open_cells, RUBBLE)]
    tm = build(w, h, layers, default=WALL)
    _seal_rim(tm, w, h)

    # processional gates on each axis of the symmetry group
    for k in range(order):
        a = 2 * math.pi * k / order
        for frac in (0.35, 0.62, 0.86):
            x = int(cx + math.cos(a) * frac * min(w, h) / 2)
            y = int(cy + math.sin(a) * frac * min(w, h) / 2)
            if tm.inb(x, y):
                tm.set(x, y, FLOOR, 'door')
    connect_components(tm, passable_mask(tm), FLOOR, width=1)
    attach(tm, sacred_intensity=r.invert(), mandala=composite,
           symmetry_phase=theta)
    tm.symmetry = {'order': order, 'rings': rings,
                   'broken_wedge': int(wedge) if spec.get('break_symmetry', True) else None}
    return tm


# ------------------------------------------- 19 Semantic Silhouette Maps

def _sdf_circle(nx, ny, cx, cy, r):
    return np.hypot(nx - cx, ny - cy) - r


def _sdf_capsule(nx, ny, ax, ay, bx, by, r):
    pax, pay = nx - ax, ny - ay
    bax, bay = bx - ax, by - ay
    denom = bax * bax + bay * bay
    t = np.clip((pax * bax + pay * bay) / max(denom, 1e-9), 0.0, 1.0)
    return np.hypot(pax - bax * t, pay - bay * t) - r


def _sdf_box(nx, ny, cx, cy, hw, hh):
    dx = np.abs(nx - cx) - hw
    dy = np.abs(ny - cy) - hh
    return (np.minimum(np.maximum(dx, dy), 0.0) +
            np.hypot(np.maximum(dx, 0.0), np.maximum(dy, 0.0)))


def _smin(a, b, k=0.12):
    hh = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1 - hh) + a * hh - k * hh * (1 - hh)


def silhouette_sdf(name, w, h, seed=0):
    """Signed-distance silhouettes assembled from primitives."""
    nx, ny = tf.coords(w, h)
    if name == 'serpent':
        d = np.full(nx.shape, 1e9, dtype=np.float32)
        prev = (-0.8, 0.0)
        for i in range(1, 15):
            t = i / 14.0
            p = (-0.8 + 1.6 * t, 0.46 * math.sin(t * math.pi * 3.4))
            d = _smin(d, _sdf_capsule(nx, ny, prev[0], prev[1], p[0], p[1],
                                      0.16 - 0.09 * t), 0.10)
            prev = p
        d = _smin(d, _sdf_circle(nx, ny, -0.80, 0.0, 0.20), 0.06)
    elif name == 'crown':
        d = _sdf_box(nx, ny, 0.0, 0.42, 0.72, 0.20)
        for k in range(-2, 3):
            d = _smin(d, _sdf_capsule(nx, ny, k * 0.34, 0.24, k * 0.34, -0.52,
                                      0.11), 0.08)
        d = _smin(d, _sdf_box(nx, ny, 0.0, 0.60, 0.80, 0.09), 0.05)
    elif name == 'skull':
        d = _sdf_circle(nx, ny, 0.0, -0.12, 0.62)
        d = _smin(d, _sdf_box(nx, ny, 0.0, 0.52, 0.34, 0.26), 0.12)
        eye_l = _sdf_circle(nx, ny, -0.24, -0.22, 0.17)
        eye_r = _sdf_circle(nx, ny, 0.24, -0.22, 0.17)
        nose = _sdf_box(nx, ny, 0.0, 0.14, 0.07, 0.12)
        d = np.maximum(d, -np.minimum(np.minimum(eye_l, eye_r), nose))
    elif name == 'sigil':
        d = np.abs(_sdf_circle(nx, ny, 0.0, 0.0, 0.66)) - 0.09
        for k in range(5):
            a0 = 2 * math.pi * k / 5 - math.pi / 2
            a1 = 2 * math.pi * ((k * 2) % 5) / 5 - math.pi / 2
            d = np.minimum(d, _sdf_capsule(
                nx, ny, math.cos(a0) * 0.62, math.sin(a0) * 0.62,
                math.cos(a1) * 0.62, math.sin(a1) * 0.62, 0.055))
    elif name == 'machine':
        d = _sdf_box(nx, ny, 0.0, 0.0, 0.52, 0.36)
        d = _smin(d, _sdf_box(nx, ny, 0.0, 0.0, 0.20, 0.68), 0.10)
        for k in (-1, 1):
            d = _smin(d, _sdf_circle(nx, ny, k * 0.62, 0.0, 0.22), 0.08)
            d = _smin(d, _sdf_box(nx, ny, k * 0.40, -0.52, 0.10, 0.22), 0.06)
    else:  # 'hand'
        d = _sdf_box(nx, ny, 0.0, 0.28, 0.36, 0.30)
        for k in range(-2, 3):
            d = _smin(d, _sdf_capsule(nx, ny, k * 0.17, 0.10,
                                      k * 0.17, -0.52 + abs(k) * 0.12, 0.075), 0.07)
        d = _smin(d, _sdf_capsule(nx, ny, -0.34, 0.34, -0.62, 0.02, 0.09), 0.07)
    return tf.Field(d.astype(np.float32), f'silhouette({name})')


SILHOUETTES = ('serpent', 'crown', 'skull', 'sigil', 'machine', 'hand')


def gen_silhouette(spec, rng):
    """A whole level that secretly depicts a symbol, creature or object, with
    its interior recursively populated by rooms."""
    w, h = dims(spec)
    seed = grammar_seed(spec, 'geo.silhouette')
    shape = spec.get('shape', 'serpent')
    sdf = silhouette_sdf(shape, w, h, seed)
    # grow the silhouette until it fills a decent share of the frame, so the
    # depicted shape reads at a glance instead of floating in a sea of wall
    inside = sdf.a < spec.get('dilate', 0.06)
    if not inside.any():
        inside = sdf.a < float(np.quantile(sdf.a, 0.35))

    interior = tf._erode(inside, spec.get('wall_thickness', 1))
    rim = inside & ~interior
    depth = tf.distance_to_mask(~inside).normalized()

    # recursively subdivide the interior into rooms
    rooms = _bsp_rooms(interior, seed, spec.get('min_room', 4),
                       spec.get('splits', 4))
    room_mask = np.zeros((h, w), dtype=bool)
    corridor = np.zeros((h, w), dtype=bool)
    centres = []
    for (x0, y0, rw, rh) in rooms:
        sub = np.zeros((h, w), dtype=bool)
        sub[y0 + 1:y0 + rh - 1, x0 + 1:x0 + rw - 1] = True
        sub &= interior
        if sub.sum() < 4:
            continue
        room_mask |= sub
        ys, xs = np.nonzero(sub)
        centres.append((int(xs.mean()), int(ys.mean())))
    for a, b in zip(centres, centres[1:]):
        _hline(corridor, a[1], a[0], b[0])
        _vline(corridor, b[0], a[1], b[1])
    corridor &= interior

    deep = interior & (depth.a > spec.get('sanctum_depth', 0.72))
    layers = [(room_mask, FLOOR), (corridor, FLOOR), (deep, COURT), (rim, WALL)]
    tm = build(w, h, layers, default=WALL)
    _seal_rim(tm, w, h)
    connect_components(tm, passable_mask(tm), FLOOR, width=1)
    attach(tm, silhouette=tf.Field(sdf.a, 'silhouette_sdf').normalized(),
           depth=depth)
    tm.silhouette = shape
    tm.rooms = rooms
    return tm


def _bsp_rooms(mask, seed, min_size, splits):
    """Binary space partition of the mask's bounding box."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return []
    nrng = np.random.default_rng(seed)
    regions = [(int(xs.min()), int(ys.min()),
                int(xs.max() - xs.min()) + 1, int(ys.max() - ys.min()) + 1)]
    for _ in range(splits):
        nxt = []
        for (x, y, rw, rh) in regions:
            if rw < min_size * 2 and rh < min_size * 2:
                nxt.append((x, y, rw, rh))
                continue
            if rw >= rh:
                cut = int(nrng.integers(min_size, max(min_size + 1, rw - min_size)))
                nxt.append((x, y, cut, rh))
                nxt.append((x + cut, y, rw - cut, rh))
            else:
                cut = int(nrng.integers(min_size, max(min_size + 1, rh - min_size)))
                nxt.append((x, y, rw, cut))
                nxt.append((x, y + cut, rw, rh - cut))
        regions = nxt
    return regions


GRAMMARS2 = {
    'space_filling':   gen_space_filling,
    'quasicrystal':    gen_quasicrystal,
    'lsystem_roots':   gen_lsystem_roots,
    'tectonic':        gen_tectonic,
    'circuit_city':    gen_circuit_city,
    'transit_network': gen_transit_network,
    'portal_maps':     gen_portal_maps,
    'mandala':         gen_mandala,
    'silhouette':      gen_silhouette,
}

tg.GRAMMARS.update(GRAMMARS2)
