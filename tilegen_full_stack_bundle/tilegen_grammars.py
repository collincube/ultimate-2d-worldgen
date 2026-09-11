"""Field-first map grammars.

Each grammar synthesises continuous fields or vector geometry first and only
then interprets that into tiles, so the same source can drive a raster TileMap
and a high-resolution render. Solvers never mention tile names; interpretation
is a separate, replaceable step.

Every grammar is registered with the signature the existing system already
uses -- ``(spec, rng) -> TileMap`` -- so make_layout, the JSON export, the socket
WFC assembler and the navigation profiles keep working untouched.
"""
from __future__ import annotations

import math

import numpy as np

import tilegen_fields as tf
from tilegen_system import TileMap

# Tile vocabulary understood by the existing renderer and runtime.
# Passable: floor platform bridge courtyard ladder_tile water rubble
# Blocking: wall bookshelf          Hazard/void: lava abyss
WALL, FLOOR, WATER, LAVA, ABYSS = 'wall', 'floor', 'water', 'lava', 'abyss'
COURT, RUBBLE, BRIDGE, PLATFORM = 'courtyard', 'rubble', 'bridge', 'platform'


# ------------------------------------------------------------------- plumbing

REFERENCE_SIZE = 96


def size_scale(w, h, ref=REFERENCE_SIZE):
    """How much larger this map is than the reference the defaults were tuned at.

    Grammars whose look comes from a characteristic feature size (fold spacing,
    stream width) must scale that feature with the canvas, or a big map is just
    the same small pattern marooned in empty space.
    """
    return max(1.0, min(w, h) / float(ref))


def dims(spec):
    size = spec.get('size', 64)
    return int(spec.get('width', size)), int(spec.get('height', size))


def grammar_seed(spec, name, index=0):
    return tf.seed_stream(spec.get('seed', spec.get('id', 0)), name, index)


def threshold_for_ratio(field, ratio):
    """Cutoff that leaves `ratio` of cells above it.

    Composite fields (a product of two normalised fields, say) pile up near
    zero, so a fixed threshold can yield a 5%-open map. Asking for an open
    ratio instead makes every parameter combination produce a usable map, and
    directly implements the atlas open-ratio quality gate.
    """
    a = field.a if isinstance(field, tf.Field) else field
    return float(np.quantile(a, 1.0 - float(np.clip(ratio, 0.01, 0.99))))


def build(w, h, layers, default=WALL, overlays=()):
    """Compose masks into a TileMap. Later layers overwrite earlier ones."""
    names = [default]
    idx = np.zeros((h, w), dtype=np.int16)
    for mask, tile in layers:
        if mask is None or not mask.any():
            continue
        if tile not in names:
            names.append(tile)
        idx[mask] = names.index(tile)

    tm = TileMap(w, h, default)
    tm.base = [[names[i] for i in row] for row in idx.tolist()]
    for mask, name in overlays:
        if mask is None:
            continue
        ys, xs = np.nonzero(mask)
        for y, x in zip(ys.tolist(), xs.tolist()):
            tm.overlay[y][x] = name
    return tm


def attach(tm, **fields):
    """Hang continuous fields off the map (atlas 6.3).

    These drive rendering and later simulation without inventing new tile enums.
    Values are numpy arrays; the runtime ignores anything it does not know.
    """
    store = getattr(tm, 'fields', None)
    if store is None:
        store = {}
        tm.fields = store
    for k, v in fields.items():
        store[k] = v.a if isinstance(v, tf.Field) else v
    return tm


def passable_mask(tm, passable=(FLOOR, COURT, WATER, RUBBLE, BRIDGE, PLATFORM, 'ladder_tile')):
    arr = np.array(tm.base, dtype=object)
    out = np.zeros(arr.shape, dtype=bool)
    for name in passable:
        out |= (arr == name)
    return out


def carve_l(tm, x0, y0, x1, y1, tile=FLOOR, width=1, overlay=None, flip=False):
    """Carve an axis-aligned L between two points.

    Deliberately not a straight line: a diagonal run is 8-connected but the
    component labeller and the runtime's N/E/S/W navigation are both
    4-connected, so a diagonal corridor would not actually join anything.
    """
    if flip:
        tm.line(x0, y0, x0, y1, width, tile, overlay)
        tm.line(x0, y1, x1, y1, width, tile, overlay)
    else:
        tm.line(x0, y0, x1, y0, width, tile, overlay)
        tm.line(x1, y0, x1, y1, width, tile, overlay)


def _l_mask(h, w, x0, y0, x1, y1, width=1, flip=False):
    m = np.zeros((h, w), dtype=bool)
    r = max(0, width // 2)

    def run(ax, ay, bx, by):
        lo_x, hi_x = sorted((ax, bx)); lo_y, hi_y = sorted((ay, by))
        m[max(0, lo_y - r):hi_y + r + 1, max(0, lo_x - r):hi_x + r + 1] = True

    if flip:
        run(x0, y0, x0, y1); run(x0, y1, x1, y1)
    else:
        run(x0, y0, x1, y0); run(x1, y0, x1, y1)
    return m


def connect_components(tm, open_mask, tile=FLOOR, width=1, overlay=None, passes=2,
                       min_component=4, seal_tile=WALL):
    """Carve minimal corridors so every open region reaches the largest one.

    Thin L-shaped corridors preserve the generator's pattern while joining
    regions. Pockets smaller than `min_component` are sealed.
    """
    for _ in range(max(1, passes)):
        labels, n = tf.connected_components(open_mask)
        if n <= 1:
            return open_mask
        counts = np.bincount(labels.ravel())
        counts[0] = 0
        main = int(counts.argmax())

        if min_component > 1:
            tiny = np.isin(labels, np.flatnonzero((counts > 0) &
                                                  (counts < min_component)))
            tiny &= labels != main
            if tiny.any():
                for y, x in zip(*np.nonzero(tiny)):
                    tm.base[y][x] = seal_tile
                    tm.overlay[y][x] = ''
                open_mask = open_mask & ~tiny
                labels, n = tf.connected_components(open_mask)
                if n <= 1:
                    return open_mask
                counts = np.bincount(labels.ravel())
                counts[0] = 0
                main = int(counts.argmax())
        main_mask = labels == main
        sy, sx = tf.nearest_seed(main_mask)
        h, w = open_mask.shape
        yy, xx = np.mgrid[0:h, 0:w]
        dist = np.hypot(xx - sx, yy - sy)

        for lab in range(1, n + 1):
            if lab == main:
                continue
            comp = labels == lab
            if not comp.any():
                continue
            d = np.where(comp, dist, np.inf)
            i = int(np.argmin(d))
            cy, cx = divmod(i, w)
            ty, tx = int(sy[cy, cx]), int(sx[cy, cx])
            if tx < 0:
                continue
            flip = ((cx + cy) % 2) == 0
            carve_l(tm, cx, cy, tx, ty, tile, width, overlay, flip)
            open_mask = open_mask | _l_mask(h, w, cx, cy, tx, ty, width, flip)
    return open_mask


def _line_mask(h, w, x0, y0, x1, y1, width=1):
    steps = max(abs(x1 - x0), abs(y1 - y0), 1) * 2
    m = np.zeros((h, w), dtype=bool)
    r = max(0, width // 2)
    for i in range(steps + 1):
        t = i / steps
        x = int(round(x0 + (x1 - x0) * t)); y = int(round(y0 + (y1 - y0) * t))
        m[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1] = True
    return m


def stamp_polyline(mask, pts, width=2, closed=True):
    """Rasterise a polyline into a boolean mask with a round brush."""
    h, w = mask.shape
    r = max(0, int(width) // 2)
    pts = np.asarray(pts, dtype=float)
    seq = np.vstack([pts, pts[:1]]) if closed and len(pts) > 2 else pts
    for (x0, y0), (x1, y1) in zip(seq, seq[1:]):
        n = max(2, int(math.hypot(x1 - x0, y1 - y0) * 2) + 1)
        for t in np.linspace(0.0, 1.0, n):
            x = int(round(x0 + (x1 - x0) * t)); y = int(round(y0 + (y1 - y0) * t))
            if r == 0:
                if 0 <= x < w and 0 <= y < h:
                    mask[y, x] = True
            else:
                mask[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1] = True
    return mask


def _mst_edges(nodes, weighted_edges):
    """Kruskal, plus the caller can add extra edges back for loops."""
    parent = {n: n for n in nodes}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    keep = []
    for wgt, a, b in sorted(weighted_edges):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
            keep.append((a, b))
    return keep


def border_between(cell, a, b):
    """Cells of region `a` that touch region `b`."""
    m = np.zeros(cell.shape, dtype=bool)
    ca, cb = cell == a, cell == b
    up, dn, lf, rt = tf._shifts(cb)
    return ca & (up | dn | lf | rt) if m is not None else m


# --------------------------------------------------------- 01 Voronoi Kingdoms

def gen_voronoi_kingdoms(spec, rng):
    """Territorial rather than rectilinear: crystalline provinces whose shared
    borders become walls, and whose selected shared edges become gates."""
    w, h = dims(spec)
    seed = grammar_seed(spec, 'geo.voronoi')
    district = spec.get('district_size', max(6, min(w, h) // 7))

    if spec.get('site_strategy', 'poisson') == 'phyllotaxis':
        sites = tf.phyllotaxis(w, h, spec.get('sites', 90))
    else:
        sites = tf.poisson_disc(w, h, district, seed=seed, margin=2)
    vor = tf.voronoi(w, h, sites, relax=spec.get('relax', 2))

    wall_th = spec.get('wall_thickness', 1)
    borders = vor.border_mask(wall_th)
    rim = tf.distance_to_border(w, h).a < spec.get('rim', 1)

    ncells = int(vor.cell.max()) + 1
    crng = np.random.default_rng(seed + 17)
    roles = crng.random(ncells)
    interior = ~borders & ~rim

    lake = np.zeros((h, w), dtype=bool)
    plaza = np.zeros((h, w), dtype=bool)
    ruin = np.zeros((h, w), dtype=bool)
    for c in range(ncells):
        cm = (vor.cell == c) & interior
        if roles[c] < spec.get('lake_rate', 0.10):
            lake |= cm
        elif roles[c] < spec.get('lake_rate', 0.10) + spec.get('plaza_rate', 0.22):
            plaza |= cm
        elif roles[c] > 1.0 - spec.get('ruin_rate', 0.12):
            ruin |= cm

    layers = [(interior, FLOOR), (plaza, COURT), (ruin, RUBBLE), (lake, WATER)]
    tm = build(w, h, layers, default=WALL)

    # gates: keep the kingdom connected through its widest shared borders
    adj = vor.adjacency()
    edges = [(-shared, a, b) for (a, b), shared in adj.items()
             if shared >= spec.get('min_gate_border', 3)]
    keep = _mst_edges(range(ncells), edges)
    extra = spec.get('extra_gates', 0.25)
    if extra and edges:
        pool = [(a, b) for _, a, b in edges if (a, b) not in keep]
        crng.shuffle(pool)
        keep += pool[:int(len(pool) * extra)]

    gate_mask = np.zeros((h, w), dtype=bool)
    gw = spec.get('gate_width', 2)
    for a, b in keep:
        seam = border_between(vor.cell, a, b) & borders
        if not seam.any():
            continue
        ys, xs = np.nonzero(seam)
        # pick the seam cell furthest from the seam's own ends: its middle
        mid = len(xs) // 2
        order = np.argsort(xs * h + ys)
        gx, gy = int(xs[order[mid]]), int(ys[order[mid]])
        y0, y1 = max(0, gy - gw // 2), gy + gw // 2 + 1
        x0, x1 = max(0, gx - gw // 2), gx + gw // 2 + 1
        gate_mask[y0:y1, x0:x1] = True
    gate_mask &= ~rim
    for y, x in zip(*np.nonzero(gate_mask)):
        tm.base[y][x] = FLOOR

    door = gate_mask & ~tf._erode(gate_mask, 1)
    for y, x in zip(*np.nonzero(door)):
        tm.overlay[y][x] = 'door'

    open_mask = passable_mask(tm)
    connect_components(tm, open_mask, FLOOR, width=1)
    attach(tm, region=vor.cell, border_tension=vor.border_tension.normalized(),
           elevation=tf.fbm(w, h, 3.0, 4, seed=seed + 5).normalized())
    tm.regions = {'count': ncells, 'adjacency': adj, 'gates': keep}
    return tm


# --------------------------------------------------- 02 Reaction-Diffusion

def gen_reaction_diffusion(spec, rng):
    """Gray-Scott chemistry read as architecture: coral, fingerprints, mould
    blooms, chemical gardens. Preset choice changes the map family, not the
    surface detail."""
    w, h = dims(spec)
    seed = grammar_seed(spec, 'geo.reaction_diffusion')
    preset = spec.get('preset', 'coral')
    v = tf.reaction_diffusion(w, h, preset,
                              steps=spec.get('steps', 2600), seed=seed,
                              feed=spec.get('feed'), kill=spec.get('kill'))
    vn = v.normalized()
    profile = spec.get('profile', 'labyrinth')

    if profile == 'garden':
        # concentration reads as moisture and planting density
        floor = vn.a < 0.72
        garden = vn.band(0.30, 0.72)
        pond = vn.a >= 0.86
        layers = [(floor, FLOOR), (garden, COURT), (pond, WATER)]
        tm = build(w, h, layers, default=WALL)
    elif profile == 'islands':
        land = vn.a > spec.get('threshold', 0.42)
        shore = tf._dilate(land, 1) & ~land
        layers = [(land, FLOOR), (shore, RUBBLE)]
        tm = build(w, h, layers, default=WATER)
    else:
        t = (spec['threshold'] if 'threshold' in spec
             else threshold_for_ratio(vn, spec.get('open_ratio', 0.46)))
        open_cells = vn.a > t
        if spec.get('invert', False):
            open_cells = ~open_cells
        detail = vn.band(t, t + 0.08)
        layers = [(open_cells, FLOOR), (detail & open_cells, COURT)]
        tm = build(w, h, layers, default=WALL)

    rim = tf.distance_to_border(w, h).a < 1
    for y, x in zip(*np.nonzero(rim)):
        tm.base[y][x] = WALL

    open_mask = passable_mask(tm)
    if open_mask.any():
        connect_components(tm, open_mask, FLOOR, width=1)
    attach(tm, chemistry=vn, moisture=vn, corruption=vn)
    return tm


# ------------------------------------------------ 03 Differential Growth

def gen_differential_growth(spec, rng):
    """A loop grown under its own pressure until it folds: brains, intestines,
    coral rims, coiled ducts."""
    w, h = dims(spec)
    seed = grammar_seed(spec, 'geo.differential_growth')
    # Covered fraction is (nodes * rest * width) / area. With feature scale `a`
    # and node count `k` relative to the reference size, holding coverage
    # constant needs k * a^2 == s^2. Splitting the difference at a = sqrt(s),
    # k = s gives bigger maps both larger folds and more of them, instead of
    # either a pure zoom or a quadratic node budget.
    s = size_scale(w, h)
    a = math.sqrt(s)
    # The loop expands outward from its seed circle, so on a large canvas it is
    # expansion-limited rather than node-limited: it runs out of steps before it
    # reaches the edges. Starting it proportionally larger buys that coverage
    # for free instead of paying for more steps.
    start_r = spec.get('start_radius', min(w, h) * min(0.30, 0.10 + 0.065 * (s - 1)))
    pts = tf.differential_growth(
        w, h,
        steps=spec.get('steps', 240), seed=seed,
        start_radius=start_r,
        max_nodes=spec.get('max_nodes', int(min(6000, 1000 * s))),
        repel_radius=spec.get('repel_radius', 5.0) * a,
        rest=spec.get('rest', 1.7) * a)

    corridor_w = max(1, int(round(spec.get('corridor_width', 2) * a)))
    curve = stamp_polyline(np.zeros((h, w), dtype=bool), pts, width=corridor_w)
    if spec.get('invert', False):
        # the fold itself is the wall, the surrounding space is playable
        layers = [(~curve, FLOOR)]
        tm = build(w, h, layers, default=WALL)
    else:
        wide = stamp_polyline(np.zeros((h, w), dtype=bool), pts, width=corridor_w + 2)
        chambers = np.zeros((h, w), dtype=bool)
        step = max(1, len(pts) // max(1, spec.get('chambers', 10)))
        for i in range(0, len(pts), step):
            x, y = int(pts[i][0]), int(pts[i][1])
            r = max(1, int(round(spec.get('chamber_radius', 3) * a)))
            chambers[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1] = True
        layers = [(curve, FLOOR), (chambers & wide, COURT)]
        tm = build(w, h, layers, default=WALL)

    rim = tf.distance_to_border(w, h).a < 1
    for y, x in zip(*np.nonzero(rim)):
        tm.base[y][x] = WALL
    connect_components(tm, passable_mask(tm), FLOOR, width=1)
    attach(tm, age=tf.distance_to_mask(curve).normalized().invert(),
           elevation=tf.fbm(w, h, 4.0, 4, seed=seed + 3).normalized())
    tm.curve = pts
    return tm


# --------------------------------------------------------- 04 Truchet Knots

def _arc_points(x0, y0, cell, corner, samples=None):
    """Quarter arc inside a cell, centred on one of its corners."""
    r = cell / 2.0
    samples = samples or max(6, int(cell))
    cx, cy = {
        'NW': (x0, y0), 'NE': (x0 + cell, y0),
        'SE': (x0 + cell, y0 + cell), 'SW': (x0, y0 + cell),
    }[corner]
    a0 = {'NW': 0.0, 'NE': math.pi / 2, 'SE': math.pi, 'SW': -math.pi / 2}[corner]
    t = np.linspace(a0, a0 + math.pi / 2, samples)
    return np.stack([cx + np.cos(t) * r, cy + np.sin(t) * r], axis=1)


def gen_truchet(spec, rng):
    """Quarter-arc motifs whose rotations are optimised for global loop
    structure, so the map is one continuous circuit rather than random pipes."""
    w, h = dims(spec)
    seed = grammar_seed(spec, 'geo.truchet')
    cell = spec.get('cell', 16)
    gw, gh = max(2, w // cell), max(2, h // cell)
    rot = tf.truchet(gw, gh, seed=seed,
                     optimize=spec.get('optimize', 240),
                     target=spec.get('target', 'long_loops'))
    loops = tf.truchet_loops(rot)

    corridor = np.zeros((h, w), dtype=bool)
    for cy in range(gh):
        for cx in range(gw):
            x0, y0 = cx * cell, cy * cell
            corners = ('NW', 'SE') if rot[cy, cx] == 0 else ('NE', 'SW')
            for corner in corners:
                stamp_polyline(corridor, _arc_points(x0, y0, cell, corner),
                               width=spec.get('corridor_width', 3), closed=False)

    junction = np.zeros((h, w), dtype=bool)
    if spec.get('chambers', True):
        jr = spec.get('chamber_radius', 2)
        for cy in range(0, gh, 2):
            for cx in range(0, gw, 2):
                x, y = cx * cell + cell // 2, cy * cell + cell // 2
                if corridor[min(y, h - 1), min(x, w - 1)]:
                    junction[max(0, y - jr):y + jr + 1, max(0, x - jr):x + jr + 1] = True
    junction &= corridor

    layers = [(corridor, FLOOR), (junction, COURT)]
    tm = build(w, h, layers, default=WALL)
    rim = tf.distance_to_border(w, h).a < 1
    for y, x in zip(*np.nonzero(rim)):
        tm.base[y][x] = WALL
    connect_components(tm, passable_mask(tm), FLOOR, width=1)
    attach(tm, elevation=tf.fbm(w, h, 5.0, 3, seed=seed + 9).normalized())
    tm.loops = {'count': len(loops),
                'largest': max((len(v) for v in loops.values()), default=0)}
    return tm


# ------------------------------------------------- 10 Watershed / River Delta

def gen_watershed(spec, rng):
    """Real geography instead of dungeon geometry: terrain, drainage, rivers,
    deltas and the settlements that follow water."""
    w, h = dims(spec)
    seed = grammar_seed(spec, 'env.watershed')
    hyd = tf.watershed(w, h, seed=seed,
                       scale=spec.get('scale', 3.0),
                       sea_level=spec.get('sea_level', 0.38),
                       river_quantile=spec.get('river_quantile', 0.97))
    e = hyd['elevation']
    moisture = tf.fbm(w, h, 4.0, 4, seed=seed + 71).normalized()

    sea = hyd['sea']
    river = tf._dilate(hyd['river'], spec.get('river_width', 1) - 1) & ~sea
    slope = hyd['slope']
    high = (e.a > spec.get('mountain_level', 0.78)) & ~sea
    cliff = (slope.a > 0.55) & ~sea & ~high
    wet = (moisture.a > 0.58) & ~sea & ~high
    beach = tf._dilate(sea, 1) & ~sea

    layers = [
        (~sea, FLOOR),
        (wet, COURT),
        (cliff, RUBBLE),
        (beach, RUBBLE),
        (high, WALL),
        (river, WATER),
        (sea, WATER),
    ]
    tm = build(w, h, layers, default=WATER)

    # fords keep the two banks reachable
    if spec.get('fords', True):
        frng = np.random.default_rng(seed + 3)
        ys, xs = np.nonzero(river)
        if len(xs):
            for i in frng.choice(len(xs), size=min(4, len(xs)), replace=False):
                x, y = int(xs[i]), int(ys[i])
                tm.set(x, y, BRIDGE, 'door')
    # Link separated landmasses with visible causeways. Carving plain floor
    # here drags straight dry corridors across open lakes, which reads as a
    # rendering fault rather than geography.
    land_mask = passable_mask(tm, (FLOOR, COURT, RUBBLE, BRIDGE, PLATFORM))
    connect_components(tm, land_mask, BRIDGE, width=1, seal_tile=WALL)
    attach(tm, elevation=e, moisture=moisture, slope=slope,
           accumulation=hyd['accumulation'])
    return tm


# ----------------------------------------------------------- 12 Flow Field

def gen_flow_field(spec, rng):
    """Curl-noise streamlines become windswept, calligraphic circulation."""
    w, h = dims(spec)
    seed = grammar_seed(spec, 'geo.flow_field')
    vec = tf.flow_field(w, h, spec.get('scale', 2.0), seed=seed)
    spacing = spec.get('spacing', max(10, min(w, h) // 6))
    seeds = [(x, y)
             for x in range(2, w, spacing)
             for y in range(2, h, spacing)]
    lines = tf.streamlines(vec, seeds,
                           steps=spec.get('steps', max(w, h)),
                           step_size=1.0)

    corridor = np.zeros((h, w), dtype=bool)
    highway = np.zeros((h, w), dtype=bool)
    lrng = np.random.default_rng(seed + 11)
    s = size_scale(w, h)
    base_w = int(round(spec.get('corridor_width', 2) * s))
    for i, line in enumerate(lines):
        wide = lrng.random() < spec.get('highway_rate', 0.12)
        stamp_polyline(highway if wide else corridor, line,
                       width=base_w + (2 * int(s) if wide else 0),
                       closed=False)
    corridor |= highway

    layers = [(corridor, FLOOR), (highway, PLATFORM)]
    tm = build(w, h, layers, default=WALL)
    rim = tf.distance_to_border(w, h).a < 1
    for y, x in zip(*np.nonzero(rim)):
        tm.base[y][x] = WALL
    connect_components(tm, passable_mask(tm), FLOOR, width=1)
    attach(tm, flow_x=vec.x, flow_y=vec.y,
           elevation=tf.fbm(w, h, 3.0, 4, seed=seed + 2).normalized())
    return tm


# ------------------------------------------------- 13 Contour Architecture

def gen_contour_terraces(spec, rng):
    """Topographic rings become architecture: terraces, retaining walls and
    stair cues where the elevation steps."""
    w, h = dims(spec)
    seed = grammar_seed(spec, 'geo.contour')
    base = tf.fbm(w, h, spec.get('scale', 2.6), 5, seed=seed).normalized()
    if spec.get('domed', True):
        dome = tf.radial(w, h).normalized().invert()
        base = base.blend(dome, spec.get('dome_weight', 0.45)).normalized()

    levels = spec.get('levels', 7)
    q, edge = base.terraces(levels)
    terrace = ~edge
    low = (q <= 0) & terrace
    high = (q >= levels - 1) & terrace
    mid = terrace & ~low & ~high

    layers = [(mid, FLOOR), (low, WATER), (high, COURT), (edge, WALL)]
    tm = build(w, h, layers, default=WALL)

    # stairs where adjacent terraces meet, so the steps are climbable
    srng = np.random.default_rng(seed + 5)
    ys, xs = np.nonzero(edge)
    if len(xs):
        pick = srng.choice(len(xs), size=min(spec.get('stairs', 26), len(xs)),
                           replace=False)
        for i in pick:
            x, y = int(xs[i]), int(ys[i])
            tm.set(x, y, FLOOR, 'stair')
    connect_components(tm, passable_mask(tm), FLOOR, width=1)
    attach(tm, elevation=base, terrace=q.astype(np.float32) / max(levels - 1, 1))
    return tm


# ------------------------------------------------- 17 Fractal Archipelago

def gen_archipelago(spec, rng):
    """Domain-warped coastlines at several scales, linked by bridges."""
    w, h = dims(spec)
    seed = grammar_seed(spec, 'geo.archipelago')
    base = tf.fractal(w, h, spec.get('scale', 2.4), seed=seed,
                      warp=spec.get('warp', 0.4)).normalized()
    falloff = tf.radial(w, h).normalized()
    e = (base - falloff * spec.get('falloff', 0.55)).normalized()

    land = e.a > spec.get('sea_level', 0.42)
    land = tf._erode(tf._dilate(land, 1), 1)
    shore = tf._dilate(land, 1) & ~land
    inland = tf._erode(land, 2)
    hills = (e.a > spec.get('hill_level', 0.72)) & inland
    grove = (tf.fbm(w, h, 5.0, 3, seed=seed + 13).a > 0.58) & inland & ~hills

    layers = [(land, FLOOR), (grove, COURT), (hills, WALL), (shore, RUBBLE)]
    tm = build(w, h, layers, default=WATER)

    # bridge the islands so the archipelago is traversable
    if spec.get('bridges', True):
        labels, n = tf.connected_components(land)
        if n > 1:
            counts = np.bincount(labels.ravel()); counts[0] = 0
            main = int(counts.argmax())
            sy, sx = tf.nearest_seed(labels == main)
            yy, xx = np.mgrid[0:h, 0:w]
            dist = np.hypot(xx - sx, yy - sy)
            for lab in range(1, n + 1):
                if lab == main:
                    continue
                comp = labels == lab
                if comp.sum() < spec.get('min_island', 12):
                    continue
                d = np.where(comp, dist, np.inf)
                i = int(np.argmin(d)); cy, cx = divmod(i, w)
                tx, ty = int(sx[cy, cx]), int(sy[cy, cx])
                if tx >= 0:
                    carve_l(tm, cx, cy, tx, ty, BRIDGE, 1, None, ((cx + cy) % 2) == 0)
    connect_components(tm, passable_mask(tm), BRIDGE, width=1, seal_tile=WALL)
    attach(tm, elevation=e, moisture=tf.fbm(w, h, 4.0, 4, seed=seed + 21).normalized())
    return tm


# ------------------------------------------------------ 07 Phyllotaxis City

def gen_phyllotaxis_city(spec, rng):
    """Golden-angle districts: sunflower spirals as urban fabric."""
    w, h = dims(spec)
    seed = grammar_seed(spec, 'geo.phyllotaxis')
    pts = tf.phyllotaxis(w, h, spec.get('nodes', 150))
    inside = ((pts[:, 0] > 1) & (pts[:, 0] < w - 2) &
              (pts[:, 1] > 1) & (pts[:, 1] < h - 2))
    pts = pts[inside]
    vor = tf.voronoi(w, h, pts, relax=spec.get('relax', 0))
    # 1-cell ward walls fragment as soon as anything else carves into them,
    # leaving a city with no legible structure at all
    borders = vor.border_mask(spec.get('wall_thickness', 1))
    plots = ~borders

    prng = np.random.default_rng(seed)
    roles = prng.random(int(vor.cell.max()) + 1)
    court = np.isin(vor.cell, np.flatnonzero(roles < 0.28)) & plots
    ruin = np.isin(vor.cell, np.flatnonzero(roles > 0.9)) & plots

    ring = tf.radial(w, h).normalized()
    avenues = (np.abs(np.sin(ring.a * math.pi * spec.get('rings', 5))) < 0.06)

    layers = [(plots, FLOOR), (court, COURT), (ruin, RUBBLE), (avenues, FLOOR)]
    tm = build(w, h, layers, default=WALL)
    rim = tf.distance_to_border(w, h).a < 1
    for y, x in zip(*np.nonzero(rim)):
        tm.base[y][x] = WALL
    connect_components(tm, passable_mask(tm), FLOOR, width=1)
    attach(tm, region=vor.cell, radial=ring)
    return tm


# ------------------------------------------------------- 09 DLA Caverns

def gen_dla_caverns(spec, rng):
    """Diffusion-limited aggregation: frost, mineral dendrites, crystal caves."""
    w, h = dims(spec)
    seed = grammar_seed(spec, 'geo.dla')
    # DLA cost scales with the particle count, which scales with area; past a
    # few hundred cells a side, grow the dendrite at reduced resolution and
    # upscale it rather than walking a quadratically growing budget
    cap = spec.get('resolution_cap', 160)
    gw, gh = min(w, cap), min(h, cap)
    dendrite = tf.dla(gw, gh, particles=spec.get('particles', int(gw * gh * 0.16)),
                      seed=seed)
    if (gw, gh) != (w, h):
        dendrite = np.repeat(np.repeat(dendrite, int(np.ceil(h / gh)), axis=0),
                             int(np.ceil(w / gw)), axis=1)[:h, :w]
    corridor = tf._dilate(dendrite, spec.get('thickness', 1))
    chamber = tf._dilate(dendrite, spec.get('thickness', 1) + 2) & \
        (tf.fbm(w, h, 5.0, 3, seed=seed + 4).a > 0.62)

    if spec.get('invert', False):
        layers = [(~corridor, FLOOR), (dendrite, WALL)]
        tm = build(w, h, layers, default=WALL)
    else:
        layers = [(corridor, FLOOR), (chamber, COURT), (dendrite, PLATFORM)]
        tm = build(w, h, layers, default=WALL)
    rim = tf.distance_to_border(w, h).a < 1
    for y, x in zip(*np.nonzero(rim)):
        tm.base[y][x] = WALL
    connect_components(tm, passable_mask(tm), FLOOR, width=1)
    attach(tm, crystal=tf.distance_to_mask(dendrite).normalized().invert())
    return tm


# ---------------------------------------------------------- 20 Hybrid Morph

FIELD_SOURCES = {
    'fbm':        lambda w, h, s, p: tf.fbm(w, h, p.get('scale', 3.0), 5, seed=s),
    'ridged':     lambda w, h, s, p: tf.ridged(w, h, p.get('scale', 3.0), 5, seed=s),
    'fractal':    lambda w, h, s, p: tf.fractal(w, h, p.get('scale', 2.5), seed=s),
    'radial':     lambda w, h, s, p: tf.radial(w, h).normalized().invert(),
    'spiral':     lambda w, h, s, p: tf.spiral(w, h, p.get('turns', 4.0)),
    'waves':      lambda w, h, s, p: tf.waves(w, h, p.get('frequency', 6.0),
                                              p.get('angle', 0.0)),
    'checker':    lambda w, h, s, p: tf.checker(w, h, p.get('cell', 8)),
    'voronoi':    lambda w, h, s, p: tf.voronoi(
        w, h, tf.poisson_disc(w, h, p.get('district_size', 9), seed=s), relax=2
    ).border_tension.normalized(),
    'reaction_diffusion': lambda w, h, s, p: tf.reaction_diffusion(
        w, h, p.get('preset', 'coral'), steps=p.get('steps', 2000), seed=s).normalized(),
    'phyllotaxis': lambda w, h, s, p: tf.distance_to_points(
        w, h, tf.phyllotaxis(w, h, p.get('nodes', 120))).normalized().invert(),
    'elevation':  lambda w, h, s, p: tf.watershed(w, h, seed=s)['elevation'],
    'growth':     lambda w, h, s, p: tf.distance_to_polylines(
        w, h, [tf.differential_growth(w, h, steps=p.get('steps', 170), seed=s,
                                      max_nodes=p.get('max_nodes', 1400)).tolist()]
    ).normalized().invert(),
}

FIELD_OPS = {
    'add':   lambda a, b, k: a + b,
    'sub':   lambda a, b, k: a - b,
    'mul':   lambda a, b, k: a * b,
    'max':   lambda a, b, k: a.maximum(b),
    'min':   lambda a, b, k: a.minimum(b),
    'blend': lambda a, b, k: a.blend(b, k),
    'mask':  lambda a, b, k: a * b.smoothstep(k - 0.15, k + 0.15),
    'warp':  lambda a, b, k: a.warp(b.curl(), k * 20.0),
}


def compose_field(w, h, spec, seed):
    """Evaluate `macro OP structure` from the pattern library (atlas 6.4)."""
    macro = spec.get('macro', 'voronoi')
    structure = spec.get('structure', 'reaction_diffusion')
    op = spec.get('op', 'mul')
    a = FIELD_SOURCES[macro](w, h, seed, spec).normalized()
    b = FIELD_SOURCES[structure](w, h, seed + 1013, spec).normalized()
    return FIELD_OPS[op](a, b, spec.get('k', 0.5)).normalized()


def gen_hybrid_morph(spec, rng):
    """Crossbreed two grammars into one field and read the result as terrain.

    This is where the catalog stops being additive: any macro x structure x
    operator triple is a new map family, not another corridor arrangement.
    """
    w, h = dims(spec)
    seed = grammar_seed(spec, 'auth.hybrid')
    field = compose_field(w, h, spec, seed)

    t = (spec['threshold'] if 'threshold' in spec
         else threshold_for_ratio(field, spec.get('open_ratio', 0.45)))
    open_cells = field.a > t
    if spec.get('invert', False):
        open_cells = ~open_cells

    detail = field.band(t, t + spec.get('detail_band', 0.10))
    deep = field.a > threshold_for_ratio(field, spec.get('deep_ratio', 0.10))
    hazard_tile = spec.get('hazard', WATER)
    hazard = ((field.a < threshold_for_ratio(field, 1.0 - spec.get('hazard_ratio', 0.12)))
              if spec.get('hazards', True) else np.zeros((h, w), dtype=bool))

    layers = [(open_cells, FLOOR), (detail & open_cells, COURT),
              (deep, PLATFORM), (hazard, hazard_tile)]
    tm = build(w, h, layers, default=WALL)
    rim = tf.distance_to_border(w, h).a < 1
    for y, x in zip(*np.nonzero(rim)):
        tm.base[y][x] = WALL
    connect_components(tm, passable_mask(tm), FLOOR, width=1)
    attach(tm, composite=field, elevation=field)
    tm.recipe = {'macro': spec.get('macro', 'voronoi'),
                 'structure': spec.get('structure', 'reaction_diffusion'),
                 'op': spec.get('op', 'mul'), 'expr': field.expr}
    return tm


GRAMMARS = {
    'voronoi_kingdoms':     gen_voronoi_kingdoms,
    'reaction_diffusion':   gen_reaction_diffusion,
    'differential_growth':  gen_differential_growth,
    'truchet':              gen_truchet,
    'watershed':            gen_watershed,
    'flow_field':           gen_flow_field,
    'contour_terraces':     gen_contour_terraces,
    'archipelago':          gen_archipelago,
    'phyllotaxis_city':     gen_phyllotaxis_city,
    'dla_caverns':          gen_dla_caverns,
    'hybrid_morph':         gen_hybrid_morph,
}
