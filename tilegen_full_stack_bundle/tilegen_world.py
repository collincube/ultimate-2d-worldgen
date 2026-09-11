"""World model: stable places, named topologies, fields, events, features.

Features exchange data through World registries. Place identifiers reproduce
for the same recipe, and declared dependencies determine feature build order.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict, deque

import numpy as np

import tilegen_fields as tf
import tilegen_grammars as tg

# ---------------------------------------------------------------- identities

def place_id(world_seed, index):
    return f"plc_{tf.seed_stream(world_seed, 'place', index):08x}"


def entity_id(world_seed, index):
    return f"ent_{tf.seed_stream(world_seed, 'entity', index):08x}"


def faction_id(world_seed, index):
    return f"fac_{tf.seed_stream(world_seed, 'faction', index):08x}"


# -------------------------------------------------------------------- places

class Place:
    """A map region referenced by an identifier derived from its seed and index."""

    __slots__ = ('id', 'kind', 'cells', 'centroid', 'area', 'stratum',
                 'district', 'function', 'tags', 'neighbours', 'meta')

    def __init__(self, pid, kind, cells, stratum=0, district=None, function=''):
        self.id = pid
        self.kind = kind                  # room | plaza | corridor | vault ...
        self.cells = cells                # boolean mask
        ys, xs = np.nonzero(cells)
        self.centroid = (float(xs.mean()), float(ys.mean())) if len(xs) else (0.0, 0.0)
        self.area = int(cells.sum())
        self.stratum = stratum
        self.district = district
        self.function = function
        self.tags = set()
        self.neighbours = set()
        self.meta = {}

    def __repr__(self):
        return f"Place({self.id} {self.kind} area={self.area} {self.function})"

    def to_dict(self):
        return {'id': self.id, 'kind': self.kind, 'area': self.area,
                'centroid': [round(c, 1) for c in self.centroid],
                'stratum': self.stratum, 'district': self.district,
                'function': self.function, 'tags': sorted(self.tags),
                'neighbours': sorted(self.neighbours), 'meta': self.meta}


# ---------------------------------------------------------------- topologies

class Topology:
    """A named graph for connections such as walking, sound, or NPC alliance."""

    __slots__ = ('name', 'directed', 'edges', 'node_meta', 'meta')

    def __init__(self, name, directed=False):
        self.name = name
        self.directed = directed
        self.edges = defaultdict(dict)     # a -> {b: weight}
        self.node_meta = {}
        self.meta = {}

    def add(self, a, b, weight=1.0, **meta):
        self.edges[a][b] = weight
        if not self.directed:
            self.edges[b][a] = weight
        if meta:
            self.node_meta.setdefault(a, {}).update(meta)

    def neighbours(self, a):
        return self.edges.get(a, {})

    @property
    def nodes(self):
        ns = set(self.edges)
        for a, nbrs in self.edges.items():
            ns.update(nbrs)
        return ns

    def edge_count(self):
        n = sum(len(v) for v in self.edges.values())
        return n if self.directed else n // 2

    def shortest_paths(self, source, cutoff=None):
        """Dijkstra from one node. Returns {node: cost}."""
        import heapq
        dist = {source: 0.0}
        heap = [(0.0, source)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, math.inf):
                continue
            if cutoff is not None and d > cutoff:
                continue
            for v, wgt in self.edges.get(u, {}).items():
                nd = d + wgt
                if nd < dist.get(v, math.inf):
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
        return dist

    def connected_parts(self):
        seen, parts = set(), []
        for start in self.nodes:
            if start in seen:
                continue
            comp, stack = set(), [start]
            seen.add(start)
            while stack:
                u = stack.pop()
                comp.add(u)
                for v in self.edges.get(u, {}):
                    if v not in seen:
                        seen.add(v)
                        stack.append(v)
            parts.append(comp)
        return parts

    def to_dict(self):
        return {'name': self.name, 'directed': self.directed,
                'nodes': len(self.nodes), 'edges': self.edge_count(),
                'adjacency': {a: {b: round(w, 3) for b, w in v.items()}
                              for a, v in self.edges.items()},
                'meta': self.meta}


# --------------------------------------------------------------------- world

class World:
    """Everything a compiled place holds. Systems read and write through here."""

    def __init__(self, tilemap, seed=0, name='world', spec=None):
        self.name = name
        self.seed = seed
        self.spec = spec or {}
        self.tilemap = tilemap
        self.h, self.w = tilemap.h, tilemap.w
        self.fields = dict(getattr(tilemap, 'fields', {}) or {})
        self.field_meta = {}
        self.topologies = {}
        self.places = {}
        self.place_order = []
        self.entities = {}
        self.factions = {}
        self.events = []
        self.findings = []
        self.artifacts = {}
        self.reports = {}
        self.time = 0

    # -- fields
    def set_field(self, name, array, units='normalised', lo=0.0, hi=1.0):
        arr = array.a if isinstance(array, tf.Field) else np.asarray(array, dtype=np.float32)
        if arr.shape != (self.h, self.w):
            raise ValueError(f"field {name} shape {arr.shape} != {(self.h, self.w)}")
        self.fields[name] = arr
        self.field_meta[name] = {'units': units, 'range': [lo, hi]}
        return arr

    def field(self, name, default=0.0):
        if name in self.fields:
            return self.fields[name]
        return np.full((self.h, self.w), default, dtype=np.float32)

    def has_field(self, name):
        return name in self.fields

    # -- topologies
    def topology(self, name, directed=False, create=True):
        if name not in self.topologies and create:
            self.topologies[name] = Topology(name, directed)
        return self.topologies.get(name)

    # -- places
    def add_place(self, kind, cells, **kw):
        pid = place_id(self.seed, len(self.place_order))
        p = Place(pid, kind, cells, **kw)
        self.places[pid] = p
        self.place_order.append(pid)
        return p

    def place_at(self, x, y):
        for pid in self.place_order:
            if self.places[pid].cells[y, x]:
                return self.places[pid]
        return None

    def places_of(self, kind=None, function=None, tag=None):
        out = []
        for pid in self.place_order:
            p = self.places[pid]
            if kind and p.kind != kind:
                continue
            if function and p.function != function:
                continue
            if tag and tag not in p.tags:
                continue
            out.append(p)
        return out

    def place_mask_field(self, values, default=0.0):
        """Paint a {place_id: value} mapping back onto the raster."""
        out = np.full((self.h, self.w), default, dtype=np.float32)
        for pid, v in values.items():
            p = self.places.get(pid)
            if p is not None:
                out[p.cells] = v
        return out

    # -- events
    def emit(self, kind, **payload):
        ev = {'seq': len(self.events), 'time': self.time, 'kind': kind}
        ev.update(payload)
        self.events.append(ev)
        return ev

    def events_of(self, kind):
        return [e for e in self.events if e['kind'] == kind]

    def finding(self, severity, rule, message, **extra):
        f = {'severity': severity, 'rule': rule, 'message': message}
        f.update(extra)
        self.findings.append(f)
        return f

    def passable(self):
        return tg.passable_mask(self.tilemap)

    def rng(self, name, index=0):
        return tf.rng_for(self.seed, name, index)

    def place_name(self, pid):
        """Readable location label with its approximate map coordinates."""
        p = self.places.get(pid)
        if p is None:
            return 'an unknown location'
        label = f"{p.district or ''} {p.function or p.kind}".strip().title()
        x, y = (round(c) for c in p.centroid)
        return f'{label} ({x}, {y})'

    def summary(self):
        return {
            'name': self.name, 'seed': self.seed,
            'size': [self.w, self.h],
            'places': len(self.places),
            'fields': sorted(self.fields),
            'topologies': {k: v.edge_count() for k, v in sorted(self.topologies.items())},
            'entities': len(self.entities),
            'factions': len(self.factions),
            'events': len(self.events),
            'findings': len(self.findings),
        }


# ------------------------------------------------------- place segmentation

def room_seeds(walk, min_clearance=2.0, min_separation=None):
    """Pick room centres as well-separated local maxima of wall clearance.

    Eroding the walkable area and taking components fails on open maps: a
    94%-open landscape erodes to one blob and yields a single 'room' for a whole
    continent. Clearance maxima find the places you could actually stand around
    in, at any openness.
    """
    dist = tf.distance_to_mask(~walk).a
    up = np.zeros_like(dist); up[1:, :] = dist[:-1, :]
    dn = np.zeros_like(dist); dn[:-1, :] = dist[1:, :]
    lf = np.zeros_like(dist); lf[:, 1:] = dist[:, :-1]
    rt = np.zeros_like(dist); rt[:, :-1] = dist[:, 1:]
    peak = walk & (dist >= min_clearance)
    peak &= (dist >= up) & (dist >= dn) & (dist >= lf) & (dist >= rt)

    ys, xs = np.nonzero(peak)
    if len(xs) == 0:
        return [], dist
    order = np.argsort(-dist[ys, xs])
    chosen = []
    for i in order:
        y, x, d = int(ys[i]), int(xs[i]), float(dist[ys[i], xs[i]])
        sep = min_separation if min_separation is not None else max(2.0, d * 1.35)
        if all((x - cx) ** 2 + (y - cy) ** 2 >= sep * sep for cx, cy, _ in chosen):
            chosen.append((x, y, d))
    return chosen, dist


def derive_places(world, min_room=6, min_clearance=2.0):
    """Segment the raster into rooms, plazas and corridors with stable ids.

    A grammar that already knows its own regions (voronoi cells, portal rooms)
    wins; otherwise rooms are grown from clearance maxima and whatever is left
    over is circulation.
    """
    tm = world.tilemap
    walk = world.passable()
    arr = np.array(tm.base, dtype=object)
    if not walk.any():
        return world.places

    elevation = world.fields.get('elevation')
    strata = None
    if elevation is not None:
        e = np.asarray(elevation, dtype=np.float32)
        rng_e = max(float(e.max() - e.min()), 1e-6)
        strata = np.clip((((e - e.min()) / rng_e) * 3).astype(int), 0, 2)

    region = world.fields.get('region')
    if region is not None and len(np.unique(region)) > 2:
        labels = np.asarray(region, dtype=np.int64) + 1
        labels = np.where(walk, labels, 0)
        seeds = None
    else:
        seeds, clearance = room_seeds(walk, min_clearance)
        if not seeds:
            labels = walk.astype(np.int64)
        else:
            seed_mask = np.zeros_like(walk)
            for (x, y, _) in seeds:
                seed_mask[y, x] = True
            sy, sx = tf.nearest_seed(seed_mask)
            index = {(y, x): i + 1 for i, (x, y, _) in enumerate(seeds)}
            labels = np.zeros(walk.shape, dtype=np.int64)
            flat = [index.get((int(a), int(b)), 0)
                    for a, b in zip(sy.ravel(), sx.ravel())]
            labels = np.array(flat, dtype=np.int64).reshape(walk.shape)
            labels = np.where(walk, labels, 0)

    room_cells = np.zeros_like(walk)
    for lab in np.unique(labels):
        if lab <= 0:
            continue
        comp = labels == lab
        # a label can straddle disconnected pockets; each is its own place
        sub, nsub = tf.connected_components(comp)
        for s in range(1, nsub + 1):
            cells = sub == s
            n = int(cells.sum())
            if n < min_room:
                continue
            kind = 'plaza' if n > min_room * 14 else 'room'
            st = int(strata[cells].mean()) if strata is not None else 0
            p = world.add_place(kind, cells, stratum=st)
            tiles = arr[cells]
            if len(tiles):
                vals, counts = np.unique(tiles.astype(str), return_counts=True)
                dominant = vals[counts.argmax()]
                p.function = {'courtyard': 'garden', 'water': 'cistern',
                              'rubble': 'ruin', 'platform': 'gallery',
                              'lava': 'forge', 'bridge': 'span'}.get(dominant, 'hall')
            room_cells |= cells

    corridor = walk & ~room_cells
    clabels, cn = tf.connected_components(corridor)
    for i in range(1, cn + 1):
        comp = clabels == i
        if comp.sum() < 2:
            continue
        world.add_place('corridor', comp, stratum=0, function='circulation')

    _link_adjacent_places(world)
    _assign_districts(world)
    return world.places


def _link_adjacent_places(world):
    """Build the base physical adjacency, and the walking topology from it."""
    owner = np.full((world.h, world.w), -1, dtype=np.int32)
    for idx, pid in enumerate(world.place_order):
        owner[world.places[pid].cells] = idx

    top = world.topology('physical.adjacency')
    seen = set()
    for A, B in ((owner[:, :-1], owner[:, 1:]), (owner[:-1, :], owner[1:, :])):
        m = (A != B) & (A >= 0) & (B >= 0)
        for a, b in zip(A[m].ravel().tolist(), B[m].ravel().tolist()):
            key = (min(a, b), max(a, b))
            if key in seen:
                continue
            seen.add(key)
            pa = world.place_order[key[0]]
            pb = world.place_order[key[1]]
            world.places[pa].neighbours.add(pb)
            world.places[pb].neighbours.add(pa)
            ca = world.places[pa].centroid
            cb = world.places[pb].centroid
            top.add(pa, pb, math.hypot(ca[0] - cb[0], ca[1] - cb[1]))
    return top


def _assign_districts(world, k=None):
    """Cluster places into districts by centroid (Lloyd on place centres)."""
    rooms = [world.places[p] for p in world.place_order
             if world.places[p].kind in ('room', 'plaza')]
    if not rooms:
        return
    k = k or max(2, min(8, len(rooms) // 4))
    pts = np.array([r.centroid for r in rooms], dtype=np.float32)
    rng = world.rng('district')
    centres = pts[rng.choice(len(pts), size=min(k, len(pts)), replace=False)]
    for _ in range(12):
        d = np.linalg.norm(pts[:, None, :] - centres[None, :, :], axis=2)
        assign = d.argmin(axis=1)
        for j in range(len(centres)):
            sel = assign == j
            if sel.any():
                centres[j] = pts[sel].mean(axis=0)
    names = ['north ward', 'east ward', 'south ward', 'west ward', 'high quarter',
             'low quarter', 'outer ring', 'inner core']
    for r, j in zip(rooms, assign):
        r.district = names[int(j) % len(names)]
    for pid in world.place_order:
        p = world.places[pid]
        if p.district is None and p.neighbours:
            for nb in p.neighbours:
                if world.places[nb].district:
                    p.district = world.places[nb].district
                    break


# ------------------------------------------------------------ field helpers

def propagate_field(seed_values, passthrough, iterations=40, decay=0.86,
                    blocked_decay=0.25):
    """Diffuse a quantity across the map, attenuated by what it passes through.

    Used by sound, light, smell, rumour and contamination. `passthrough` is a
    0..1 array: 1 flows freely, 0 blocks. Blocked cells still leak a little, so
    a thick wall muffles rather than silences.
    """
    cur = np.asarray(seed_values, dtype=np.float32).copy()
    pt = np.asarray(passthrough, dtype=np.float32)
    step = decay * pt + blocked_decay * (1.0 - pt)
    for _ in range(iterations):
        nb = (np.roll(cur, 1, 0) + np.roll(cur, -1, 0) +
              np.roll(cur, 1, 1) + np.roll(cur, -1, 1)) * 0.25
        nb[0, :] *= 0.5; nb[-1, :] *= 0.5; nb[:, 0] *= 0.5; nb[:, -1] *= 0.5
        cur = np.maximum(cur, nb * step)
    return cur


def line_of_sight(passable, origin, radius=24, samples=None):
    """Shadow-cast visibility from a point, by ray marching to the perimeter."""
    h, w = passable.shape
    vis = np.zeros((h, w), dtype=bool)
    ox, oy = origin
    samples = samples or max(64, int(radius * 6))
    for i in range(samples):
        a = 2 * math.pi * i / samples
        dx, dy = math.cos(a), math.sin(a)
        x, y = ox + 0.5, oy + 0.5
        for _ in range(int(radius)):
            x += dx
            y += dy
            xi, yi = int(x), int(y)
            if not (0 <= xi < w and 0 <= yi < h):
                break
            vis[yi, xi] = True
            if not passable[yi, xi]:
                break
    vis[oy, ox] = True
    return vis


# ----------------------------------------------------------- feature registry

class Feature:
    """One of the 40 depth features with declared inputs and outputs."""

    __slots__ = ('id', 'number', 'name', 'context', 'consumes', 'produces',
                 'fn', 'part', 'last')

    def __init__(self, fid, number, name, context, consumes, produces, fn, part,
                 last=False):
        self.id = fid
        self.number = number
        self.name = name
        self.context = context
        self.consumes = tuple(consumes)
        self.produces = tuple(produces)
        self.fn = fn
        self.part = part
        self.last = last          # terminal features run after everything else

    def __repr__(self):
        return f"Feature({self.number:02d} {self.id})"

    def to_dict(self):
        return {'id': self.id, 'number': self.number, 'name': self.name,
                'context': self.context, 'part': self.part,
                'consumes': list(self.consumes), 'produces': list(self.produces)}


FEATURES = {}


def feature(fid, number, name, context, consumes=(), produces=(), part='II',
            last=False):
    """Register a feature. Dependencies are declared, not implied by import order.

    `last=True` marks a terminal feature such as the world compiler, which must
    observe the finished world. Declaring only what it consumes is not enough:
    nothing depends on its output, so the topological sort is free to place it
    first, and it then snapshots an empty world.
    """
    def deco(fn):
        FEATURES[fid] = Feature(fid, number, name, context, consumes, produces,
                                fn, part, last)
        return fn
    return deco


def resolve_order(feature_ids=None):
    """Topologically sort features by produces -> consumes. Refuses cycles."""
    all_ids = list(feature_ids or FEATURES)
    terminal = [f for f in all_ids if FEATURES[f].last]
    ids = [f for f in all_ids if not FEATURES[f].last]
    producer = {}
    for fid in ids:
        for art in FEATURES[fid].produces:
            producer.setdefault(art, fid)

    deps = {fid: set() for fid in ids}
    for fid in ids:
        for art in FEATURES[fid].consumes:
            src = producer.get(art)
            if src and src != fid and src in deps:
                deps[fid].add(src)

    order, ready = [], deque(sorted(f for f in ids if not deps[f]))
    remaining = {f: set(d) for f, d in deps.items()}
    while ready:
        fid = ready.popleft()
        order.append(fid)
        for other, d in remaining.items():
            if fid in d:
                d.discard(fid)
                if not d and other not in order and other not in ready:
                    ready.append(other)
    missing = [f for f in ids if f not in order]
    if missing:
        # a cycle: run the rest in declaration order rather than silently dropping
        order.extend(sorted(missing, key=lambda f: FEATURES[f].number))
    order.extend(sorted(terminal, key=lambda f: FEATURES[f].number))
    return order


def run_features(world, feature_ids=None, verbose=False):
    """Execute features in dependency order, recording what each produced."""
    order = resolve_order(feature_ids)
    ran = []
    world.artifacts['features_run'] = ran
    for fid in order:
        feat = FEATURES[fid]
        try:
            feat.fn(world)
            ran.append(fid)
            if verbose:
                print(f"  {feat.number:02d} {feat.id}")
        except Exception as exc:                                  # noqa: BLE001
            world.finding('error', f'feature.{fid}',
                          f"{type(exc).__name__}: {exc}")
    world.artifacts['features_run'] = ran
    return ran
