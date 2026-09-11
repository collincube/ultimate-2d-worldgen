"""Part II, features 21-39: the overlapping maps that make a place lived-in.

Each feature publishes fields, topologies or artifacts. None of them invents a
tile type and none reaches into another feature's state - they communicate
through the World registries, in the dependency order the registry resolves.

Feature 40, the world compiler, lives in tilegen_compiler.py.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

import tilegen_fields as tf
import tilegen_grammars as tg
import tilegen_world as tw
from tilegen_world import feature, propagate_field, line_of_sight

# Physical properties of surfaces, kept in one table so acoustics, light,
# thermal and contamination all agree about what a wall is.
SURFACE = {
    #            sound  light   heat   flow   footfall_wear
    'wall':      (0.06, 0.00, 0.20, 0.00, 0.00),
    'bookshelf': (0.18, 0.02, 0.35, 0.02, 0.05),
    'floor':     (1.00, 1.00, 1.00, 1.00, 1.00),
    'platform':  (0.95, 1.00, 0.95, 0.90, 1.10),
    'ladder_tile': (0.90, 0.90, 0.90, 0.60, 1.30),
    'courtyard': (0.90, 1.00, 1.10, 0.95, 0.70),
    'rubble':    (0.70, 0.85, 0.90, 0.55, 1.40),
    'bridge':    (0.85, 1.00, 0.90, 0.80, 1.25),
    'water':     (0.75, 0.90, 0.60, 1.00, 0.10),
    'lava':      (0.60, 1.00, 3.00, 0.40, 0.00),
    'abyss':     (0.30, 0.20, 0.50, 1.00, 0.00),
}
PROP_INDEX = {'sound': 0, 'light': 1, 'heat': 2, 'flow': 3, 'wear': 4}


def surface_property(world, prop):
    """Per-cell physical property array for the current tile layout."""
    arr = np.array(world.tilemap.base, dtype=object)
    idx = PROP_INDEX[prop]
    out = np.ones(arr.shape, dtype=np.float32)
    for tile in np.unique(arr):
        out[arr == tile] = SURFACE.get(str(tile), (0.8,) * 5)[idx]
    return out


def _place_field_mean(world, field_name):
    """Average a field over each place - the usual bridge from raster to place."""
    f = world.field(field_name)
    return {pid: float(f[world.places[pid].cells].mean())
            for pid in world.place_order if world.places[pid].area}


def _seed_from_tiles(world, tiles, strength=1.0):
    arr = np.array(world.tilemap.base, dtype=object)
    seed = np.zeros(arr.shape, dtype=np.float32)
    for t in tiles:
        seed[arr == t] = strength
    return seed


# ------------------------------------------------------------- 21 Acoustic

@feature('plc.acoustic', 21, 'Acoustic Map', 'Place Semantics',
         consumes=('places',), produces=('fields.sound_pressure',
                                         'topology.perception.acoustic'))
def acoustic_map(world):
    """Sound propagation: what can be heard from where, and how muffled.

    Walls attenuate rather than silence, so a forge two rooms away is a rumour
    of a forge. The acoustic topology deliberately disagrees with the walking
    graph: sound goes through a wall that a body cannot.
    """
    passthrough = surface_property(world, 'sound')
    sources = np.zeros((world.h, world.w), dtype=np.float32)
    arr = np.array(world.tilemap.base, dtype=object)
    sources[arr == 'lava'] = 1.0                       # forges roar
    sources[arr == 'water'] = 0.30                     # water runs
    # Crowd noise scales with how big a gathering place is relative to the
    # largest one. Giving every plaza the same loud source saturates the whole
    # field, and then nowhere is quiet enough for the difference to matter.
    plazas = world.places_of('plaza')
    if plazas:
        biggest = max(p.area for p in plazas)
        for p in plazas:
            sources[p.cells] = np.maximum(sources[p.cells],
                                          0.15 + 0.55 * (p.area / max(biggest, 1)))
    for p in world.places_of(function='forge'):
        sources[p.cells] = 1.0
    if sources.max() <= 0.0:
        # no forge, water or plaza: people themselves are the sound source, so
        # fall back to the largest inhabited places rather than a silent world
        biggest = sorted(world.places.values(), key=lambda p: -p.area)[:6]
        for p in biggest:
            sources[p.cells] = 0.6

    pressure = propagate_field(sources, passthrough, iterations=48,
                               decay=0.88, blocked_decay=0.30)
    # Sound leaks through walls by design, so with sources in many places the
    # raw field saturates near its own maximum and has almost no dynamic range
    # left to threshold against. Rescale to the observed span so "loud" and
    # "quiet" mean something relative to this world.
    lo, hi = float(pressure.min()), float(pressure.max())
    if hi - lo > 1e-6:
        pressure = (pressure - lo) / (hi - lo)
    world.set_field('sound_pressure', np.clip(pressure, 0, 1), 'relative', 0, 1)

    top = world.topology('perception.acoustic')
    means = _place_field_mean(world, 'sound_pressure')
    for pid in world.place_order:
        p = world.places[pid]
        for nb in p.neighbours:
            # cost is how much sound is lost between the two, not distance
            loss = abs(means.get(pid, 0) - means.get(nb, 0)) + 0.05
            top.add(pid, nb, round(float(loss), 4))
    # Tag relative to this world: an absolute cutoff on a skewed field labels
    # every place loud, which tells a designer nothing.
    if means:
        vals = sorted(means.values())
        hi = vals[int(len(vals) * 0.75)]
        lo = vals[int(len(vals) * 0.25)]
        for pid, v in means.items():
            if v >= hi:
                world.places[pid].tags.add('loud')
            elif v <= lo:
                world.places[pid].tags.add('quiet')
    return top


# ---------------------------------------------------------------- 22 Light

@feature('prs.light', 22, 'Light Map', 'Presentation & Style',
         consumes=('places',), produces=('fields.light', 'fields.shadow'))
def light_map(world):
    """Light transport from emissive surfaces, sky wells and placed lamps.

    Drives both rendering and concealment: what is lit is what can be seen, so
    the guard and stealth systems both read this field.
    """
    arr = np.array(world.tilemap.base, dtype=object)
    passthrough = surface_property(world, 'light')
    emit = np.zeros((world.h, world.w), dtype=np.float32)
    emit[arr == 'lava'] = 1.0
    emit[arr == 'courtyard'] = 0.85                    # open to the sky
    emit[arr == 'water'] = 0.30                        # reflected sky

    # a lamp in the middle of each sizable interior room
    rng = world.rng('light.lamps')
    for p in world.places_of('room'):
        if p.area < 12 or 'lit' in p.tags:
            continue
        if rng.random() < 0.65:
            x, y = int(p.centroid[0]), int(p.centroid[1])
            if 0 <= x < world.w and 0 <= y < world.h:
                emit[y, x] = max(emit[y, x], 0.9)

    light = propagate_field(emit, passthrough, iterations=40, decay=0.80,
                            blocked_decay=0.02)
    light = np.clip(light, 0, 1)
    world.set_field('light', light, 'relative', 0, 1)
    world.set_field('shadow', 1.0 - light, 'relative', 0, 1)
    lm = _place_field_mean(world, 'light')
    if lm:
        vals = sorted(lm.values())
        hi = vals[int(len(vals) * 0.75)]
        lo = vals[int(len(vals) * 0.25)]
        for pid, v in lm.items():
            if v <= lo:
                world.places[pid].tags.add('dark')
            elif v >= hi:
                world.places[pid].tags.add('bright')
    return light


# --------------------------------------------------- 23 Social Circulation

ROLES = ('resident', 'servant', 'guard', 'guest', 'outsider')


@feature('eco.circulation', 23, 'Social Circulation Map', 'Ecology & Inhabitants',
         consumes=('places', 'fields.light'),
         produces=('topology.circulation.public', 'topology.circulation.service',
                   'fields.privacy'))
def social_circulation(world):
    """Public and service circulation as separate graphs.

    The grand route and the servants' route may connect the same two halls and
    share no edges. Privacy falls out of how far a place sits from the public
    graph. Hidden shops and stealth encounters use this spatial privacy field.
    """
    public = world.topology('circulation.public')
    service = world.topology('circulation.service')
    light = world.field('light')

    scores = {}
    for pid in world.place_order:
        p = world.places[pid]
        lit = float(light[p.cells].mean()) if p.area else 0.0
        # big, bright, well-connected places carry the public route
        scores[pid] = (0.5 * min(p.area / 60.0, 1.0) + 0.3 * lit +
                       0.2 * min(len(p.neighbours) / 4.0, 1.0))

    # Split at the median rather than an absolute score. Service access is defined
    # relative to the grand route: with a fixed cutoff a uniformly grand map
    # puts every edge on the public graph and leaves the service graph -- and
    # everything downstream of it, logistics and market access -- empty.
    edges = []
    for pid in world.place_order:
        for nb in world.places[pid].neighbours:
            if pid < nb:
                edges.append((pid, nb, (scores[pid] + scores[nb]) / 2.0))
    cut = float(np.median([e[2] for e in edges])) if edges else 0.5
    for pid, nb, grand in edges:
        if grand >= cut:
            public.add(pid, nb, round(1.0 / max(grand, 0.05), 3))
        else:
            service.add(pid, nb, round(1.0 / max(1.0 - grand, 0.05), 3))
        # a few links belong to both: the doors staff and guests share
        if abs(grand - cut) < 0.03:
            (service if grand >= cut else public).add(pid, nb, 2.0)

    pubnodes = set(public.nodes)
    privacy = {}
    for pid in world.place_order:
        if pid in pubnodes:
            d = 0.0
        else:
            d = 1.0
        p = world.places[pid]
        privacy[pid] = min(1.0, d * 0.6 + (1.0 - scores[pid]) * 0.4)
        if privacy[pid] > 0.6:
            p.tags.add('private')
        if pid in pubnodes and scores[pid] > 0.6:
            p.tags.add('public')
    world.set_field('privacy', world.place_mask_field(privacy, 0.5))
    world.artifacts['circulation_scores'] = scores
    return public, service


# ------------------------------------------------------------- 24 Schedule

DAY_PHASES = ('dawn', 'morning', 'midday', 'dusk', 'night', 'deep night')


@feature('eco.schedule', 24, 'Schedule Map', 'Ecology & Inhabitants',
         consumes=('places', 'topology.circulation.public'),
         produces=('artifacts.schedule', 'fields.crowding'))
def schedule_map(world):
    """Per-place occupancy across the day, so a place is busy or empty at a time
    rather than in the abstract. Crowding is the daily mean."""
    rng = world.rng('schedule')
    schedule = {}
    profiles = {
        'hall':        (0.2, 0.7, 0.9, 0.8, 0.4, 0.1),
        'garden':      (0.3, 0.6, 0.7, 0.5, 0.1, 0.05),
        'cistern':     (0.4, 0.5, 0.3, 0.3, 0.1, 0.05),
        'forge':       (0.5, 0.9, 0.9, 0.7, 0.2, 0.05),
        'ruin':        (0.05, 0.1, 0.15, 0.2, 0.3, 0.35),
        'gallery':     (0.1, 0.4, 0.6, 0.6, 0.3, 0.1),
        'circulation': (0.3, 0.8, 0.8, 0.7, 0.3, 0.1),
        'span':        (0.2, 0.6, 0.6, 0.5, 0.2, 0.1),
    }
    for pid in world.place_order:
        p = world.places[pid]
        base = profiles.get(p.function, profiles['hall'])
        jitter = rng.normal(0, 0.06, len(DAY_PHASES))
        curve = np.clip(np.array(base) + jitter, 0.0, 1.0)
        if 'private' in p.tags:
            curve *= 0.45
        if 'dark' in p.tags:
            curve = curve * 0.7 + np.array([0, 0, 0, .05, .15, .2])
        schedule[pid] = {phase: round(float(v), 3)
                         for phase, v in zip(DAY_PHASES, curve)}
    world.artifacts['schedule'] = schedule
    world.set_field('crowding', world.place_mask_field(
        {pid: float(np.mean(list(v.values()))) for pid, v in schedule.items()}))
    return schedule


# ------------------------------------------------------------ 25 Logistics

GOODS = ('water', 'grain', 'ore', 'fuel', 'cloth', 'medicine')


@feature('eco.logistics', 25, 'Logistics Map', 'Economy & Logistics',
         consumes=('places', 'topology.circulation.service'),
         produces=('topology.logistics.supply', 'fields.market_access',
                   'artifacts.logistics'))
def logistics_map(world):
    """Where goods are produced, where they are consumed, and the routes between.

    Routes run on the service graph, not the grand one - supply moves through
    the back of the house.
    """
    # Route on service edges by preference, but keep the public ones available
    # at a penalty. A producer whose every link happens to sit on the grand
    # route is not a node of the service graph at all, and routing from it finds
    # nothing -- which silently emptied the supply network on some maps.
    service = tw.Topology('logistics.routing')
    for name, penalty in (('circulation.service', 1.0), ('circulation.public', 2.5)):
        top = world.topologies.get(name)
        if not top:
            continue
        for a, nbrs in top.edges.items():
            for b, wgt in nbrs.items():
                if b not in service.edges.get(a, {}):
                    service.add(a, b, wgt * penalty)
    if service.edge_count() < 2:
        service = world.topology('physical.adjacency')
    supply = world.topology('logistics.supply', directed=True)
    sources, sinks = defaultdict(list), defaultdict(list)
    for pid in world.place_order:
        p = world.places[pid]
        if p.function == 'cistern':
            sources['water'].append(pid)
        elif p.function == 'forge':
            sources['ore'].append(pid); sources['fuel'].append(pid)
        elif p.function == 'garden':
            sources['grain'].append(pid); sources['medicine'].append(pid)
        elif p.area > 12 and p.kind in ('room', 'plaza'):
            for g in GOODS:
                sinks[g].append(pid)

    if not any(sources.values()):
        depots = sorted(world.places.values(), key=lambda p: -p.area)[:4]
        for i, p in enumerate(depots):
            sources[GOODS[i % len(GOODS)]].append(p.id)
            p.tags.add('depot')
    if not any(sinks.values()):
        src_ids = {pid for v in sources.values() for pid in v}
        for p in sorted(world.places.values(), key=lambda p: -p.area):
            if p.id in src_ids:
                continue
            for gd in GOODS:
                sinks[gd].append(p.id)
            if len(sinks[GOODS[0]]) >= 8:
                break
    routes = []
    access = defaultdict(float)
    for good in GOODS:
        for src in sources.get(good, []):
            # no cutoff: a long thin map (space-filling curves) has legitimate
            # route costs far above any fixed budget, and clipping them silently
            # produced a world with no supply network at all
            dist = service.shortest_paths(src)
            good_sinks = set(sinks.get(good, ()))
            reachable = [(d, pid) for pid, d in dist.items() if pid in good_sinks]
            reachable.sort()
            for d, dst in reachable[:6]:
                supply.add(src, dst, round(d, 2), good=good)
                routes.append({'good': good, 'from': src, 'to': dst,
                               'cost': round(d, 2)})
                access[dst] += 1.0 / (1.0 + d)
    if access:
        mx = max(access.values())
        access = {k: v / mx for k, v in access.items()}
    world.set_field('market_access', world.place_mask_field(dict(access), 0.0))
    world.artifacts['logistics'] = {'routes': routes,
                                    'sources': {k: v for k, v in sources.items()},
                                    'goods': list(GOODS)}
    return supply


# ------------------------------------------------------------- 26 Security

@feature('soc.security', 26, 'Security Map', 'Society & Institutions',
         consumes=('places', 'fields.light', 'topology.circulation.public'),
         produces=('topology.society.guard_patrol', 'fields.guard_attention'))
def security_map(world):
    """Patrol circuits, coverage and response time.

    Attention is what smuggler encounters and dungeon missions have to route
    around, so it is a field rather than a boolean.
    """
    public = world.topology('circulation.public')
    patrol = world.topology('society.guard_patrol', directed=True)
    light = world.field('light')

    # posts go where routes converge and the light is good
    candidates = []
    for pid in world.place_order:
        p = world.places[pid]
        if not p.area:
            continue
        val = len(p.neighbours) * 0.5 + float(light[p.cells].mean())
        candidates.append((val, pid))
    candidates.sort(reverse=True)
    nposts = max(2, len(world.place_order) // 12)
    posts = [pid for _, pid in candidates[:nposts]]

    attention = defaultdict(float)
    for i, post in enumerate(posts):
        dist = public.shortest_paths(post, cutoff=40)
        ring = sorted(dist.items(), key=lambda kv: kv[1])[:10]
        prev = post
        for pid, d in ring:
            if pid != prev:
                patrol.add(prev, pid, round(d, 2))
                prev = pid
            attention[pid] += 1.0 / (1.0 + d)
        if prev != post:
            patrol.add(prev, post, 1.0)
        world.places[post].tags.add('guard post')
    if attention:
        mx = max(attention.values())
        attention = {k: v / mx for k, v in attention.items()}
    world.set_field('guard_attention', world.place_mask_field(dict(attention)))
    world.artifacts['security'] = {'posts': posts, 'patrol_edges': patrol.edge_count()}
    return patrol


# --------------------------------------------------------------- 27 Rumour

@feature('nar.rumor', 27, 'Rumor and Knowledge Map', 'Incidents & Narrative',
         consumes=('places', 'artifacts.schedule'),
         produces=('topology.society.rumor', 'fields.rumor_reach'))
def rumor_map(world):
    """Belief spreads where people overlap in time, not where walls connect.

    Two rooms adjacent on the map but busy at opposite hours barely exchange
    gossip; two distant plazas busy at midday do.
    """
    rumor = world.topology('society.rumor')
    sched = world.artifacts.get('schedule', {})
    ids = [pid for pid in world.place_order if world.places[pid].area >= 4]
    vectors = {pid: np.array([sched.get(pid, {}).get(ph, 0.0) for ph in DAY_PHASES])
               for pid in ids}
    for i, a in enumerate(ids):
        pa = world.places[a]
        for b in list(pa.neighbours) + ids[max(0, i - 3):i]:
            if b not in vectors or b == a:
                continue
            overlap = float(np.minimum(vectors[a], vectors[b]).sum())
            if overlap > 0.35:
                rumor.add(a, b, round(1.0 / overlap, 3))
    reach = {}
    for pid in ids:
        reach[pid] = len(rumor.shortest_paths(pid, cutoff=6)) / max(len(ids), 1)
    world.set_field('rumor_reach', world.place_mask_field(reach))
    return rumor


# -------------------------------------------------------------- 28 Ecology

SPECIES = (
    ('rat',    {'moisture': 0.4, 'light': -0.6, 'crowding': 0.3}),
    ('bat',    {'moisture': 0.2, 'light': -0.9, 'crowding': -0.4}),
    ('moss',   {'moisture': 0.9, 'light': -0.2, 'crowding': -0.3}),
    ('bird',   {'moisture': 0.1, 'light': 0.8, 'crowding': -0.1}),
    ('beetle', {'moisture': 0.5, 'light': -0.2, 'crowding': -0.2}),
)


@feature('eco.ecology', 28, 'Ecological Map', 'Ecology & Inhabitants',
         consumes=('places', 'fields.light', 'fields.crowding'),
         produces=('fields.habitat', 'topology.ecology.routes',
                   'artifacts.ecology'))
def ecology_map(world):
    """Habitat suitability per species, and the routes they actually use.

    Species routes are their own topology: a rat run crosses a wall gap no
    person fits through.
    """
    moisture = world.field('moisture', 0.4)
    if moisture.max() == moisture.min():
        moisture = tf.fbm(world.w, world.h, 4.0, 4,
                          seed=tf.seed_stream(world.seed, 'eco.moisture')).normalized().a
    light = world.field('light', 0.5)
    crowding = world.field('crowding', 0.3)
    walk = world.passable()

    habitats = {}
    for name, weights in SPECIES:
        score = (0.5 + weights['moisture'] * (moisture - 0.5)
                 + weights['light'] * (light - 0.5)
                 + weights['crowding'] * (crowding - 0.5))
        habitats[name] = np.clip(score, 0, 1).astype(np.float32) * walk
    best = max(habitats.items(), key=lambda kv: float(kv[1].mean()))
    world.set_field('habitat', best[1])
    for name, arr in habitats.items():
        world.set_field(f'habitat_{name}', arr)

    routes = world.topology('ecology.routes')
    means = {}
    for pid in world.place_order:
        p = world.places[pid]
        if not p.area:
            continue
        means[pid] = float(habitats['rat'][p.cells].mean())
    for pid, v in means.items():
        for nb in world.places[pid].neighbours:
            if means.get(nb, 0) > 0.35 and v > 0.35:
                routes.add(pid, nb, round(2.0 - v - means[nb], 3))
    world.artifacts['ecology'] = {
        'species': [s[0] for s in SPECIES],
        'dominant': best[0],
        'mean_suitability': {k: round(float(v.mean()), 3) for k, v in habitats.items()}}
    return habitats


# -------------------------------------------------------------- 29 Weather

@feature('env.weather', 29, 'Weather Map', 'Environment & Infrastructure',
         consumes=('places',), produces=('fields.temperature', 'fields.exposure',
                                         'fields.wind', 'artifacts.weather'))
def weather_map(world):
    """Exposure, wind shadow and temperature, and what they close off.

    Exposure is derived from what is open to the sky, so weather affects the
    courtyard and the bridge and leaves the vault alone.
    """
    arr = np.array(world.tilemap.base, dtype=object)
    seed = tf.seed_stream(world.seed, 'env.weather')
    open_sky = np.isin(arr.astype(str), ['courtyard', 'water', 'bridge', 'abyss'])
    exposure = propagate_field(open_sky.astype(np.float32),
                               surface_property(world, 'flow'),
                               iterations=14, decay=0.72, blocked_decay=0.05)
    exposure = np.clip(exposure, 0, 1)

    wind = tf.flow_field(world.w, world.h, 2.0, seed=seed)
    speed = tf.Field(np.hypot(wind.x, wind.y)).normalized().a * exposure

    elevation = world.field('elevation', 0.5)
    heat = surface_property(world, 'heat')
    temperature = np.clip(0.55 - 0.35 * elevation + 0.25 * (heat - 1.0)
                          - 0.20 * exposure, 0, 1).astype(np.float32)

    world.set_field('exposure', exposure)
    world.set_field('wind', speed.astype(np.float32))
    world.set_field('temperature', temperature, 'normalised 0=frozen 1=scorching')
    world.artifacts['weather'] = {
        'seasons': ['dry', 'storm', 'frost'],
        'mean_exposure': round(float(exposure.mean()), 3),
        'sheltered_places': [pid for pid in world.place_order
                             if float(exposure[world.places[pid].cells].mean()) < 0.1][:40]}
    return exposure


# -------------------------------------------------------- 30 Material Aging

@feature('env.aging', 30, 'Material-Aging Map', 'Environment & Infrastructure',
         consumes=('places', 'fields.crowding', 'fields.exposure'),
         produces=('fields.wear', 'fields.age', 'artifacts.maintenance'))
def material_aging(world):
    """Wear accumulates where feet fall, water sits and weather reaches.

    Maintenance debt is wear minus whatever attention the place gets, which is
    what later lets a neglected quarter visibly rot.
    """
    footfall = surface_property(world, 'wear')
    crowding = world.field('crowding', 0.3)
    exposure = world.field('exposure', 0.2)
    moisture = world.field('moisture', 0.3)
    walk = world.passable()

    wear = np.clip(0.45 * crowding * footfall + 0.30 * exposure +
                   0.25 * moisture, 0, 1).astype(np.float32) * walk
    base_age = world.field('age')
    if base_age.max() == base_age.min():
        base_age = tf.fbm(world.w, world.h, 2.5, 4,
                          seed=tf.seed_stream(world.seed, 'env.age')).normalized().a
    age = np.clip(0.6 * base_age + 0.4 * wear, 0, 1).astype(np.float32)

    world.set_field('wear', wear)
    world.set_field('age', age)
    debt = {}
    for pid in world.place_order:
        p = world.places[pid]
        if not p.area:
            continue
        w = float(wear[p.cells].mean())
        attention = 0.6 if 'public' in p.tags else 0.25
        debt[pid] = round(max(0.0, w - attention), 3)
        if debt[pid] > 0.35:
            p.tags.add('dilapidated')
    world.artifacts['maintenance'] = debt
    return wear


# --------------------------------------------------------- 31 Transformation

@feature('nar.transformation', 31, 'Transformation Map', 'Incidents & Narrative',
         consumes=('places', 'fields.wear'),
         produces=('artifacts.transformations', 'fields.structural_stress'))
def transformation_map(world):
    """Which parts of the fabric can change, and what each change would do.

    Persistent structural state: a breached wall stays breached, so the
    chronicle and the navigation graph must both be able to learn about it.
    """
    wear = world.field('wear')
    arr = np.array(world.tilemap.base, dtype=object)
    solid = np.isin(arr.astype(str), ['wall', 'bookshelf'])
    walk = world.passable()

    # a wall is breachable where it is thin and separates two open regions
    thin = solid & (tf.distance_to_mask(walk).a <= 1.5)
    stress = np.clip(tf.distance_to_mask(~solid).a / 4.0, 0, 1).astype(np.float32)
    stress = np.where(solid, 1.0 - stress, 0.0).astype(np.float32)
    world.set_field('structural_stress', stress)

    rng = world.rng('transformation')
    ys, xs = np.nonzero(thin)
    transforms = []
    if len(xs):
        pick = rng.choice(len(xs), size=min(30, len(xs)), replace=False)
        for i in pick:
            x, y = int(xs[i]), int(ys[i])
            transforms.append({
                'kind': 'breach', 'at': [x, y],
                'difficulty': round(float(1.0 - wear[y, x]), 3),
                'effect': 'connects two regions permanently'})
    for p in world.places_of(tag='dilapidated'):
        transforms.append({'kind': 'collapse', 'place': p.id,
                           'difficulty': 0.4,
                           'effect': 'removes the place and blocks its exits'})
    world.artifacts['transformations'] = transforms
    return transforms


# --------------------------------------------------------------- 32 Memory

@feature('nar.memory', 32, 'Memory Map', 'Incidents & Narrative',
         consumes=('places', 'topology.society.rumor'),
         produces=('fields.memory', 'artifacts.memories'))
def memory_map(world):
    """Event residue: places remember what happened in them, and the memory
    spreads along the rumour graph rather than through walls."""
    rumor = world.topology('society.rumor')
    rng = world.rng('memory')
    candidates = [p for p in world.places.values() if p.area >= 6]
    memories = []
    if candidates:
        kinds = ['a death', 'a betrayal', 'a wedding', 'a fire', 'a verdict',
                 'a disappearance', 'a miracle', 'a riot']
        n = min(len(candidates), max(3, len(candidates) // 8))
        for p in rng.choice(candidates, size=n, replace=False):
            kind = kinds[int(rng.integers(len(kinds)))]
            memories.append({'place': p.id, 'event': kind,
                             'age': round(float(rng.random()), 3),
                             'weight': round(float(rng.random() * 0.7 + 0.3), 3)})
            p.tags.add('remembered')
            world.emit('memory.recorded', place=p.id, event=kind)

    strength = defaultdict(float)
    for m in memories:
        strength[m['place']] += m['weight']
        for pid, d in rumor.shortest_paths(m['place'], cutoff=5).items():
            strength[pid] += m['weight'] / (1.0 + d)
    if strength:
        mx = max(strength.values())
        strength = {k: v / mx for k, v in strength.items()}
    world.set_field('memory', world.place_mask_field(dict(strength)))
    world.artifacts['memories'] = memories
    return memories


# ------------------------------------------------------- 33 Traversal Verbs

VERBS = {
    'walk':  {'floor', 'platform', 'courtyard', 'bridge', 'rubble', 'ladder_tile'},
    'swim':  {'water'},
    'climb': {'ladder_tile', 'rubble', 'wall'},
    'crawl': {'floor', 'rubble', 'water', 'courtyard'},
    'fly':   {'floor', 'platform', 'courtyard', 'bridge', 'water', 'abyss',
              'lava', 'rubble', 'ladder_tile'},
}


@feature('top.traversal', 33, 'Traversal-Verb Map', 'Topology & Traversal',
         consumes=('places',), produces=('topology.traversal',))
def traversal_verbs(world):
    """One topology per capability. A chasm is a wall to a walker, a road to a
    flier, and a slow crawl to something that climbs."""
    arr = np.array(world.tilemap.base, dtype=object).astype(str)
    built = {}
    for verb, tiles in VERBS.items():
        mask = np.isin(arr, list(tiles))
        top = world.topology(f'traversal.{verb}')
        for pid in world.place_order:
            p = world.places[pid]
            if not p.area:
                continue
            frac = float(mask[p.cells].mean())
            if frac < 0.25:
                continue
            top.node_meta[pid] = {'coverage': round(frac, 3)}
            for nb in p.neighbours:
                q = world.places[nb]
                if q.area and float(mask[q.cells].mean()) >= 0.25:
                    top.add(pid, nb, round(2.0 - frac, 3))
        built[verb] = top
    world.artifacts['traversal'] = {
        v: {'nodes': len(t.nodes), 'edges': t.edge_count()} for v, t in built.items()}
    return built


# --------------------------------------------------- 34 Gravity/Orientation

@feature('top.gravity', 34, 'Gravity and Orientation Map', 'Topology & Traversal',
         consumes=('places',), produces=('fields.gravity_x', 'fields.gravity_y',
                                         'artifacts.frames'))
def gravity_map(world):
    """Local reference frames and gravity volumes.

    Most of a map is down-is-down. Shafts, spires and voids get their own frame,
    which is what makes a vertical level legible instead of merely tall.
    """
    arr = np.array(world.tilemap.base, dtype=object).astype(str)
    gx = np.zeros((world.h, world.w), dtype=np.float32)
    gy = np.ones((world.h, world.w), dtype=np.float32)

    elevation = world.field('elevation', 0.5)
    grad = tf.Field(elevation).gradient()
    steep = tf.Field(elevation).slope().normalized().a > 0.55
    mag = np.hypot(grad.x, grad.y)
    safe = np.where(mag < 1e-6, 1.0, mag)
    gx = np.where(steep, grad.x / safe, gx).astype(np.float32)
    gy = np.where(steep, grad.y / safe, gy).astype(np.float32)

    void = np.isin(arr, ['abyss'])
    frames = [{'id': 'frame.default', 'gravity': [0, 1], 'cells': int((~void).sum())}]
    if void.any():
        frames.append({'id': 'frame.void', 'gravity': [0, 0],
                       'cells': int(void.sum()),
                       'note': 'free orientation; entities choose their own down'})
        gy[void] = 0.0
    world.set_field('gravity_x', gx, 'unit vector', -1, 1)
    world.set_field('gravity_y', gy, 'unit vector', -1, 1)
    world.artifacts['frames'] = frames
    return frames


# ----------------------------------------------------------- 35 Perceptual

@feature('plc.perceptual', 35, 'Perceptual Map', 'Place Semantics',
         consumes=('places', 'fields.light', 'fields.sound_pressure'),
         produces=('artifacts.perception',))
def perceptual_map(world):
    """What a specific observer, standing somewhere specific, can know.

    Sight is line-of-sight limited by light; hearing is not blocked by walls.
    The two disagree, which is the entire point of modelling them separately.
    """
    walk = world.passable()
    light = world.field('light')
    sound = world.field('sound_pressure')
    observers = [p for p in world.places_of('plaza')][:3] or \
                [p for p in world.places_of('room')][:3]
    out = []
    for p in observers:
        x, y = int(p.centroid[0]), int(p.centroid[1])
        x = max(0, min(world.w - 1, x)); y = max(0, min(world.h - 1, y))
        vis = line_of_sight(walk, (x, y), radius=min(28, min(world.w, world.h) // 2))
        seen = vis & (light > 0.10)
        # audible means distinctly above the ambient level, not merely non-zero:
        # a keep with sources in every plaza has faint sound everywhere, so a
        # near-zero threshold reports the entire map as audible and the
        # heard-but-unseen figure stops meaning anything
        ambient = float(np.quantile(sound[walk], 0.70)) if walk.any() else 0.5
        heard = sound > max(0.30, ambient)
        out.append({
            'observer_place': p.id, 'at': [x, y],
            'visible_cells': int(seen.sum()),
            'audible_cells': int(heard.sum()),
            'heard_but_unseen': int((heard & ~seen).sum()),
            'visible_places': sorted({world.place_order[i] for i in range(0)}),
        })
        # which places they can see any part of
        vp = []
        for pid in world.place_order:
            if (seen & world.places[pid].cells).any():
                vp.append(pid)
        out[-1]['visible_places'] = vp[:40]
    world.artifacts['perception'] = out
    return out


# -------------------------------------------- 36 Mechanical Infrastructure

@feature('env.machines', 36, 'Mechanical Infrastructure Map',
         'Environment & Infrastructure',
         consumes=('places', 'topology.logistics.supply'),
         produces=('topology.infrastructure.power', 'artifacts.machines'))
def mechanical_infrastructure(world):
    """Machines with real dependencies: a pump needs power, a lift needs the
    pump, and cutting the power cascades in a defined order."""
    power = world.topology('infrastructure.power', directed=True)
    rng = world.rng('machines')
    machines = []

    biggest = sorted(world.places.values(), key=lambda p: -p.area)
    generators = ([p for p in world.places_of(function='forge')] or
                  [p for p in world.places_of('plaza')][:2] or biggest[:2])
    pumps = [p for p in world.places_of(function='cistern')][:6] or biggest[2:6]
    lifts = [p for p in world.places.values() if 'ladder_tile' in str(p.function)][:4]

    for g in generators[:4]:
        machines.append({'id': f'mch_gen_{g.id[-6:]}', 'kind': 'generator',
                         'place': g.id, 'requires': [], 'output': 'power',
                         'condition': round(float(rng.random() * 0.5 + 0.5), 2)})
        g.tags.add('machine')
    gen_ids = [m['id'] for m in machines]
    for p in pumps:
        mid = f'mch_pump_{p.id[-6:]}'
        req = gen_ids[:1]
        machines.append({'id': mid, 'kind': 'pump', 'place': p.id,
                         'requires': req, 'output': 'water pressure',
                         'condition': round(float(rng.random() * 0.6 + 0.4), 2)})
        p.tags.add('machine')
        for r in req:
            power.add(r, mid, 1.0)
    pump_ids = [m['id'] for m in machines if m['kind'] == 'pump']
    for i, p in enumerate((world.places_of('plaza') or biggest)[:3]):
        mid = f'mch_lift_{p.id[-6:]}'
        req = pump_ids[:1] or gen_ids[:1]
        machines.append({'id': mid, 'kind': 'lift', 'place': p.id,
                         'requires': req, 'output': 'vertical access',
                         'condition': round(float(rng.random() * 0.6 + 0.4), 2)})
        for r in req:
            power.add(r, mid, 1.0)

    # what fails if each generator stops
    cascade = {}
    for g in gen_ids:
        cascade[g] = sorted(set(power.shortest_paths(g).keys()) - {g})
    world.artifacts['machines'] = {'machines': machines, 'cascade': cascade}
    return machines


# ---------------------------------------------------------------- 37 Crisis

CRISES = ('fire', 'flood', 'siege', 'blackout', 'contagion')


@feature('nar.crisis', 37, 'Crisis Map', 'Incidents & Narrative',
         consumes=('places', 'fields.exposure', 'artifacts.machines'),
         produces=('artifacts.crises',))
def crisis_map(world):
    """Scenario overlays: for each crisis, where it starts, how it spreads and
    which places offer shelter or contain hazards."""
    arr = np.array(world.tilemap.base, dtype=object).astype(str)
    walk = world.passable()
    flow = surface_property(world, 'flow')
    heat = surface_property(world, 'heat')
    exposure = world.field('exposure', 0.3)
    elevation = world.field('elevation', 0.5)

    out = {}
    for crisis in CRISES:
        if crisis == 'fire':
            seed = (arr == 'lava').astype(np.float32)
            if seed.max() <= 0:
                # nothing molten: fire starts in the flammable fabric instead,
                # otherwise the whole scenario is identically zero
                seed = np.isin(arr, ['bookshelf', 'bridge', 'rubble']).astype(np.float32)
            if seed.max() <= 0 and world.place_order:
                hot = max(world.places.values(), key=lambda p: p.area)
                seed = np.zeros_like(seed); seed[hot.cells] = 1.0
            fuel = np.clip(heat / 3.0, 0.15, 1.0) * walk
            spread = propagate_field(seed, fuel, iterations=30, decay=0.85)
        elif crisis == 'flood':
            seed = (arr == 'water').astype(np.float32)
            spread = propagate_field(seed, flow * (1.0 - elevation),
                                     iterations=34, decay=0.90)
        elif crisis == 'siege':
            spread = exposure.copy()
        elif crisis == 'blackout':
            spread = 1.0 - world.field('light', 0.5)
        else:
            crowd = world.field('crowding', 0.3)
            spread = propagate_field(crowd * walk, flow, iterations=22, decay=0.86)
        spread = np.clip(spread, 0, 1).astype(np.float32)
        world.set_field(f'crisis_{crisis}', spread)

        danger, refuge = [], []
        for pid in world.place_order:
            p = world.places[pid]
            if not p.area:
                continue
            v = float(spread[p.cells].mean())
            if v > 0.55:
                danger.append(pid)
            elif v < 0.08:
                refuge.append(pid)
        out[crisis] = {'danger_places': danger[:30], 'refuge_places': refuge[:30],
                       'mean_severity': round(float(spread.mean()), 3)}
    world.artifacts['crises'] = out
    return out


# ------------------------------------------------------ 38 Emotional Topology

AFFECTS = ('dread', 'awe', 'safety', 'melancholy', 'tension')


@feature('plc.emotion', 38, 'Emotional Topology', 'Place Semantics',
         consumes=('places', 'fields.light', 'fields.sound_pressure',
                   'fields.memory'),
         produces=('fields.dread', 'fields.awe', 'fields.safety',
                   'artifacts.pacing'))
def emotional_topology(world):
    """Atmosphere fields and locations ranked by their suspense values.

    Dread combines darkness, isolation, and memory; awe combines scale and
    light. The pacing artifact lists the lowest and highest values.
    """
    light = world.field('light', 0.5)
    sound = world.field('sound_pressure', 0.2)
    memory = world.field('memory', 0.0)
    crowding = world.field('crowding', 0.3)
    walk = world.passable()
    openness = tf.distance_to_mask(~walk).a
    openness = openness / max(float(openness.max()), 1e-6)

    dread = np.clip(0.45 * (1 - light) + 0.25 * memory +
                    0.20 * (1 - crowding) + 0.10 * (1 - sound), 0, 1)
    awe = np.clip(0.5 * openness + 0.3 * light + 0.2 * sound, 0, 1)
    safety = np.clip(0.4 * light + 0.35 * crowding +
                     0.25 * world.field('guard_attention', 0.2), 0, 1)
    melancholy = np.clip(0.5 * world.field('wear', 0.3) + 0.5 * (1 - crowding), 0, 1)
    tension = np.clip(0.5 * dread + 0.5 * (1 - safety), 0, 1)

    for name, arr_ in (('dread', dread), ('awe', awe), ('safety', safety),
                       ('melancholy', melancholy), ('tension', tension)):
        world.set_field(name, (arr_ * walk).astype(np.float32))

    seq = []
    for pid in world.place_order:
        p = world.places[pid]
        if p.area < 6:
            continue
        seq.append({'place': pid,
                    'dread': round(float(dread[p.cells].mean()), 3),
                    'awe': round(float(awe[p.cells].mean()), 3),
                    'safety': round(float(safety[p.cells].mean()), 3)})
    seq.sort(key=lambda r: r['dread'])
    world.artifacts['pacing'] = {
        'calmest': seq[:5], 'most_suspenseful': seq[-5:],
        'contrast': round((seq[-1]['dread'] - seq[0]['dread']) if seq else 0.0, 3)}
    return seq


# ------------------------------------------- 39 Architectural Archaeology

ERAS = ('foundation', 'expansion', 'occupation', 'decline', 'reuse')


@feature('plc.archaeology', 39, 'Architectural Archaeology', 'Place Semantics',
         consumes=('places', 'fields.age'),
         produces=('fields.era', 'artifacts.build_phases'))
def architectural_archaeology(world):
    """Build phases and preserved fragments.

    Older fabric survives where nothing has needed to change: deep, low-traffic,
    structurally sound. That gives ruins a reason to be where they are.
    """
    age = world.field('age', 0.5)
    wear = world.field('wear', 0.3)
    walk = world.passable()
    depth = tf.distance_to_mask(~walk).a
    depth = depth / max(float(depth.max()), 1e-6)

    era_value = np.clip(0.55 * age + 0.25 * depth + 0.20 * (1 - wear), 0, 1)
    era_index = np.clip((era_value * len(ERAS)).astype(int), 0, len(ERAS) - 1)
    world.set_field('era', (era_index / (len(ERAS) - 1)).astype(np.float32),
                    'era index normalised')

    phases = defaultdict(list)
    for pid in world.place_order:
        p = world.places[pid]
        if not p.area:
            continue
        e = int(np.round(era_index[p.cells].mean()))
        name = ERAS[min(e, len(ERAS) - 1)]
        p.meta['era'] = name
        phases[name].append(pid)
        if name in ('foundation', 'expansion'):
            p.tags.add('ancient fabric')
    world.artifacts['build_phases'] = {k: v[:40] for k, v in phases.items()}
    world.artifacts['build_order'] = list(ERAS)
    return dict(phases)
