"""Overworld: one heightfield read as climate, biomes, rivers, settlements and roads.

Map family 22. A landmass is raised from warped noise and eroded. Latitude and
altitude set the temperature; a wind sweep rains moisture out on windward
slopes and leaves rain shadows behind the ranges; Whittaker-style bands turn
temperature and moisture into biomes; rivers follow rain-weighted drainage to
the sea; settlements take the best sites tier by tier; and roads are surveyed
between them with a hiking-cost A* that shares trunks and bridges rivers.

Everything is interpreted into the existing tile vocabulary, so the runtime,
the socket WFC assembler, the world compiler and the six styles consume an
overworld unchanged. The continuous layers ride along as fields, and the
settlements and routes as plain records.

    import tilegen_overworld as ow
    world = ow.build_overworld({'size': 128, 'seed': 7})       # every layer
    tm = ow.gen_overworld({'size': 96, 'seed': 7}, None)        # the grammar
    ow.render_overworld(tm).save('overworld.png')               # a map picture

Siting follows central-place spacing (Christaller 1933) with soft suitability
terms; route cost follows Tobler's hiking function (Tobler 1993); the biome
bands follow Whittaker's temperature-precipitation diagram (Whittaker 1975).
Thresholds are taken by target ratio, so every seed yields a usable map.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field as dc_field

import numpy as np

import tilegen_fields as tf
import tilegen_grammars as tg
from tilegen_grammars import (BRIDGE, COURT, FLOOR, PLATFORM, RUBBLE, WALL, WATER,
                              attach, build, dims, grammar_seed, passable_mask,
                              size_scale)

F32 = np.float32


# ------------------------------------------------------------------ biomes

@dataclass(frozen=True)
class Biome:
    id: int
    name: str
    tile: str
    color: tuple
    label: str


BIOMES = (
    Biome(0, 'deep_ocean', WATER, (24, 52, 94), 'Deep ocean'),
    Biome(1, 'ocean', WATER, (38, 86, 140), 'Ocean'),
    Biome(2, 'lake', WATER, (60, 120, 170), 'Lake'),
    Biome(3, 'river', WATER, (66, 128, 178), 'River'),
    Biome(4, 'beach', RUBBLE, (220, 206, 160), 'Beach'),
    Biome(5, 'desert', RUBBLE, (226, 200, 138), 'Desert'),
    Biome(6, 'savanna', FLOOR, (186, 178, 98), 'Savanna'),
    Biome(7, 'shrubland', FLOOR, (158, 154, 106), 'Shrubland'),
    Biome(8, 'grassland', FLOOR, (140, 172, 90), 'Grassland'),
    Biome(9, 'temperate_forest', COURT, (80, 132, 70), 'Temperate forest'),
    Biome(10, 'temperate_rainforest', COURT, (52, 114, 76), 'Temperate rainforest'),
    Biome(11, 'tropical_forest', COURT, (68, 142, 60), 'Tropical seasonal forest'),
    Biome(12, 'jungle', COURT, (36, 106, 52), 'Tropical rainforest'),
    Biome(13, 'taiga', COURT, (74, 108, 88), 'Taiga'),
    Biome(14, 'tundra', FLOOR, (154, 162, 144), 'Tundra'),
    Biome(15, 'marsh', COURT, (94, 126, 102), 'Marsh'),
    Biome(16, 'alpine', WALL, (134, 126, 116), 'Alpine'),
    Biome(17, 'snow', WALL, (236, 240, 244), 'Snow and ice'),
)
BIOME = {b.name: b.id for b in BIOMES}
WATER_BIOMES = (BIOME['deep_ocean'], BIOME['ocean'], BIOME['lake'], BIOME['river'])
FARMLAND = (BIOME['grassland'], BIOME['savanna'], BIOME['shrubland'])

# Cover = biome with the built environment stamped over it.
ROAD, TOWN = 18, 19
COVER_NAMES = tuple(b.name for b in BIOMES) + ('road', 'town')
COVER_COLORS = {**{b.id: b.color for b in BIOMES}, ROAD: (150, 112, 70), TOWN: (196, 92, 70)}

# Temperature at the top and bottom edge of the map (0 polar .. 1 equatorial);
# 'pole_to_pole' puts the equator across the middle.
CLIMATES = {
    'temperate': (0.20, 0.80),
    'tropical': (0.62, 0.98),
    'polar': (0.04, 0.50),
    'pole_to_pole': None,
}
WINDS = ('west', 'east', 'north', 'south')       # where the prevailing wind comes from
SHAPES = ('continent', 'archipelago', 'highlands', 'peninsula')


def biome_palette():
    """{cover id: (r, g, b)} for styles and renderers (biomes, then road and town)."""
    return dict(COVER_COLORS)


# ------------------------------------------------------------- small helpers

def _bell(x, lo, hi, soft):
    """1 inside [lo, hi], falling linearly to 0 over `soft` outside it."""
    x = np.asarray(x, dtype=np.float64)
    below = np.clip(1.0 - (lo - x) / soft, 0.0, 1.0)
    above = np.clip(1.0 - (x - hi) / soft, 0.0, 1.0)
    return np.where(x < lo, below, np.where(x > hi, above, 1.0))


def _box_sum(a, r):
    """Sum over the (2r+1)^2 window around each cell, edges padded by replication."""
    a = np.asarray(a, dtype=np.float64)
    p = np.pad(a, r, mode='edge')
    c = np.pad(p.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    k = 2 * r + 1
    return c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]


def _box_mean(a, r):
    return _box_sum(a, r) / float((2 * r + 1) ** 2)


def _box_max(a, r):
    a = np.asarray(a, dtype=np.float64)
    h, w = a.shape
    p = np.pad(a, r, mode='edge')
    rows = np.max([p[:, r + d:r + d + w] for d in range(-r, r + 1)], axis=0)
    return np.max([rows[r + d:r + d + h, :] for d in range(-r, r + 1)], axis=0)


def _rank_normalize(a, mask):
    """Values under `mask` replaced by their rank in [0, 1]; the rest keep 0."""
    out = np.zeros(a.shape, dtype=np.float64)
    vals = a[mask]
    if vals.size:
        order = np.argsort(vals, kind='stable')
        ranks = np.empty(vals.size, dtype=np.float64)
        ranks[order] = np.linspace(0.0, 1.0, vals.size)
        out[mask] = ranks
    return out


def _cut_for_share(values, share):
    """Cutoff leaving `share` of `values` at or above it (thresholds by ratio)."""
    if values.size == 0:
        return math.inf
    return float(np.quantile(values, 1.0 - float(np.clip(share, 1e-4, 0.9999))))


# ---------------------------------------------------------------- landmass

def landmass(w, h, seed, shape='continent', scale=2.6, warp=0.55):
    """Raw elevation in [0, 1] whose high ground forms the requested shape."""
    if shape not in SHAPES:
        raise ValueError(f"shape must be one of {SHAPES}, got {shape!r}")
    base = tf.fbm(w, h, scale, 6, seed=seed).normalized()
    ridge = tf.ridged(w, h, scale * 1.4, 4, seed=seed + 31).normalized()
    e = base.blend(ridge, 0.35)
    if warp:
        swirl = tf.fbm(w, h, scale * 0.7, 3, seed=seed + 57).curl()
        e = e.warp(swirl, warp * min(w, h) * 0.06)
    e = e.normalized()

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    nx = (xx + 0.5) / w * 2.0 - 1.0
    ny = (yy + 0.5) / h * 2.0 - 1.0
    wobble = tf.fbm(w, h, 1.6, 3, seed=seed + 91).normalized().a.astype(np.float64)
    if shape == 'continent':
        r = np.hypot(nx * 0.92, ny) * (0.82 + 0.36 * wobble)
        fall = 1.0 - np.clip((r - 0.35) / 0.70, 0.0, 1.0) ** 1.6
    elif shape == 'peninsula':
        rng = np.random.default_rng(tf.seed_stream(seed, 'geo.overworld.peninsula'))
        ang = rng.uniform(0.0, 2.0 * math.pi)
        along = nx * math.cos(ang) + ny * math.sin(ang)
        across = -nx * math.sin(ang) + ny * math.cos(ang)
        neck = np.clip((along + 1.0) / 2.0, 0.0, 1.0)
        width = 0.75 - 0.45 * neck + 0.25 * (wobble - 0.5)
        fall = np.clip(1.0 - np.abs(across) / np.maximum(width, 0.12), 0.0, 1.0) ** 0.7
        fall *= np.clip((1.15 - along) / 0.45, 0.0, 1.0)
    elif shape == 'archipelago':
        rad = max(6.0, min(w, h) / 6.5)
        sites = tf.poisson_disc(w, h, rad, seed=tf.seed_stream(seed, 'geo.overworld.isles'),
                                margin=int(rad * 0.6))
        fall = np.zeros((h, w))
        if len(sites):
            rng = np.random.default_rng(tf.seed_stream(seed, 'geo.overworld.isle_size'))
            ragged = tf.fbm(w, h, 5.0, 4, seed=seed + 113).normalized().a.astype(np.float64)
            for (sx, sy), k in zip(sites, rng.uniform(0.45, 1.25, size=len(sites))):
                d = np.hypot(xx - sx, yy - sy) / (rad * k) * (0.70 + 0.60 * ragged)
                fall = np.maximum(fall, np.clip(1.0 - d, 0.0, 1.0) ** 1.2)
        fall = fall * (0.70 + 0.6 * wobble)
    else:                                       # highlands: one long range across the land
        rng = np.random.default_rng(tf.seed_stream(seed, 'geo.overworld.range'))
        ang = rng.uniform(-0.6, 0.6) + math.pi / 2.0
        off = rng.uniform(-0.25, 0.25)
        across = nx * math.cos(ang) + ny * math.sin(ang) - off
        r = np.hypot(nx * 0.95, ny * 0.95) * (0.85 + 0.3 * wobble)
        fall = 1.0 - np.clip((r - 0.45) / 0.65, 0.0, 1.0) ** 1.4
        belt = np.exp(-(across / 0.16) ** 2)
        e = (e.a * 0.65 + 0.55 * belt * (0.6 + 0.4 * ridge.a)).astype(F32)
        e = tf.Field(e)
    out = (e.a.astype(np.float64) * (0.30 + 0.70 * fall)) + 0.35 * fall
    return tf.Field(out.astype(F32), f'landmass({shape})').normalized()


# ----------------------------------------------------------------- climate

def temperature(height, sea_level, seed, climate='temperate', lapse=0.45, jitter=0.06):
    """0 (polar) .. 1 (equatorial): latitude band, minus a lapse rate with altitude."""
    if climate not in CLIMATES:
        raise ValueError(f"climate must be one of {tuple(CLIMATES)}, got {climate!r}")
    h, w = height.shape
    y = (np.arange(h, dtype=np.float64) + 0.5) / h
    band = CLIMATES[climate]
    if band is None:
        lat = 1.0 - np.abs(2.0 * y - 1.0)
    else:
        lat = band[0] + (band[1] - band[0]) * y
    alt = np.clip((height - sea_level) / max(1.0 - sea_level, 1e-6), 0.0, 1.0)
    noise = tf.fbm(w, h, 3.0, 3, seed=seed).normalized().a.astype(np.float64) - 0.5
    return np.clip(lat[:, None] - lapse * alt + 2.0 * jitter * noise, 0.0, 1.0)


def _to_wind(a, wind):
    """View `a` so the wind blows along +column (from column 0)."""
    if wind == 'west':
        return a
    if wind == 'east':
        return a[:, ::-1]
    if wind == 'north':
        return a.T
    return a.T[:, ::-1]                         # south


def _from_wind(a, wind):
    if wind == 'west':
        return a
    if wind == 'east':
        return a[:, ::-1]
    if wind == 'north':
        return a.T
    return a[:, ::-1].T


def precipitation(height, sea, wind='west', rain=0.030, orographic=7.0, recharge=0.30,
                  carry=0.55):
    """Orographic rainfall from one prevailing wind, in [0, 1].

    Air enters at the windward edge carrying `carry` humidity, recharges over
    the sea and rains out over land: a little everywhere (`rain`) and much more
    where it is forced uphill (`orographic` times the rise). Air that has
    crossed a range arrives dry, so its lee lies in a rain shadow. Humidity
    mixes sideways between neighbouring rows each step, which keeps the
    shadows soft instead of streaked.
    """
    if wind not in WINDS:
        raise ValueError(f"wind must be one of {WINDS}, got {wind!r}")
    H = np.ascontiguousarray(_to_wind(np.asarray(height, dtype=np.float64), wind))
    S = np.ascontiguousarray(_to_wind(np.asarray(sea, dtype=bool), wind))
    rows, cols = H.shape
    q = np.full(rows, float(carry))
    out = np.zeros((rows, cols))
    prev = H[:, 0].copy()
    for c in range(cols):
        hc, sc = H[:, c], S[:, c]
        rise = np.maximum(hc - prev, 0.0)
        rate = np.where(sc, 0.0, np.clip(rain + orographic * rise, 0.0, 0.9))
        p = q * rate
        q = q - p
        q = np.where(sc, q + recharge * (1.0 - q), q)
        qp = np.pad(q, 1, mode='edge')
        q = 0.25 * qp[:-2] + 0.5 * qp[1:-1] + 0.25 * qp[2:]
        out[:, c] = p
        prev = hc
    out = _from_wind(out, wind)
    out = tf.Field(out.astype(F32)).blur(1).a.astype(np.float64)
    peak = float(out.max())
    return out / peak if peak > 0 else out


def moisture(precip, land, fresh, aridity=0.0, riparian=0.45, reach=2.5):
    """Land moisture in [0, 1]: rainfall plus a riparian boost near rivers and
    lakes. Half of it is the cell's rank over the land, so every map spans dry
    to wet ('tags relative to the world'); half is absolute, so a rain-soaked
    archipelago still reads lush. `aridity` skews it (-1 lush .. +1 parched)."""
    near = np.zeros(land.shape)
    if fresh.any():
        d = tf.distance_to_mask(fresh).a.astype(np.float64)
        near = np.exp(-d / max(reach, 0.5))
    wet = np.asarray(precip, dtype=np.float64) + riparian * near
    ref = float(np.quantile(wet[land], 0.9)) if land.any() else 1.0
    m = 0.55 * _rank_normalize(wet, land) + 0.45 * np.clip(wet / max(ref, 1e-9), 0.0, 1.0) * land
    gamma = 2.0 ** (float(np.clip(aridity, -1.0, 1.0)) * 1.5)
    return np.clip(m, 0.0, 1.0) ** gamma


# --------------------------------------------------------------- hydrology

def drainage(height, rain, seed):
    """Pit-filled surface, D8 receivers and rain-weighted flow accumulation."""
    h, w = height.shape
    jitter = np.random.default_rng(tf.seed_stream(seed, 'env.overworld.tiebreak')).random(
        (h, w)) * 1e-3
    filled = tf.fill_depressions(np.asarray(height, dtype=np.float64) + jitter, 2e-4).a
    _, receiver = tf.flow_accumulation(filled)
    order = np.argsort(-filled.ravel(), kind='stable').tolist()
    acc = (0.05 + np.asarray(rain, dtype=np.float64).ravel()).tolist()
    rec = receiver.tolist()
    for i in order:
        r = rec[i]
        if r != i:
            acc[r] += acc[i]
    return np.asarray(filled, dtype=np.float64), receiver, np.asarray(acc).reshape(h, w)


def lakes_from(basin_height, sea, land_cells, lake_ratio=0.02, min_cells=4):
    """Lakes in the basins of the uneroded landmass, deepest first, kept to about
    `lake_ratio` of the land. Erosion drains almost every pit, so the basins
    are read from the surface before it ran, as glacial and tectonic hollows."""
    raw = np.asarray(basin_height, dtype=np.float64)
    depth = np.where(sea, 0.0, np.asarray(tf.fill_depressions(raw, 2e-4).a, dtype=np.float64) - raw)
    wet = depth > 1e-3
    if not wet.any() or lake_ratio <= 0:
        return np.zeros(sea.shape, dtype=bool)
    budget = max(1, int(lake_ratio * land_cells))
    vals = depth[wet]
    cut = float(np.sort(vals)[-budget]) if vals.size > budget else float(vals.min())
    lake = (depth >= cut) & wet
    lake = tf.remove_small_regions(lake, min_cells)
    lake, _ = tf.smooth_lakes(lake, passes=2)
    return lake & ~sea


def trace_rivers(acc, receiver, sea, lake, land, river_ratio=0.03):
    """Channels from the wettest `river_ratio` of the land down to sea or lake,
    4-connected by construction (D8 corners are filled)."""
    vals = acc[land & ~lake]
    if vals.size == 0:
        return np.zeros(acc.shape, dtype=bool)
    sources = (acc >= _cut_for_share(vals, river_ratio)) & land & ~lake
    stop = sea | lake
    river = tf.trace_channels(acc, receiver, acc.shape, sources, stop_mask=stop) & ~stop
    # coastal stubs of a cell or two read as stray marks, not rivers
    return tf.remove_small_regions(river, max(3, int(round(min(acc.shape) / 32))))


# ------------------------------------------------------------------ biomes

def peak_level(height, sea, mountain_ratio=0.09, min_wall=0.03):
    """Height above which land counts as mountain: the highest `mountain_ratio`
    of the land, and never less than `min_wall` of the whole map."""
    land_h = np.asarray(height)[~sea]
    if not land_h.size:
        return math.inf
    share = max(mountain_ratio, min_wall * height.size / land_h.size)
    return _cut_for_share(land_h, min(share, 0.5))


def classify_biomes(height, sea_level, temp, moist, sea, lake, river, *,
                    mountain_ratio=0.09, min_wall=0.03, seed=0):
    """Biome id per cell (int16) from temperature and moisture bands, with
    altitude, coast and wetland overrides taken by ratio."""
    h, w = height.shape
    land = ~sea
    out = np.full((h, w), BIOME['grassland'], dtype=np.int16)
    T, M = temp, moist

    def put(mask, name):
        out[mask & land] = BIOME[name]

    # Whittaker bands, coldest first; later rows overwrite earlier ones.
    cold = T < 0.25
    put(cold & (M < 0.45), 'tundra')
    put(cold & (M >= 0.45), 'taiga')
    cool = (T >= 0.25) & (T < 0.45)
    put(cool & (M < 0.20), 'shrubland')
    put(cool & (M >= 0.20) & (M < 0.45), 'grassland')
    put(cool & (M >= 0.45), 'temperate_forest')
    put(cool & (T < 0.33) & (M >= 0.55), 'taiga')
    put(cool & (M >= 0.82), 'temperate_rainforest')
    warm = (T >= 0.45) & (T < 0.70)
    put(warm & (M < 0.15), 'desert')
    put(warm & (M >= 0.15) & (M < 0.32), 'shrubland')
    put(warm & (M >= 0.32) & (M < 0.55), 'grassland')
    put(warm & (M >= 0.55) & (M < 0.80), 'temperate_forest')
    put(warm & (M >= 0.80), 'temperate_rainforest')
    hot = T >= 0.70
    put(hot & (M < 0.18), 'desert')
    put(hot & (M >= 0.18) & (M < 0.42), 'savanna')
    put(hot & (M >= 0.42) & (M < 0.72), 'tropical_forest')
    put(hot & (M >= 0.72), 'jungle')

    land_h = height[land]
    slope = tf.Field(height.astype(F32)).slope().a.astype(np.float64)
    if land_h.size:
        low = land & (height <= np.quantile(land_h, 0.22))
        ocean_rim = tf.dilate(sea, 1) & land
        put(ocean_rim & low & (T >= 0.22), 'beach')
        flat = slope <= np.quantile(slope[land], 0.35)
        fresh = tf.distance_to_mask(river | lake).a <= 2.0 if (river | lake).any() \
            else np.zeros((h, w), dtype=bool)
        put(low & flat & fresh & (M >= 0.72) & (T >= 0.2) & ~ocean_rim, 'marsh')
        peak = land & (height >= peak_level(height, sea, mountain_ratio, min_wall))
        put(peak, 'alpine')
        put(peak & (T < 0.16), 'snow')
    put(land & (T < 0.05), 'snow')

    sea_h = height[sea]
    if sea_h.size:
        deep = sea & (height <= np.quantile(sea_h, 0.55))
        out[sea] = BIOME['ocean']
        out[deep] = BIOME['deep_ocean']
    out[lake] = BIOME['lake']
    out[river & land] = BIOME['river']
    return out


# ------------------------------------------------------------- settlements

TIERS = ('hamlet', 'village', 'town', 'city')
SPACING = {3: 0.0, 2: 0.20, 1: 0.11, 0: 0.065}   # min distance to any site, share of the map side
RING = (0.10, 0.30)                               # preferred distance to the nearest larger site
POPULATION = {3: (4800, 10500), 2: (900, 2800), 1: (150, 600), 0: (25, 100)}
WEIGHTS = dict(water=2.0, flat=1.7, farm=2.0, mix=0.6, harbour=1.3, shelter=0.8,
               river=1.4, central=0.7, ring=0.9, climate=1.2)

_PREFIX = ('Ash', 'Oak', 'Thorn', 'Wolf', 'Raven', 'Elder', 'Stone', 'Frost', 'Amber',
           'Marsh', 'Iron', 'Salt', 'Wind', 'Black', 'White', 'Red', 'Green', 'Gold',
           'Silver', 'Elm', 'Briar', 'Hawk', 'Fern', 'Mist', 'Copper', 'Willow', 'Heron',
           'Bram', 'Cold', 'Sun', 'Moon', 'Dun', 'Kings', 'High', 'Long', 'Fair', 'Grey',
           'Old', 'Hazel', 'Swan', 'Wren', 'Larch', 'Cinder', 'Rook', 'Oxen', 'Bell')
_SUFFIX = {
    'port': ('port', 'haven', 'mouth', 'wick', 'strand', 'quay'),
    'river': ('ford', 'bridge', 'brook', 'mere', 'water', 'weir'),
    'hill': ('hill', 'crest', 'mount', 'watch', 'tor', 'fell'),
    'market': ('ton', 'ham', 'stead', 'by', 'bury', 'field', 'worth', 'ley'),
}


def _namer(seed):
    rng = np.random.default_rng(tf.seed_stream(seed, 'soc.overworld.names'))
    used = set()

    def name(setting):
        suffixes = _SUFFIX.get(setting, _SUFFIX['market'])
        for _ in range(64):
            n = _PREFIX[int(rng.integers(len(_PREFIX)))] + suffixes[int(rng.integers(len(suffixes)))]
            if n not in used:
                used.add(n)
                return n
        n = f"{_PREFIX[int(rng.integers(len(_PREFIX)))]}{suffixes[0]} {len(used) + 1}"
        used.add(n)
        return n
    return name


def site_terms(ow):
    """Suitability terms per cell as a dict of arrays, plus the candidate mask.

    Soft terms (each 0..1): fresh water (river or lake, not the sea) near but
    not on the bank, level ground,
    farmland share and land-cover variety nearby, a harbour with a sheltered
    shore (a bay rather than open coast), river size nearby (confluences rank
    higher), a central position and a liveable climate.
    """
    h, w = ow.shape
    s = size_scale(w, h)
    land = ~ow.sea
    water_any = ow.sea | ow.lake | ow.river
    fresh = ow.lake | ow.river
    blocked = np.isin(ow.biome, (BIOME['alpine'], BIOME['snow'])) | water_any
    far = np.full((h, w), float(max(h, w)))
    d_wet = tf.distance_to_mask(water_any).a.astype(np.float64) if water_any.any() else far
    d_fresh = tf.distance_to_mask(fresh).a.astype(np.float64) if fresh.any() else far
    d_sea = tf.distance_to_mask(ow.sea).a.astype(np.float64) if ow.sea.any() \
        else np.full((h, w), float(max(h, w)))
    r = max(2, int(round(3 * s)))
    slope = ow.slope
    land_slope = slope[land & ~blocked]
    ref = float(np.quantile(land_slope, 0.75)) if land_slope.size else 1.0
    flat = np.clip(1.0 - _box_mean(slope, r) / max(ref, 1e-6), 0.0, 1.0)
    farm = _box_mean(np.isin(ow.biome, FARMLAND), r)
    kinds = sum((_box_max(ow.biome == b.id, r) > 0).astype(np.float64)
                for b in BIOMES if b.id not in WATER_BIOMES)
    mix = np.clip((kinds - 1.0) / 3.0, 0.0, 1.0)
    water = _bell(d_fresh, 1.5 * s, 4.5 * s, 4.0 * s)
    harbour = _bell(d_sea, 0.0, 1.5 * s, 1.5 * s) * (ow.height < ow.sea_level + 0.10)
    ring_r = max(3.0, 5.0 * s)
    yy, xx = np.mgrid[0:h, 0:w]
    sea_share = np.zeros((h, w))
    for k in range(16):
        a = k * math.pi / 8.0
        oy = np.clip(yy + int(round(ring_r * math.sin(a))), 0, h - 1)
        ox = np.clip(xx + int(round(ring_r * math.cos(a))), 0, w - 1)
        sea_share += ow.sea[oy, ox]
    shelter = _bell(sea_share / 16.0, 0.15, 0.55, 0.25) * harbour
    fresh_acc = np.where(ow.river | ow.lake, ow.accumulation, 0.0)
    top = float(fresh_acc.max()) or 1.0
    river = np.sqrt(np.clip(_box_max(fresh_acc, r) / top, 0.0, 1.0))
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    central = 1.0 - 0.5 * np.hypot(xx - cx, yy - cy) / max(math.hypot(cx, cy), 1.0)
    climate = _bell(ow.temperature, 0.30, 0.80, 0.25) * _bell(ow.moisture, 0.20, 0.85, 0.30)
    ring_mean = _box_mean(ow.height, 2 * r) * 1.0
    prominence = ow.height - ring_mean
    edge = max(2, int(round(min(w, h) * 0.05)))
    cand = land & ~blocked & (d_wet >= 1.0)
    cand[:edge, :] = cand[-edge:, :] = False
    cand[:, :edge] = cand[:, -edge:] = False
    if land_slope.size:
        cand &= slope <= np.quantile(land_slope, 0.80)
    return dict(water=water, flat=flat, farm=farm, mix=mix, harbour=harbour,
                shelter=shelter, river=river, central=central, climate=climate,
                prominence=prominence, d_wet=d_wet), cand


def default_counts(w, h):
    """Settlements per tier for a map of this size (cities, towns, villages, hamlets)."""
    area = (w * h) / float(96 * 96)
    return (1, max(1, int(round(3 * area ** 0.5))), max(2, int(round(6 * area ** 0.6))),
            max(2, int(round(7 * area ** 0.6))))


def place_settlements(ow, counts=None, step=2):
    """Settlements as dicts, best site first (city, towns, villages, hamlets).

    Candidates are ranked by the weighted terms plus a little seeded jitter;
    each tier is taken greedily with a minimum spacing, and a smaller tier
    prefers to sit 10-30% of the map side from the nearest larger place, so
    towns feed on cities instead of piling on them.
    """
    h, w = ow.shape
    terms, cand = site_terms(ow)
    rng = np.random.default_rng(tf.seed_stream(ow.seed, 'soc.overworld.sites'))
    jitter = rng.uniform(0.0, 0.25, size=(h, w))
    score = sum(WEIGHTS[k] * terms[k] for k in
                ('water', 'flat', 'farm', 'mix', 'harbour', 'river', 'central', 'climate'))
    score = score + WEIGHTS['shelter'] * terms['shelter'] + jitter
    grid = np.zeros((h, w), dtype=bool)
    grid[::step, ::step] = True
    ys, xs = np.nonzero(cand & grid)
    if not len(xs):
        ys, xs = np.nonzero(cand)
    if not len(xs):
        return []
    sc = score[ys, xs]
    order = np.lexsort((xs, ys, -sc))
    ys, xs, sc = ys[order], xs[order], sc[order]
    size = float(max(w, h))
    s = size_scale(w, h)
    radius = {3: 2.0 * s, 2: 1.4 * s, 1: 0.8 * s, 0: 0.0}
    counts = counts or default_counts(w, h)
    name = _namer(ow.seed)
    pop_rng = np.random.default_rng(tf.seed_stream(ow.seed, 'soc.overworld.population'))
    placed = []
    for tier, count in zip((3, 2, 1, 0), counts):
        need = SPACING[tier] * size
        for _ in range(int(count)):
            ok = np.ones(len(xs), dtype=bool)
            for p in placed:
                gap = max(need, p['radius'] + radius[tier] + SPACING[0] * size * 0.6)
                ok &= np.hypot(xs - p['x'], ys - p['y']) >= gap
            idx = np.flatnonzero(ok)[:400]
            if not len(idx):
                break
            total = sc[idx].copy()
            bigger = [p for p in placed if p['tier'] > tier]
            if bigger:
                d = np.min([np.hypot(xs[idx] - p['x'], ys[idx] - p['y']) for p in bigger],
                           axis=0) / size
                total = total + WEIGHTS['ring'] * _bell(d, RING[0], RING[1], 0.15)
            best = idx[int(np.argmax(total))]
            x, y = int(xs[best]), int(ys[best])
            setting = ('port' if terms['harbour'][y, x] > 0.5 else
                       'river' if terms['d_wet'][y, x] < 3.0 * s and terms['river'][y, x] > 0.15
                       else 'hill' if terms['prominence'][y, x] > 0.035 else 'market')
            lo, hi = POPULATION[tier]
            placed.append(dict(
                id=f"s{len(placed)}", name=name(setting), tier=tier, kind=TIERS[tier],
                x=x, y=y, setting=setting, radius=round(radius[tier], 2),
                population=int(pop_rng.integers(lo, hi + 1)),
                biome=BIOMES[int(ow.biome[y, x])].name, score=round(float(sc[best]), 3)))
    return placed


# ------------------------------------------------------------------- roads

MODES = {'road': 1.0, 'lane': 1.25, 'track': 1.6}
TOBLER = 3.5
_BIOME_COST = {
    'desert': 0.5, 'savanna': 0.1, 'shrubland': 0.15, 'grassland': 0.0, 'beach': 0.3,
    'temperate_forest': 0.35, 'temperate_rainforest': 0.6, 'tropical_forest': 0.45,
    'jungle': 0.9, 'taiga': 0.45, 'tundra': 0.3, 'marsh': 2.0, 'alpine': 4.0, 'snow': 6.0,
    'river': 2.4,
}


class _Survey:
    """Hiking-cost A* over the 4-neighbour grid, shared by every route of one map.

    A step costs Tobler's walking-time factor for its grade (the optimum is a
    slight downhill), plus the ground it crosses, plus a river crossing priced
    by how wide the river looks; sea and lakes cannot be crossed. Steps along
    an existing route are discounted, so branches join trunks instead of
    running beside them. Four neighbours only: every route is 4-connected.
    """

    def __init__(self, ow, relief=8.0):
        h, w = ow.shape
        self.w, self.h = w, h
        self.height = (np.asarray(ow.height, dtype=np.float64) * relief).ravel().tolist()
        extra = np.zeros((h, w))
        for name, cost in _BIOME_COST.items():
            extra[ow.biome == BIOME[name]] = cost
        if ow.river.any():
            wide = _box_sum(ow.river, 1) - 1.0
            extra[ow.river] += 0.5 * np.clip(wide[ow.river], 0.0, 8.0)
        blocked = ow.sea | ow.lake
        self.extra = extra.ravel().tolist()
        self.blocked = blocked.ravel().tolist()

    def plan(self, a, b, owner, weight=1.0):
        """Cell indices of the cheapest route from a to b, or None."""
        w, h = self.w, self.h
        if self.blocked[a] or self.blocked[b]:
            return None
        hgt, extra, blocked = self.height, self.extra, self.blocked
        ax, ay = a % w, a // w
        bx, by = b % w, b // w
        ux, uy = bx - ax, by - ay
        seg2 = float(ux * ux + uy * uy) or 1.0
        scale = 0.8 * weight                       # near-admissible: trunk discounts make it greedy
        dist = {a: 0.0}
        prev = {}
        heap = [(0.0, 0, a)]
        tick = 1
        done = set()
        base = math.exp(TOBLER * 0.05)
        cap = 6 * w * h
        while heap and cap:
            cap -= 1
            _, _, k = heapq.heappop(heap)
            if k in done:
                continue
            if k == b:
                break
            done.add(k)
            x, y = k % w, k // w
            dk, hk = dist[k], hgt[k]
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if not (0 <= nx < w and 0 <= ny < h):
                    continue
                m = ny * w + nx
                if m in done or blocked[m]:
                    continue
                s = hgt[m] - hk
                cost = (math.exp(TOBLER * abs(s + 0.05)) / base + extra[m]) * weight
                if s > 0.35 or s < -0.35:
                    cost += 6.0 * (abs(s) - 0.35) / 0.35 * weight
                # a faint pull toward the straight line: among equal-cost staircases
                # take the one that reads as a diagonal, not an L
                t = min(1.0, max(0.0, ((nx - ax) * ux + (ny - ay) * uy) / seg2))
                cost += 0.02 * math.hypot(nx - ax - t * ux, ny - ay - t * uy)
                if m in owner:
                    cost *= 0.4
                nd = dk + cost
                if nd < dist.get(m, 1e30):
                    dist[m] = nd
                    prev[m] = k
                    heapq.heappush(heap, (nd + (abs(bx - nx) + abs(by - ny)) * scale, tick, m))
                    tick += 1
        if b not in prev and a != b:
            return None
        path = [b]
        while path[-1] != a:
            path.append(prev[path[-1]])
        return path[::-1]


def _mode(a, b):
    m = min(a['tier'], b['tier'])
    return 'road' if m >= 2 else 'lane' if m == 1 else 'track'


def road_network(ow, sites, loops=1, exits=1):
    """Join the settlements of each landmass into one network.

    Each site links to the nearest site already on the network (largest
    first); `loops` extra links close the shortest missing pairs between
    villages and above; `exits` roads run from the capital to the map edge
    where dry land reaches it. Returns route dicts with their cells.
    """
    if not sites:
        return []
    h, w = ow.shape
    survey = _Survey(ow)
    roadable = ~(ow.sea | ow.lake)
    labels, _ = tf.connected_components(roadable)
    order = sorted(sites, key=lambda s: (-s['tier'], -s['score'], s['id']))
    owner = set()
    routes = []

    def link(sa, sb, mode, ida, idb):
        a = sa[1] * w + sa[0]
        b = sb[1] * w + sb[0]
        path = survey.plan(a, b, owner, MODES[mode])
        if path is None or len(path) < 2:
            return None
        cells = [(k % w, k // w) for k in path]
        fresh = sum(1 for k in path if k not in owner)
        owner.update(path)
        bridges = sum(1 for x, y in cells if ow.river[y, x])
        r = dict(id=f"r{len(routes)}", a=ida, b=idb, mode=mode, cells=cells,
                 length=len(cells), new_cells=fresh, bridges=bridges)
        routes.append(r)
        return r

    linked = []
    for s in order:
        same = [t for t in linked if labels[t['y'], t['x']] == labels[s['y'], s['x']]]
        if same:
            t = min(same, key=lambda o: (math.hypot(o['x'] - s['x'], o['y'] - s['y']), o['id']))
            link((s['x'], s['y']), (t['x'], t['y']), _mode(s, t), s['id'], t['id'])
        linked.append(s)

    big = [s for s in order if s['tier'] >= 1]
    for _ in range(int(loops)):
        have = {frozenset((r['a'], r['b'])) for r in routes}
        pairs = sorted(
            (math.hypot(a['x'] - b['x'], a['y'] - b['y']), a['id'], b['id'], a, b)
            for i, a in enumerate(big) for b in big[i + 1:]
            if frozenset((a['id'], b['id'])) not in have
            and labels[a['y'], a['x']] == labels[b['y'], b['x']])
        if not pairs:
            break
        _, _, _, a, b = pairs[0]
        link((a['x'], a['y']), (b['x'], b['y']), _mode(a, b), a['id'], b['id'])

    top = order[0]
    lab = labels[top['y'], top['x']]
    rng = np.random.default_rng(tf.seed_stream(ow.seed, 'soc.overworld.exits'))
    ends = []
    for k in range(8):
        a = k * math.pi / 4.0 + rng.uniform(-0.2, 0.2)
        dx, dy = math.cos(a), math.sin(a)
        reach = []
        if abs(dx) > 1e-9:
            reach.append(((w - 1 - top['x']) if dx > 0 else -top['x']) / dx)
        if abs(dy) > 1e-9:
            reach.append(((h - 1 - top['y']) if dy > 0 else -top['y']) / dy)
        t = min(reach)
        ex = int(np.clip(round(top['x'] + dx * t), 0, w - 1))
        ey = int(np.clip(round(top['y'] + dy * t), 0, h - 1))
        if roadable[ey, ex] and labels[ey, ex] == lab:
            ends.append((math.hypot(ex - top['x'], ey - top['y']), ex, ey))
    for _, ex, ey in sorted(ends, reverse=True)[:int(exits)]:
        link((top['x'], top['y']), (ex, ey), 'road', top['id'], 'edge')
    return routes


# -------------------------------------------------------------- assembly

@dataclass
class Overworld:
    """Every layer of one overworld. Arrays are (h, w)."""
    seed: int
    spec: dict
    height: np.ndarray
    sea_level: float
    sea: np.ndarray
    lake: np.ndarray
    river: np.ndarray
    accumulation: np.ndarray
    receiver: np.ndarray
    slope: np.ndarray
    temperature: np.ndarray
    precipitation: np.ndarray
    moisture: np.ndarray
    biome: np.ndarray
    settlements: list = dc_field(default_factory=list)
    routes: list = dc_field(default_factory=list)
    cover: np.ndarray = None

    @property
    def shape(self):
        return self.height.shape

    def biome_counts(self):
        ids, counts = np.unique(self.biome, return_counts=True)
        return {BIOMES[int(i)].name: int(c) for i, c in zip(ids, counts)}


def build_overworld(spec):
    """Run every stage for `spec` and return an Overworld.

    Spec knobs: 'size' / 'width' / 'height', 'seed', 'shape' (continent,
    archipelago, highlands, peninsula), 'land_ratio', 'scale', 'warp',
    'erosion' (stream-power passes), 'climate' (temperate, tropical, polar,
    pole_to_pole), 'wind' (west, east, north, south), 'lapse', 'aridity'
    (-1..1), 'lake_ratio', 'river_ratio', 'mountain_ratio', 'settlements'
    ((cities, towns, villages, hamlets) or 0 for none), 'loops', 'exits'.
    """
    w, h = dims(spec)
    seed = grammar_seed(spec, 'geo.overworld')
    shape = spec.get('shape', 'continent')
    raw = landmass(w, h, seed, shape, spec.get('scale', 2.6), spec.get('warp', 0.55))
    passes = int(spec.get('erosion', 8))
    height = np.asarray(raw.a, dtype=np.float64)
    if passes:
        # Stream-power erosion fills pits as it routes flow; a filled basin is a
        # perfectly flat L1 cone (a diamond plateau). Taking the minimum with
        # the uneroded surface keeps the incision and gives the basins back.
        carved = tf.thermal_erode(tf.stream_power_erode(raw, iterations=passes), iterations=6)
        height = np.minimum(np.asarray(carved.a, dtype=np.float64), height)
        height = (height - height.min()) / max(float(np.ptp(height)), 1e-9)
    default_land = {'archipelago': 0.34, 'highlands': 0.55, 'peninsula': 0.42}.get(shape, 0.46)
    land_ratio = float(np.clip(spec.get('land_ratio', default_land), 0.08, 0.92))
    sea_level = tg.threshold_for_ratio(height, land_ratio)
    sea = height < sea_level
    land = ~sea
    if land.any():
        main = tf.connected_components(land)[0]
        tiny = land & np.isin(main, np.flatnonzero(np.bincount(main.ravel()) < 6)) & (main > 0)
        sea = sea | tiny
        land = ~sea

    wind = spec.get('wind', 'west')
    precip = precipitation(height, sea, wind, orographic=spec.get('orographic', 7.0))
    filled, receiver, acc = drainage(height, precip, seed)
    lake = lakes_from(raw.a, sea, int(land.sum()), spec.get('lake_ratio', 0.02))
    land_dry = land & ~lake
    river = trace_rivers(acc, receiver, sea, lake, land, spec.get('river_ratio', 0.03))
    temp = temperature(height, sea_level, tf.seed_stream(seed, 'env.overworld.temperature'),
                       spec.get('climate', 'temperate'), spec.get('lapse', 0.45))
    moist = moisture(precip, land_dry, river | lake, spec.get('aridity', 0.0))
    slope = tf.Field(height.astype(F32)).slope().a.astype(np.float64)
    biome = classify_biomes(height, sea_level, temp, moist, sea, lake, river,
                            mountain_ratio=spec.get('mountain_ratio', 0.09), seed=seed)
    ow = Overworld(seed=seed, spec=dict(spec), height=height, sea_level=float(sea_level),
                   sea=sea, lake=lake, river=river, accumulation=acc, receiver=receiver,
                   slope=slope, temperature=temp, precipitation=precip, moisture=moist,
                   biome=biome)
    counts = spec.get('settlements')
    if counts != 0:
        ow.settlements = place_settlements(ow, counts)
        ow.routes = road_network(ow, ow.settlements, spec.get('loops', 1), spec.get('exits', 1))
    ow.cover = _cover(ow)
    return ow


def _footprint(ow, s):
    """Cells of a settlement's built-up disc (dry land only)."""
    h, w = ow.shape
    r = s['radius']
    ri = int(math.ceil(r))
    cells = []
    for y in range(max(0, s['y'] - ri), min(h, s['y'] + ri + 1)):
        for x in range(max(0, s['x'] - ri), min(w, s['x'] + ri + 1)):
            if (x - s['x']) ** 2 + (y - s['y']) ** 2 <= r * r + 0.25 and not (
                    ow.sea[y, x] or ow.lake[y, x] or ow.river[y, x]):
                cells.append((x, y))
    return cells or [(s['x'], s['y'])]


def _cover(ow):
    cover = ow.biome.astype(np.int16).copy()
    for r in ow.routes:
        for x, y in r['cells']:
            if not ow.river[y, x]:
                cover[y, x] = ROAD
    for s in ow.settlements:
        for x, y in _footprint(ow, s):
            cover[y, x] = TOWN
    return cover


def to_tilemap(ow):
    """Interpret an Overworld into the shared tile vocabulary."""
    h, w = ow.shape
    layers = []
    for b in BIOMES:
        layers.append((ow.biome == b.id, b.tile))
    road = np.zeros((h, w), dtype=bool)
    for r in ow.routes:
        for x, y in r['cells']:
            road[y, x] = True
    town = np.zeros((h, w), dtype=bool)
    for s in ow.settlements:
        for x, y in _footprint(ow, s):
            town[y, x] = True
    layers += [(road & ~ow.river, PLATFORM), (road & ow.river, BRIDGE), (town, PLATFORM)]
    overlays = [(np.zeros((h, w), dtype=bool), 'district')]
    for s in ow.settlements:
        overlays[0][0][s['y'], s['x']] = True
    tm = build(w, h, layers, default=WATER, overlays=overlays)
    # Valleys walled in by peaks get a pass, so the whole map stays one walkable piece.
    tg.connect_components(tm, passable_mask(tm), PLATFORM, width=1, seal_tile=WALL,
                          strategy='prim')
    accn = np.log1p(ow.accumulation)
    accn = accn / max(float(accn.max()), 1e-9)
    attach(tm, elevation=ow.height.astype(F32), temperature=ow.temperature.astype(F32),
           moisture=ow.moisture.astype(F32), precipitation=ow.precipitation.astype(F32),
           accumulation=accn.astype(F32), slope=ow.slope.astype(F32),
           biome=ow.biome.astype(F32), cover=ow.cover.astype(F32))
    tm.settlements = [dict(s) for s in ow.settlements]
    tm.routes = [dict(r, cells=[list(c) for c in r['cells']]) for r in ow.routes]
    tm.biome_names = list(COVER_NAMES)
    tm.field_palettes = {'cover': biome_palette()}
    tm.overworld = dict(sea_level=round(ow.sea_level, 6), land_ratio=round(float((~ow.sea).mean()), 4),
                        climate=ow.spec.get('climate', 'temperate'), wind=ow.spec.get('wind', 'west'),
                        shape=ow.spec.get('shape', 'continent'), biomes=ow.biome_counts())
    tm.overworld_layers = ow
    return tm


def gen_overworld(spec, rng):
    """Grammar 22, Overworld: (spec, rng) -> TileMap. `rng` is unused; every
    draw comes from named streams under the spec seed."""
    return to_tilemap(build_overworld(spec))


def _climate_field(w, h, seed, params):
    """Hybrid Morph source: moisture of an overworld's land, 0 at sea."""
    ow = build_overworld({'width': w, 'height': h, 'seed': seed, 'settlements': 0,
                          'erosion': params.get('erosion', 4),
                          'shape': params.get('shape', 'continent')})
    return tf.Field(ow.moisture.astype(F32), 'overworld.moisture')


tg.GRAMMARS['overworld'] = gen_overworld
tg.FIELD_SOURCES['overworld'] = _climate_field


# ------------------------------------------------------------------ export

def overworld_record(tm):
    """JSON-ready record of an overworld TileMap: settlements, routes and stats."""
    info = dict(getattr(tm, 'overworld', {}) or {})
    return {
        'schema': 'tilegen.overworld/1',
        'width': tm.w, 'height': tm.h,
        'overworld': info,
        'settlements': list(getattr(tm, 'settlements', []) or []),
        'routes': [dict(r) for r in (getattr(tm, 'routes', []) or [])],
        'cover_names': list(getattr(tm, 'biome_names', COVER_NAMES)),
    }


# ------------------------------------------------------------------ render

def _font(size):
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                              # Pillow < 10.1: bitmap font only
        return ImageFont.load_default()


def _simplify(points, eps):
    """Ramer-Douglas-Peucker polyline simplification (for drawing only)."""
    pts = [tuple(map(float, p)) for p in points]
    if len(pts) < 3:
        return pts
    keep = [False] * len(pts)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        i, j = stack.pop()
        (x0, y0), (x1, y1) = pts[i], pts[j]
        dx, dy = x1 - x0, y1 - y0
        n = math.hypot(dx, dy) or 1.0
        best, idx = 0.0, -1
        for k in range(i + 1, j):
            px, py = pts[k]
            dist = abs(dy * (px - x0) - dx * (py - y0)) / n
            if dist > best:
                best, idx = dist, k
        if idx > 0 and best > eps:
            keep[idx] = True
            stack += [(i, idx), (idx, j)]
    return [p for p, k in zip(pts, keep) if k]


def _chaikin(points, rounds=2):
    pts = [tuple(map(float, p)) for p in points]
    for _ in range(rounds):
        if len(pts) < 3:
            return pts
        out = [pts[0]]
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            out.append((0.75 * x0 + 0.25 * x1, 0.75 * y0 + 0.25 * y1))
            out.append((0.25 * x0 + 0.75 * x1, 0.25 * y0 + 0.75 * y1))
        out.append(pts[-1])
        pts = out
    return pts


def render_overworld(source, upscale=6, labels=True, legend=True, title=None):
    """A finished map picture of an overworld (TileMap from gen_overworld, or an
    Overworld): soft biome colours, hillshade, depth-tinted water, rivers drawn
    along their drainage, roads, settlements and their names."""
    from PIL import Image, ImageDraw
    ow = source if isinstance(source, Overworld) else getattr(source, 'overworld_layers', None)
    if ow is None:
        raise ValueError('render_overworld needs an Overworld or a TileMap from gen_overworld')
    h, w = ow.shape
    U = int(max(2, upscale))
    H, W = h * U, w * U
    ys, xs = np.mgrid[0:H, 0:W]
    sx, sy = (xs + 0.5) / U - 0.5, (ys + 0.5) / U - 0.5
    z = tf.Field(ow.height.astype(F32)).sample(sx, sy).astype(np.float64)

    pal = np.zeros((len(BIOMES), 3))
    for b in BIOMES:
        pal[b.id] = b.color
    pal[BIOME['snow']] = pal[BIOME['alpine']]     # snow is painted per pixel below
    land_biome = ow.biome.copy()
    water_ids = np.isin(land_biome, WATER_BIOMES)
    if (~water_ids).any():                      # extend land colours under the coast for the blur
        near = tf.nearest_seed(~water_ids)
        land_biome = land_biome[near[0], near[1]]
    col = pal[land_biome][(ys // U).clip(0, h - 1), (xs // U).clip(0, w - 1)]
    k = max(1, U // 2)
    for _ in range(2):
        col = np.stack([_box_mean(col[..., c], k) for c in range(3)], axis=-1)

    t_px = tf.Field(ow.temperature.astype(F32)).sample(sx, sy).astype(np.float64)
    cut = peak_level(ow.height, ow.sea, ow.spec.get('mountain_ratio', 0.09))
    if math.isfinite(cut):
        high = np.clip((z - cut) / 0.03 + 0.5, 0.0, 1.0)
        snow = np.clip((0.17 - t_px) / 0.04, 0.0, 1.0) * high + np.clip((0.06 - t_px) / 0.03, 0.0, 1.0)
        snow = np.clip(snow, 0.0, 1.0)[..., None] * (z >= ow.sea_level)[..., None]
        col = col * (1 - snow) + np.array([240, 243, 247.]) * snow

    gy, gx = np.gradient(z * 46.0)
    nz = 1.0 / np.sqrt(gx * gx + gy * gy + 1.0)
    lx, ly = -0.6, -0.7
    ln = math.sqrt(lx * lx + ly * ly + 0.8)
    shade = np.clip((-gx * lx - gy * ly) * nz / ln + 0.8 * nz / ln, 0.0, 1.0)
    col = col * (0.50 + 0.70 * shade[..., None])

    lake_soft = tf.Field(tf.smooth_lakes(ow.lake, passes=3)[1]).sample(sx, sy) if ow.lake.any() \
        else np.zeros((H, W))
    sea_px = z < ow.sea_level
    depth = np.clip((ow.sea_level - z) / max(ow.sea_level, 1e-6), 0.0, 1.0)
    shallow, deep = np.array([70, 132, 176.]), np.array([18, 44, 86.])
    water = shallow + (deep - shallow) * (depth[..., None] ** 0.6)
    col = np.where(sea_px[..., None], water, col)
    lake_a = np.clip(lake_soft, 0.0, 1.0)[..., None] * (~sea_px)[..., None]
    col = col * (1 - lake_a) + np.array([72, 132, 178.]) * lake_a
    # Shore rim: sea pixels with a land neighbour. The map edge is not a shore,
    # so pad by replication (a zero pad would draw a pale frame round the map).
    sp = np.pad(sea_px, 1, mode='edge')
    coast = sea_px & ~(sp[:-2, 1:-1] & sp[2:, 1:-1] & sp[1:-1, :-2] & sp[1:-1, 2:])
    col[coast] = col[coast] * 0.55 + np.array([232, 226, 196.]) * 0.45

    img = Image.fromarray(np.clip(col, 0, 255).astype(np.uint8), 'RGB')
    d = ImageDraw.Draw(img)

    def centre(x, y):
        return ((x + 0.5) * U, (y + 0.5) * U)

    # rivers along their D8 receivers, widening downstream
    acc = ow.accumulation
    top = float(acc[ow.river].max()) if ow.river.any() else 1.0
    ys_r, xs_r = np.nonzero(ow.river)
    for y, x in sorted(zip(ys_r.tolist(), xs_r.tolist())):
        nxt = int(ow.receiver[y * w + x])
        ny, nx = divmod(nxt, w)
        width = max(1, int(round(U * (0.22 + 0.55 * math.sqrt(acc[y, x] / top)))))
        d.line([centre(x, y), centre(nx, ny)], fill=(60, 120, 172), width=width)

    road_col = {'road': ((86, 58, 34), (226, 196, 140), 0.42),
                'lane': ((92, 66, 40), (214, 186, 136), 0.30),
                'track': ((104, 82, 58), (200, 176, 132), 0.22)}
    for casing in (True, False):
        for r in ow.routes:
            dark, light, frac = road_col[r['mode']]
            pts = _chaikin(_simplify([centre(x, y) for x, y in r['cells']], U * 0.9), 3)
            wd = max(1, int(round(U * frac)))
            d.line(pts, fill=dark if casing else light, width=wd + (2 if casing else 0), joint='curve')

    marker = {3: 0.95, 2: 0.70, 1: 0.48, 0: 0.32}
    for s in sorted(ow.settlements, key=lambda s: s['tier']):
        cx, cy = centre(s['x'], s['y'])
        r = max(2.0, U * marker[s['tier']])
        fill = (198, 64, 52) if s['tier'] >= 2 else (236, 222, 196)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill, outline=(28, 22, 18),
                  width=max(1, U // 4))
        if s['tier'] == 3:
            d.ellipse([cx - r * 0.45, cy - r * 0.45, cx + r * 0.45, cy + r * 0.45],
                      fill=(250, 232, 190))

    if labels:
        # Largest places first; a label tries the right of its marker, then the
        # left, and is dropped rather than drawn over another one.
        min_tier = 1 if U >= 4 and min(w, h) >= 120 else 2 if U < 4 else 1
        taken = []
        for s in sorted(ow.settlements, key=lambda s: (-s['tier'], s['id'])):
            if s['tier'] < min_tier:
                continue
            f = _font(max(9, int(U * (2.3 if s['tier'] == 3 else 1.9 if s['tier'] == 2 else 1.5))))
            cx, cy = centre(s['x'], s['y'])
            box = d.textbbox((0, 0), s['name'], font=f)
            tw, th = box[2] - box[0], box[3] - box[1]
            for tx in (cx + U * 1.1, cx - U * 1.1 - tw):
                ty = cy - th / 2 - box[1]
                rect = (tx - 2, ty + box[1] - 2, tx + tw + 2, ty + box[3] + 2)
                inside = rect[0] >= 0 and rect[2] <= W and rect[1] >= 0 and rect[3] <= H
                if inside and not any(rect[0] < r[2] and r[0] < rect[2] and rect[1] < r[3] and r[1] < rect[3]
                                      for r in taken):
                    taken.append(rect)
                    for ox, oy in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)):
                        d.text((tx + ox, ty + oy), s['name'], font=f, fill=(20, 18, 16))
                    d.text((tx, ty), s['name'], font=f, fill=(250, 244, 230))
                    break

    if legend:
        present = [b for b in BIOMES if b.id not in (BIOME['river'],) and (ow.biome == b.id).any()]
        f = _font(max(9, int(U * 1.6)))
        sw = max(8, int(U * 1.6))
        pad = max(4, U)
        rows = len(present)
        box_w = int(sw + pad * 3 + max(d.textlength(b.label, font=f) for b in present)) if present else 0
        box_h = rows * (sw + 3) + pad * 2
        x0, y0 = W - box_w - pad, H - box_h - pad
        if present and x0 > 0 and y0 > 0:
            d.rectangle([x0, y0, x0 + box_w, y0 + box_h], fill=(246, 240, 224), outline=(60, 50, 40))
            for i, b in enumerate(present):
                yy = y0 + pad + i * (sw + 3)
                d.rectangle([x0 + pad, yy, x0 + pad + sw, yy + sw], fill=b.color, outline=(60, 50, 40))
                d.text((x0 + pad * 2 + sw, yy), b.label, font=f, fill=(40, 34, 28))

    if title:
        # The bar is sized from the text's real extent (ascenders to descenders)
        # plus padding, so a heading never touches or overlaps the map; a title
        # too wide for the map steps its font down instead of running off.
        pad = max(5, U * 2)
        size = max(14, int(U * 3.4))
        f = _font(size)
        while size > 9 and d.textlength(title, font=f) > W - 2 * pad:
            size -= 1
            f = _font(size)
        box = d.textbbox((0, 0), title, font=f)
        bar = (box[3] - box[1]) + 2 * pad
        canvas = Image.new('RGB', (W, H + bar), (28, 26, 24))
        canvas.paste(img, (0, bar))
        cd = ImageDraw.Draw(canvas)
        cd.text((pad - box[0], pad - box[1]), title, font=f, fill=(240, 230, 210))
        cd.line([(0, bar - 1), (W, bar - 1)], fill=(64, 60, 54))
        return canvas
    return img
