"""Autotiles drawn procedurally: Wang-16 edge sheets and 47-case blob corners.

    import tilegen_autotile as ta
    atlas = ta.make_edge_atlas('dungeon', tile_px=16, seed=3)
    ta.save_tile_atlas(atlas, 'out', 'dungeon')          # sheet PNG + atlas JSON
    img = ta.render_autotile(tm, 'cave', scheme='blob')   # corners from 8 neighbours

Tile index == mask: N=1 E=2 S=4 W=8, bit set = the neighbour on that side is
the same terrain, so the tile joins it; bit clear = a rim on that side. The
saved sheet is 4x4, row-major (mask 0 top-left, mask 15 bottom-right), with an
atlas JSON in the 'wang-edge-16' scheme that `tilegen_style.load_tile_atlas`
reads, so generated sheets and sheets from other tools are interchangeable.

Two rules keep any arrangement continuous. The ground and the rim are each
one tile-periodic texture shared by every tile, so a tile differs from its
siblings only in where the rim sits; and the rim's wobble fades to zero at the
tile border, so a rim leaves one tile exactly where it enters the next. A
4-bit mask cannot see diagonals, so the blob scheme reads the 8-bit mask and
adds inner-corner notches where two joined sides meet a diagonal gap (47
distinct cases). Pixels are palette-limited (four ground tones, four rim
tones, two accents) and ordered-dithered, so tiles stay crisp pixel art.

Four skins: dungeon (flagstones, masonry), cave (gravel, strata), shore
(sand, water with foam) and tech (deck plates, wall panels), each with four
palettes: classic, ember, frost and verdant.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

N, E, S, W = 1, 2, 4, 8
NE, SE, SW, NW = 16, 32, 64, 128
SKINS = ('dungeon', 'cave', 'shore', 'tech')
LABELS = {0: 'isolated', 1: 'N', 2: 'E', 3: 'NE', 4: 'S', 5: 'NS', 6: 'ES', 7: 'NES',
          8: 'W', 9: 'NW', 10: 'EW', 11: 'NEW', 12: 'SW', 13: 'NSW', 14: 'ESW', 15: 'center'}

# (ground x4 dark..light, rim x4 dark..light, accent x2) per skin and palette.
PALETTES = {
    'dungeon': {
        'classic': (('#3a2c1e', '#6e5a3f', '#8f7852', '#b89d6e'), ('#16181f', '#3b3f4c', '#565b6b', '#8a90a0'), ('#4f6b3a', '#7f9f4f')),
        'ember': (('#2b0f0a', '#5c1a0e', '#8f2b10', '#d0541a'), ('#0a0a10', '#1e1d2b', '#33324a', '#5a5878'), ('#e0661c', '#ffb347')),
        'frost': (('#171a2a', '#2b3150', '#444d78', '#6a76a6'), ('#2c4a66', '#5e8fb3', '#9cc7e2', '#e4f4ff'), ('#7fd0ff', '#e8fbff')),
        'verdant': (('#1c2a1e', '#33523a', '#4e7a50', '#7ba66c'), ('#1b1a14', '#403b2a', '#6b6347', '#a39a72'), ('#8bff4a', '#d8ff9a')),
    },
    'cave': {
        'classic': (('#4a3d2f', '#6f5e48', '#92805f', '#b8a57c'), ('#14131a', '#2e2c3a', '#4a4757', '#767285'), ('#c9b98f', '#e6dcc0')),
        'ember': (('#2a0d08', '#561a0c', '#8a2e0e', '#cf5a18'), ('#0b0a0f', '#1d1b27', '#34303f', '#5b5566'), ('#ff8a2a', '#ffd27a')),
        'frost': (('#14182c', '#252d4d', '#3b4a78', '#5d72a8'), ('#2a4561', '#4f7ea1', '#8bbbd6', '#d6efff'), ('#a8e1ff', '#ffffff')),
        'verdant': (('#1a2a14', '#2f4d22', '#4f7a2e', '#85b84a'), ('#1c130c', '#3b2a1a', '#5e4529', '#8a6a40'), ('#c2ff6a', '#f1ffb8')),
    },
    'shore': {
        'classic': (('#6f5c3c', '#a48d5f', '#cdb27a', '#ecd9a4'), ('#0c2a4a', '#164c7d', '#2477ad', '#4aa3d1'), ('#f2fbff', '#a8e6ff')),
        'ember': (('#241822', '#3d2a38', '#5a4054', '#7d5e74'), ('#2a0608', '#560d12', '#8c1a1c', '#c2402f'), ('#ffc7b8', '#ff8d6b')),
        'frost': (('#8a9bb8', '#b4c3d8', '#d6e1ee', '#f4f8fc'), ('#10304a', '#1d5b86', '#3f8eb8', '#7cc4e0'), ('#ffffff', '#c8f0ff')),
        'verdant': (('#25301a', '#3f5226', '#62783a', '#8fa353'), ('#13261a', '#1f4a2c', '#33803f', '#63b857'), ('#e8ffb0', '#b8ff5a')),
    },
    'tech': {
        'classic': (('#1d2128', '#363c47', '#505968', '#7a8596'), ('#0e1015', '#242932', '#3a424f', '#5f6b7d'), ('#ffb42a', '#fff0b0')),
        'ember': (('#15121a', '#2b2633', '#433c4f', '#665c76'), ('#08070b', '#18151f', '#2c2736', '#463f55'), ('#ff5a1e', '#ffc24a')),
        'frost': (('#232b38', '#46566d', '#6f86a3', '#a6bdd8'), ('#101821', '#26364a', '#3f5873', '#6c8bad'), ('#6fe3ff', '#e2fbff')),
        'verdant': (('#1b2418', '#364a2e', '#53703f', '#7e9e5c'), ('#0b1209', '#1a2815', '#2c4222', '#4a6638'), ('#9bff3a', '#e4ffb0')),
    },
}

# Boundary wobble in pixels at 16 px (dungeon is hewn, cave and shore organic, tech straight).
WOBBLE = {'dungeon': 0.8, 'cave': 2.4, 'shore': 1.6, 'tech': 0.0}

_BAYER = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]],
                  dtype=np.float64) / 16.0


def _rgb(c):
    return tuple(int(c[i:i + 2], 16) for i in (1, 3, 5))


def _bayer(n):
    reps = (n + 3) // 4
    return np.tile(_BAYER, (reps, reps))[:n, :n]


def periodic_noise(n, cells, seed, octaves=3):
    """Tile-periodic value noise on an n x n tile, in [0, 1]: the right edge
    continues into the left and the bottom into the top."""
    rng = np.random.default_rng(seed)
    out = np.zeros((n, n))
    amp, total = 1.0, 0.0
    for o in range(octaves):
        c = max(1, int(cells) * (2 ** o))
        lat = rng.random((c, c))
        pos = (np.arange(n) + 0.5) * c / n - 0.5
        i0 = np.floor(pos).astype(int)
        f = pos - i0
        f = f * f * (3.0 - 2.0 * f)
        a, b = i0 % c, (i0 + 1) % c
        top = lat[np.ix_(a, a)] * (1 - f[None, :]) + lat[np.ix_(a, b)] * f[None, :]
        bot = lat[np.ix_(b, a)] * (1 - f[None, :]) + lat[np.ix_(b, b)] * f[None, :]
        out += amp * (top * (1 - f[:, None]) + bot * f[:, None])
        total += amp
        amp *= 0.5
    return out / total


def _ramp(t, n, lo=0, hi=3):
    """Quantise [0, 1] onto ramp indices lo..hi with ordered dither."""
    v = np.clip(t, 0.0, 1.0) * (hi - lo + 1) + (_bayer(n) - 0.5) * 0.9
    return np.clip(np.floor(v), 0, hi - lo).astype(np.int16) + lo


# ----------------------------------------------------------------- textures
# Each returns an index map: 0..3 into the four tones, 4 and 5 for the accents.

def _cells(seed, k=64):
    """A table of per-cell draws (stone shade, brick shade), indexed modulo k."""
    return np.random.default_rng(seed).random((k, k))


def _ground(skin, n, seed):
    yy, xx = np.mgrid[0:n, 0:n]
    t = periodic_noise(n, max(2, n // 8), seed)
    if skin == 'dungeon':
        stone = max(2, n // 2)
        row = yy // stone
        sx = (xx + row * (stone // 2)) % n
        lx, ly = sx % stone, yy % stone
        pick = _cells(seed + 1)[row % 64, (sx // stone) % 64]
        idx = np.where(pick > 0.5, 2, 1).astype(np.int16)
        idx = np.where((lx == 1) | (ly == 1), idx + 1, idx)
        idx = np.where((lx == stone - 1) | (ly == stone - 1), idx - 1, idx)
        crack = np.abs(periodic_noise(n, 4, seed + 2, 2) - 0.5) < 0.025
        idx = np.where(crack, 4, idx)
        idx = np.where((lx == 0) | (ly == 0), 0, idx)
        return idx.astype(np.int16)
    if skin == 'cave':
        idx = _ramp(0.25 + 0.6 * t, n, 0, 3)
        stones = periodic_noise(n, 3, seed + 3, 3) > 0.74
        return np.where(stones, 4, idx).astype(np.int16)
    if skin == 'shore':
        ripple = 0.5 + 0.5 * np.sin(2 * math.pi * (yy * 2 / n + 0.35 * t))
        idx = _ramp(0.35 + 0.35 * ripple + 0.3 * (t - 0.5), n, 1, 3)
        glint = periodic_noise(n, max(2, n // 3), seed + 4, 1) > 0.86
        return np.where(glint, 5, idx).astype(np.int16)
    # tech: deck plates with rivets and a conduit
    plate = max(4, n // 2)
    lx, ly = xx % plate, yy % plate
    idx = np.where((lx == 0) | (ly == 0), 0, 2).astype(np.int16)
    idx = np.where(((lx == 1) | (ly == 1)) & (idx > 0), 3, idx)
    idx = np.where(((lx == plate - 1) | (ly == plate - 1)) & (idx > 0), 1, idx)
    rivet = ((lx == 2) | (lx == plate - 2)) & ((ly == 2) | (ly == plate - 2))
    idx = np.where(rivet, 3, idx)
    conduit = (yy == n // 2 + 1) & (t > 0.35)
    return np.where(conduit, 4, idx).astype(np.int16)


def _rim(skin, n, seed):
    yy, xx = np.mgrid[0:n, 0:n]
    t = periodic_noise(n, max(2, n // 8), seed + 10)
    if skin == 'dungeon':
        bh, bw = max(2, n // 4), max(2, n // 2)
        row = yy // bh
        ly = yy % bh
        sx = (xx + (row % 2) * (bw // 2)) % n
        lx = sx % bw
        pick = _cells(seed + 11)[row % 64, (sx // bw) % 64]
        idx = np.where(pick > 0.5, 2, 1).astype(np.int16)
        idx = np.where(ly == 1, idx + 1, idx)
        idx = np.where(ly == bh - 1, idx - 1, idx)
        return np.where((ly == 0) | (lx == 0), 0, idx).astype(np.int16)
    if skin == 'cave':
        strata = 0.5 + 0.5 * np.sin(2 * math.pi * (yy * 3 / n + 0.6 * t))
        return _ramp(0.15 + 0.55 * strata + 0.3 * (t - 0.5), n, 0, 3)
    if skin == 'shore':
        wave = 0.5 + 0.5 * np.sin(2 * math.pi * (yy * 2 / n + xx / n + 0.5 * t))
        return _ramp(0.2 + 0.6 * wave, n, 0, 3)
    panel = max(4, n // 2)
    lx, ly = xx % panel, yy % panel
    idx = np.where((lx == 0) | (ly == 0), 0, 1).astype(np.int16)
    idx = np.where((ly == panel // 2) & (lx > 1) & (lx < panel - 1), 2, idx)
    return np.where((lx == panel - 1) & (ly > 0), 3, idx).astype(np.int16)


# ------------------------------------------------------------------- shapes

def _wobbles(skin, n, seed):
    """Per-side boundary offset in pixels, fading to zero at the tile corners."""
    amp = WOBBLE[skin] * n / 16.0
    pos = (np.arange(n) + 0.5) / n
    fade = np.sin(math.pi * pos)
    out = {}
    for k, side in enumerate('NESW'):
        line = periodic_noise(n, 3, seed + 20 + k, 2)[n // 2]
        out[side] = (line - 0.5) * 2.0 * amp * fade
    return out


def _rim_depth(n):
    return max(2.0, n / 4.0)


def _shape(mask, n, wob, blob):
    """(inside, lit): signed depth into the rim (>0 = rim) and whether the
    nearest rim faces the light (north and west rims are lit)."""
    yy, xx = np.mgrid[0:n, 0:n] + 0.5
    d0 = _rim_depth(n)
    inside = np.full((n, n), -1e9)
    lit = np.zeros((n, n), dtype=bool)
    perp = {'N': yy, 'S': n - yy, 'W': xx, 'E': n - xx}
    along = {'N': (xx - 0.5).astype(int), 'S': (xx - 0.5).astype(int),
             'W': (yy - 0.5).astype(int), 'E': (yy - 0.5).astype(int)}
    for side, bit in (('N', N), ('E', E), ('S', S), ('W', W)):
        if mask & bit:
            continue
        v = d0 + wob[side][along[side]] - perp[side]
        better = v > inside
        inside = np.where(better, v, inside)
        lit = np.where(better, side in 'NW', lit)
    if blob:
        for diag, (a, b, dbit) in (('NE', (N, E, NE)), ('SE', (S, E, SE)),
                                   ('SW', (S, W, SW)), ('NW', (N, W, NW))):
            if (mask & a) and (mask & b) and not (mask & dbit):
                dx = xx if 'W' in diag else n - xx
                dy = yy if 'N' in diag else n - yy
                v = d0 * 0.9 - (dx + dy) * 0.75
                better = v > inside
                inside = np.where(better, v, inside)
                lit = np.where(better, diag[0] == 'N' or diag[1] == 'W', lit)
    return inside, lit


# -------------------------------------------------------------------- tiles

class _Kit:
    """Textures and palette shared by every tile of one skin/palette/seed."""

    def __init__(self, skin, palette, n, seed):
        if skin not in PALETTES:
            raise ValueError(f"skin must be one of {SKINS}, got {skin!r}")
        if palette not in PALETTES[skin]:
            raise ValueError(f"palette must be one of {tuple(PALETTES[skin])}, got {palette!r}")
        ground, rim, accent = PALETTES[skin][palette]
        self.skin, self.palette, self.n, self.seed = skin, palette, int(n), int(seed)
        self.g = np.array([_rgb(c) for c in ground + accent], dtype=np.uint8)
        self.r = np.array([_rgb(c) for c in rim + accent], dtype=np.uint8)
        self.ground = _ground(skin, self.n, self.seed)
        self.rim = _rim(skin, self.n, self.seed)
        self.wob = _wobbles(skin, self.n, self.seed)

    def tile(self, mask, blob=False):
        n = self.n
        inside, lit = _shape(mask, n, self.wob, blob)
        rim = inside > 0
        lip = rim & (inside <= 1.0)
        shadow = ~rim & (inside > -1.25) & lit
        gi = np.where(shadow, np.maximum(self.ground - 1, 0), self.ground)
        gi = np.where(shadow & (self.ground >= 4), 0, gi)
        out = self.g[gi]
        rim_rgb = self.r[self.rim]
        if self.skin == 'shore':
            lip_rgb = np.broadcast_to(self.r[4], rim_rgb.shape)
        else:
            lip_rgb = np.where(lit[..., None], self.r[3], self.r[0])
        out = np.where(rim[..., None], np.where(lip[..., None], lip_rgb, rim_rgb), out)
        alpha = np.full((n, n, 1), 255, dtype=np.uint8)
        return Image.fromarray(np.concatenate([out.astype(np.uint8), alpha], axis=-1), 'RGBA')

    def solid(self):
        """A full tile of rim material, for cells outside the terrain."""
        n = self.n
        alpha = np.full((n, n, 1), 255, dtype=np.uint8)
        return Image.fromarray(np.concatenate([self.r[self.rim], alpha], axis=-1), 'RGBA')


def make_edge_atlas(skin='dungeon', tile_px=16, seed=0, palette='classic'):
    """The 16 Wang edge tiles of one skin: {'tiles': {mask: RGBA Image}, 'tile': px, ...},
    the same shape `tilegen_style.load_tile_atlas` returns, so `render_tiles` takes it."""
    kit = _Kit(skin, palette, tile_px, seed)
    return {'tiles': {m: kit.tile(m) for m in range(16)}, 'tile': kit.n,
            'scheme': 'wang-edge-16', 'skin': skin, 'palette': palette, 'seed': int(seed)}


def canonical_blob(mask):
    """Drop diagonal bits whose two adjacent sides are not both joined: those
    diagonals cannot change the tile. Leaves 47 distinct masks out of 256."""
    m = mask & 15
    for dbit, a, b in ((NE, N, E), (SE, S, E), (SW, S, W), (NW, N, W)):
        if (mask & dbit) and (mask & a) and (mask & b):
            m |= dbit
    return m


BLOB_MASKS = tuple(sorted({canonical_blob(m) for m in range(256)}))


def make_blob_atlas(skin='dungeon', tile_px=16, seed=0, palette='classic'):
    """The 47 blob tiles: {'tiles': {canonical 8-bit mask: RGBA Image}, 'tile': px, ...}."""
    kit = _Kit(skin, palette, tile_px, seed)
    return {'tiles': {m: kit.tile(m, blob=True) for m in BLOB_MASKS}, 'tile': kit.n,
            'scheme': 'blob-47', 'skin': skin, 'palette': palette, 'seed': int(seed)}


def blob_masks(open_mask):
    """Canonical 8-bit blob mask per cell (N=1 E=2 S=4 W=8 NE=16 SE=32 SW=64
    NW=128); cells outside the terrain get -1. Out of bounds is not terrain."""
    a = np.asarray(open_mask, dtype=bool)
    p = np.pad(a, 1)
    n_, s_ = p[:-2, 1:-1], p[2:, 1:-1]
    w_, e_ = p[1:-1, :-2], p[1:-1, 2:]
    ne, se = p[:-2, 2:], p[2:, 2:]
    sw, nw = p[2:, :-2], p[:-2, :-2]
    m = (n_ * N + e_ * E + s_ * S + w_ * W).astype(int)
    m += (ne & n_ & e_) * NE + (se & s_ & e_) * SE + (sw & s_ & w_) * SW + (nw & n_ & w_) * NW
    return np.where(a, m, -1)


def render_autotile(tm, skin='dungeon', palette='classic', tile_px=16, scheme='edge',
                    open_mask=None, seed=0):
    """Render a TileMap's terrain (passable cells, or `open_mask`) with generated
    autotiles; everything else is rim material. scheme='edge' (16 tiles, 4-bit masks) or 'blob' (47 tiles,
    8-bit masks)."""
    if scheme not in ('edge', 'blob'):
        raise ValueError(f"scheme must be 'edge' or 'blob', got {scheme!r}")
    if open_mask is None:
        import tilegen_grammars as tg
        open_mask = tg.passable_mask(tm)
    kit = _Kit(skin, palette, tile_px, seed)
    n = kit.n
    if scheme == 'edge':
        p = np.pad(np.asarray(open_mask, dtype=bool), 1)
        a = p[1:-1, 1:-1]
        masks = np.where(a, (p[:-2, 1:-1] * N + p[1:-1, 2:] * E + p[2:, 1:-1] * S
                             + p[1:-1, :-2] * W).astype(int), -1)
    else:
        masks = blob_masks(open_mask)
    cache = {}
    img = Image.new('RGBA', (tm.w * n, tm.h * n))
    solid = kit.solid()
    for y in range(tm.h):
        for x in range(tm.w):
            m = int(masks[y][x])
            if m < 0:
                img.paste(solid, (x * n, y * n))
                continue
            if m not in cache:
                cache[m] = kit.tile(m, blob=(scheme == 'blob'))
            img.paste(cache[m], (x * n, y * n))
    return img.convert('RGB')


def save_tile_atlas(atlas, directory, stem='tiles'):
    """Write `<stem>_sheet.png` (4x4) and `<stem>_atlas.json` (scheme
    'wang-edge-16', bits N=1 E=2 S=4 W=8). Returns (atlas_path, sheet_path)."""
    if sorted(atlas['tiles']) != list(range(16)):
        raise ValueError('a Wang-16 atlas needs tiles for masks 0..15')
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    t = int(atlas['tile'])
    sheet = Image.new('RGBA', (4 * t, 4 * t))
    frames = {}
    for m in range(16):
        x, y = (m % 4) * t, (m // 4) * t
        sheet.paste(atlas['tiles'][m], (x, y))
        frames[f"wang_{m}_{LABELS[m]}"] = {'x': x, 'y': y, 'w': t, 'h': t}
    sheet_path = out / f"{stem}_sheet.png"
    atlas_path = out / f"{stem}_atlas.json"
    sheet.save(sheet_path, optimize=True)
    meta = {
        'autotile': {'scheme': 'wang-edge-16', 'tileSize': t, 'columns': 4,
                     'bits': {'1': 'north', '2': 'east', '4': 'south', '8': 'west'}},
        'frames': dict(sorted(frames.items())),
        'generator': {'name': 'tilegen_autotile', 'skin': atlas.get('skin'),
                      'palette': atlas.get('palette'), 'seed': atlas.get('seed')},
    }
    atlas_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding='utf-8')
    return atlas_path, sheet_path


def sheet_preview(skins=SKINS, palette='classic', tile_px=16, seed=0, scale=3, gap=6):
    """One picture of each skin's 16-tile sheet side by side (for docs)."""
    sheets = []
    for skin in skins:
        at = make_edge_atlas(skin, tile_px, seed, palette)
        t = at['tile']
        s = Image.new('RGBA', (4 * t, 4 * t))
        for m in range(16):
            s.paste(at['tiles'][m], ((m % 4) * t, (m // 4) * t))
        sheets.append(s.resize((4 * t * scale, 4 * t * scale), Image.NEAREST))
    wd = sum(s.width for s in sheets) + gap * (len(sheets) + 1)
    ht = max(s.height for s in sheets) + 2 * gap
    canvas = Image.new('RGB', (wd, ht), (16, 18, 22))
    x = gap
    for s in sheets:
        canvas.paste(s, (x, gap), s)
        x += s.width + gap
    return canvas
