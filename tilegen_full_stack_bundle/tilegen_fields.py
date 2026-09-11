"""Continuous field substrate for tilegen grammars.

Grammars think in scalar fields, vector fields and vector geometry. The TileMap
raster is a projection of that, not the source of truth.

Nothing in this module knows what a 'wall' is. No tile vocabulary appears here;
interpretation lives in tilegen_grammars.py. Everything is deterministic given a
seed.
"""
from __future__ import annotations

import math

import numpy as np

F32 = np.float32


# ---------------------------------------------------------------- seed streams

def seed_stream(world_seed, name, index=0):
    """Named deterministic substream, so changing one grammar does not reshuffle
    every other one (atlas 3.9). FNV-1a with explicit 64-bit wrapping."""
    h = 1469598103934665603
    for ch in f"{world_seed}|{name}|{index}":
        h ^= ord(ch)
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h & 0x7FFFFFFF


def rng_for(world_seed, name, index=0):
    return np.random.default_rng(seed_stream(world_seed, name, index))


# ---------------------------------------------------------------------- fields

class Field:
    """A 2D scalar field, shape (h, w), float32.

    Arithmetic composes fields with each other and with scalars. `expr` carries a
    human-readable provenance string so a designer can ask why a cell looks the
    way it does.
    """

    __slots__ = ('a', 'expr')

    def __init__(self, a, expr='field'):
        a = np.asarray(a, dtype=F32)
        if a.ndim != 2:
            raise ValueError(f"Field must be 2D, got shape {a.shape}")
        self.a = a
        self.expr = expr

    # -- basics
    @property
    def h(self): return self.a.shape[0]

    @property
    def w(self): return self.a.shape[1]

    @property
    def shape(self): return self.a.shape

    def copy(self): return Field(self.a.copy(), self.expr)

    def __repr__(self):
        return (f"Field({self.w}x{self.h} "
                f"[{float(self.a.min()):.3f},{float(self.a.max()):.3f}] {self.expr})")

    def _pair(self, other):
        if isinstance(other, Field):
            if other.shape != self.shape:
                raise ValueError(f"shape mismatch {self.shape} vs {other.shape}")
            return other.a, other.expr
        return other, repr(other)

    # -- arithmetic
    def __add__(self, o):
        b, e = self._pair(o); return Field(self.a + b, f"({self.expr}+{e})")

    def __radd__(self, o):
        b, e = self._pair(o); return Field(b + self.a, f"({e}+{self.expr})")

    def __sub__(self, o):
        b, e = self._pair(o); return Field(self.a - b, f"({self.expr}-{e})")

    def __rsub__(self, o):
        b, e = self._pair(o); return Field(b - self.a, f"({e}-{self.expr})")

    def __mul__(self, o):
        b, e = self._pair(o); return Field(self.a * b, f"({self.expr}*{e})")

    def __rmul__(self, o):
        b, e = self._pair(o); return Field(b * self.a, f"({e}*{self.expr})")

    def __truediv__(self, o):
        b, e = self._pair(o)
        return Field(self.a / np.where(np.abs(b) < 1e-9, 1e-9, b), f"({self.expr}/{e})")

    def __pow__(self, k):
        return Field(np.clip(self.a, 0.0, None) ** k, f"({self.expr}^{k})")

    def __neg__(self): return Field(-self.a, f"(-{self.expr})")

    def __abs__(self): return Field(np.abs(self.a), f"abs({self.expr})")

    # -- comparisons produce boolean masks
    def __gt__(self, o):
        b, _ = self._pair(o); return self.a > b

    def __lt__(self, o):
        b, _ = self._pair(o); return self.a < b

    def __ge__(self, o):
        b, _ = self._pair(o); return self.a >= b

    def __le__(self, o):
        b, _ = self._pair(o); return self.a <= b

    # -- range shaping
    def normalized(self):
        lo, hi = float(self.a.min()), float(self.a.max())
        if hi - lo < 1e-9:
            return Field(np.zeros_like(self.a), f"norm({self.expr})")
        return Field((self.a - lo) / (hi - lo), f"norm({self.expr})")

    def clip(self, lo=0.0, hi=1.0):
        return Field(np.clip(self.a, lo, hi), f"clip({self.expr},{lo},{hi})")

    def remap(self, lo, hi):
        n = self.normalized()
        return Field(n.a * (hi - lo) + lo, f"remap({self.expr},{lo},{hi})")

    def smoothstep(self, e0=0.0, e1=1.0):
        t = np.clip((self.a - e0) / max(e1 - e0, 1e-9), 0.0, 1.0)
        return Field(t * t * (3.0 - 2.0 * t), f"smoothstep({self.expr})")

    def quantize(self, levels):
        q = np.floor(np.clip(self.a, 0.0, 0.999999) * levels) / max(levels - 1, 1)
        return Field(q, f"quantize({self.expr},{levels})")

    def invert(self):
        return Field(1.0 - self.a, f"invert({self.expr})")

    # -- masks
    def threshold(self, t):
        return self.a > t

    def band(self, lo, hi):
        return (self.a >= lo) & (self.a <= hi)

    def mask_where(self, mask, value=0.0):
        out = self.a.copy()
        out[~mask] = value
        return Field(out, f"mask({self.expr})")

    # -- combination
    def blend(self, other, t):
        tv = t.a if isinstance(t, Field) else t
        te = t.expr if isinstance(t, Field) else repr(t)
        return Field(self.a * (1.0 - tv) + other.a * tv,
                     f"blend({self.expr},{other.expr},{te})")

    def maximum(self, other):
        b, e = self._pair(other); return Field(np.maximum(self.a, b), f"max({self.expr},{e})")

    def minimum(self, other):
        b, e = self._pair(other); return Field(np.minimum(self.a, b), f"min({self.expr},{e})")

    # -- spatial operators
    def blur(self, iterations=1):
        a = self.a
        for _ in range(iterations):
            a = (a
                 + np.roll(a, 1, 0) + np.roll(a, -1, 0)
                 + np.roll(a, 1, 1) + np.roll(a, -1, 1)) / 5.0
        return Field(a, f"blur({self.expr},{iterations})")

    def gradient(self):
        gy, gx = np.gradient(self.a)
        return VectorField(gx, gy, f"grad({self.expr})")

    def slope(self):
        gy, gx = np.gradient(self.a)
        return Field(np.hypot(gx, gy), f"slope({self.expr})")

    def curl(self):
        """Gradient rotated 90 degrees - divergence-free flow around contours."""
        gy, gx = np.gradient(self.a)
        return VectorField(gy, -gx, f"curl({self.expr})")

    def sample(self, xs, ys):
        """Bilinear sample at float coordinates (clamped)."""
        h, w = self.shape
        xs = np.clip(xs, 0, w - 1.001)
        ys = np.clip(ys, 0, h - 1.001)
        x0 = np.floor(xs).astype(np.int32); y0 = np.floor(ys).astype(np.int32)
        fx = (xs - x0).astype(F32); fy = (ys - y0).astype(F32)
        x1 = np.minimum(x0 + 1, w - 1); y1 = np.minimum(y0 + 1, h - 1)
        a = self.a
        top = a[y0, x0] * (1 - fx) + a[y0, x1] * fx
        bot = a[y1, x0] * (1 - fx) + a[y1, x1] * fx
        return top * (1 - fy) + bot * fy

    def warp(self, vec, amount=1.0):
        """Displace this field by a vector field (atlas: 'phyllotaxis warped_by flow')."""
        h, w = self.shape
        yy, xx = np.mgrid[0:h, 0:w].astype(F32)
        return Field(self.sample(xx + vec.x * amount, yy + vec.y * amount),
                     f"warp({self.expr},{vec.expr},{amount})")

    def contour_mask(self, level, width=1.0):
        """Cells where the field crosses `level` - marching-squares style edges."""
        s = self.a - level
        cross = np.zeros(self.shape, dtype=bool)
        cross[:, :-1] |= (s[:, :-1] * s[:, 1:]) <= 0
        cross[:-1, :] |= (s[:-1, :] * s[1:, :]) <= 0
        if width > 1:
            cross = _dilate(cross, int(width) - 1)
        return cross

    def terraces(self, levels):
        """Quantised elevation plus the wall mask between adjacent terraces."""
        q = np.floor(np.clip(self.normalized().a, 0, 0.999999) * levels).astype(np.int32)
        edge = np.zeros(self.shape, dtype=bool)
        edge[:, :-1] |= q[:, :-1] != q[:, 1:]
        edge[:-1, :] |= q[:-1, :] != q[1:, :]
        return q, edge


class VectorField:
    """A 2D vector field stored as two component arrays."""

    __slots__ = ('x', 'y', 'expr')

    def __init__(self, x, y, expr='vec'):
        self.x = np.asarray(x, dtype=F32)
        self.y = np.asarray(y, dtype=F32)
        self.expr = expr

    @property
    def shape(self): return self.x.shape

    def magnitude(self):
        return Field(np.hypot(self.x, self.y), f"|{self.expr}|")

    def angle(self):
        return Field(np.arctan2(self.y, self.x), f"angle({self.expr})")

    def normalize(self):
        m = np.hypot(self.x, self.y)
        m = np.where(m < 1e-9, 1.0, m)
        return VectorField(self.x / m, self.y / m, f"unit({self.expr})")

    def __mul__(self, k):
        return VectorField(self.x * k, self.y * k, f"({self.expr}*{k})")

    __rmul__ = __mul__

    def __add__(self, o):
        return VectorField(self.x + o.x, self.y + o.y, f"({self.expr}+{o.expr})")

    def rotate(self, radians):
        c, s = math.cos(radians), math.sin(radians)
        return VectorField(self.x * c - self.y * s, self.x * s + self.y * c,
                           f"rot({self.expr},{radians:.2f})")


# ------------------------------------------------------------- mask utilities

def _shifts(mask):
    up = np.zeros_like(mask); up[:-1, :] = mask[1:, :]
    dn = np.zeros_like(mask); dn[1:, :] = mask[:-1, :]
    lf = np.zeros_like(mask); lf[:, :-1] = mask[:, 1:]
    rt = np.zeros_like(mask); rt[:, 1:] = mask[:, :-1]
    return up, dn, lf, rt


def _dilate(mask, iterations=1):
    m = mask
    for _ in range(max(0, iterations)):
        up, dn, lf, rt = _shifts(m)
        m = m | up | dn | lf | rt
    return m


def _erode(mask, iterations=1):
    m = mask
    for _ in range(max(0, iterations)):
        up, dn, lf, rt = _shifts(m)
        m = m & up & dn & lf & rt
    return m


def dilate(mask, iterations=1): return _dilate(mask, iterations)


def erode(mask, iterations=1): return _erode(mask, iterations)


def outline(mask):
    """Cells inside `mask` that touch something outside it."""
    return mask & ~_erode(mask, 1)


def connected_components(mask):
    """Label 4-connected components. Returns (labels, count); background is 0."""
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    current = 0
    flat = mask.ravel()
    lab = labels.ravel()
    stack = []
    for start in np.flatnonzero(flat):
        if lab[start]:
            continue
        current += 1
        stack.append(int(start))
        lab[start] = current
        while stack:
            i = stack.pop()
            y, x = divmod(i, w)
            if x > 0 and flat[i - 1] and not lab[i - 1]:
                lab[i - 1] = current; stack.append(i - 1)
            if x < w - 1 and flat[i + 1] and not lab[i + 1]:
                lab[i + 1] = current; stack.append(i + 1)
            if y > 0 and flat[i - w] and not lab[i - w]:
                lab[i - w] = current; stack.append(i - w)
            if y < h - 1 and flat[i + w] and not lab[i + w]:
                lab[i + w] = current; stack.append(i + w)
    return labels, current


def largest_component(mask):
    labels, n = connected_components(mask)
    if n == 0:
        return np.zeros_like(mask)
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    return labels == int(counts.argmax())


# ------------------------------------------------------- distance (jump flood)

def nearest_seed(seed_mask):
    """Jump-flooding: for every cell, the coordinates of the nearest seed cell.

    Fully vectorised - log2(n) passes of 9 neighbour comparisons. Returns
    (sy, sx) integer arrays; cells with no reachable seed hold -1.
    """
    h, w = seed_mask.shape
    sx = np.full((h, w), -1, dtype=np.int32)
    sy = np.full((h, w), -1, dtype=np.int32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.int32)
    sx[seed_mask] = xx[seed_mask]
    sy[seed_mask] = yy[seed_mask]

    def d2(cy, cx):
        valid = cx >= 0
        dd = (xx - cx).astype(np.float64) ** 2 + (yy - cy).astype(np.float64) ** 2
        return np.where(valid, dd, np.inf)

    step = 1 << max(0, int(math.ceil(math.log2(max(w, h, 2)))) - 1)
    best = d2(sy, sx)
    while step >= 1:
        for dy in (-step, 0, step):
            for dx in (-step, 0, step):
                if dx == 0 and dy == 0:
                    continue
                cx = np.roll(np.roll(sx, dy, axis=0), dx, axis=1)
                cy = np.roll(np.roll(sy, dy, axis=0), dx, axis=1)
                # roll wraps; blank out the wrapped band so seeds do not teleport
                if dy > 0:
                    cx[:dy, :] = -1; cy[:dy, :] = -1
                elif dy < 0:
                    cx[dy:, :] = -1; cy[dy:, :] = -1
                if dx > 0:
                    cx[:, :dx] = -1; cy[:, :dx] = -1
                elif dx < 0:
                    cx[:, dx:] = -1; cy[:, dx:] = -1
                cand = d2(cy, cx)
                better = cand < best
                if better.any():
                    best = np.where(better, cand, best)
                    sx = np.where(better, cx, sx)
                    sy = np.where(better, cy, sy)
        step //= 2
    return sy, sx


def distance_to_mask(mask):
    """Euclidean distance from every cell to the nearest True cell."""
    if not mask.any():
        return Field(np.full(mask.shape, float(max(mask.shape)), dtype=F32),
                     'distance_to_mask(empty)')
    sy, sx = nearest_seed(mask)
    h, w = mask.shape
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.hypot(xx - sx, yy - sy).astype(F32)
    return Field(d, 'distance_to_mask')


def distance_to_border(w, h):
    """Distance to the nearest edge of the map."""
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.minimum.reduce([xx, yy, (w - 1) - xx, (h - 1) - yy]).astype(F32)
    return Field(d, 'distance_to_border')


def distance_to_points(w, h, points):
    """Distance to the nearest of an explicit point set."""
    pts = np.asarray(points, dtype=F32).reshape(-1, 2)
    if len(pts) == 0:
        return Field(np.full((h, w), float(max(w, h)), dtype=F32), 'distance_to_points(empty)')
    yy, xx = np.mgrid[0:h, 0:w]
    best = np.full((h, w), np.inf, dtype=np.float64)
    for i in range(0, len(pts), 256):
        chunk = pts[i:i + 256]
        d = np.sqrt((xx[None, ...] - chunk[:, 0, None, None]) ** 2 +
                    (yy[None, ...] - chunk[:, 1, None, None]) ** 2)
        best = np.minimum(best, d.min(axis=0))
    return Field(best.astype(F32), 'distance_to_points')


def distance_to_polylines(w, h, polylines, samples_per_unit=1.0):
    """Distance to a set of polylines, by densely sampling them."""
    pts = []
    for line in polylines:
        for (x0, y0), (x1, y1) in zip(line, line[1:]):
            n = max(2, int(math.hypot(x1 - x0, y1 - y0) * samples_per_unit) + 1)
            t = np.linspace(0.0, 1.0, n)
            pts.append(np.stack([x0 + (x1 - x0) * t, y0 + (y1 - y0) * t], axis=1))
    if not pts:
        return Field(np.full((h, w), float(max(w, h)), dtype=F32), 'distance_to_polylines(empty)')
    return Field(distance_to_points(w, h, np.concatenate(pts)).a, 'distance_to_polylines')


# ----------------------------------------------------------- pattern producers

def coords(w, h, aspect=True):
    """Normalised coordinate grids centred on the map."""
    yy, xx = np.mgrid[0:h, 0:w].astype(F32)
    nx = (xx / max(w - 1, 1)) * 2.0 - 1.0
    ny = (yy / max(h - 1, 1)) * 2.0 - 1.0
    if aspect and w != h:
        if w > h:
            ny *= h / w
        else:
            nx *= w / h
    return nx, ny


def constant(w, h, value=0.0):
    return Field(np.full((h, w), value, dtype=F32), f"constant({value})")


def radial(w, h, cx=0.0, cy=0.0, power=1.0):
    """Distance from a normalised centre, 0 at the centre."""
    nx, ny = coords(w, h)
    d = np.hypot(nx - cx, ny - cy)
    return Field(d ** power, f"radial({cx},{cy})")


def angular(w, h, cx=0.0, cy=0.0):
    """Bearing around a centre, normalised to 0..1."""
    nx, ny = coords(w, h)
    a = np.arctan2(ny - cy, nx - cx)
    return Field((a / (2 * math.pi)) + 0.5, f"angular({cx},{cy})")


def spiral(w, h, turns=3.0, cx=0.0, cy=0.0, tightness=1.0):
    nx, ny = coords(w, h)
    r = np.hypot(nx - cx, ny - cy)
    a = np.arctan2(ny - cy, nx - cx)
    v = np.sin(a * turns + r * tightness * 2.0 * math.pi * turns)
    return Field(v * 0.5 + 0.5, f"spiral({turns})")


def waves(w, h, frequency=6.0, angle=0.0, phase=0.0):
    nx, ny = coords(w, h)
    d = nx * math.cos(angle) + ny * math.sin(angle)
    return Field(np.sin(d * frequency * math.pi + phase) * 0.5 + 0.5,
                 f"waves({frequency},{angle:.2f})")


def checker(w, h, size=8):
    yy, xx = np.mgrid[0:h, 0:w]
    v = ((xx // max(size, 1)) + (yy // max(size, 1))) % 2
    return Field(v.astype(F32), f"checker({size})")


_GRAD2 = np.array([[1, 1], [-1, 1], [1, -1], [-1, -1],
                   [1, 0], [-1, 0], [0, 1], [0, -1]], dtype=F32)


def _perm_table(seed):
    p = np.random.default_rng(seed).permutation(256).astype(np.int32)
    return np.concatenate([p, p])


def _fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


def perlin(w, h, scale=8.0, seed=0):
    """Gradient noise in 0..1. Smooth and isotropic, unlike the per-cell hash
    noise in tilegen_system, which is uncorrelated between neighbours."""
    p = _perm_table(seed)
    xs = np.arange(w, dtype=F32) * (scale / max(w, 1))
    ys = np.arange(h, dtype=F32) * (scale / max(h, 1))
    X, Y = np.meshgrid(xs, ys)
    xi = np.floor(X).astype(np.int32); yi = np.floor(Y).astype(np.int32)
    xf = (X - xi).astype(F32); yf = (Y - yi).astype(F32)
    u, v = _fade(xf), _fade(yf)

    def grad(ix, iy, dx, dy):
        g = _GRAD2[p[(p[ix & 255] + (iy & 255)) & 511] & 7]
        return g[..., 0] * dx + g[..., 1] * dy

    n00 = grad(xi, yi, xf, yf)
    n10 = grad(xi + 1, yi, xf - 1, yf)
    n01 = grad(xi, yi + 1, xf, yf - 1)
    n11 = grad(xi + 1, yi + 1, xf - 1, yf - 1)
    a = n00 + u * (n10 - n00)
    b = n01 + u * (n11 - n01)
    return Field((a + v * (b - a)) * 0.5 + 0.5, f"perlin({scale},{seed})")


def fbm(w, h, scale=4.0, octaves=5, persistence=0.5, lacunarity=2.0, seed=0):
    """Fractional Brownian motion - the workhorse for terrain and organics."""
    total = np.zeros((h, w), dtype=F32)
    amp, freq, norm = 1.0, 1.0, 0.0
    for i in range(octaves):
        total += perlin(w, h, scale * freq, seed + i * 977).a * amp
        norm += amp
        amp *= persistence
        freq *= lacunarity
    return Field(total / max(norm, 1e-9), f"fbm({scale},{octaves},{seed})")


def ridged(w, h, scale=4.0, octaves=5, persistence=0.5, lacunarity=2.0, seed=0):
    """Ridged multifractal - mountain crests and canyon walls."""
    total = np.zeros((h, w), dtype=F32)
    amp, freq, norm = 1.0, 1.0, 0.0
    for i in range(octaves):
        n = perlin(w, h, scale * freq, seed + i * 977).a
        total += (1.0 - np.abs(n * 2.0 - 1.0)) * amp
        norm += amp
        amp *= persistence
        freq *= lacunarity
    return Field(total / max(norm, 1e-9), f"ridged({scale},{octaves},{seed})")


def billow(w, h, scale=4.0, octaves=5, seed=0):
    total = np.zeros((h, w), dtype=F32)
    amp, freq, norm = 1.0, 1.0, 0.0
    for i in range(octaves):
        n = perlin(w, h, scale * freq, seed + i * 977).a
        total += np.abs(n * 2.0 - 1.0) * amp
        norm += amp
        amp *= 0.5
        freq *= 2.0
    return Field(total / max(norm, 1e-9), f"billow({scale},{seed})")


def fractal(w, h, scale=3.0, seed=0, warp=0.35):
    """Domain-warped fbm - coastlines and continents at several scales."""
    base = fbm(w, h, scale, 6, 0.5, 2.0, seed)
    wx = fbm(w, h, scale * 1.7, 4, 0.5, 2.0, seed + 4001).a - 0.5
    wy = fbm(w, h, scale * 1.7, 4, 0.5, 2.0, seed + 8009).a - 0.5
    amount = warp * min(w, h) * 0.25
    return Field(base.warp(VectorField(wx, wy), amount).a, f"fractal({scale},{seed})")


def flow_field(w, h, scale=3.0, seed=0):
    """Curl of a noise field: smooth, swirling, divergence-free streamlines."""
    potential = fbm(w, h, scale, 4, 0.5, 2.0, seed)
    return potential.curl().normalize()


def streamlines(vec, seeds, steps=200, step_size=1.0, bounds=None):
    """Integrate paths through a vector field. Returns a list of polylines."""
    h, w = vec.shape
    lines = []
    fx = Field(vec.x); fy = Field(vec.y)
    for sx, sy in seeds:
        x, y = float(sx), float(sy)
        pts = [(x, y)]
        for _ in range(steps):
            dx = float(fx.sample(np.array([x]), np.array([y]))[0])
            dy = float(fy.sample(np.array([x]), np.array([y]))[0])
            m = math.hypot(dx, dy)
            if m < 1e-6:
                break
            x += (dx / m) * step_size
            y += (dy / m) * step_size
            if not (0 <= x < w and 0 <= y < h):
                break
            pts.append((x, y))
        if len(pts) > 2:
            lines.append(pts)
    return lines


# ------------------------------------------------------------------- point sets

def poisson_disc(w, h, radius, seed=0, k=24, margin=0):
    """Bridson Poisson-disc sampling - evenly spaced sites without clumping."""
    rng = np.random.default_rng(seed)
    cell = radius / math.sqrt(2)
    gw, gh = int(math.ceil(w / cell)), int(math.ceil(h / cell))
    grid = -np.ones((gh, gw), dtype=np.int32)
    pts, active = [], []

    def emit(p):
        pts.append(p)
        active.append(len(pts) - 1)
        grid[int(p[1] / cell), int(p[0] / cell)] = len(pts) - 1

    emit((float(rng.uniform(margin, w - margin)), float(rng.uniform(margin, h - margin))))
    while active:
        idx = int(rng.integers(len(active)))
        px, py = pts[active[idx]]
        placed = False
        for _ in range(k):
            ang = float(rng.uniform(0, 2 * math.pi))
            rad = float(rng.uniform(radius, 2 * radius))
            nx, ny = px + math.cos(ang) * rad, py + math.sin(ang) * rad
            if not (margin <= nx < w - margin and margin <= ny < h - margin):
                continue
            gx, gy = int(nx / cell), int(ny / cell)
            ok = True
            for yy in range(max(0, gy - 2), min(gh, gy + 3)):
                for xx in range(max(0, gx - 2), min(gw, gx + 3)):
                    j = grid[yy, xx]
                    if j >= 0 and math.hypot(pts[j][0] - nx, pts[j][1] - ny) < radius:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                emit((nx, ny))
                placed = True
                break
        if not placed:
            active.pop(idx)
    return np.array(pts, dtype=F32)


def phyllotaxis(w, h, count=200, spread=None, cx=None, cy=None):
    """Golden-angle spiral point set - sunflower and nautilus arrangements."""
    cx = w / 2 if cx is None else cx
    cy = h / 2 if cy is None else cy
    spread = (min(w, h) * 0.48 / math.sqrt(max(count, 1))) if spread is None else spread
    i = np.arange(count, dtype=F32)
    ang = i * math.radians(137.507764)
    r = spread * np.sqrt(i)
    return np.stack([cx + np.cos(ang) * r, cy + np.sin(ang) * r], axis=1).astype(F32)


def lloyd_relax(sites, w, h, iterations=2):
    """Move each site to the centroid of its Voronoi cell - evens out the mesh."""
    sites = np.asarray(sites, dtype=F32).copy()
    if len(sites) == 0:
        return sites
    yy, xx = np.mgrid[0:h, 0:w]
    for _ in range(max(0, iterations)):
        cell = _nearest_site_index(xx, yy, sites)
        for i in range(len(sites)):
            m = cell == i
            if m.any():
                sites[i, 0] = xx[m].mean()
                sites[i, 1] = yy[m].mean()
    return sites


def _nearest_site_index(xx, yy, sites):
    best_d = np.full(xx.shape, np.inf)
    best_i = np.zeros(xx.shape, dtype=np.int32)
    for i in range(0, len(sites), 128):
        chunk = sites[i:i + 128]
        d = ((xx[None, ...] - chunk[:, 0, None, None]) ** 2 +
             (yy[None, ...] - chunk[:, 1, None, None]) ** 2)
        local_i = d.argmin(axis=0)
        local_d = d.min(axis=0)
        better = local_d < best_d
        best_d = np.where(better, local_d, best_d)
        best_i = np.where(better, local_i + i, best_i)
    return best_i


class VoronoiResult:
    """Cell ids plus the two nearest-site distances (F1/F2 give crisp borders)."""

    __slots__ = ('cell', 'f1', 'f2', 'sites')

    def __init__(self, cell, f1, f2, sites):
        self.cell = cell
        self.f1 = Field(f1, 'voronoi.f1')
        self.f2 = Field(f2, 'voronoi.f2')
        self.sites = sites

    @property
    def border_tension(self):
        """Small near a shared edge, large deep inside a cell (atlas GEO-01)."""
        return Field(self.f2.a - self.f1.a, 'voronoi.border_tension')

    def border_mask(self, width=1):
        c = self.cell
        edge = np.zeros(c.shape, dtype=bool)
        edge[:, :-1] |= c[:, :-1] != c[:, 1:]
        edge[:-1, :] |= c[:-1, :] != c[1:, :]
        return _dilate(edge, width - 1) if width > 1 else edge

    def adjacency(self):
        """Region adjacency graph: {(a,b): shared border cell count}."""
        c = self.cell
        pairs = {}
        for A, B in ((c[:, :-1], c[:, 1:]), (c[:-1, :], c[1:, :])):
            m = A != B
            for a, b in zip(A[m].ravel(), B[m].ravel()):
                key = (int(min(a, b)), int(max(a, b)))
                pairs[key] = pairs.get(key, 0) + 1
        return pairs


def voronoi(w, h, sites, relax=0):
    """Voronoi partition over an explicit site list."""
    sites = np.asarray(sites, dtype=F32).reshape(-1, 2)
    if relax:
        sites = lloyd_relax(sites, w, h, relax)
    yy, xx = np.mgrid[0:h, 0:w]
    d1 = np.full((h, w), np.inf); d2 = np.full((h, w), np.inf)
    cell = np.zeros((h, w), dtype=np.int32)
    for i in range(0, len(sites), 64):
        chunk = sites[i:i + 64]
        d = np.sqrt((xx[None, ...] - chunk[:, 0, None, None]) ** 2 +
                    (yy[None, ...] - chunk[:, 1, None, None]) ** 2)
        for j in range(len(chunk)):
            dj = d[j]
            closer = dj < d1
            d2 = np.where(closer, d1, np.minimum(d2, dj))
            cell = np.where(closer, i + j, cell)
            d1 = np.where(closer, dj, d1)
    return VoronoiResult(cell, d1, d2, sites)


# --------------------------------------------------------- reaction diffusion

RD_PRESETS = {
    #                feed    kill
    'coral':        (0.0545, 0.0620),
    'mitosis':      (0.0367, 0.0649),
    'fingerprint':  (0.0370, 0.0600),
    'maze':         (0.0290, 0.0570),
    'holes':        (0.0390, 0.0580),
    'worms':        (0.0580, 0.0630),
    'spots':        (0.0350, 0.0650),
    'chaos':        (0.0260, 0.0510),
    'flower':       (0.0550, 0.0620),
}


def reaction_diffusion(w, h, preset='coral', steps=3000, seed=0,
                       du=0.16, dv=0.08, dt=1.0, feed=None, kill=None,
                       seed_count=None, mask=None):
    """Gray-Scott reaction-diffusion. Returns the V concentration field.

    Different (feed, kill) pairs produce genuinely different map families, not
    cosmetic variation - see RD_PRESETS.
    """
    f, k = RD_PRESETS.get(preset, RD_PRESETS['coral'])
    if feed is not None:
        f = feed
    if kill is not None:
        k = kill
    rng = np.random.default_rng(seed)
    u = np.ones((h, w), dtype=np.float32)
    v = np.zeros((h, w), dtype=np.float32)

    n = seed_count if seed_count is not None else max(3, (w * h) // 900)
    for _ in range(n):
        cx = int(rng.integers(2, max(3, w - 2)))
        cy = int(rng.integers(2, max(3, h - 2)))
        r = int(rng.integers(2, 5))
        y0, y1 = max(0, cy - r), min(h, cy + r + 1)
        x0, x1 = max(0, cx - r), min(w, cx + r + 1)
        u[y0:y1, x0:x1] = 0.50
        v[y0:y1, x0:x1] = 0.25

    for _ in range(steps):
        lu = (np.roll(u, 1, 0) + np.roll(u, -1, 0) +
              np.roll(u, 1, 1) + np.roll(u, -1, 1) - 4.0 * u)
        lv = (np.roll(v, 1, 0) + np.roll(v, -1, 0) +
              np.roll(v, 1, 1) + np.roll(v, -1, 1) - 4.0 * v)
        uvv = u * v * v
        u += (du * lu - uvv + f * (1.0 - u)) * dt
        v += (dv * lv + uvv - (f + k) * v) * dt
        if mask is not None:
            v[~mask] = 0.0
        np.clip(u, 0.0, 1.0, out=u)
        np.clip(v, 0.0, 1.0, out=v)

    return Field(v, f"reaction_diffusion({preset},{steps},{seed})")


# ----------------------------------------------------------------- hydrology

def fill_depressions(elevation, epsilon=1e-4):
    """Priority-flood: raise pits so every cell drains to the map edge."""
    import heapq
    a = np.asarray(elevation, dtype=np.float64)
    h, w = a.shape
    filled = np.full_like(a, np.inf)
    heap = []
    for x in range(w):
        for y in (0, h - 1):
            filled[y, x] = a[y, x]; heapq.heappush(heap, (a[y, x], y, x))
    for y in range(h):
        for x in (0, w - 1):
            if filled[y, x] == np.inf:
                filled[y, x] = a[y, x]; heapq.heappush(heap, (a[y, x], y, x))
    while heap:
        e, y, x = heapq.heappop(heap)
        if e > filled[y, x]:
            continue
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and filled[ny, nx] == np.inf:
                filled[ny, nx] = max(a[ny, nx], e + epsilon)
                heapq.heappush(heap, (filled[ny, nx], ny, nx))
    return Field(filled.astype(F32), 'fill_depressions')


def flow_accumulation(elevation):
    """D8 drainage. Returns (accumulation, receiver_index) - accumulation
    highlights river channels and confluences."""
    a = np.asarray(elevation, dtype=np.float64)
    h, w = a.shape
    n = h * w
    flat = a.ravel()
    receiver = np.arange(n, dtype=np.int64)
    best_slope = np.zeros(n)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            src_y0, src_y1 = max(0, -dy), h - max(0, dy)
            src_x0, src_x1 = max(0, -dx), w - max(0, dx)
            src = (np.arange(src_y0, src_y1)[:, None] * w +
                   np.arange(src_x0, src_x1)[None, :]).ravel()
            dst = src + dy * w + dx
            drop = (flat[src] - flat[dst]) / math.hypot(dx, dy)
            better = drop > best_slope[src]
            idx = src[better]
            best_slope[idx] = drop[better]
            receiver[idx] = dst[better]

    order = np.argsort(-flat, kind='stable')
    acc = np.ones(n, dtype=np.float64)
    for i in order:
        r = receiver[i]
        if r != i:
            acc[r] += acc[i]
    return Field(acc.reshape(h, w).astype(F32), 'flow_accumulation'), receiver


def trace_channels(acc, receiver, shape, source_mask, stop_mask=None, max_len=None):
    """Follow the D8 receiver graph downstream from each source.

    Thresholding flow accumulation directly picks isolated local maxima and
    yields scattered puddles; walking the receiver graph produces channels that
    are connected by construction and actually reach the sea.
    """
    h, w = shape
    river = np.zeros(h * w, dtype=bool)
    stop = stop_mask.ravel() if stop_mask is not None else np.zeros(h * w, dtype=bool)
    limit = max_len or (h + w) * 2
    for start in np.flatnonzero(source_mask.ravel()):
        i = int(start)
        for _ in range(limit):
            if river[i] or stop[i]:
                break
            river[i] = True
            nxt = int(receiver[i])
            if nxt == i:
                break
            # D8 steps diagonally, but connectivity and the runtime's navigation
            # are both 4-connected, so a diagonal hop would leave the channel as
            # a chain of disjoint fragments. Fill the corner it passes through.
            y0, x0 = divmod(i, w)
            y1, x1 = divmod(nxt, w)
            if y0 != y1 and x0 != x1:
                river[y1 * w + x0] = True
            i = nxt
    return river.reshape(h, w)


def watershed(w, h, seed=0, scale=3.0, octaves=6, sea_level=0.38, river_quantile=0.97):
    """Full hydrology pass: terrain, filled basins, drainage, rivers, sea."""
    elevation = fbm(w, h, scale, octaves, 0.5, 2.0, seed).normalized()
    ridge = ridged(w, h, scale * 1.3, 4, 0.5, 2.0, seed + 31).normalized()
    elevation = elevation.blend(ridge, 0.35).normalized()

    # Depression filling leaves perfectly flat pools, and D8 breaks ties on a
    # flat surface the same way in every cell, so drainage there comes out as
    # long straight axis-aligned runs. A jitter far below the terrain relief but
    # well above the fill epsilon breaks those ties. It must be applied *before*
    # filling: added afterwards it punches micro-pits into the flats that have
    # no downhill neighbour, and every channel entering one dead-ends there.
    jitter = np.random.default_rng(seed_stream(seed, 'env.flat_tiebreak')).random(
        (h, w)).astype(F32) * 1e-3
    filled = fill_depressions(elevation.a + jitter)
    acc, receiver = flow_accumulation(filled.a)
    logacc = Field(np.log1p(acc.a), 'log_accumulation').normalized()

    sea = elevation.a < sea_level
    sources = (logacc.a > np.quantile(logacc.a, river_quantile)) & ~sea
    river = trace_channels(acc.a, receiver, (h, w), sources, stop_mask=sea) & ~sea
    return {
        'elevation': elevation,
        'filled': filled,
        'accumulation': logacc,
        'sea': sea,
        'river': river,
        'slope': elevation.slope().normalized(),
    }


# ------------------------------------------------------- differential growth

def _neighbour_candidates(pts, radius, w, h, K=24):
    """Bucket points into a uniform grid and return, for every point, the index
    list of points in its own and the 8 surrounding cells.

    Turns the O(n^2) repulsion in differential growth into O(n*K). Cells that
    overflow K simply drop the extra candidates, which costs a little repulsion
    accuracy in the densest folds and never breaks correctness.
    """
    n = len(pts)
    gw = max(1, int(w / radius) + 1)
    gh = max(1, int(h / radius) + 1)
    cx = np.clip((pts[:, 0] / radius).astype(np.int64), 0, gw - 1)
    cy = np.clip((pts[:, 1] / radius).astype(np.int64), 0, gh - 1)
    cell = cy * gw + cx

    order = np.argsort(cell, kind='stable')
    sorted_cells = cell[order]
    first = np.empty(n, dtype=bool)
    first[0] = True
    np.not_equal(sorted_cells[1:], sorted_cells[:-1], out=first[1:])
    starts = np.flatnonzero(first)
    counts = np.diff(np.append(starts, n))
    rank = np.arange(n) - np.repeat(starts, counts)
    keep = rank < K

    grid = np.full((gh * gw, K), -1, dtype=np.int64)
    grid[sorted_cells[keep], rank[keep]] = order[keep]

    out = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            ny = np.clip(cy + dy, 0, gh - 1)
            nx = np.clip(cx + dx, 0, gw - 1)
            out.append(grid[ny * gw + nx])
    return np.concatenate(out, axis=1)


def differential_growth(w, h, steps=220, seed=0, start_radius=None,
                        max_nodes=2400, repel_radius=5.0, rest=1.7,
                        attract=0.28, repel=0.55, growth_rate=0.055,
                        margin=3.0, bounds_mask=None):
    """Grow a closed loop that folds under its own pressure.

    Nodes are held at a rest spacing by springs along the curve and pushed apart
    by non-neighbours within repel_radius. New nodes are inserted into the
    longest segments every step, so the curve lengthens faster than it can stay
    convex and buckles into brain/coral/intestine folds. Returns the polyline.
    """
    rng = np.random.default_rng(seed)
    cx, cy = w / 2.0, h / 2.0
    r = start_radius if start_radius is not None else min(w, h) * 0.10
    n0 = 24
    ang = np.linspace(0, 2 * math.pi, n0, endpoint=False)
    pts = np.stack([cx + np.cos(ang) * r, cy + np.sin(ang) * r], axis=1).astype(np.float64)
    pts += rng.normal(0, 0.25, pts.shape)

    for _ in range(steps):
        n = len(pts)
        if n < 4:
            break
        force = np.zeros_like(pts)

        # spring toward the rest spacing along the curve, in both directions
        for shift in (1, -1):
            d = np.roll(pts, shift, axis=0) - pts
            L = np.linalg.norm(d, axis=1, keepdims=True)
            L = np.where(L < 1e-9, 1e-9, L)
            force += (d / L) * (L - rest) * attract

        # non-neighbour repulsion is what forces the folding
        cand = _neighbour_candidates(pts, repel_radius, w, h)
        valid = cand >= 0
        cj = np.where(valid, cand, 0)
        idx = np.arange(n)[:, None]
        skip = (~valid) | (cj == idx) | (cj == (idx + 1) % n) | (cj == (idx - 1) % n)
        delta = pts[:, None, :] - pts[cj]
        dist = np.sqrt((delta ** 2).sum(-1))
        dist = np.where(skip, np.inf, dist)
        close = dist < repel_radius
        if close.any():
            safe = np.where(dist < 1e-6, 1e-6, dist)
            weight = np.where(close, (repel_radius - dist) / repel_radius, 0.0)
            force += (delta / safe[..., None] * weight[..., None]).sum(axis=1) * repel

        pts = pts + force
        pts[:, 0] = np.clip(pts[:, 0], margin, w - 1 - margin)
        pts[:, 1] = np.clip(pts[:, 1], margin, h - 1 - margin)

        if bounds_mask is not None:
            gx = np.clip(pts[:, 0].astype(int), 0, w - 1)
            gy = np.clip(pts[:, 1].astype(int), 0, h - 1)
            outside = ~bounds_mask[gy, gx]
            if outside.any():
                pull = np.stack([cx - pts[:, 0], cy - pts[:, 1]], axis=1)
                nrm = np.linalg.norm(pull, axis=1, keepdims=True)
                pts[outside] += (pull / np.where(nrm < 1e-6, 1e-6, nrm))[outside] * 0.8

        # insert nodes into the longest segments so the curve keeps lengthening
        if len(pts) < max_nodes:
            seg = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
            k = min(max(1, int(len(pts) * growth_rate)), max_nodes - len(pts))
            long_idx = np.sort(np.argsort(-seg)[:k])
            mids = (pts[long_idx] + pts[(long_idx + 1) % len(pts)]) * 0.5
            pts = np.insert(pts, long_idx + 1, mids, axis=0)

    return pts.astype(F32)


# ------------------------------------------------------------------- truchet

def truchet(w_cells, h_cells, seed=0, optimize=0, target='long_loops'):
    """Assign a rotation to every cell of a Truchet quarter-arc lattice.

    Quarter-arc motifs always meet edge-midpoint to edge-midpoint, so every
    assignment is valid and the plane fills with closed loops. Optimisation
    hill-climbs rotations toward a global objective instead of leaving it random.
    """
    rng = np.random.default_rng(seed)
    rot = rng.integers(0, 2, size=(h_cells, w_cells)).astype(np.int8)

    def score(r):
        loops = truchet_loops(r)
        if not loops:
            return 0.0
        sizes = sorted((len(v) for v in loops.values()), reverse=True)
        if target == 'long_loops':
            return sizes[0]
        if target == 'many_loops':
            return len(sizes)
        return -abs(len(sizes) - 12)

    if optimize:
        best = score(rot)
        for _ in range(optimize):
            y = int(rng.integers(h_cells)); x = int(rng.integers(w_cells))
            rot[y, x] ^= 1
            s = score(rot)
            # strictly-better only: accepting equal scores lets the search
            # random-walk across the plateau to the same canonical optimum from
            # every seed, which silently destroys seed variety on small grids
            if s > best:
                best = s
            else:
                rot[y, x] ^= 1
    return rot


def truchet_loops(rot):
    """Trace the closed loops of a Truchet rotation grid.

    Each cell carries two arcs joining its four edge midpoints. Rotation 0 joins
    N-W and S-E; rotation 1 joins N-E and S-W. Walking those joins partitions all
    edge midpoints into cycles. Returns {loop_id: [edge keys]}.
    """
    h, w = rot.shape

    def edge_key(y, x, side):
        # canonical id for the edge midpoint shared by two neighbouring cells
        if side == 'N':
            return ('H', y, x)
        if side == 'S':
            return ('H', y + 1, x)
        if side == 'W':
            return ('V', y, x)
        return ('V', y, x + 1)

    # for each cell, which pairs of sides its two arcs connect
    pair_by_rot = {0: (('N', 'W'), ('S', 'E')), 1: (('N', 'E'), ('S', 'W'))}

    adj = {}
    for y in range(h):
        for x in range(w):
            for a, b in pair_by_rot[int(rot[y, x])]:
                ka, kb = edge_key(y, x, a), edge_key(y, x, b)
                adj.setdefault(ka, []).append(kb)
                adj.setdefault(kb, []).append(ka)

    loops, seen, lid = {}, set(), 0
    for start in adj:
        if start in seen:
            continue
        lid += 1
        stack, members = [start], []
        seen.add(start)
        while stack:
            k = stack.pop()
            members.append(k)
            for nb in adj.get(k, ()):
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        loops[lid] = members
    return loops


# ------------------------------------------------------------------------- DLA

def dla(w, h, particles=3000, seed=0, stick=1.0, start_mask=None, max_steps=3000):
    """Diffusion-limited aggregation - dendritic mineral/frost growth.

    Walkers reflect off the map boundary rather than being discarded, so the
    aggregate actually reaches the requested particle count; letting them escape
    silently wastes most of the budget and yields a tiny cluster.
    """
    rng = np.random.default_rng(seed)
    grid = np.zeros((h, w), dtype=bool)
    if start_mask is not None:
        grid |= start_mask
    else:
        grid[h // 2, w // 2] = True
    cx, cy = w / 2.0, h / 2.0
    radius = 3.0
    max_r = math.hypot(cx, cy)
    # pre-draw the random walk in bulk; indexing into it is far cheaper than
    # calling the generator once per step
    for _ in range(particles):
        ang = rng.uniform(0, 2 * math.pi)
        r = min(radius + 4.0, max_r)
        x = int(np.clip(cx + math.cos(ang) * r, 1, w - 2))
        y = int(np.clip(cy + math.sin(ang) * r, 1, h - 2))
        walk = rng.integers(-1, 2, size=(max_steps, 2))
        for sx, sy in walk:
            x += int(sx); y += int(sy)
            # reflect at the border so no walker is lost
            if x < 1:
                x = 1
            elif x > w - 2:
                x = w - 2
            if y < 1:
                y = 1
            elif y > h - 2:
                y = h - 2
            if (grid[y - 1, x] or grid[y + 1, x] or
                    grid[y, x - 1] or grid[y, x + 1]):
                if stick >= 1.0 or rng.random() <= stick:
                    grid[y, x] = True
                    radius = max(radius, math.hypot(x - cx, y - cy) + 2.0)
                    break
    return grid
