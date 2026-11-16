"""Vector geometry for the field engine: exact Delaunay triangulation, clipped
Voronoi cells, Lloyd relaxation, the proximity graphs built on them, contour
tracing of masks and scalar fields into closed rings, and radial basis function
fields through scattered samples.

Clean-room implementation from M. de Berg, O. Cheong, M. van Kreveld and
M. Overmars, Computational Geometry: Algorithms and Applications, 3rd ed.
(2008), chapters 7 and 9; A. Bowyer, The Computer Journal 24(2), 1981;
D. F. Watson, The Computer Journal 24(2), 1981; S. Lloyd, IEEE Trans.
Information Theory 28(2), 1982; Q. Du, V. Faber and M. Gunzburger, SIAM Review
41(4), 1999; A. Secord, 'Weighted Voronoi stippling', NPAR 2002; K. R. Gabriel
and R. R. Sokal, Systematic Zoology 18(3), 1969; G. T. Toussaint, 'The relative
neighbourhood graph of a finite planar set', Pattern Recognition 12, 1980;
J. R. Shewchuk, 'Adaptive precision floating-point arithmetic and fast robust
geometric predicates', 1997; H. Edelsbrunner and E. P. Mucke, 'Simulation of
simplicity', ACM TOG 9(1), 1990; W. E. Lorensen and H. E. Cline, 'Marching
cubes', SIGGRAPH 1987 (the 2D case, marching squares); C. Maple, 'Geometric
design and space planning using the marching squares and marching cube
algorithms', GMAG 2003; S. Suzuki and K. Abe, 'Topological structural analysis
of digitized binary images by border following', CVGIP 30(1), 1985; A.
Rosenfeld, 'Connectivity in digital pictures', JACM 17(1), 1970; H. Wendland,
Scattered Data Approximation, Cambridge UP 2005; J. Duchon, 'Splines minimizing
rotation-invariant semi-norms in Sobolev spaces', 1977; C. A. Micchelli,
'Interpolation of scattered data: distance matrices and conditionally positive
definite functions', Constructive Approximation 2, 1986; J. C. Carr et al.,
'Reconstruction and representation of 3D objects with radial basis functions',
SIGGRAPH 2001.
Feature list informed by Sverchok (GPL-3.0); no Sverchok code was consulted.

Coordinates are (x, y) in the same pixel frame as `tilegen_fields`: a sites
array uses the units of `voronoi`/`lloyd_relax`, so a pixel (x, y) sits at the
point (x, y). Nothing here knows about tiles. Everything is deterministic; the
only randomness is an optional named stream that permutes the insertion order,
and the result does not depend on it.

Imports downward only: this module needs `tilegen_fields` and nothing else.
"""
from __future__ import annotations

import math

import numpy as np

import tilegen_fields as tf

I32 = np.int32
F64 = np.float64
_GHOST = -1                 # the single symbolic point at infinity
_LATTICE_BITS = 24          # snapped coordinates are integers in [0, 2**24]


# --------------------------------------------------------------- triangulation

class Triangulation:
    """Delaunay triangulation of a de-duplicated point set.

    points     (n, 2) float64: first occurrence of every distinct input point
    index_map  (n_in,) int32: kept index of every input point
    tris       (m, 3) int32: counter-clockwise, smallest index first, sorted
    neighbors  (m, 3) int32: triangle across the edge opposite each vertex, or -1
    hull       (k,) int32: counter-clockwise boundary vertices, collinear ones
               included, starting at the smallest index. With fewer than three
               points, or when all points are collinear, there are no
               triangles and `hull` lists all points along the line, sorted.
    """

    __slots__ = ('points', 'index_map', 'tris', 'neighbors', 'hull', 'lattice')

    def __init__(self, points, index_map, tris, neighbors, hull, lattice):
        self.points = points
        self.index_map = index_map
        self.tris = tris
        self.neighbors = neighbors
        self.hull = hull
        self.lattice = lattice          # (xs, ys): exact integer coordinates

    def __repr__(self):
        return (f"Triangulation({len(self.points)} points, {len(self.tris)} triangles, "
                f"hull {len(self.hull)})")

    @property
    def degenerate(self):
        return len(self.tris) == 0

    def edges(self):
        """Delaunay edges as a sorted (e, 2) int32 array of index pairs i < j."""
        n = len(self.points)
        if n < 2:
            return np.zeros((0, 2), dtype=I32)
        if self.degenerate:
            line = self.hull.astype(I32)
            return np.sort(np.stack([line[:-1], line[1:]], axis=1), axis=1).astype(I32)
        t = self.tris
        both = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
        both = np.sort(both, axis=1)
        return np.unique(both, axis=0).astype(I32)

    def vertex_neighbors(self):
        """Delaunay neighbours of every vertex, as sorted lists."""
        out = [[] for _ in range(len(self.points))]
        for a, b in self.edges().tolist():
            out[a].append(b)
            out[b].append(a)
        return [sorted(v) for v in out]

    def circumcenters(self):
        """Circumcentres of the triangles as an (m, 2) float64 array."""
        if self.degenerate:
            return np.zeros((0, 2), dtype=F64)
        a, b, c = (self.points[self.tris[:, i]] for i in range(3))
        d = 2.0 * (a[:, 0] * (b[:, 1] - c[:, 1]) + b[:, 0] * (c[:, 1] - a[:, 1])
                   + c[:, 0] * (a[:, 1] - b[:, 1]))
        d = np.where(d == 0.0, 1e-300, d)
        a2, b2, c2 = ((p ** 2).sum(axis=1) for p in (a, b, c))
        ux = (a2 * (b[:, 1] - c[:, 1]) + b2 * (c[:, 1] - a[:, 1]) + c2 * (a[:, 1] - b[:, 1])) / d
        uy = (a2 * (c[:, 0] - b[:, 0]) + b2 * (a[:, 0] - c[:, 0]) + c2 * (b[:, 0] - a[:, 0])) / d
        return np.stack([ux, uy], axis=1)


def _empty_triangulation(points=None, index_map=None):
    pts = np.zeros((0, 2), dtype=F64) if points is None else points
    imap = np.zeros(0, dtype=I32) if index_map is None else index_map
    return Triangulation(pts, imap, np.zeros((0, 3), dtype=I32), np.zeros((0, 3), dtype=I32),
                         np.zeros(0, dtype=I32), ([], []))


def _dedupe(pts, tol):
    """Keep the first occurrence of every point; later ones within `tol`
    (Chebyshev) of a kept point map to it."""
    kept, index_map = [], np.zeros(len(pts), dtype=I32)
    buckets = {}
    inv = 1.0 / tol
    for i, (x, y) in enumerate(pts.tolist()):
        cx, cy = int(math.floor(x * inv)), int(math.floor(y * inv))
        found = -1
        for gx in (cx - 1, cx, cx + 1):
            for gy in (cy - 1, cy, cy + 1):
                for j in buckets.get((gx, gy), ()):
                    kx, ky = kept[j]
                    if abs(kx - x) <= tol and abs(ky - y) <= tol:
                        found = j
                        break
                if found >= 0:
                    break
            if found >= 0:
                break
        if found < 0:
            found = len(kept)
            kept.append((x, y))
            buckets.setdefault((cx, cy), []).append(found)
        index_map[i] = found
    return np.array(kept, dtype=F64).reshape(-1, 2), index_map


def delaunay(points, seed_name=None):
    """Exact 2D Delaunay triangulation by randomized incremental insertion.

    Bowyer-Watson cavity updates on a triangulation of the plane plus one
    symbolic point at infinity, so the hull needs no super-triangle. Points are
    snapped to a power-of-two lattice of at most 2**-24 times the extent and
    every orientation and in-circle test is evaluated in exact integer
    arithmetic. Cocircular ties are broken by symbolic perturbation of the
    lifted heights, ranked by lattice order, which makes the result the unique
    Delaunay triangulation of the perturbed set: it does not depend on the
    insertion order, the input order or the seed.

    `seed_name` selects the insertion permutation: None inserts in
    lexicographic order, a string draws the permutation from
    `tf.rng_for(0, name)`, and a (world_seed, name[, index]) tuple from
    `tf.rng_for(*tuple)`. It only affects speed.

    Points closer than max(1e-9, extent * 2**-24) (Chebyshev) are merged into
    their first occurrence; `index_map` records the merge.
    """
    pts = np.asarray(points, dtype=F64)
    if pts.size == 0:
        return _empty_triangulation()
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points must have shape (n, 2), got {pts.shape}")
    if not np.isfinite(pts).all():
        raise ValueError("points must be finite")

    lo = pts.min(axis=0)
    extent = float((pts.max(axis=0) - lo).max())
    if extent == 0.0:
        return Triangulation(pts[:1].copy(), np.zeros(len(pts), dtype=I32), np.zeros((0, 3), dtype=I32),
                             np.zeros((0, 3), dtype=I32), np.zeros(1, dtype=I32), ([0], [0]))
    exp2 = int(math.ceil(math.log2(extent))) - _LATTICE_BITS
    step = math.ldexp(1.0, exp2)                      # lattice spacing, a power of two
    tol = max(1e-9, step)
    kept, index_map = _dedupe(pts, tol)
    m = len(kept)

    inv = math.ldexp(1.0, -exp2)
    xs = [int(v) for v in np.floor((kept[:, 0] - lo[0]) * inv + 0.5)]
    ys = [int(v) for v in np.floor((kept[:, 1] - lo[1]) * inv + 0.5)]
    lattice = (xs, ys)

    if m < 3:
        order = sorted(range(m), key=lambda i: (xs[i], ys[i]))
        return Triangulation(kept, index_map, np.zeros((0, 3), dtype=I32), np.zeros((0, 3), dtype=I32),
                             np.array(order, dtype=I32), lattice)

    lex = sorted(range(m), key=lambda i: (xs[i], ys[i]))
    rank = [0] * m
    for r, i in enumerate(lex):
        rank[i] = r
    if seed_name is None:
        insertion = lex
    else:
        rng = tf.rng_for(*seed_name) if isinstance(seed_name, tuple) else tf.rng_for(0, str(seed_name))
        insertion = [int(i) for i in rng.permutation(m)]

    built = _incremental(xs, ys, rank, insertion)
    if built is None:                                   # every point is collinear
        return Triangulation(kept, index_map, np.zeros((0, 3), dtype=I32), np.zeros((0, 3), dtype=I32),
                             np.array(lex, dtype=I32), lattice)
    tris, nbrs, hull = built
    tris = np.array(tris, dtype=I32).reshape(-1, 3)
    nbrs = np.array(nbrs, dtype=I32).reshape(-1, 3)

    # canonical form: rotate so the smallest index leads, then sort rows
    rot = tris.argmin(axis=1)
    cols = (rot[:, None] + np.arange(3)[None, :]) % 3
    rows = np.arange(len(tris))[:, None]
    tris, nbrs = tris[rows, cols], nbrs[rows, cols]
    order = np.lexsort((tris[:, 2], tris[:, 1], tris[:, 0]))
    inverse = np.empty(len(order), dtype=I32)
    inverse[order] = np.arange(len(order), dtype=I32)
    tris = tris[order]
    nbrs = nbrs[order]
    nbrs = np.where(nbrs >= 0, inverse[np.maximum(nbrs, 0)], -1).astype(I32)
    hull = np.array(hull, dtype=I32)
    hull = np.roll(hull, -int(hull.argmin()))
    return Triangulation(kept, index_map, tris, nbrs, hull, lattice)


def _incremental(X, Y, rank, insertion):
    """Bowyer-Watson on integer coordinates. Returns (tris, neighbours, hull)
    for the real triangles, or None when every point is collinear."""
    G = _GHOST
    m = len(X)

    def orient(a, b, c):
        return (X[b] - X[a]) * (Y[c] - Y[a]) - (Y[b] - Y[a]) * (X[c] - X[a])

    def inside_circle(a, b, c, d):
        """d strictly inside the (perturbed) circumcircle of CCW triangle abc."""
        adx, ady = X[a] - X[d], Y[a] - Y[d]
        bdx, bdy = X[b] - X[d], Y[b] - Y[d]
        cdx, cdy = X[c] - X[d], Y[c] - Y[d]
        det = ((adx * adx + ady * ady) * (bdx * cdy - cdx * bdy)
               + (bdx * bdx + bdy * bdy) * (cdx * ady - adx * cdy)
               + (cdx * cdx + cdy * cdy) * (adx * bdy - bdx * ady))
        if det:
            return det > 0
        # Symbolic perturbation of the lifted heights: the first non-zero
        # cofactor, taken in rank order, decides. The cofactor of d is the
        # orientation of a, b, c, which is never zero for a real triangle.
        terms = ((rank[a], orient(b, c, d)), (rank[b], -orient(a, c, d)),
                 (rank[c], orient(a, b, d)), (rank[d], -orient(a, b, c)))
        for _, cofactor in sorted(terms):
            if cofactor:
                return cofactor > 0
        return False

    # ---- first non-degenerate triangle from the insertion order
    first = insertion[0]
    second = insertion[1]
    third = -1
    for v in insertion[2:]:
        if orient(first, second, v):
            third = v
            break
    if third < 0:
        return None
    a, b, c = first, second, third
    if orient(a, b, c) < 0:
        b, c = c, b

    TV, TN = [], []              # vertices / neighbours per triangle id
    free = []

    def new_triangle(v0, v1, v2):
        if free:
            t = free.pop()
            TV[t] = (v0, v1, v2)
            TN[t] = [-1, -1, -1]
        else:
            t = len(TV)
            TV.append((v0, v1, v2))
            TN.append([-1, -1, -1])
        return t

    t0 = new_triangle(a, b, c)
    g_ab = new_triangle(b, a, G)
    g_bc = new_triangle(c, b, G)
    g_ca = new_triangle(a, c, G)
    TN[t0][:] = [g_bc, g_ca, g_ab]
    TN[g_ab][:] = [g_ca, g_bc, t0]
    TN[g_bc][:] = [g_ab, g_ca, t0]
    TN[g_ca][:] = [g_bc, g_ab, t0]
    vertex_tri = [-1] * m
    vertex_tri[a] = vertex_tri[b] = vertex_tri[c] = t0

    def in_conflict(t, p):
        v = TV[t]
        if G not in v:
            return inside_circle(v[0], v[1], v[2], p)
        gi = v.index(G)
        u, w = v[(gi + 1) % 3], v[(gi + 2) % 3]        # hull edge runs w -> u
        s = orient(w, u, p)
        if s:
            return s < 0
        return ((X[p] - X[w]) * (X[u] - X[w]) + (Y[p] - Y[w]) * (Y[u] - Y[w]) > 0
                and (X[p] - X[u]) * (X[w] - X[u]) + (Y[p] - Y[u]) * (Y[w] - Y[u]) > 0)

    # coarse bucket grid: latest inserted vertex per cell, to start the walk
    side = max(1, int(math.sqrt(m / 2.0)))
    span = (1 << _LATTICE_BITS) + 1
    bucket = {}

    def bucket_of(v):
        return (X[v] * side // span, Y[v] * side // span)

    for v in (a, b, c):
        bucket[bucket_of(v)] = v
    last = c

    def locate(p, start):
        """Walk to a triangle in conflict with p."""
        t = start
        if G in TV[t]:
            t = TN[t][TV[t].index(G)]
        turn = 0
        limit = 8 * len(TV) + 64
        for _ in range(limit):
            v = TV[t]
            moved = False
            for k in range(3):
                i = (turn + k) % 3
                u, w = v[(i + 1) % 3], v[(i + 2) % 3]
                if orient(u, w, p) < 0:
                    t = TN[t][i]
                    if G in TV[t]:
                        return t
                    moved = True
                    break
            turn += 1
            if not moved:
                return t
        for t in range(len(TV)):                       # safety net, never expected
            if TV[t] is not None and in_conflict(t, p):
                return t
        raise RuntimeError("point location failed")

    for p in insertion:
        if p in (a, b, c):
            continue
        key = bucket_of(p)
        hint = bucket.get(key, last)
        t_start = vertex_tri[hint]
        seed_t = locate(p, t_start)

        # ---- cavity: every triangle in conflict with p, connected to seed_t
        status = {seed_t: True}
        cavity = [seed_t]
        stack = [seed_t]
        while stack:
            t = stack.pop()
            for n in TN[t]:
                if n not in status:
                    hit = in_conflict(n, p)
                    status[n] = hit
                    if hit:
                        cavity.append(n)
                        stack.append(n)
        boundary = []                                  # (u, w, outside triangle)
        for t in cavity:
            v, nb = TV[t], TN[t]
            for i in range(3):
                if not status[nb[i]]:
                    boundary.append((v[(i + 1) % 3], v[(i + 2) % 3], nb[i]))
        for t in cavity:
            TV[t] = None
            free.append(t)

        # ---- retriangulate: one new triangle (u, w, p) per boundary edge
        by_u, by_w = {}, {}
        made = []
        for (u, w, out) in boundary:
            t = new_triangle(u, w, p)
            made.append(t)
            by_u[u] = t
            by_w[w] = t
            ov = TV[out]
            for j in range(3):
                if ov[(j + 1) % 3] == w and ov[(j + 2) % 3] == u:
                    TN[out][j] = t
                    break
            TN[t][2] = out
        for t in made:
            u, w, _ = TV[t]
            TN[t][0] = by_u[w]
            TN[t][1] = by_w[u]
            for v in (u, w):
                if v != G:
                    vertex_tri[v] = t
        vertex_tri[p] = made[0]
        bucket[key] = p
        last = p

    # ---- extract real triangles and the hull
    tris, nbrs, ident = [], [], {}
    for t, v in enumerate(TV):
        if v is not None and G not in v:
            ident[t] = len(tris)
            tris.append(v)
            nbrs.append(TN[t])
    out_nbrs = []
    hull_next = {}
    for t, v in enumerate(TV):
        if v is not None and G in v:
            gi = v.index(G)
            hull_next[v[(gi + 2) % 3]] = v[(gi + 1) % 3]
    for nb in nbrs:
        out_nbrs.append([ident.get(n, -1) for n in nb])
    start = next(iter(hull_next))
    hull, cur = [start], hull_next[start]
    while cur != start:
        hull.append(cur)
        cur = hull_next[cur]
    return tris, out_nbrs, hull


# ------------------------------------------------------------------- Voronoi

def _check_bounds(bounds):
    x0, y0, x1, y1 = (float(v) for v in bounds)
    if not (math.isfinite(x0 + y0 + x1 + y1) and x1 > x0 and y1 > y0):
        raise ValueError(f"bounds must be (x0, y0, x1, y1) with positive area, got {tuple(bounds)}")
    return x0, y0, x1, y1


def _clip_halfplane(poly, nx, ny, mx, my):
    """Clip a convex polygon to {q : n . (q - m) <= 0} (Sutherland-Hodgman)."""
    out = []
    count = len(poly)
    for i in range(count):
        sx, sy = poly[i]
        ex, ey = poly[(i + 1) % count]
        ds = nx * (sx - mx) + ny * (sy - my)
        de = nx * (ex - mx) + ny * (ey - my)
        if ds <= 0.0:
            out.append((sx, sy))
        if (ds < 0.0 < de) or (de < 0.0 < ds):
            t = ds / (ds - de)
            out.append((sx + t * (ex - sx), sy + t * (ey - sy)))
    return out


def _polygon_area(poly):
    area = 0.0
    for i in range(len(poly)):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % len(poly)]
        area += x0 * y1 - x1 * y0
    return 0.5 * area


def _tidy(poly, eps):
    """Drop consecutive duplicate vertices."""
    out = []
    for p in poly:
        if not out or abs(p[0] - out[-1][0]) > eps or abs(p[1] - out[-1][1]) > eps:
            out.append(p)
    if len(out) > 1 and abs(out[0][0] - out[-1][0]) <= eps and abs(out[0][1] - out[-1][1]) <= eps:
        out.pop()
    return out


def voronoi_cells(tri, bounds):
    """Voronoi cells of the triangulation's sites, clipped to `bounds`.

    Returns one convex counter-clockwise (k, 2) polygon per kept site, empty
    when the cell misses the bounds (or only touches them). Each cell starts
    as the bounds rectangle and is clipped by the bisector half-plane of every
    Delaunay neighbour, which is all the exact cell depends on. A site outside
    `bounds` takes part in the triangulation and owns whatever part of the box
    it is nearest to. `bounds` is (x0, y0, x1, y1) and must have positive area.
    """
    x0, y0, x1, y1 = _check_bounds(bounds)
    box = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    area_tol = 1e-14 * (x1 - x0) * (y1 - y0)
    eps = 1e-12 * max(x1 - x0, y1 - y0)
    pts = tri.points
    cells = []
    if len(pts) == 0:
        return cells
    nbrs = tri.vertex_neighbors()
    for i in range(len(pts)):
        px, py = pts[i]
        poly = list(box)
        for j in sorted(nbrs[i], key=lambda j: (pts[j, 0] - px) ** 2 + (pts[j, 1] - py) ** 2):
            qx, qy = pts[j]
            poly = _clip_halfplane(poly, qx - px, qy - py, 0.5 * (px + qx), 0.5 * (py + qy))
            if len(poly) < 3:
                break
        poly = _tidy(poly, eps) if len(poly) >= 3 else []
        if len(poly) < 3 or _polygon_area(poly) <= area_tol:
            cells.append(np.zeros((0, 2), dtype=F64))
        else:
            cells.append(np.array(poly, dtype=F64))
    return cells


# -------------------------------------------------------------------- Lloyd

def _polygon_centroid(poly):
    ox, oy = poly[0]
    xs, ys = poly[:, 0] - ox, poly[:, 1] - oy
    xn, yn = np.roll(xs, -1), np.roll(ys, -1)
    cross = xs * yn - xn * ys
    area = 0.5 * cross.sum()
    if abs(area) < 1e-300:
        return poly.mean(axis=0)
    cx = ((xs + xn) * cross).sum() / (6.0 * area)
    cy = ((ys + yn) * cross).sum() / (6.0 * area)
    return np.array([ox + cx, oy + cy])


def _weighted_centroid(poly, density, bounds):
    """Mass centroid of the density pixels whose centres fall in a convex
    counter-clockwise polygon, or None when the mass is ~0. Centres that sit on
    an edge are assigned by a fixed rule, so neighbouring cells never share one."""
    x0, y0, x1, y1 = bounds
    rows, cols = density.shape
    sx, sy = (x1 - x0) / cols, (y1 - y0) / rows
    c_lo = max(0, int(math.floor((poly[:, 0].min() - x0) / sx - 0.5)))
    c_hi = min(cols - 1, int(math.ceil((poly[:, 0].max() - x0) / sx - 0.5)))
    r_lo = max(0, int(math.floor((poly[:, 1].min() - y0) / sy - 0.5)))
    r_hi = min(rows - 1, int(math.ceil((poly[:, 1].max() - y0) / sy - 0.5)))
    if c_lo > c_hi or r_lo > r_hi:
        return None
    cx = x0 + (np.arange(c_lo, c_hi + 1) + 0.5) * sx
    cy = y0 + (np.arange(r_lo, r_hi + 1) + 0.5) * sy
    gx, gy = np.meshgrid(cx, cy)
    keep = np.ones(gx.shape, dtype=bool)
    nxt = np.roll(poly, -1, axis=0)
    for (ax, ay), (bx, by) in zip(poly, nxt):
        dx, dy = bx - ax, by - ay
        cross = dx * (gy - ay) - dy * (gx - ax)
        on_edge = cross == 0.0
        owns = dy < 0.0 or (dy == 0.0 and dx > 0.0)
        keep &= (cross > 0.0) | (on_edge & owns)
    block = density[r_lo:r_hi + 1, c_lo:c_hi + 1]
    mass = float(block[keep].sum())
    if mass <= 1e-12 * max(1.0, float(density.max())):
        return None
    return np.array([float((gx * block)[keep].sum()) / mass, float((gy * block)[keep].sum()) / mass])


def lloyd_relax_vector(points, bounds, iterations=2, density=None):
    """Lloyd / centroidal Voronoi relaxation on exact cell polygons.

    Each iteration moves every site to the centroid of its clipped Voronoi
    cell and clamps it to `bounds`. With `density`, an (h, w) non-negative
    array laid over the bounds, the centroid is mass weighted over the pixels
    whose centres fall in the cell (the polygon centroid if the mass is ~0).
    Returns an (n, 2) float64 array aligned with `points`; input points that
    were merged by the triangulation come back identical.
    """
    pts = np.asarray(points, dtype=F64).reshape(-1, 2)
    if not np.isfinite(pts).all():
        raise ValueError("points must be finite")
    box = _check_bounds(bounds)
    dens = None
    if density is not None:
        dens = np.asarray(density.a if isinstance(density, tf.Field) else density, dtype=F64)
        if dens.ndim != 2 or dens.size == 0 or not np.isfinite(dens).all() or (dens < 0).any():
            raise ValueError("density must be a finite non-negative 2D array")
    cur = pts.copy()
    for _ in range(max(0, int(iterations))):
        if len(cur) == 0:
            break
        tri = delaunay(cur)
        moved = tri.points.copy()
        for k, cell in enumerate(voronoi_cells(tri, box)):
            if len(cell) == 0:
                continue
            target = None
            if dens is not None:
                target = _weighted_centroid(cell, dens, box)
            moved[k] = _polygon_centroid(cell) if target is None else target
        moved[:, 0] = np.clip(moved[:, 0], box[0], box[2])
        moved[:, 1] = np.clip(moved[:, 1], box[1], box[3])
        cur = moved[tri.index_map]
    return cur


# --------------------------------------------------------- proximity graphs

def _find(parent, a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


def _lune_free(X, Y, edges):
    """Edges (i, j) whose open lune max(|ir|, |jr|) < |ij| holds no other site.

    Exact on the integer lattice. The lune lies inside the disk of radius
    sqrt(3)/2 |ij| about the midpoint, which a uniform grid answers in time
    proportional to the sites nearby.
    """
    if not edges:
        return []
    n = len(X)
    lengths = sorted((X[i] - X[j]) ** 2 + (Y[i] - Y[j]) ** 2 for i, j in edges)
    cell = max(1, int(math.sqrt(lengths[len(lengths) // 2])))
    grid = {}
    for v in range(n):
        grid.setdefault((X[v] // cell, Y[v] // cell), []).append(v)
    keep = []
    for i, j in edges:
        length = (X[i] - X[j]) ** 2 + (Y[i] - Y[j]) ** 2
        mx, my = 0.5 * (X[i] + X[j]), 0.5 * (Y[i] + Y[j])
        reach = math.sqrt(0.75 * length) + 1.0
        c0, c1 = int(math.floor((mx - reach) / cell)), int(math.floor((mx + reach) / cell))
        r0, r1 = int(math.floor((my - reach) / cell)), int(math.floor((my + reach) / cell))
        if (c1 - c0 + 1) * (r1 - r0 + 1) <= len(grid):
            buckets = (grid.get((cx, cy), ()) for cx in range(c0, c1 + 1) for cy in range(r0, r1 + 1))
        else:
            buckets = (v for key, v in grid.items() if c0 <= key[0] <= c1 and r0 <= key[1] <= r1)
        free = True
        for bucket in buckets:
            for r in bucket:
                if r == i or r == j:
                    continue
                if ((X[i] - X[r]) ** 2 + (Y[i] - Y[r]) ** 2 < length
                        and (X[j] - X[r]) ** 2 + (Y[j] - Y[r]) ** 2 < length):
                    free = False
                    break
            if not free:
                break
        if free:
            keep.append((i, j))
    return keep


def proximity_graphs(tri):
    """Delaunay, Gabriel, relative-neighbourhood and minimum spanning graphs.

    Returns {'delaunay', 'gabriel', 'rng', 'mst'}, each a sorted (e, 2) int32
    array of index pairs i < j. All tests run on the exact lattice
    coordinates, so ties resolve identically for every input order:

    * Gabriel keeps an edge when no other site lies in the closed disk on it
      as diameter. Only the apexes of its one or two triangles need testing.
    * RNG keeps an edge pq when no site r has max(|pr|, |qr|) < |pq| (open
      lune). The lune can hold a site that is a Delaunay neighbour of neither
      p nor q, so it is queried through a uniform grid over all sites.
    * MST is Kruskal over the Delaunay edges, ties broken by index.

    These satisfy MST <= RNG <= Gabriel <= Delaunay (Toussaint 1980).
    """
    delaunay_edges = tri.edges()
    if len(delaunay_edges) == 0:
        none = np.zeros((0, 2), dtype=I32)
        return {'delaunay': none, 'gabriel': none, 'rng': none, 'mst': none}
    X, Y = tri.lattice

    def d2(i, j):
        return (X[i] - X[j]) ** 2 + (Y[i] - Y[j]) ** 2

    edge_list = [tuple(e) for e in delaunay_edges.tolist()]
    if tri.degenerate:                    # points on a line: every graph is the chain
        return {k: delaunay_edges.copy() for k in ('delaunay', 'gabriel', 'rng', 'mst')}

    apex = {e: [] for e in edge_list}
    for (a, b, c) in tri.tris.tolist():
        for u, w, o in ((a, b, c), (b, c, a), (c, a, b)):
            apex[(u, w) if u < w else (w, u)].append(o)

    gabriel = []
    for (i, j) in edge_list:
        if all((X[i] - X[o]) * (X[j] - X[o]) + (Y[i] - Y[o]) * (Y[j] - Y[o]) > 0 for o in apex[(i, j)]):
            gabriel.append((i, j))

    rng_edges = _lune_free(X, Y, gabriel)

    parent = list(range(len(tri.points)))
    mst = []
    for length, i, j in sorted((d2(i, j), i, j) for (i, j) in edge_list):
        ri, rj = _find(parent, i), _find(parent, j)
        if ri != rj:
            parent[ri] = rj
            mst.append((i, j))

    def pack(edges):
        return np.array(sorted(edges), dtype=I32).reshape(-1, 2)

    return {'delaunay': delaunay_edges, 'gabriel': pack(gabriel), 'rng': pack(rng_edges),
            'mst': pack(mst)}


# ---------------------------------------------------------- contour tracing
#
# Frame: sample (row r, column c) sits at the point (x = c, y = r), the same
# frame the sites of this module and tf.voronoi use. In 'edges' mode the unit
# square of a pixel is centred on its sample, so ring vertices fall on half
# integers. Shoelace area 0.5 * sum(x_i * y_(i+1) - x_(i+1) * y_i) is positive
# for outer rings and negative for holes in this (x = column, y = row) frame.

_STEP = ((1, 0), (0, 1), (-1, 0), (0, -1))   # top, right, bottom, left pixel edge directions


def signed_area(ring):
    """Shoelace area of a ring with x = column and y = row: positive for outer
    rings, negative for holes."""
    ring = np.asarray(ring, dtype=F64)
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * float((x * np.roll(y, -1) - np.roll(x, -1) * y).sum())


def _walk(nxt, labels=None):
    """Follow next-pointers (a permutation) into cycles, lowest start first. Each
    cycle lists its indices, or `labels[index]` when labels are given."""
    nxt = nxt.tolist()
    labels = None if labels is None else labels.tolist()
    seen = bytearray(len(nxt))
    cycles = []
    for start in range(len(nxt)):
        if seen[start]:
            continue
        cycle, cur = [], start
        while not seen[cur]:
            seen[cur] = 1
            cycle.append(cur if labels is None else labels[cur])
            cur = nxt[cur]
        cycles.append(cycle)
    return cycles


def _split_pinches(seq):
    """Split a closed walk that revisits vertices into simple loops (a stack:
    each time a vertex repeats, the stretch since its first visit is a loop)."""
    loops, stack, where = [], [], {}
    for v in seq:
        if v in where:
            j = where[v]
            loop = stack[j:]
            for u in loop:
                del where[u]
            del stack[j:]
            loops.append(loop)
        where[v] = len(stack)
        stack.append(v)
    loops.append(stack)
    return loops


def _canonical(rings):
    """Rotate each ring to start at its lexicographically smallest vertex and
    sort the rings by it; drop rings with fewer than three vertices or no area.
    (The edge tracer does this itself, faster, on vertex ids.)"""
    out = []
    for ring in rings:
        ring = np.asarray(ring, dtype=F64)
        keep = np.ones(len(ring), dtype=bool)
        keep[1:] = (np.abs(np.diff(ring, axis=0)) > 0).any(axis=1)
        ring = ring[keep]
        if len(ring) > 1 and (ring[0] == ring[-1]).all():
            ring = ring[:-1]
        if len(ring) < 3 or abs(signed_area(ring)) < 1e-12:
            continue
        first = min(range(len(ring)), key=lambda i: (ring[i, 0], ring[i, 1]))
        out.append(np.roll(ring, -first, axis=0))
    out.sort(key=lambda r: tuple(r[:3].ravel()))
    return out


def _edge_rings(inside, simple):
    """Boundary of a pixel set along pixel edges, with 4-connected foreground."""
    h, w = inside.shape
    padded = np.pad(inside, 1)
    core = padded[1:-1, 1:-1]
    corner = lambda i, j: i * (w + 1) + j
    ii, jj = np.mgrid[0:h, 0:w]
    starts, ends, kinds = [], [], []
    for kind, outside in enumerate((~padded[:-2, 1:-1], ~padded[1:-1, 2:], ~padded[2:, 1:-1], ~padded[1:-1, :-2])):
        sel = core & outside
        i, j = ii[sel], jj[sel]
        if kind == 0:
            a, b = corner(i, j), corner(i, j + 1)
        elif kind == 1:
            a, b = corner(i, j + 1), corner(i + 1, j + 1)
        elif kind == 2:
            a, b = corner(i + 1, j + 1), corner(i + 1, j)
        else:
            a, b = corner(i + 1, j), corner(i, j)
        starts.append(a); ends.append(b); kinds.append(np.full(len(a), kind))
    start, end, kind = (np.concatenate(v) for v in (starts, ends, kinds))
    if len(start) == 0:
        return []
    order = np.argsort(start, kind="stable")
    sorted_start = start[order]
    lo = np.searchsorted(sorted_start, end, side="left")
    hi = np.searchsorted(sorted_start, end, side="right")
    first = order[lo]
    second = order[np.minimum(lo + 1, len(order) - 1)]
    step = np.array(_STEP)
    d_in = step[kind]
    cross_first = d_in[:, 0] * step[kind[first]][:, 1] - d_in[:, 1] * step[kind[first]][:, 0]
    cross_second = d_in[:, 0] * step[kind[second]][:, 1] - d_in[:, 1] * step[kind[second]][:, 0]
    # at a pinch (two outgoing edges) turn toward the foreground, which keeps it 4-connected
    nxt = np.where((hi - lo == 2) & (cross_second > cross_first), second, first)

    # rings as vertex-id lists, rotated to their lexicographically smallest (x, y) vertex
    width1, height1 = w + 1, h + 1
    loops = []
    for walk in _walk(nxt, start):
        for loop in (_split_pinches(walk) if simple and len(set(walk)) < len(walk) else [walk]):
            keys = [(v % width1) * height1 + v // width1 for v in loop]
            at = keys.index(min(keys))
            loops.append((keys[at:] + keys[:at], loop[at:] + loop[:at]))
    loops.sort(key=lambda kv: kv[0][:3])
    return [np.stack([np.array(ids) % width1 - 0.5, np.array(ids) // width1 - 0.5], axis=1)
            for _, ids in loops]


# marching-squares segments per corner case (TL = bit 0, TR = 1, BR = 2, BL = 3); an edge is
# 0 top, 1 right, 2 bottom, 3 left. A segment runs from the crossing where an inside run
# leaves the cell boundary to the one where it entered, so inside stays on the positive side.
def _case_segments(case, connected):
    corners = [(case >> k) & 1 for k in range(4)]
    runs = []
    for a in range(4):
        if corners[a] and not corners[(a - 1) % 4]:
            b = a
            while corners[(b + 1) % 4]:
                b = (b + 1) % 4
            runs.append((a, b))
    if len(runs) == 2 and connected:
        (a1, b1), (a2, b2) = runs
        return [(b1, (a2 - 1) % 4), (b2, (a1 - 1) % 4)]
    return [(b, (a - 1) % 4) for a, b in runs]


_SEGMENTS = np.full((16, 2, 2, 2), -1, dtype=np.int64)       # case, connected, segment, (from, to)
for _case in range(16):
    for _conn in (0, 1):
        for _k, _seg in enumerate(_case_segments(_case, bool(_conn))):
            _SEGMENTS[_case, _conn, _k] = _seg


def _interp_rings(values, level, saddle, pad):
    h, w = values.shape
    inside = np.pad(values >= level, 1)
    real = np.pad(np.ones((h, w), dtype=bool), 1)
    field = np.pad(values, 1, constant_values=level)
    tl, tr, br, bl = (inside[:-1, :-1], inside[:-1, 1:], inside[1:, 1:], inside[1:, :-1])
    case = tl.astype(np.int64) | (tr.astype(np.int64) << 1) | (br.astype(np.int64) << 2) | (bl.astype(np.int64) << 3)
    centre = (field[:-1, :-1] + field[:-1, 1:] + field[1:, 1:] + field[1:, :-1]) / 4.0
    connected = ((case == 5) | (case == 10)) & (centre >= level) & (saddle == "center")
    ci, cj = np.nonzero((case != 0) & (case != 15))
    segs = _SEGMENTS[case[ci, cj], connected[ci, cj].astype(np.int64)]       # (n, 2, 2)
    n_h = (h + 2) * (w + 1)

    def edge_id(code, i, j):
        top, bottom = i * (w + 1) + j, (i + 1) * (w + 1) + j
        left, right = n_h + i * (w + 2) + j, n_h + i * (w + 2) + j + 1
        return np.select([code == 0, code == 1, code == 2], [top, right, bottom], left)

    use = segs[:, :, 0] >= 0
    rows = np.repeat(ci, 2).reshape(-1, 2)[use]
    cols = np.repeat(cj, 2).reshape(-1, 2)[use]
    a = edge_id(segs[:, :, 0][use], rows, cols)
    b = edge_id(segs[:, :, 1][use], rows, cols)
    if len(a) == 0:
        return []
    lookup = np.full(n_h + (h + 1) * (w + 2), -1, dtype=np.int64)
    lookup[a] = np.arange(len(a))
    nxt = lookup[b]
    if (nxt < 0).any():
        raise RuntimeError("open contour: the contour does not close")

    # position of every crossing, in the sample frame (padded index minus one)
    ids = np.unique(a)
    horizontal = ids < n_h
    hi_, hj_ = ids[horizontal] // (w + 1), ids[horizontal] % (w + 1)
    vid = ids[~horizontal] - n_h
    vi_, vj_ = vid // (w + 2), vid % (w + 2)

    def fraction(v0, v1, real0, real1):
        both = real0 & real1
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(both, (level - v0) / (v1 - v0), 0.0)
        if pad:
            t = np.where(both, t, 0.5)
        else:
            t = np.where(both, t, np.where(real0, 0.0, 1.0))
        return np.clip(np.nan_to_num(t, nan=0.5), 0.0, 1.0)

    px = np.zeros(len(ids)); py = np.zeros(len(ids))
    t = fraction(field[hi_, hj_], field[hi_, hj_ + 1], real[hi_, hj_], real[hi_, hj_ + 1])
    px[horizontal], py[horizontal] = hj_ + t - 1.0, hi_ - 1.0
    t = fraction(field[vi_, vj_], field[vi_ + 1, vj_], real[vi_, vj_], real[vi_ + 1, vj_])
    px[~horizontal], py[~horizontal] = vj_ - 1.0, vi_ + t - 1.0
    position = np.searchsorted(ids, a)                      # per segment: its start crossing
    rings = []
    for cycle in _walk(nxt):
        rings.append(np.stack([px[position[cycle]], py[position[cycle]]], axis=1))
    return rings


def trace_rings(source, level=0.5, mode="edges", pad=True, saddle="center", simple=True):
    """Closed rings (polygons) around the foreground of a mask or a scalar field.

    `source` is a 2D bool array (or 0/1 mask) or a scalar field (array or
    `Field`). Rings are (k, 2) float64 arrays of (x = column, y = row)
    vertices without a repeated first vertex, each starting at its
    lexicographically smallest vertex and sorted by it. Shoelace area is
    positive for outer rings and negative for holes. Foreground is
    `source >= level` (a bool array is used as it is).

    mode 'edges': the exact boundary along pixel edges, foreground
    4-connected, so pixels that only touch at a corner are separate rings.
    Every lattice point on the boundary is kept (a 2 x 2 block has 8
    vertices). The unit square of a pixel is centred on its sample.
    With `simple` a boundary that touches itself at a corner is split there
    into simple loops (outer ring plus a hole touching it); without it the
    walk is kept whole and may repeat that vertex.

    mode 'interp': marching squares, with linear interpolation of the field at
    `level`. `saddle` decides an ambiguous cell: 'center' joins the two inside
    corners when the cell average is >= level, 'low' always keeps them apart.
    Zero-area rings (a sample touching the level exactly) are dropped. Rings
    here are not promised to be simple: two vertices coincide where a sample
    sits exactly on the level.

    `pad` surrounds the data with a below-level margin so every ring closes:
    in 'interp' mode True puts the closing half a sample beyond the border
    (so a ring that meets the border gets its corners cut diagonally) and
    False on the border samples; 'edges' rings always close along the array
    edge.

    To trace the region below a level, pass `-field` and `-level`.
    """
    if mode not in ("edges", "interp"):
        raise ValueError(f"mode must be 'edges' or 'interp', got {mode!r}")
    if saddle not in ("center", "low"):
        raise ValueError(f"saddle must be 'center' or 'low', got {saddle!r}")
    arr = np.asarray(source.a if isinstance(source, tf.Field) else source)
    if arr.ndim != 2:
        raise ValueError(f"source must be 2D, got shape {arr.shape}")
    if arr.dtype != np.bool_ and not np.isfinite(arr.astype(F64)).all():
        raise ValueError("source must be finite")
    if arr.size == 0:
        return []
    if mode == "edges":
        inside = arr if arr.dtype == np.bool_ else arr >= level
        if not inside.any():
            return []
        return _edge_rings(inside, simple)
    values = arr.astype(F64)
    if not (values >= level).any():
        return []
    return _canonical(_interp_rings(values, float(level), saddle, pad))


def _inside_ring(ring, point):
    """Even-odd point-in-polygon with the half-open crossing rule."""
    px, py = point
    x, y = ring[:, 0], ring[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    straddle = (y > py) != (yn > py)
    with np.errstate(divide="ignore", invalid="ignore"):
        cross = x + (py - y) * (xn - x) / (yn - y)
    return bool(((cross > px) & straddle).sum() % 2)


def ring_tree(rings):
    """Outer / hole hierarchy: {outer index: [hole indices]}. Rings with positive
    area are outers, negative are holes; a hole belongs to the smallest outer
    that contains it, tested at the midpoint of the hole's first edge. Outer
    rings nested inside holes appear as keys of their own."""
    areas = [signed_area(r) for r in rings]
    tree = {i: [] for i, a in enumerate(areas) if a > 0}
    for hole, a in enumerate(areas):
        if a >= 0:
            continue
        probe = 0.5 * (rings[hole][0] + rings[hole][1])
        parents = [o for o in tree if areas[o] > -a and _inside_ring(rings[o], probe)]
        if parents:
            tree[min(parents, key=lambda o: areas[o])].append(hole)
    return tree


# ------------------------------------------------ scattered-data interpolation
#
# A radial basis function interpolant s(x) = sum_i w_i phi(|x - c_i|) + p(x), p a
# polynomial of degree <= `degree`, found from the symmetric system
#     [[Phi + lambda I, P], [P^T, 0]] [w; q] = [f; 0].
# Kernels that are only conditionally positive definite of order m (thin plate,
# cubic: 2; linear, multiquadric: 1) need a polynomial of degree >= m - 1 for the
# system to have a unique solution.

RBF_KERNELS = ("thin_plate", "cubic", "linear", "gaussian", "multiquadric", "inverse_multiquadric")
RBF_MAX_CENTERS = 1500
_RBF_MIN_DEGREE = {"thin_plate": 1, "cubic": 1, "linear": 0, "multiquadric": 0, "gaussian": -1,
                   "inverse_multiquadric": -1}
_RBF_SHAPE = ("gaussian", "multiquadric", "inverse_multiquadric")      # kernels that read epsilon
_RBF_CHUNK = 4_000_000                                                 # kernel evaluations per block
_RBF_COND_LIMIT = 1e13
_RBF_RIDGE = 1e-10


def _rbf_phi(kernel, r2, epsilon):
    """The kernel as a function of the squared distance `r2` (float64)."""
    if kernel == "thin_plate":                       # r^2 ln r, zero at r = 0
        inner = r2 > 0.0
        return np.where(inner, 0.5 * r2 * np.log(np.where(inner, r2, 1.0)), 0.0)
    if kernel == "cubic":
        return r2 * np.sqrt(r2)
    if kernel == "linear":
        return np.sqrt(r2)
    if kernel == "gaussian":
        return np.exp(-(epsilon * epsilon) * r2)
    if kernel == "multiquadric":
        return np.sqrt(1.0 + (epsilon * epsilon) * r2)
    return 1.0 / np.sqrt(1.0 + (epsilon * epsilon) * r2)


def _first_duplicate(cx, cy):
    """Indices (i, j), i < j, of the first pair of identical centers, or None."""
    order = np.lexsort((cy, cx))
    same = (cx[order][1:] == cx[order][:-1]) & (cy[order][1:] == cy[order][:-1])
    if not same.any():
        return None
    k = int(np.argmax(same))
    return tuple(sorted((int(order[k]), int(order[k + 1]))))


def rbf_field(centers, values, w, h, kernel="thin_plate", epsilon=1.0, smoothing=0.0, degree=1, bounds=None):
    """A smooth `Field` through (or near) scattered samples: designer pins such as
    'a mountain here, a sea there'.

    `centers` is (n, 2) of (x, y) points in the pixel frame of this module (pixel
    (x, y) sits at the point (x, y)); `values` is (n,) for one `Field` or (n, d)
    for a list of d of them. The grid is w x h samples; `bounds` = (x0, y0, x1, y1)
    gives the points of the first and last pixel (default (0, 0, w - 1, h - 1)).

    kernel: 'thin_plate' r^2 ln r, 'cubic' r^3, 'linear' r (all independent of
    `epsilon`), 'gaussian' exp(-(eps r)^2), 'multiquadric' sqrt(1 + (eps r)^2),
    'inverse_multiquadric' 1 / sqrt(1 + (eps r)^2). The gaussian wants a small
    epsilon: when 1 / eps is much shorter than the spacing of the centers the
    weights equal the data and the field is about zero in between.
    `smoothing` >= 0 is a ridge on the kernel matrix (0 interpolates exactly,
    large values tend to the least-squares polynomial). `degree` -1 (none), 0
    (constant) or 1 (plane) augments the kernel; thin_plate and cubic need 1,
    linear and multiquadric at least 0, otherwise ValueError.

    Duplicate centers, non-finite input or more than 1500 centers raise
    ValueError. No centers give a constant 0 field. With degree 1 and fewer than
    three non-collinear centers the degree drops to 0 (thin_plate and cubic,
    which cannot do without a linear term, keep a ramp along the line the
    centers span instead). A system that is ill-conditioned (1-norm condition
    estimate above 1e13, as a gaussian with a small epsilon is) gets a ridge of
    1e-10 times its mean kernel magnitude, so it no longer passes through every
    value. Both cases are named in the `Field.expr`. Pure and deterministic; the
    solution is the same when centers and bounds move together.
    """
    if kernel not in RBF_KERNELS:
        raise ValueError(f"kernel must be one of {RBF_KERNELS}, got {kernel!r}")
    if degree not in (-1, 0, 1):
        raise ValueError(f"degree must be -1, 0 or 1, got {degree!r}")
    degree = int(degree)
    if degree < _RBF_MIN_DEGREE[kernel]:
        raise ValueError(f"kernel {kernel!r} needs degree >= {_RBF_MIN_DEGREE[kernel]}, got {degree}")
    if not (np.isfinite(epsilon) and epsilon > 0):
        raise ValueError(f"epsilon must be a positive number, got {epsilon!r}")
    if not (np.isfinite(smoothing) and smoothing >= 0):
        raise ValueError(f"smoothing must be >= 0, got {smoothing!r}")
    w, h = int(w), int(h)
    if w < 1 or h < 1:
        raise ValueError(f"w and h must be >= 1, got {w} x {h}")
    if bounds is None:
        bounds = (0.0, 0.0, w - 1.0, h - 1.0)
    bounds = np.asarray(bounds, dtype=F64)
    if bounds.shape != (4,) or not np.isfinite(bounds).all():
        raise ValueError(f"bounds must be four finite numbers (x0, y0, x1, y1), got {bounds}")

    pts = np.asarray(centers, dtype=F64)
    if pts.size == 0:
        pts = pts.reshape(0, 2)
    f = np.asarray(values, dtype=F64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"centers must have shape (n, 2), got {pts.shape}")
    n = len(pts)
    if f.ndim not in (1, 2) or len(f) != n:
        raise ValueError(f"values must have shape ({n},) or ({n}, d), got {f.shape}")
    if not (np.isfinite(pts).all() and np.isfinite(f).all()):
        raise ValueError("centers and values must be finite")
    if n > RBF_MAX_CENTERS:
        raise ValueError(f"at most {RBF_MAX_CENTERS} centers are supported, got {n}")
    dup = _first_duplicate(pts[:, 0], pts[:, 1])
    if dup is not None:
        raise ValueError(f"centers {dup[0]} and {dup[1]} coincide at ({pts[dup[0], 0]:g}, {pts[dup[0], 1]:g})")
    columns = f[:, None] if f.ndim == 1 else f
    d = columns.shape[1]

    notes = []
    if n == 0:
        out = [tf.Field(np.zeros((h, w), dtype=np.float32), "rbf(none)") for _ in range(d)]
        return out[0] if f.ndim == 1 else out

    # the polynomial part works in centered, scaled coordinates (same space, better conditioned)
    mean = pts.mean(axis=0)
    spread = float(np.sqrt(((pts - mean) ** 2).sum(axis=1).mean()))
    scale = spread if spread > 0 else 1.0
    basis, direction = ("none", "constant", "plane")[degree + 1], None
    if degree == 1:
        _, sv, vt = np.linalg.svd((pts - mean) / scale, full_matrices=False)
        if n < 3 or sv[-1] <= 1e-9 * sv[0]:
            if _RBF_MIN_DEGREE[kernel] >= 1 and n >= 2:
                # these kernels need a linear term; along the line the centers span, that is all there is
                basis, direction = "line", vt[0]
                notes.append("warning: centers are collinear, the plane term is a ramp along their line")
            else:
                degree, basis = 0, "constant"
                notes.append("warning: degree 1 -> 0, fewer than 3 non-collinear centers")

    def poly(x, y):
        if basis == "none":
            return np.zeros((len(x), 0))
        cols = [np.ones(len(x))]
        if basis == "plane":
            cols += [(x - mean[0]) / scale, (y - mean[1]) / scale]
        elif basis == "line":
            cols.append(((x - mean[0]) * direction[0] + (y - mean[1]) * direction[1]) / scale)
        return np.stack(cols, axis=1)

    m = {"none": 0, "constant": 1, "line": 2, "plane": 3}[basis]
    dx = pts[:, 0][:, None] - pts[:, 0][None, :]
    dy = pts[:, 1][:, None] - pts[:, 1][None, :]
    kernel_matrix = _rbf_phi(kernel, dx * dx + dy * dy, epsilon) + smoothing * np.eye(n)
    # the polynomial block is scaled to the kernel's magnitude; the coefficients are scaled back below
    kappa = float(np.abs(kernel_matrix).mean())
    kappa = kappa if kappa > 0 else 1.0
    P = poly(pts[:, 0], pts[:, 1])
    system = np.zeros((n + m, n + m))
    system[:n, :n] = kernel_matrix
    system[:n, n:] = kappa * P
    system[n:, :n] = kappa * P.T
    with np.errstate(all="ignore"):
        cond = float(np.linalg.cond(system, 1))
    if not cond <= _RBF_COND_LIMIT:
        system[np.arange(n), np.arange(n)] += _RBF_RIDGE * kappa
        notes.append(f"warning: ill-conditioned (cond ~ {cond:.1e}), ridge {_RBF_RIDGE:g} added")
    rhs = np.zeros((n + m, d))
    rhs[:n] = columns
    try:
        solution = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"the interpolation system is singular ({exc})") from exc
    weights, coeffs = solution[:n], kappa * solution[n:]

    xs = np.linspace(bounds[0], bounds[2], w) if w > 1 else np.array([bounds[0]])
    ys = np.linspace(bounds[1], bounds[3], h) if h > 1 else np.array([bounds[1]])
    total = w * h
    step = max(1, _RBF_CHUNK // n)
    result = np.empty((d, total), dtype=np.float32)
    for start in range(0, total, step):
        index = np.arange(start, min(total, start + step))
        gx, gy = xs[index % w], ys[index // w]
        ex, ey = gx[:, None] - pts[:, 0][None, :], gy[:, None] - pts[:, 1][None, :]
        block = _rbf_phi(kernel, ex * ex + ey * ey, epsilon) @ weights
        if m:
            block = block + poly(gx, gy) @ coeffs
        result[:, start:start + len(index)] = block.T

    eps = f", eps={epsilon:g}" if kernel in _RBF_SHAPE else ""
    ridge = f", smoothing={smoothing:g}" if smoothing else ""
    expr = f"rbf({kernel}, n={n}, degree={degree}{eps}{ridge})" + "".join(f" [{t}]" for t in notes)
    fields = [tf.Field(result[k].reshape(h, w), expr if d == 1 else f"{expr}[{k}]") for k in range(d)]
    return fields[0] if f.ndim == 1 else fields
