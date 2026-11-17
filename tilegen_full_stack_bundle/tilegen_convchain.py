"""ConvChain: example-driven bilevel synthesis by Metropolis sampling of N x N windows.

Idea-level port of the ConvChain node of MarkovJunior (MIT licence, Copyright (C)
2022 Maxim Gumin, github.com/mxgmn/MarkovJunior@42aaf24bcf54ae164fba49c0a59348297904a676,
source/ConvChain.cs), after S. Geman and D. Geman, 'Stochastic relaxation, Gibbs
distributions, and the Bayesian restoration of images', IEEE PAMI 6(6), 1984,
and N. Metropolis et al., 'Equation of state calculations by fast computing
machines', J. Chem. Phys. 21(6), 1953. Rewritten for numpy with colour-class
updates instead of random single-cell picks; no source text was carried over.

The model. A bilevel sample (open / closed, e.g. the passable mask of one of the
100 shipped layouts) is cut into every N x N window, the dihedral symmetries are
added and the occurrences of each pattern are counted. `tilegen_wfc.PatternModel`
does that cutting and counting, so the overlapping WFC and this sampler learn
local structure with the same code. A pattern the sample never shows weighs
`floor` (0.1, as in the original) rather than 0. A grid scores the product of
the weights of all its windows, and grids are drawn with probability proportional
to score ** (1 / temperature): temperature 1 reproduces the statistics of the
sample, lower values sharpen towards its commonest patterns, higher ones relax
towards noise.

The sampler. Metropolis flips of one cell at a time. A flip only changes the N x N
windows that cover the cell, so the ratio q of the two scores is the product, over
those windows, of weight(after) / weight(before); the flip is taken when q >= 1
and otherwise with probability q ** (1 / temperature). Two cells that are N or
more apart in x or y share no window, so their updates are independent and the
cells of one colour class (x mod N, y mod N; a periodic grid whose side is not a
multiple of N uses unequal blocks instead) are updated together with numpy. One
sweep proposes every free cell exactly once, the classes in an order drawn
anew each sweep; each class update is a Metropolis kernel for the target
distribution, so a sweep leaves it invariant, like the original's random picks.

Randomness: `tf.rng_for(seed, 'convchain')`. Draw order: the initial fill (a
(h, w) array of uniform draws, unless `init` is given), then per sweep one
class permutation and, per class, one uniform draw for each of its free cells
in row-major order.

Imports downward only: `tilegen_fields`, `tilegen_wfc` and, for the TileMap
wrapper, `tilegen_grammars`.
"""
from __future__ import annotations

import numpy as np

import tilegen_fields as tf
import tilegen_grammars as tg
from tilegen_wfc import PatternModel


def ring_colours(length, n, periodic):
    """A colour for every position of a row such that two positions closer than n
    (counting around the ring when `periodic`) never share one. Returns
    (colours, how many). Positions mod n do it, except on a ring whose length
    is not a multiple of n, which is cut into floor(length / n) blocks of nearly
    equal size (each at least n long) and coloured by offset within the block."""
    if not periodic or length % n == 0:
        colours = np.arange(length) % n
        return colours, int(colours.max()) + 1
    blocks = length // n
    base, extra = divmod(length, blocks)
    colours = np.empty(length, dtype=np.int64)
    pos = 0
    for b in range(blocks):
        size = base + (1 if b < extra else 0)
        colours[pos:pos + size] = np.arange(size)
        pos += size
    return colours, int(colours.max()) + 1


class ConvChainModel:
    """Window weights learned from a bilevel sample.

    sample      2D array of 0 / 1 (or bool)
    n           window size, 1..5
    symmetry    1 identity, 2 + left-right mirror, 4 + quarter turns, 8 all
                eight (the original's default)
    periodic_in wrap the sample when cutting windows
    floor       weight of a pattern the sample does not contain

    `patterns` is the `tilegen_wfc.PatternModel` the weights come from (distinct
    windows, how often each occurs); a pattern's code is sum(bit << (dy * n + dx)),
    so `keys` are the sorted codes of the distinct patterns and `weights` their
    counts.
    """

    def __init__(self, sample, n=3, symmetry=8, periodic_in=True, floor=0.1):
        arr = np.asarray(sample)
        if arr.ndim != 2 or arr.size == 0:
            raise ValueError(f"sample must be a non-empty 2D array, got shape {arr.shape}")
        if arr.dtype != np.bool_ and not (np.issubdtype(arr.dtype, np.integer)
                                          and ((arr == 0) | (arr == 1)).all()):
            raise ValueError("sample must be bool or hold only 0 and 1")
        if not float(floor) > 0.0:
            raise ValueError(f"floor must be > 0, got {floor!r}")
        self.patterns = PatternModel(arr.astype(np.uint8), n, periodic_in, symmetry)
        self.n, self.floor = self.patterns.n, float(floor)
        bits = np.left_shift(1, np.arange(self.n * self.n, dtype=np.int64))
        codes = (self.patterns.patterns.reshape(self.patterns.count, -1).astype(np.int64) * bits).sum(axis=1)
        order = np.argsort(codes)
        self.keys = codes[order]
        self.weights = self.patterns.weights[order].astype(np.float64)
        self._log = np.log(self.weights)
        self._log_floor = float(np.log(self.floor))
        self.sample_open = float(arr.mean())

    def log_weight_of(self, codes):
        """log weight of each pattern code (an int64 array of any shape)."""
        codes = np.asarray(codes, dtype=np.int64)
        pos = np.minimum(np.searchsorted(self.keys, codes), len(self.keys) - 1)
        return np.where(self.keys[pos] == codes, self._log[pos], self._log_floor)

    def window_codes(self, grid, periodic=False):
        """Pattern code of every window of a bool grid: (h, w) windows, one per
        top-left cell, wrapping when `periodic`; otherwise (h - n + 1, w - n + 1)."""
        grid = np.asarray(grid, dtype=bool)
        n = self.n
        h, w = grid.shape
        if min(h, w) < n:
            raise ValueError(f"grid {w}x{h} is smaller than the window size {n}")
        if periodic:
            parts = [np.roll(grid, (-dy, -dx), axis=(0, 1)) for dy in range(n) for dx in range(n)]
        else:
            parts = [grid[dy:dy + h - n + 1, dx:dx + w - n + 1] for dy in range(n) for dx in range(n)]
        codes = np.zeros(parts[0].shape, dtype=np.int64)
        for bit, part in enumerate(parts):
            codes |= part.astype(np.int64) << bit
        return codes

    def log_weight(self, grid, periodic=False):
        """log of the score of a grid: the sum of log weights over all its windows."""
        return float(self.log_weight_of(self.window_codes(grid, periodic)).sum())

    def coverage(self, grid, periodic=False):
        """Share of the grid's windows whose pattern occurs in the sample."""
        codes = self.window_codes(grid, periodic)
        pos = np.minimum(np.searchsorted(self.keys, codes), len(self.keys) - 1)
        return float((self.keys[pos] == codes).mean())

    def flip_log_ratio(self, grid, cells, periodic=False):
        """For each (x, y) in `cells`: the change of log score if that cell alone
        were flipped, read from the N x N windows that cover it. Equals the
        difference of two `log_weight` calls."""
        grid = np.array(grid, dtype=bool)
        cells = np.asarray(cells, dtype=np.int64).reshape(-1, 2)
        h, w = grid.shape
        if len(cells) and not ((cells[:, 0] >= 0).all() and (cells[:, 0] < w).all()
                               and (cells[:, 1] >= 0).all() and (cells[:, 1] < h).all()):
            raise ValueError("cells must lie inside the grid")
        return _Windows(self, grid, periodic).flip_log_ratio(cells[:, 1], cells[:, 0])

    def sample(self, out_w, out_h, temperature=1.0, sweeps=30, periodic=False, seed=0,
               init=None, free=None, fill=0.5, on_sweep=None):
        """Draw an (out_h, out_w) bool grid (True = open). Returns (grid, stats).

        temperature  > 0; the target is (product of window weights) ** (1 / T)
        sweeps       how many times every free cell is proposed once
        periodic     score the wrapped windows too (the grid is then a torus)
        init         bool (h, w) start state
        free         bool (h, w) cells allowed to flip (default: all); the rest
                     keep their `init` value (False without one) but still count
                     in every window, so a fixed rim or a pinned room shapes
                     what grows beside it
        fill         without `init`, the chance that a free cell starts open
                     (0.5 is the original's fair coin)
        on_sweep     optional callback(sweep, grid) after each sweep (counted
                     from 1); `grid` is the live array, copy it to keep it

        stats: 'sweeps', 'flips' (accepted per sweep, a list) and 'log_weight'
        (score of the result). Equal arguments give equal grids.

        The model has no term for how much of the grid is open, so that share
        drifts to wherever the patterns favour: from a sparse sample it stays
        near `fill` at temperature 1 for N = 3 but runs away at lower
        temperatures or larger N. Steer it with `fill`, the temperature, the
        sweep count, or by pinning cells with `init` and `free`. N = 5 needs
        about 60 sweeps before its large patterns nucleate.
        """
        out_w, out_h, sweeps = int(out_w), int(out_h), int(sweeps)
        n = self.n
        if not float(temperature) > 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature!r}")
        if sweeps < 0:
            raise ValueError(f"sweeps must be >= 0, got {sweeps}")
        if not 0.0 <= float(fill) <= 1.0:
            raise ValueError(f"fill must be between 0 and 1, got {fill!r}")
        if out_w < n or out_h < n:
            raise ValueError(f"output {out_w}x{out_h} is smaller than the window size {n}")
        shape = (out_h, out_w)
        free = np.ones(shape, dtype=bool) if free is None else np.asarray(free, dtype=bool)
        if free.shape != shape:
            raise ValueError(f"free has shape {free.shape}, expected {shape}")
        rng = tf.rng_for(seed, 'convchain')
        if init is None:
            grid = (rng.random(shape) < float(fill)) & free
        else:
            grid = np.array(init, dtype=bool)
            if grid.shape != shape:
                raise ValueError(f"init has shape {grid.shape}, expected {shape}")

        state = _Windows(self, grid, periodic)
        classes = state.classes(free)
        inv_t = 1.0 / float(temperature)
        flips = []
        for sweep in range(1, sweeps + 1):
            taken = 0
            for k in rng.permutation(len(classes)):
                ys, xs = classes[k]
                if not len(ys):
                    continue
                accept = rng.random(len(ys)) < np.exp(np.minimum(state.flip_log_ratio(ys, xs) * inv_t, 0.0))
                state.flip(ys[accept], xs[accept])
                taken += int(accept.sum())
            flips.append(taken)
            if on_sweep is not None:
                on_sweep(sweep, state.grid)
        return state.grid, {"sweeps": sweeps, "flips": flips,
                            "log_weight": self.log_weight(state.grid, periodic)}


class _Windows:
    """A grid with the pattern code of every window kept up to date, so the
    effect of flipping a cell is read from the N x N windows that cover it."""

    def __init__(self, model, grid, periodic):
        self.model, self.grid, self.periodic = model, grid, bool(periodic)
        self.codes = model.window_codes(grid, periodic)

    def classes(self, free):
        """Free cells grouped by colour class, each class as (ys, xs) in row-major order."""
        h, w = self.grid.shape
        n = self.model.n
        cx, nx = ring_colours(w, n, self.periodic)
        cy, ny = ring_colours(h, n, self.periodic)
        yy, xx = np.nonzero(free)
        which = cy[yy] * nx + cx[xx]
        return [(yy[which == k], xx[which == k]) for k in range(nx * ny)]

    def _windows(self, ys, xs, oy, ox):
        """Windows in which cell (y, x) sits at offset (oy, ox): their indices into
        `codes` and which of them exist (all do on a torus)."""
        wy, wx = ys - oy, xs - ox
        if self.periodic:
            h, w = self.grid.shape
            return wy % h, wx % w, None
        hh, ww = self.codes.shape
        ok = (wy >= 0) & (wy < hh) & (wx >= 0) & (wx < ww)
        return wy[ok], wx[ok], ok

    def flip_log_ratio(self, ys, xs):
        """log of the score ratio (after / before) of flipping each given cell,
        every cell on its own: the sum over its covering windows."""
        n, model = self.model.n, self.model
        total = np.zeros(len(ys))
        for oy in range(n):
            for ox in range(n):
                wy, wx, ok = self._windows(ys, xs, oy, ox)
                code = self.codes[wy, wx]
                gain = model.log_weight_of(code ^ (1 << (oy * n + ox))) - model.log_weight_of(code)
                if ok is None:
                    total += gain
                else:
                    total[ok] += gain
        return total

    def flip(self, ys, xs):
        """Flip the cells; they must be pairwise at least n apart in x or y."""
        n = self.model.n
        for oy in range(n):
            for ox in range(n):
                wy, wx, _ = self._windows(ys, xs, oy, ox)
                self.codes[wy, wx] ^= 1 << (oy * n + ox)
        self.grid[ys, xs] ^= True


def convchain(sample, out_w, out_h, n=3, temperature=1.0, sweeps=30, symmetry=8,
              periodic_in=True, periodic_out=False, floor=0.1, seed=0, init=None, free=None,
              fill=0.5):
    """Learn windows from `sample` and draw a new (out_h, out_w) bool grid in its
    style. Returns (grid, stats); see `ConvChainModel` and `ConvChainModel.sample`.
    To draw several grids from one sample, build the model once."""
    model = ConvChainModel(sample, n, symmetry, periodic_in, floor)
    return model.sample(out_w, out_h, temperature, sweeps, periodic_out, seed, init, free, fill)


def convchain_tilemap(sample, out_w, out_h, n=3, temperature=1.0, sweeps=30, seed=0,
                      rim=True, join='prim', fill=None, symmetry=8, periodic_in=True, floor=0.1):
    """A new TileMap in the style of `sample`: open cells are floor, the rest wall.

    `sample` is a bool array or a TileMap (its passable cells are the sample).
    The chain starts with `fill` of the cells open (default: the sample's own
    open share, which keeps a sparse sample's output from filling up). With
    `rim` the outermost ring is fixed to wall, like the grammars' sealed edges.
    The sampled blobs are then joined into one 4-connected region by
    `tilegen_grammars.connect_components` (`join`: 'prim' or 'star'), which also
    walls off pockets under four cells. What carries over from the sample is
    local structure (corridor widths, room shapes, how walls meet), not layout:
    a hub with spokes does not come back as a hub."""
    if isinstance(sample, tg.TileMap):
        sample = tg.passable_mask(sample)
    model = ConvChainModel(sample, n, symmetry, periodic_in, floor)
    free = None
    if rim:
        free = np.ones((int(out_h), int(out_w)), dtype=bool)
        free[0, :] = free[-1, :] = False
        free[:, 0] = free[:, -1] = False
    grid, _ = model.sample(out_w, out_h, temperature, sweeps, False, seed, None, free,
                           model.sample_open if fill is None else fill)
    tm = tg.build(int(out_w), int(out_h), [(grid, tg.FLOOR)], default=tg.WALL)
    tg.connect_components(tm, tg.passable_mask(tm), tg.FLOOR, width=1, strategy=join)
    return tm
