"""Overlapping-model wave function collapse: learn local structure from a drawn
exemplar and synthesise a grid whose every N x N window appears in it.

Clean-room implementation from A. Efros and T. Leung, 'Texture synthesis by
non-parametric sampling', ICCV 1999; P. Merrell, 'Example-based model
synthesis', I3D 2007; A. Mackworth, 'Consistency in networks of relations'
(AC-3), Artificial Intelligence 8, 1977; and the published description of
M. Gumin's WaveFunctionCollapse (2016, MIT), none of whose source was
consulted. Feature list informed by Sverchok (GPL-3.0); no Sverchok code was
consulted.

The method: extract every N x N window of the exemplar (wrapping when
`periodic_in`), add the chosen symmetries, merge duplicates (weight = how often
a pattern occurs), and let pattern b sit beside pattern a in a direction when
their overlapping strips agree. The output wave has one cell per pattern
position, each holding the set of patterns still possible there. Repeatedly the
undecided cell with the least Shannon entropy collapses to one pattern sampled
by weight, and arc consistency propagates the consequences; an empty cell is a
contradiction and the attempt restarts on the next named sub-stream.

Domains are Python-int bitmasks and propagation, count quotas and the
contradiction hook are the shared core in `tilegen_csp`, the same one the
socket solver extensions use, so there is one restart-and-quota vocabulary.

Randomness: attempt i draws from `tf.seed_stream(seed, 'wfc', i)`.
Imports downward only: `tilegen_fields` and `tilegen_csp`.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

import tilegen_fields as tf
from tilegen_csp import BitGrid, Failures, Quota, iter_bits, quota_scopes, settle_quotas

# neighbour offsets of a pattern position: left, down, right, up
DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))

# symmetry -> (clockwise quarter turns, mirrored left-right first) transforms
SYMMETRIES = {
    1: ((0, False),),
    2: ((0, False), (0, True)),
    4: tuple((q, False) for q in range(4)),
    8: tuple((q, m) for m in (False, True) for q in range(4)),
}


def _transform(window, quarter, mirrored):
    out = window[:, ::-1] if mirrored else window
    return np.rot90(out, -quarter) if quarter else out


class PatternModel:
    """Patterns, weights and adjacency learned from an exemplar.

    patterns  (P, n, n): distinct windows in order of first occurrence
    weights   (P,): how many times each occurs (symmetric copies included)
    ground    pattern ids found on the exemplar's last row (identity and
              mirrored copies only: a rotation would no longer be on the ground)
    compat    {direction index: [bitmask of patterns allowed that way]}
    """

    def __init__(self, exemplar, n=3, periodic=True, symmetry=1):
        ex = np.asarray(exemplar)
        if ex.ndim != 2 or ex.size == 0:
            raise ValueError(f"exemplar must be a non-empty 2D array, got shape {ex.shape}")
        if not (np.issubdtype(ex.dtype, np.integer) or ex.dtype == np.bool_):
            raise ValueError("exemplar must hold integer tile ids")
        if int(n) != n or not 1 <= n <= 5:
            raise ValueError(f"n must be an integer from 1 to 5, got {n!r}")
        n = int(n)
        if symmetry not in SYMMETRIES:
            raise ValueError(f"symmetry must be one of {sorted(SYMMETRIES)}, got {symmetry!r}")
        rows, cols = ex.shape
        if n > rows or n > cols:
            raise ValueError(f"pattern size {n} is larger than the {rows}x{cols} exemplar")
        self.n, self.exemplar, self.periodic_in, self.symmetry = n, ex, bool(periodic), symmetry

        source = np.pad(ex, ((0, n - 1), (0, n - 1)), mode="wrap") if periodic else ex
        span_y, span_x = (rows, cols) if periodic else (rows - n + 1, cols - n + 1)
        index, order, counts, ground = {}, [], [], set()
        for y in range(span_y):
            for x in range(span_x):
                window = source[y:y + n, x:x + n]
                for quarter, mirrored in SYMMETRIES[symmetry]:
                    variant = _transform(window, quarter, mirrored)
                    key = variant.tobytes()
                    pid = index.get(key)
                    if pid is None:
                        pid = index[key] = len(order)
                        order.append(np.ascontiguousarray(variant))
                        counts.append(0)
                    counts[pid] += 1
                    if y == rows - n and quarter == 0:
                        ground.add(pid)
        self.patterns = np.stack(order)
        self.weights = np.array(counts, dtype=np.int64)
        self.ground = sorted(ground)
        self.compat = {d: table for d, table in enumerate(self._adjacency())}

    @property
    def count(self):
        return len(self.patterns)

    def _adjacency(self):
        """Bitmask per pattern per direction: b may sit at offset (dx, dy) from a
        when a's strip facing that way equals b's strip facing back."""
        patterns, n = self.patterns, self.n
        tables = []
        for dx, dy in DIRECTIONS:
            ys0, ys1 = max(0, dy), n + min(0, dy)
            xs0, xs1 = max(0, dx), n + min(0, dx)
            heads = {}
            for b, pattern in enumerate(patterns):
                key = pattern[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx].tobytes()
                heads[key] = heads.get(key, 0) | (1 << b)
            tables.append([heads.get(pattern[ys0:ys1, xs0:xs1].tobytes(), 0) for pattern in patterns])
        return tables

    def states_with_tile(self, tiles, anchor=(0, 0)):
        """Bitmask of the patterns whose tile at `anchor` (x, y in the window)
        is one of `tiles`. Anchor (0, 0) is the tile a pattern writes to the
        output cell it sits at."""
        wanted = set(int(t) for t in tiles)
        ax, ay = anchor
        mask = 0
        for pid, pattern in enumerate(self.patterns):
            if int(pattern[ay, ax]) in wanted:
                mask |= 1 << pid
        return mask

    def wave_size(self, out_w, out_h, periodic_out):
        return (out_w, out_h) if periodic_out else (out_w - self.n + 1, out_h - self.n + 1)


def tile_quota(model, tiles, op, count, anchor=(0, 0), cells=None, name=""):
    """A quota on how many wave cells hold a pattern whose `anchor` tile is in
    `tiles`. With a periodic output and anchor (0, 0) that is the number of
    output cells showing one of those tiles; `cells` are wave (x, y) cells."""
    return Quota(model.states_with_tile(tiles, anchor), op, count, cells=cells, name=name)


class WaveSolver:
    """One attempt: the wave and the observe / propagate loop.

    `rng` needs `random(size=None)` like a numpy Generator: an array of
    `size` uniform draws, or a single draw. Each step draws one noise value
    below 1e-6 per cell for tie-breaking, then one draw to sample the pattern.
    """

    def __init__(self, model, wave_w, wave_h, periodic=False, ground=False, quotas=(), rng=None):
        self.model, self.width, self.height = model, wave_w, wave_h
        self.cells = wave_w * wave_h
        self.rng = rng
        self.ground = bool(ground)
        self.quotas = list(quotas)
        self.scopes = quota_scopes(self.quotas, self._cell_index)
        self.grid = BitGrid(model.compat, wave_w, wave_h,
                            [(d, dx, dy) for d, (dx, dy) in enumerate(DIRECTIONS)], periodic)
        full = (1 << model.count) - 1
        self.doms = [full] * self.cells
        self.ent = np.zeros(self.cells)
        self.und = np.zeros(self.cells, dtype=bool)
        self._w = [float(w) for w in model.weights]
        self._wlogw = [w * math.log(w) for w in self._w]
        self._entropy_cache = {}
        self.collapses = 0
        self.reason = None
        self.last = None            # (cell, pattern) of the latest collapse

    def _cell_index(self, cell):
        x, y = cell
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise ValueError(f"cell {cell} is outside the {self.width}x{self.height} wave")
        return y * self.width + x

    def entropy(self, domain):
        """Shannon entropy of the weighted set of patterns in a domain:
        log(sum w) - sum(w log w) / sum(w)."""
        hit = self._entropy_cache.get(domain)
        if hit is None:
            if domain & (domain - 1) == 0:
                hit = 0.0
            else:
                total = sum_log = 0.0
                for pid in iter_bits(domain):
                    total += self._w[pid]
                    sum_log += self._wlogw[pid]
                hit = math.log(total) - sum_log / total
            self._entropy_cache[domain] = hit
        return hit

    def _refresh(self, touched):
        for idx in touched:
            d = self.doms[idx]
            self.ent[idx] = self.entropy(d)
            self.und[idx] = d & (d - 1) != 0

    def initialize(self):
        """Ground rule, then arc consistency over the whole wave, then quotas, before
        any choice is made. 'ok' or 'contradiction'."""
        model = self.model
        if self.ground:
            ground = sum(1 << pid for pid in model.ground)
            for y in range(self.height):
                for x in range(self.width):
                    self.doms[y * self.width + x] &= ground if y == self.height - 1 else ~ground
        queue = deque(range(self.cells))
        if 0 in self.doms or not self.grid.propagate(self.doms, queue):
            self.reason = "initial"
            return "contradiction"
        if self.quotas and not settle_quotas(self.grid, self.quotas, self.scopes, self.doms):
            self.reason = "initial"
            return "contradiction"
        self._refresh(range(self.cells))
        return "ok"

    def fork(self, rng):
        """A fresh attempt that starts from this solver's initialised wave."""
        other = object.__new__(WaveSolver)
        other.__dict__.update(self.__dict__)
        other.doms = list(self.doms)
        other.ent = self.ent.copy()
        other.und = self.und.copy()
        other.grid = BitGrid(self.model.compat, self.width, self.height, self.grid.directions,
                             self.grid.periodic)
        other.grid._allowed = self.grid._allowed
        other.rng = rng
        other.collapses = 0
        other.reason = None
        other.last = None
        return other

    def pick_cell(self):
        """Undecided cell of least entropy (ties and noise from the stream), or None."""
        if not self.und.any():
            return None
        noise = np.asarray(self.rng.random(self.cells)) * 1e-6
        return int(np.argmin(np.where(self.und, self.ent + noise, np.inf)))

    def sample_pattern(self, cell):
        domain = self.doms[cell]
        ids = list(iter_bits(domain))
        total = sum(self._w[i] for i in ids)
        r = float(self.rng.random()) * total
        acc = 0.0
        for pid in ids:
            acc += self._w[pid]
            if acc >= r:
                return pid
        return ids[-1]

    def step(self):
        """Collapse the least-entropy undecided cell and propagate.
        'done' when every cell is decided, 'collapsed', or 'contradiction'."""
        cell = self.pick_cell()
        if cell is None:
            return "done"
        pid = self.sample_pattern(cell)
        self.doms[cell] = 1 << pid
        self.collapses += 1
        self.last = (cell, pid)
        touched = {cell}
        if not self.grid.propagate(self.doms, deque([cell]), touched):
            self.reason = "propagation"
            return "contradiction"
        if self.quotas and not settle_quotas(self.grid, self.quotas, self.scopes, self.doms, touched):
            self.reason = "quota"
            return "contradiction"
        self._refresh(touched)
        return "collapsed"

    def wave(self):
        """The wave as a boolean (height, width, patterns) array."""
        out = np.zeros((self.height, self.width, self.model.count), dtype=bool)
        for idx, d in enumerate(self.doms):
            for pid in iter_bits(d):
                out[idx // self.width, idx % self.width, pid] = True
        return out

    def image(self, out_w, out_h):
        """Paste each cell's chosen window: the output is the cell grid, plus for a
        non-periodic output the last n-1 rows and columns from the edge cells."""
        ids = np.array([d.bit_length() - 1 for d in self.doms]).reshape(self.height, self.width)
        ys, xs = np.arange(out_h), np.arange(out_w)
        cy, cx = np.minimum(ys, self.height - 1), np.minimum(xs, self.width - 1)
        grid = self.model.patterns[ids[cy[:, None], cx[None, :]], (ys - cy)[:, None], (xs - cx)[None, :]]
        return grid.astype(self.model.exemplar.dtype)


def overlapping_wfc(exemplar, out_w, out_h, n=3, periodic_in=True, periodic_out=False,
                    symmetry=1, ground=False, seed=0, max_tries=10, *, quotas=(),
                    on_contradiction=None):
    """Synthesise an (out_h, out_w) grid from an exemplar. Returns (grid or None, stats).

    exemplar      2D integer array of tile ids
    n             pattern size, 1..5
    periodic_in   wrap the exemplar when extracting patterns
    periodic_out  wrap the output (it is then also the wave size)
    symmetry      1 identity; 2 + left-right mirror; 4 + the quarter turns;
                  8 the whole dihedral group
    ground        the output's last row of windows may only hold patterns from
                  the exemplar's last row, and those patterns may sit nowhere
                  else
    seed          attempt i draws from tf.seed_stream(seed, 'wfc', i)
    max_tries     contradictions restart on the next stream; None after this many
    quotas        `Quota` objects over pattern ids, or callables f(model) ->
                  Quota (see `tile_quota`), enforced after every propagation
    on_contradiction(attempt, reason)  called for every failed attempt

    stats: tries, collapses, propagations, restarts and 'failures' (one reason
    per failed attempt). When the ground rule or a quota makes the wave
    impossible before any choice, every attempt fails at once and None comes
    back after exactly `max_tries` attempts. Every N x N window of the result
    (wrapping windows too when `periodic_out`) is a pattern of the exemplar.
    """
    model = PatternModel(exemplar, n, periodic_in, symmetry)
    out_w, out_h = int(out_w), int(out_h)
    if out_w < model.n or out_h < model.n:
        raise ValueError(f"output {out_w}x{out_h} is smaller than the pattern size {model.n}")
    resolved = [q(model) if callable(q) else q for q in quotas]
    wave_w, wave_h = model.wave_size(out_w, out_h, periodic_out)
    failures = Failures(on_contradiction)

    template = WaveSolver(model, wave_w, wave_h, periodic_out, ground, resolved)
    ready = template.initialize() == "ok"
    stats = {"tries": 0, "collapses": 0, "propagations": template.grid.pops, "restarts": 0}
    for attempt in range(max_tries):
        stats["tries"] += 1
        if not ready:
            failures.record(attempt, template.reason)
            stats["restarts"] += 1
            continue
        rng = np.random.default_rng(tf.seed_stream(seed, "wfc", attempt))
        solver = template.fork(rng)
        status = "collapsed"
        while status == "collapsed":
            status = solver.step()
        stats["collapses"] += solver.collapses
        stats["propagations"] += solver.grid.pops
        if status == "done":
            stats["failures"] = list(failures)
            return solver.image(out_w, out_h), stats
        failures.record(attempt, solver.reason)
        stats["restarts"] += 1
    stats["failures"] = list(failures)
    return None, stats
