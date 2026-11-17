"""Bitmask constraint core shared by the wave-function-collapse solvers.

Both the socket (tiled) extensions in `tilegen_socket_ext` and the overlapping
model in `tilegen_wfc` keep one Python int per cell as its domain (bit i set =
state i still possible) and need the same three things, so they share them
here instead of each inventing its own:

* `BitGrid`: arc consistency over a grid, with the supported-set of a domain
  cached per (direction, domain).
* `Quota`: at most / at least / exactly k cells whose state is in a set, by
  tri-state counting, with `settle_quotas` applying a list of them to a
  fixpoint.
* `Failures`: the contradiction hook. Every failed attempt is recorded with a
  reason and passed to an optional `on_contradiction(attempt, reason)`
  callback, so a restart policy is observable the same way in every solver.

The count constraint is an idea-level port of DeBroglie's CountConstraint (MIT
licence, github.com/BorisTheBrave/DeBroglie, v2.1.0); see also M. Regin,
'Generalized arc consistency for global cardinality constraints', AAAI-96.
Rewritten for bitmask domains; no source text was carried over.

Depends on nothing but the standard library.
"""
from __future__ import annotations

from collections import deque

QUOTA_OPS = ("at_most", "at_least", "exactly")


def iter_bits(mask):
    """Indices of the set bits of `mask`, lowest first."""
    while mask:
        low = mask & -mask
        yield low.bit_length() - 1
        mask ^= low


class Failures:
    """Reasons of the failed attempts so far, with an optional observer.

    A plain class (iterable, sized) rather than a `list` subclass so the
    module stays portable to the JS and GDScript ports (`forge.py port`)."""

    def __init__(self, on_contradiction=None):
        self.items = []
        self.callback = on_contradiction

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)

    def record(self, attempt, reason):
        self.items.append(reason)
        observer = self.callback
        if observer is not None:
            observer(attempt, reason)


class BitGrid:
    """Arc consistency for bitmask domains on a width x height grid.

    `compat[key][state]` is the bitmask of states allowed in the neighbouring
    cell in the direction `key`. `directions` lists (key, dx, dy) in the order
    neighbours are visited. With `periodic`, neighbours wrap around the edges.
    `pops` counts the cells taken off the work queue.
    """

    def __init__(self, compat, width, height, directions, periodic=False):
        self.compat, self.width, self.height = compat, width, height
        self.directions = tuple(directions)
        self.periodic = periodic
        self.pops = 0
        self._allowed = {}

    def allowed(self, key, domain):
        """States some member of `domain` permits next to it in direction `key`."""
        hit = self._allowed.get((key, domain))
        if hit is None:
            hit = 0
            table = self.compat[key]
            for state in iter_bits(domain):
                hit |= table[state]
            if len(self._allowed) > 200000:
                self._allowed.clear()
            self._allowed[(key, domain)] = hit
        return hit

    def propagate(self, doms, queue, touched=None):
        """Narrow neighbours until nothing changes. False if a domain empties.
        Cells whose domain shrank are added to `touched` when given."""
        width, height, periodic = self.width, self.height, self.periodic
        while queue:
            cell = queue.popleft()
            self.pops += 1
            x, y = cell % width, cell // width
            domain = doms[cell]
            for key, dx, dy in self.directions:
                nx, ny = x + dx, y + dy
                if periodic:
                    nx, ny = nx % width, ny % height
                elif not (0 <= nx < width and 0 <= ny < height):
                    continue
                nidx = ny * width + nx
                old = doms[nidx]
                new = old & self.allowed(key, domain)
                if new != old:
                    if new == 0:
                        return False
                    doms[nidx] = new
                    queue.append(nidx)
                    if touched is not None:
                        touched.add(nidx)
        return True


class Quota:
    """At most / at least / exactly `count` cells whose state is in a set.

    Tri-state counting over the cell domains: a cell is *yes* when every state
    it can still take is in the set, *no* when none is, *maybe* otherwise.
    'at_most': more than `count` yes cells is a contradiction, and at exactly
    `count` the set is banned from every maybe cell. 'at_least': fewer than
    `count` yes + maybe cells is a contradiction, and at exactly `count` the
    set is selected in every maybe cell. 'exactly' does both. `cells` limits
    the count to a list of (x, y) cells (default: the whole grid). Counted
    this way a quota also bites before a cell is collapsed.
    """

    __slots__ = ("mask", "op", "count", "cells", "name")

    def __init__(self, states, op, count, cells=None, name=""):
        if op not in QUOTA_OPS:
            raise ValueError(f"op must be one of {QUOTA_OPS}, got {op!r}")
        if int(count) < 0:
            raise ValueError(f"count must be >= 0, got {count}")
        if isinstance(states, int):
            self.mask = states
        else:
            self.mask = 0
            for state in states:
                self.mask |= 1 << state
        self.op, self.count, self.name = op, int(count), name
        self.cells = None if cells is None else sorted({tuple(c) for c in cells})

    def __repr__(self):
        return f"Quota({self.name or hex(self.mask)} {self.op} {self.count})"

    def tally(self, doms, indices=None):
        """(yes, maybe) cell counts for the given domains."""
        yes = maybe = 0
        for idx in (range(len(doms)) if indices is None else indices):
            domain = doms[idx]
            inside = domain & self.mask
            if inside == domain:
                yes += 1
            elif inside:
                maybe += 1
        return yes, maybe

    def apply(self, doms, changed, indices=None):
        """Enforce the quota on `doms` in place. Appends the cells it narrowed to
        `changed`; returns False on a contradiction."""
        yes, maybe = self.tally(doms, indices)
        scope = range(len(doms)) if indices is None else indices
        if self.op in ("at_most", "exactly"):
            if yes > self.count:
                return False
            if yes == self.count and maybe:
                for idx in scope:
                    domain = doms[idx]
                    if domain & self.mask and domain & ~self.mask:
                        doms[idx] = domain & ~self.mask
                        changed.append(idx)
                maybe = 0
        if self.op in ("at_least", "exactly"):
            if yes + maybe < self.count:
                return False
            if yes + maybe == self.count and maybe:
                for idx in scope:
                    domain = doms[idx]
                    if domain & self.mask and domain & ~self.mask:
                        doms[idx] = domain & self.mask
                        changed.append(idx)
        return True


def quota_scopes(quotas, cell_index):
    """Resolve each quota's (x, y) cell list to indices with `cell_index((x, y))`."""
    return [None if q.cells is None else [cell_index(c) for c in q.cells] for q in quotas]


def settle_quotas(grid, quotas, scopes, doms, touched=None):
    """Apply every quota, re-propagate what they narrowed, and repeat until
    nothing changes. False on a contradiction."""
    while True:
        changed = deque()
        for quota, scope in zip(quotas, scopes):
            if not quota.apply(doms, changed, scope):
                return False
        if not changed:
            return True
        if touched is not None:
            touched.update(changed)
        if not grid.propagate(doms, changed, touched):
            return False
