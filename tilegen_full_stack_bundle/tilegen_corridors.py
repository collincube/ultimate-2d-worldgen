"""Corridor networks: retract dead ends, count them, and add loops.

`TileMap.carve_maze` in the byte-frozen `tilegen_system` leaves dead ends, and
the compiler's design lint only looks at open ratio and connectivity. This
module is the post-pass that was missing, working on boolean open-cell masks
(or a TileMap through the small wrappers) without touching the frozen file:

* `retract_dead_ends` repeatedly removes open cells with at most one open
  4-neighbour. Removing such a cell takes away one cell and at most one link,
  so the cycle rank of the network never changes; trees melt away entirely
  unless `keep` protects cells (rooms, a start, a goal).
* `corridor_metrics` reports cells, links, components, dead ends, junctions and
  the cycle rank E - V + C, the number of independent loops.
* `insert_cycles` opens a few single walls between corridor cells that are
  already connected, each adding exactly one loop.
* `lint_corridors` puts those numbers on a compiled world as an opt-in finding.
* `kiss_blocks`, `count_kisses` and `open_diagonal_kisses` find and clear
  'diagonal kisses': 2 x 2 blocks whose only open cells touch at a corner. The
  4-connected navigation cannot cross one, yet it reads as a passage on a
  map. Cellular-automaton caves are full of them, and `connect_components`
  repairs components but never kisses. One of the two blocking cells is opened,
  so a kiss becomes a three-cell corner and nothing is ever closed.

Idea-level port from MarkovJunior (MIT licence, Copyright (C) 2022 Maxim Gumin,
github.com/mxgmn/MarkovJunior@42aaf24bcf54ae164fba49c0a59348297904a676:
models/NystromDungeon.xml 'insert cycles' and 'retract dead ends',
models/NoDeadEnds.xml, and the kiss rule `<all in="AD/DA" out="AA/DA"/>` of
models/Cave.xml), after Bob Nystrom, 'Rooms and Mazes' (2014). Rewritten as
numpy mask operations; no source text was carried over.

Metrics are meant for one-cell-wide corridors: in open rooms every 2 x 2 block
of floor counts as a loop. All randomness is a named stream
(`tf.rng_for(seed, 'corridors.cycles')`, `tf.rng_for(seed, 'corridors.kisses')`).
"""
from __future__ import annotations

import numpy as np

import tilegen_fields as tf
import tilegen_grammars as tg


def _as_mask(mask):
    arr = np.asarray(mask, dtype=bool)
    if arr.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {arr.shape}")
    return arr


def _neighbours(values):
    """Value of the north, south, west and east neighbour of every cell (zero or
    False beyond the edge), for a boolean mask or an integer label array."""
    north = np.zeros_like(values); north[1:, :] = values[:-1, :]
    south = np.zeros_like(values); south[:-1, :] = values[1:, :]
    west = np.zeros_like(values); west[:, 1:] = values[:, :-1]
    east = np.zeros_like(values); east[:, :-1] = values[:, 1:]
    return north, south, west, east


def open_degree(mask):
    """Number of open 4-neighbours of every cell, as an int8 array."""
    mask = _as_mask(mask)
    n, s, w, e = _neighbours(mask)
    return n.astype(np.int8) + s + w + e


def dead_end_cells(mask, keep=None):
    """Open cells with at most one open 4-neighbour (isolated cells included),
    except cells in `keep`."""
    mask = _as_mask(mask)
    found = mask & (open_degree(mask) <= 1)
    return found & ~_as_mask(keep) if keep is not None else found


def retract_dead_ends(mask, keep=None, max_passes=None):
    """Remove dead ends until none are left (or `max_passes` rounds have run).

    Each round removes every open cell with at most one open 4-neighbour that is
    not in `keep`. Returns the new mask; the input is not modified. The result
    is the same however the rounds are batched, and cycle rank is unchanged.
    """
    cur = _as_mask(mask).copy()
    protected = None if keep is None else _as_mask(keep)
    rounds = 0
    while max_passes is None or rounds < max_passes:
        peel = dead_end_cells(cur, protected)
        if not peel.any():
            break
        cur &= ~peel
        rounds += 1
    return cur


def corridor_metrics(mask):
    """Counts for a corridor network on a boolean mask.

    vertices (open cells), edges (4-adjacent open pairs), components
    (4-connected), cycle_rank = edges - vertices + components (independent
    loops), dead_ends (open cells with exactly one neighbour), isolated (none),
    junctions (three or four neighbours).
    """
    mask = _as_mask(mask)
    deg = open_degree(mask)
    vertices = int(mask.sum())
    edges = int((mask[:, :-1] & mask[:, 1:]).sum() + (mask[:-1, :] & mask[1:, :]).sum())
    components = tf.connected_components(mask)[1]
    return {
        "vertices": vertices,
        "edges": edges,
        "components": int(components),
        "cycle_rank": edges - vertices + int(components),
        "dead_ends": int((mask & (deg == 1)).sum()),
        "isolated": int((mask & (deg == 0)).sum()),
        "junctions": int((mask & (deg >= 3)).sum()),
    }


def insert_cycles(mask, count, seed=0):
    """Open up to `count` walls to add that many independent loops.

    A wall cell qualifies when its two open neighbours face each other across
    it (north and south, or west and east), its other two sides are closed and
    both neighbours already belong to the same component, so opening it adds
    one cell and two links and the cycle rank grows by exactly one. No opened
    cell touches another (diagonals included), which keeps the corridors one
    cell wide. Candidates are taken in an order drawn from the named stream
    'corridors.cycles'. Returns (new mask, [(x, y), ...] opened cells); fewer
    than `count` are opened if the candidates run out.
    """
    cur = _as_mask(mask).copy()
    count = int(count)
    if count < 0:
        raise ValueError(f"count must be >= 0, got {count}")
    labels, _ = tf.connected_components(cur)
    n, s, w, e = _neighbours(cur)
    ln, ls, lw, le = _neighbours(labels)
    vertical = ~cur & n & s & ~w & ~e & (ln == ls)
    horizontal = ~cur & w & e & ~n & ~s & (lw == le)
    ys, xs = np.nonzero(vertical | horizontal)
    order = tf.rng_for(seed, "corridors.cycles").permutation(len(xs))
    blocked = np.zeros_like(cur)
    opened = []
    for i in order:
        if len(opened) >= count:
            break
        x, y = int(xs[i]), int(ys[i])
        if blocked[y, x]:
            continue
        cur[y, x] = True
        opened.append((x, y))
        blocked[max(0, y - 1):y + 2, max(0, x - 1):x + 2] = True
    return cur, opened


def retract_tilemap(tm, keep=None, wall=tg.WALL):
    """Retract the dead ends of a TileMap's passable cells in place, turning the
    removed cells into `wall` and clearing their overlays. Returns how many."""
    open_cells = tg.passable_mask(tm)
    removed = open_cells & ~retract_dead_ends(open_cells, keep)
    for y, x in zip(*np.nonzero(removed)):
        tm.base[y][x] = wall
        tm.overlay[y][x] = ""
    return int(removed.sum())


def insert_cycles_tilemap(tm, count, seed=0, tile=tg.FLOOR):
    """`insert_cycles` on a TileMap's passable cells, opening cells as `tile`.
    Returns the opened (x, y) cells."""
    _, opened = insert_cycles(tg.passable_mask(tm), count, seed)
    for x, y in opened:
        tm.base[y][x] = tile
    return opened


def kiss_blocks(mask):
    """Top-left corners of every diagonal kiss: the 2 x 2 blocks that hold two
    open cells on one diagonal and two closed cells on the other. Returns
    (xs, ys) index arrays, in row-major order of the corner."""
    m = _as_mask(mask)
    tl, tr, bl, br = m[:-1, :-1], m[:-1, 1:], m[1:, :-1], m[1:, 1:]
    ys, xs = np.nonzero((tl & br & ~tr & ~bl) | (tr & bl & ~tl & ~br))
    return xs, ys


def count_kisses(mask):
    """How many diagonal kisses (see `kiss_blocks`) the open mask has."""
    return len(kiss_blocks(mask)[0])


def open_diagonal_kisses(mask, seed=0, keep_closed=None):
    """Open one blocking cell of every diagonal kiss until none is left.

    For a kiss with open cells p and q, either of the two closed cells r and s
    next to both turns the contact into a 4-connected corner. One is picked at
    random per kiss (the rule of the original, which also leaves the choice to
    chance); opening it can create a kiss elsewhere, so the mask is scanned
    again until a scan finds none, which is certain to happen because every
    round opens at least one cell. Cells in `keep_closed` (a rim, a sealed
    wall) are never opened: a kiss with one such blocker opens the other, and a
    kiss whose two blockers are both kept stays. The same seed on the same mask
    gives the same result; the draws come from the named stream
    'corridors.kisses', per round one permutation of the kisses and one coin
    each.

    Only closed cells become open, so no region is ever split or removed and
    components can only merge: each 8-connected region of the input ends up
    inside one 4-connected region of the output. Returns (new mask,
    [(x, y), ...] opened cells in the order they were opened).
    """
    cur = _as_mask(mask).copy()
    kept = None if keep_closed is None else _as_mask(keep_closed)
    if kept is not None and kept.shape != cur.shape:
        raise ValueError(f"keep_closed shape {kept.shape} != mask shape {cur.shape}")
    rng = tf.rng_for(seed, "corridors.kisses")
    opened = []
    while True:
        xs, ys = kiss_blocks(cur)
        if not len(xs):
            break
        order = rng.permutation(len(xs))
        coins = rng.random(len(xs))
        progress = False
        for i in order:
            x, y = int(xs[i]), int(ys[i])
            tl, tr, bl, br = cur[y, x], cur[y, x + 1], cur[y + 1, x], cur[y + 1, x + 1]
            if tl and br and not tr and not bl:
                blockers = [(x + 1, y), (x, y + 1)]
            elif tr and bl and not tl and not br:
                blockers = [(x, y), (x + 1, y + 1)]
            else:
                continue                                    # an earlier opening cleared it
            if kept is not None:
                blockers = [(bx, by) for bx, by in blockers if not kept[by, bx]]
            if not blockers:
                continue
            bx, by = blockers[int(coins[i] * len(blockers))]
            cur[by, bx] = True
            opened.append((bx, by))
            progress = True
        if not progress:
            break
    return cur, opened


def open_kisses_tilemap(tm, seed=0, tile=tg.FLOOR, openable=(tg.WALL,), keep_closed=None):
    """`open_diagonal_kisses` on a TileMap's passable cells, in place: the
    blocking cells it opens become `tile`. Only cells whose base tile is in
    `openable` (default: plain walls) may be opened, so lava, abyss and other
    closed tiles are never overwritten; `keep_closed` adds more protected
    cells. An opened cell loses its overlay. Returns the opened (x, y) cells."""
    base = np.array(tm.base, dtype=object)
    protected = np.ones(base.shape, dtype=bool)
    for name in openable:
        protected &= base != name
    if keep_closed is not None:
        protected = protected | _as_mask(keep_closed)
    _, opened = open_diagonal_kisses(tg.passable_mask(tm), seed, protected)
    for x, y in opened:
        tm.base[y][x] = tile
        tm.overlay[y][x] = ""
    return opened


def lint_corridors(world, add_findings=False, severity="info"):
    """Corridor metrics for a compiled world's passable cells. With
    `add_findings`, also records a `geometry.corridors` finding. Not part of
    `tilegen_compiler.validate_world`, so existing findings are unchanged."""
    metrics = corridor_metrics(world.passable())
    if add_findings:
        ratio = metrics["dead_ends"] / max(metrics["vertices"], 1)
        world.finding(severity, "geometry.corridors",
                      f"{metrics['dead_ends']} dead ends ({ratio:.1%} of open cells), "
                      f"cycle rank {metrics['cycle_rank']}", **metrics)
    return metrics
