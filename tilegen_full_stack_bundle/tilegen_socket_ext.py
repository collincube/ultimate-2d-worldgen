"""Extensions for the socket WFC in tilegen_lowend_runtime, without touching it.

The legacy runtime (one fixed orientation per layout, a solver that always
starts with every state possible, no way to pin a module or cap a theme) stays
byte for byte as it is. Everything here is opt-in and built from its public
helpers:

* D4 module variants: every 16x16 module in its dihedral orientations as extra
  WFC states, sockets recomputed with the owner's own `side_slot_metrics`.
* Solver hooks: per-cell initial domains, pinned cells, per-state weights and
  priorities, uniqueness by source layout, a contradiction callback, and a
  reroll that redoes only the cells a new pin invalidates.
* Count quotas: at most / at least / exactly k cells whose state is in a set,
  by tri-state counting (yes, maybe, no) over the cell domains.
* Backtracking: `backtrack=` runs each attempt as a chronological backtracking
  search with banning (`tilegen_backtrack`) instead of discarding it at the
  first contradiction, with uniqueness propagated as soon as a cell is decided.

The bitmask arc consistency, the quota and the contradiction hook are the shared
core in `tilegen_csp`, which the overlapping model in `tilegen_wfc` uses too, so
there is one restart-and-quota vocabulary across both solvers.

Idea-level ports from DeBroglie (MIT licence, github.com/BorisTheBrave/DeBroglie,
v2.1.0): the rotation group and tile rotation of DeBroglie/Rot, the border and
fixed-tile constraints, the array-priority picker, the dirty index picker, the
tri-state count constraint and, in `tilegen_backtrack`, the backtracking. Also
M. Regin, 'Generalized arc consistency for global cardinality constraints',
AAAI-96. The code is rewritten for bitmask domains and this runtime's data
model; no source text was carried over.
"""
from __future__ import annotations

import random
from collections import Counter, deque

import tilegen_backtrack as bt
import tilegen_lowend_runtime as rt
from tilegen_csp import (QUOTA_OPS, BitGrid, Failures, Quota,  # noqa: F401  (re-exported)
                         quota_scopes, settle_quotas)

OPPOSITE = rt.OPPOSITE
DIRECTIONS = "NESW"

# The eight dihedral orientations of a square, as (clockwise quarter turns,
# mirrored). A mirrored orientation reflects left-right first, then rotates.
ORIENTATIONS = tuple((quarter, mirrored) for mirrored in (False, True) for quarter in range(4))


def orientation_name(quarter, mirrored):
    return ("m" if mirrored else "r") + str(90 * quarter)


# ------------------------------------------------------------------ D4 variants

def transform_grid(grid, quarter=0, mirrored=False):
    """Rotate a square grid (list of rows) clockwise by `quarter` quarter turns,
    after reflecting it left-right when `mirrored`. The input is not modified."""
    out = [list(row) for row in grid]
    if mirrored:
        out = [row[::-1] for row in out]
    for _ in range(quarter % 4):
        n = len(out)
        out = [[out[n - 1 - x][y] for x in range(n)] for y in range(n)]
    return out


def module_sockets(mask):
    """Socket fields for a walkable mask, computed exactly as build_module_library
    does: four-slot metrics per side, the two cheapest slots form the side mask."""
    side_metrics = {side: rt.side_slot_metrics(mask, side, rt.MODULE_SLOTS) for side in DIRECTIONS}
    side_masks, side_anchors = {}, {}
    for side, entries in side_metrics.items():
        bits = 0
        for entry in entries[:2]:
            bits |= 1 << entry["slot"]
        side_masks[side] = int(bits)
        side_anchors[side] = {str(entry["slot"]): entry["anchor"] for entry in entries}
    return {
        "side_masks": side_masks,
        "side_slots": {side: rt.bitmask_to_slots(bits) for side, bits in side_masks.items()},
        "side_anchor_map": side_anchors,
        "side_metrics": side_metrics,
    }


def orient_module(module, quarter=0, mirrored=False):
    """One module in one orientation: grids transformed, sockets recomputed from
    the transformed largest-component mask. Identity returns an equal copy."""
    mask = transform_grid(module["largest_component_mask"], quarter, mirrored)
    variant = dict(module)
    variant["base"] = transform_grid(module["base"], quarter, mirrored)
    variant["overlay"] = transform_grid(module["overlay"], quarter, mirrored)
    variant["largest_component_mask"] = mask
    variant.update(module_sockets(mask))
    variant["source_state"] = module.get("source_state", module["state_index"])
    variant["orientation"] = orientation_name(quarter, mirrored)
    if (quarter % 4, bool(mirrored)) != (0, False):
        variant["source_render"] = ""       # the source image shows the other way round
    return variant


def compute_compat(modules):
    """Adjacency bitmasks per direction, identical to build_module_library: state
    j may sit next to state i in direction d when their facing side masks share a slot."""
    compat = {}
    for d in DIRECTIONS:
        table = {}
        for mine in {m["side_masks"][d] for m in modules}:
            bits = 0
            for j, other in enumerate(modules):
                if mine & other["side_masks"][OPPOSITE[d]]:
                    bits |= 1 << j
            table[mine] = bits
        compat[d] = [table[m["side_masks"][d]] for m in modules]
    return compat


def expand_d4(modules, orientations=ORIENTATIONS, dedupe=True):
    """Expand every module into its dihedral orientations as extra WFC states.

    Returns (variants, compat): the new module library and its adjacency
    bitmasks, in the shape the legacy solver and assembler take. A variant
    carries `source_state` (index in the input library) and `orientation`
    ('r0', 'r90', 'r180', 'r270', 'm0', ...). With `dedupe`, orientations of one
    module that draw the same tiles are kept once (the first in `orientations`
    order, so the identity orientation wins), which collapses symmetric layouts.
    """
    if not orientations:
        raise ValueError("orientations must not be empty")
    variants = []
    for source, module in enumerate(modules):
        seen = set()
        for quarter, mirrored in orientations:
            variant = orient_module(dict(module, state_index=source), quarter, mirrored)
            key = (tuple(map(tuple, variant["base"])), tuple(map(tuple, variant["overlay"])))
            if dedupe and key in seen:
                continue
            seen.add(key)
            variant["state_index"] = len(variants)
            variants.append(variant)
    compat = compute_compat(variants)
    for index, variant in enumerate(variants):
        variant["compat_degree"] = int(sum(rt.bit_count(compat[d][index]) for d in DIRECTIONS))
    return variants, compat


def source_groups(variants):
    """State -> source layout, for 'one of each layout' uniqueness across orientations."""
    return [v["source_state"] for v in variants]


# ----------------------------------------------------------------- solver hooks

def states_where(modules, predicate):
    """Bitmask of the states whose module satisfies `predicate(module)`."""
    mask = 0
    for index, module in enumerate(modules):
        if predicate(module):
            mask |= 1 << index
    return mask


def all_states(modules):
    return (1 << len(modules)) - 1


def border_cells(grid_w, grid_h, sides="NESW"):
    """Cells (x, y) on the given sides of the grid, row by row."""
    out = []
    for y in range(grid_h):
        for x in range(grid_w):
            if (("N" in sides and y == 0) or ("S" in sides and y == grid_h - 1)
                    or ("W" in sides and x == 0) or ("E" in sides and x == grid_w - 1)):
                out.append((x, y))
    return out


def allow_states(cells, allowed_mask):
    """Initial-domain rule: these cells may only hold the states in `allowed_mask`."""
    return {tuple(cell): allowed_mask for cell in cells}


def ban_states(modules, cells, banned_mask):
    """Initial-domain rule: these cells may hold anything except `banned_mask`."""
    return {tuple(cell): all_states(modules) & ~banned_mask for cell in cells}


def merge_domains(*rules):
    """Combine initial-domain rules; a cell named twice keeps only states both allow."""
    merged = {}
    for rule in rules:
        for cell, mask in rule.items():
            merged[cell] = merged[cell] & mask if cell in merged else mask
    return merged


def default_weight(x, y, state, module, used_counts):
    """The legacy choice weight: compatibility degree, damped by how often the
    state is already used. Wrap it to bias a region or a theme."""
    return max(1, module["compat_degree"]) * (1 / (1 + used_counts.get(state, 0)))


def _cell_index(cell, grid_w, grid_h):
    x, y = cell
    if not (0 <= x < grid_w and 0 <= y < grid_h):
        raise ValueError(f"cell {cell} is outside the {grid_w}x{grid_h} grid")
    return y * grid_w + x


def _initial_domains(domains, grid_w, grid_h, all_mask):
    count = grid_w * grid_h
    out = [all_mask] * count
    if domains is None:
        return out
    if callable(domains):
        for y in range(grid_h):
            for x in range(grid_w):
                mask = domains(x, y)
                if mask is not None:
                    out[y * grid_w + x] = all_mask & int(mask)
    elif isinstance(domains, dict):
        for cell, mask in domains.items():
            out[_cell_index(cell, grid_w, grid_h)] &= int(mask)
    else:
        masks = list(domains)
        if len(masks) != count:
            raise ValueError(f"domains has {len(masks)} entries for {count} cells")
        out = [all_mask & int(m) for m in masks]
    return out


def _pins(fixed, grid_w, grid_h, state_count):
    pins = {}
    if not fixed:
        return pins
    items = fixed.items() if isinstance(fixed, dict) else ((c[:2], c[2]) for c in fixed)
    for cell, state in items:
        index = _cell_index(tuple(cell), grid_w, grid_h)
        if not 0 <= state < state_count:
            raise ValueError(f"state {state} is not in the library of {state_count}")
        if pins.get(index, state) != state:
            raise ValueError(f"cell {tuple(cell)} is pinned to two states")
        pins[index] = int(state)
    return pins


def _groups(groups, state_count):
    group_of = list(range(state_count)) if groups is None else [groups[s] for s in range(state_count)]
    members = {}
    for state, group in enumerate(group_of):
        members[group] = members.get(group, 0) | (1 << state)
    return group_of, [members[g] for g in group_of]


def _pick_state(domain, x, y, modules, rng, used_counts, weight, priority):
    """Weighted choice among the states of a domain; the legacy function when no
    hook is set, so the random stream is unchanged."""
    if weight is None and priority is None:
        return rt.choose_state_from_domain(domain, modules, rng, used_counts)
    states = list(rt.iter_bits(domain))
    if not states:
        return None
    if priority is not None:
        ranks = [priority(x, y, s, modules[s]) for s in states]
        top = max(ranks)
        states = [s for s, r in zip(states, ranks) if r == top]
    fn = weight or default_weight
    weights = [max(0.0, float(fn(x, y, s, modules[s], used_counts))) for s in states]
    total = sum(weights)
    if total <= 0.0:
        weights, total = [1.0] * len(states), float(len(states))
    r = rng.random() * total
    acc = 0.0
    for state, w in zip(states, weights):
        acc += w
        if acc >= r:
            return state
    return states[-1]


def quota_where(modules, predicate, op, count, cells=None, name=""):
    """A quota over the states whose module satisfies `predicate(module)`."""
    return Quota(states_where(modules, predicate), op, count, cells=cells, name=name)


_DIRECTIONS = tuple((key, dx, dy) for key, (dx, dy) in rt.DIRS.items())


def _socket_grid(compat, grid_w, grid_h):
    """Arc consistency over the grid, visiting neighbours N, E, S, W like the legacy solver."""
    return BitGrid(compat, grid_w, grid_h, _DIRECTIONS)


def check_placements(compat, grid_w, grid_h, placements):
    """True when every pair of neighbouring cells is allowed by `compat`."""
    for y in range(grid_h):
        for x in range(grid_w):
            a = placements[y * grid_w + x]
            if x + 1 < grid_w and not (compat["E"][a] >> placements[y * grid_w + x + 1]) & 1:
                return False
            if y + 1 < grid_h and not (compat["S"][a] >> placements[(y + 1) * grid_w + x]) & 1:
                return False
    return True


class _SocketProblem(bt.GridProblem):
    """The socket solver as a problem for `tilegen_backtrack.search`: the legacy
    cell picker, the weight / priority hooks for the choice, and the usage counts
    the weights read, kept right through undone decisions."""

    def __init__(self, grid, doms, rng, rules, scopes, modules, grid_w, used_counts, weight, priority):
        super().__init__(grid, doms, rng, rules, scopes)
        self.modules, self.grid_w, self.used_counts = modules, grid_w, used_counts
        self.weight, self.priority = weight, priority

    def choose(self, cell):
        return _pick_state(self.doms[cell], cell % self.grid_w, cell // self.grid_w, self.modules,
                           self.rng, self.used_counts, self.weight, self.priority)

    def decide(self, cell, state):
        self.used_counts[state] += 1
        return super().decide(cell, state)

    def snapshot(self):
        return list(self.doms), Counter(self.used_counts)

    def restore(self, snapshot):
        doms, counts = snapshot
        self.doms[:] = doms
        self.used_counts.clear()
        self.used_counts.update(counts)


def solve_socket_wfc_ext(modules, compat, grid_w, grid_h, seed=0, unique_states=False,
                         max_restarts=64, *, domains=None, fixed=None, weight=None,
                         priority=None, groups=None, quotas=(), on_contradiction=None,
                         backtrack=None):
    """The legacy `solve_socket_wfc` with hooks. With no hook it makes the same
    random draws and returns the same placements. Cell (x, y) is index
    y * grid_w + x.

    domains   initial domain per cell: a dict {(x, y): allowed_mask}, a list of
              masks, or a callable (x, y) -> mask or None. Build rules with
              `allow_states`, `ban_states` (e.g. keep a set off the border via
              `border_cells`) and `merge_domains`.
    fixed     pinned cells: {(x, y): state} or [(x, y, state), ...].
    weight    f(x, y, state, module, used_counts) -> weight for the random
              choice of a state (see `default_weight`).
    priority  f(x, y, state, module) -> rank; only states of the highest rank
              in a cell's domain are eligible for the choice.
    groups    state -> group id, so `unique_states` means 'one per group'
              (for instance `source_groups(variants)`: each layout once, in
              any orientation).
    quotas    `Quota` objects (see `quota_where`) enforced by tri-state counting
              after every propagation. A preset that must use every module
              once ('all100_unique') cannot also cap a theme.
    backtrack None (default) discards an attempt at its first contradiction and
              starts the next, as the legacy solver does. True, an integer (the
              budget) or a `tilegen_backtrack.Backtrack` runs each attempt as a
              chronological backtracking search instead: a contradiction undoes
              the latest decision and bans its state at that cell (see
              `tilegen_backtrack`; with `patience` it may undo several). The
              draws, the weights and the pins are the same, so a run that never
              contradicts is draw for draw the legacy run when `unique_states`
              is off; with it on, uniqueness is propagated as soon as a cell is
              decided (the state, or group, of a decided cell leaves every other
              domain) because a duplicate found only at the end would send the
              search back to decisions that did not cause it. A search ends as
              'unsat' (no decision is left to undo: no solution exists under
              these hooks, so the solve stops at once without using the other
              restarts), 'exhausted' (a patience search that gave up, having
              jumped back over part of the tree or being asked to jump past
              its first decision: it proves nothing) or 'budget' (more
              contradictions than the budget); the last two start the next
              attempt.
    on_contradiction(attempt, reason) is called for every failed attempt;
    reasons are 'domains', 'fixed', 'propagation', 'quota', 'no-choice' and
    'unique', and with `backtrack` also 'unsat', 'exhausted' and 'budget'.

    Returns {'success', 'attempts', 'placements', 'failures'}; `failures`
    lists the reason of each failed attempt. With `backtrack` the result also
    has 'backtracks' and 'backjumps', the totals over all attempts.
    """
    rng = random.Random(seed)
    state_count = len(modules)
    all_mask = (1 << state_count) - 1
    cell_count = grid_w * grid_h
    start = _initial_domains(domains, grid_w, grid_h, all_mask)
    pins = _pins(fixed, grid_w, grid_h, state_count)
    group_of, group_mask = _groups(groups, state_count)
    restricted = [i for i in range(cell_count) if start[i] != all_mask]
    grid = _socket_grid(compat, grid_w, grid_h)
    quotas = list(quotas)
    scopes = quota_scopes(quotas, lambda cell: _cell_index(cell, grid_w, grid_h))
    failures = Failures(on_contradiction)
    fail = failures.record
    policy = bt.Backtrack.coerce(backtrack)
    rules, rule_scopes = list(quotas), list(scopes)
    if policy is not None and unique_states:
        rules.append(bt.AllDifferent(group_of))
        rule_scopes.append(None)
    searched = {"backtracks": 0, "backjumps": 0}

    def result(success, attempt, placements):
        out = {"success": success, "attempts": attempt, "placements": placements,
               "failures": list(failures)}
        if policy is not None:
            out.update(searched)
        return out

    for attempt in range(max_restarts):
        doms = list(start)
        if 0 in doms:
            fail(attempt, "domains")
            continue
        collapsed = [False] * cell_count
        used_mask = 0
        used_counts = Counter()

        def enqueue_unique(queue_init):
            nonlocal used_mask
            dq = deque(queue_init)
            if not unique_states:
                return True, dq
            for idx in range(cell_count):
                if collapsed[idx]:
                    continue
                new_domain = doms[idx] & ~used_mask
                if new_domain != doms[idx]:
                    if new_domain == 0:
                        return False, None
                    doms[idx] = new_domain
                    dq.append(idx)
            return True, dq

        queue = list(restricted)
        broken = None
        for idx, state in pins.items():
            if not doms[idx] >> state & 1:
                broken = "fixed"
                break
            if unique_states and used_mask & (1 << state):
                broken = "unique"
                break
            doms[idx] = 1 << state
            collapsed[idx] = True
            used_counts[state] += 1
            if unique_states:
                used_mask |= group_mask[state]
            queue.append(idx)
        if broken:
            fail(attempt, broken)
            continue
        ok, dq = enqueue_unique(queue)
        if not ok or not grid.propagate(doms, dq):
            fail(attempt, "propagation")
            continue
        if quotas and not settle_quotas(grid, quotas, scopes, doms):
            fail(attempt, "quota")
            continue

        if policy is not None:
            if unique_states and not settle_quotas(grid, rules, rule_scopes, doms):
                fail(attempt, "unique")
                continue
            problem = _SocketProblem(grid, doms, rng, rules, rule_scopes, modules, grid_w, used_counts,
                                     weight, priority)
            found = bt.search(problem, policy)
            searched["backtracks"] += found["backtracks"]
            searched["backjumps"] += found["backjumps"]
            if found["status"] == "solved":
                return result(True, attempt + 1, [doms[idx].bit_length() - 1 for idx in range(cell_count)])
            fail(attempt, found["status"])
            if found["status"] == "unsat":
                return result(False, attempt + 1, [])
            continue

        while True:
            pending = [(rt.bit_count(doms[idx]), idx) for idx in range(cell_count) if not collapsed[idx]]
            if not pending:
                return result(True, attempt + 1, [doms[idx].bit_length() - 1 for idx in range(cell_count)])
            non_singletons = [item for item in pending if item[0] > 1]
            if not non_singletons:
                if unique_states and len({group_of[doms[idx].bit_length() - 1]
                                          for idx in range(cell_count)}) != cell_count:
                    fail(attempt, "unique")
                    break
                return result(True, attempt + 1, [doms[idx].bit_length() - 1 for idx in range(cell_count)])
            min_entropy = min(item[0] for item in non_singletons)
            cell = rng.choice([idx for count, idx in non_singletons if count == min_entropy])
            chosen = _pick_state(doms[cell], cell % grid_w, cell // grid_w, modules, rng, used_counts,
                                 weight, priority)
            if chosen is None:
                fail(attempt, "no-choice")
                break
            doms[cell] = 1 << chosen
            collapsed[cell] = True
            used_counts[chosen] += 1
            if unique_states:
                used_mask |= group_mask[chosen]
            ok, dq = enqueue_unique([cell])
            if not ok or not grid.propagate(doms, dq):
                fail(attempt, "propagation")
                break
            if quotas and not settle_quotas(grid, quotas, scopes, doms):
                fail(attempt, "quota")
                break
    return result(False, max_restarts, [])


def solve_preset(modules, compat, preset, **hooks):
    """Solve one of `tilegen_lowend_runtime.WORLD_PRESETS` (or any dict with its
    keys), optionally with solver hooks; the legacy `build_worlds` passes none."""
    return solve_socket_wfc_ext(modules, compat, preset["grid_w"], preset["grid_h"],
                                seed=preset["seed"], unique_states=preset["unique_states"],
                                max_restarts=preset["max_restarts"], **hooks)


def reroll_after_pin(modules, compat, grid_w, grid_h, previous, pins, seed=0,
                     unique_states=False, max_restarts=64, **hooks):
    """Pin cells in an existing solution and redo only what the pins invalidate.

    After the pins and `hooks` (see `solve_socket_wfc_ext`, except `fixed`)
    propagate, a cell whose previous state is still possible stays as it was;
    the rest are rerolled. If the kept cells leave the rerolled ones no
    solution, the rerolled region grows by one ring of neighbours and the solve
    is retried, down to rerolling every cell but the pins.

    Returns the solver result plus 'changed' (indices that differ from
    `previous`) and 'rings' (how many times the region had to grow).
    """
    if "fixed" in hooks:
        raise ValueError("pass pinned cells through `pins`, not `fixed`")
    cell_count = grid_w * grid_h
    if len(previous) != cell_count:
        raise ValueError(f"previous has {len(previous)} placements for {cell_count} cells")
    pin_map = _pins(pins, grid_w, grid_h, len(modules))
    all_mask = (1 << len(modules)) - 1

    def as_cells(by_index):
        return {(i % grid_w, i // grid_w): state for i, state in by_index.items()}

    def solve(by_index):
        return solve_socket_wfc_ext(modules, compat, grid_w, grid_h, seed, unique_states,
                                    max_restarts, fixed=as_cells(by_index), **hooks)

    doms = _initial_domains(hooks.get("domains"), grid_w, grid_h, all_mask)
    queue = deque(i for i in range(cell_count) if doms[i] != all_mask)
    for idx, state in pin_map.items():
        doms[idx] &= 1 << state
        queue.append(idx)
    if 0 in doms or not _socket_grid(compat, grid_w, grid_h).propagate(doms, queue):
        out = solve(pin_map)
        out.update(changed=[], rings=0)
        return out
    free = set(range(cell_count)) - set(pin_map)
    dirty = {i for i in free if not doms[i] >> previous[i] & 1}
    rings = 0
    while True:
        keep = {i: previous[i] for i in free - dirty}
        keep.update(pin_map)
        out = solve(keep)
        if out["success"]:
            break
        grown = set(dirty)
        for i in dirty:
            x, y = i % grid_w, i // grid_w
            for dx, dy in rt.TURN_DIRS:
                if 0 <= x + dx < grid_w and 0 <= y + dy < grid_h:
                    grown.add((y + dy) * grid_w + x + dx)
        grown &= free
        if grown == dirty:
            if dirty == free:
                break
            grown = set(free)
        dirty = grown
        rings += 1
    out["changed"] = [i for i in range(cell_count) if out["success"] and out["placements"][i] != previous[i]]
    out["rings"] = rings
    return out
