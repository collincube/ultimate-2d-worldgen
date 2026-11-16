"""Chronological backtracking with banning, for the bitmask solvers.

The restart loop of the socket solver throws a whole attempt away on the first
contradiction. When the compatibility sets are tight (sockets tightened, a theme
capped, every module used once) most attempts die long after the cause. This is
the usual cure from constraint solving, maintained arc consistency with
backtracking search (S. Russell and P. Norvig, 'Artificial Intelligence: A
Modern Approach', constraint satisfaction chapter): on a contradiction undo the
latest decision, remove its state from that cell for good ('ban'), propagate
the ban, and carry on; if the ban contradicts too, undo the decision before it,
and so on. The search is complete, so when it runs out of decisions to undo the
problem has no solution under the constraints given, which restarts can never
show.

Idea-level port from DeBroglie (MIT licence, Copyright (c) 2018 Adam Newgas,
github.com/BorisTheBrave/DeBroglie@27b37fa7a54f7c532b52634622f36eddd2a3c7da,
v2.1.0): the backtrack loop and the ban of the failed choice in
DeBroglie/Wfc/WavePropagator.cs (TryBacktrackUntilNoContradiction), the
patience backjump of DeBroglie/Wfc/IBacktrackPolicy.cs, and the two
negative-path ideas of DeBroglie.Test/WavePropagatorTest.cs (an odd periodic
chessboard must contradict; a constraint that forces backtracking). Rewritten
for bitmask domains with a snapshot of the domains per decision instead of an
undo trail; no source text was carried over.

The pieces:

* `search(problem, options)` is the loop. A problem offers `pick`, `choose`,
  `decide`, `forbid`, `snapshot` and `restore` (see `GridProblem`), so the loop
  knows nothing about modules, patterns or grids.
* `Backtrack` holds the options: a budget of undone decisions per search and the
  optional patience backjump (after 10 failed backtracks jump back 4 levels,
  then 8 after 20, doubling both; progress to a new greatest depth resets the
  schedule). A backjump discards several decisions without banning anything,
  so after one an exhausted search is no longer a proof (status 'exhausted'
  instead of 'unsat'); and since the schedule keeps escalating, a search with
  patience gives up for good once it asks to jump back past the first decision.
* `GridProblem` is the ready-made problem for bitmask domains on a
  `tilegen_csp.BitGrid`, with quotas or any other rule that has
  `apply(doms, changed, indices)`; `tilegen_socket_ext` builds on it.
* `AllDifferent` is such a rule: at most one cell per group of states, counted
  as soon as a cell is decided (its whole domain inside one group), the
  all-different forward check. `Quota(group, 'at_most', 1)` says the same but
  has to be built and scanned per group.

From the socket solver it is one argument:
`solve_socket_wfc_ext(modules, compat, w, h, unique_states=True, backtrack=True)`.

Measured on the shipped library (100 layouts, compatibility density 0.94) over
seeds 0 to 59 of the 10 x 10 all-different solve of the 'all100_unique' preset
(`test_backtracking` repeats the first eight): the restart loop needs 1 to 5
attempts (mean 2.0), the search one attempt every time with 1.1 backtracks on
average (44 of the 60 need none), and the all-different check alone, with
restarts (`backtrack=0`), 1.4 attempts. Where the restart loop gives up (10 x 10
tables of 100 states used once with a planted solution, 8 restarts, density 0.5
to 0.3) the search finds a valid arrangement in one attempt after 0 to 61
backtracks, 784 for the hardest table tried; a pigeonhole instance (n + 1 cells
that must differ, n states) takes exactly n! - 1 backtracks to refute.

Depends on `tilegen_csp` and the standard library only.
"""
from __future__ import annotations

from collections import deque

from tilegen_csp import iter_bits, settle_quotas


class Backtrack:
    """Options of a backtracking search.

    max_backtracks  how many contradictions one search may process (each undoes
                    a decision, or several in a backjump) before it gives up
                    with status 'budget' (default 2000: a restart with fresh
                    draws beats a very long search); None is unlimited (the
                    search still ends: plain backtracking is complete and the
                    patience schedule gives up). 0 gives up at the first
                    contradiction, which is a restart.
    patience        use the patience backjump (see the module docstring)
    """

    __slots__ = ("max_backtracks", "patience")

    def __init__(self, max_backtracks=2000, patience=False):
        if max_backtracks is not None:
            try:
                whole = max_backtracks >= 0 and int(max_backtracks) == max_backtracks
            except (TypeError, ValueError, OverflowError):
                whole = False
            if not whole:
                raise ValueError(f"max_backtracks must be None or an integer >= 0, got {max_backtracks!r}")
            max_backtracks = int(max_backtracks)
        self.max_backtracks, self.patience = max_backtracks, bool(patience)

    def __repr__(self):
        return f"Backtrack(max_backtracks={self.max_backtracks}, patience={self.patience})"

    @classmethod
    def coerce(cls, value):
        """None or False -> None (no backtracking); True -> the defaults; an
        integer -> that budget; a `Backtrack` -> itself."""
        if value is None or value is False:
            return None
        if value is True:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, int):
            return cls(value)
        raise ValueError(f"backtrack must be None, True, a budget or a Backtrack, got {value!r}")


class _Patience:
    """The patience backjump schedule. `counter` counts every decision undone,
    `depth` the decisions standing, `deepest` the greatest depth reached and
    `start` the counter at that moment. Level k expires `10 * 2**k` undone
    decisions after `start` and, once it has, asks to jump back to depth
    `deepest - 4 * 2**k`; the highest expired level wins and every expired level
    is re-armed for another `10 * 2**j` undone decisions. Reaching a new deepest
    depth forgets all levels."""

    def __init__(self):
        self.counter = self.depth = self.deepest = self.start = 0
        self.levels = []                          # [target depth, timeout]

    def chose(self):
        self.depth += 1
        if self.depth > self.deepest:
            self.deepest, self.levels, self.start = self.depth, [], self.counter

    def undone(self):
        self.counter += 1
        self.depth -= 1

    def backjump(self):
        """How many decisions to undo for the contradiction at hand (1 = the
        ordinary backtrack)."""
        i = 0
        while i < len(self.levels) and self.levels[i][1] <= self.counter:
            i += 1
        if i == len(self.levels):
            self.levels.append([self.deepest - 4 * 2 ** i, self.start + 10 * 2 ** i])
        if i == 0:
            return 1
        amount = self.depth - self.levels[i - 1][0]
        for j in range(i):
            self.levels[j][1] = self.counter + 10 * 2 ** j
        return max(1, amount)


def search(problem, options=None):
    """Run chronological backtracking with banning on `problem`.

    The problem is already consistent when this is called (its constraints
    propagated). It must offer:

    pick()                 the next cell to decide, or None when all are decided
    choose(cell)           a state to try there (it may draw from a random
                           stream), or None when nothing is left
    snapshot()             an opaque copy of everything decide and forbid change
    decide(cell, state)    fix the cell and propagate; False on a contradiction
    forbid(cell, state)    remove the state from the cell and propagate; False on
                           a contradiction
    restore(snapshot)      go back to a snapshot

    Each decision is recorded with the snapshot taken just before it. On a
    contradiction the latest decision is undone and its state forbidden at its
    cell (counted as one backtrack); if that contradicts as well the next
    decision down is undone, and so on. With the patience option a contradiction
    may instead undo several decisions at once (a backjump) without forbidding
    anything; a backjump that would reach past the first decision ends the
    search, so a search always ends, even with no budget.

    Returns {'status', 'backtracks', 'backjumps', 'decisions', 'max_depth'}.
    status is 'solved'; 'unsat' (no decision left to undo and no backjump has
    skipped a part of the tree: the problem has no solution); 'exhausted' (the
    same after backjumps, or a backjump past the first decision: nothing is
    proved); or 'budget' (`max_backtracks` reached).
    """
    options = Backtrack.coerce(options) or Backtrack()
    patience = _Patience() if options.patience else None
    stack = []
    backtracks = backjumps = decisions = deepest = 0

    def outcome(status):
        return {"status": status, "backtracks": backtracks, "backjumps": backjumps,
                "decisions": decisions, "max_depth": deepest}

    broken = False
    while True:
        if broken:
            if not stack:
                return outcome("unsat" if backjumps == 0 else "exhausted")
            if options.max_backtracks is not None and backtracks + backjumps >= options.max_backtracks:
                return outcome("budget")
            amount = 1
            if patience is not None:
                amount = patience.backjump()
                if amount > len(stack):
                    return outcome("exhausted")
            for _ in range(amount):
                saved, cell, state = stack.pop()
                problem.restore(saved)
                if patience is not None:
                    patience.undone()
            if amount == 1:
                backtracks += 1
                broken = not problem.forbid(cell, state)
            else:
                backjumps += 1
                broken = False
            continue
        cell = problem.pick()
        if cell is None:
            return outcome("solved")
        state = problem.choose(cell)
        if state is None:                   # nothing left to try here: undo the decision before
            broken = True
            continue
        stack.append((problem.snapshot(), cell, state))
        decisions += 1
        deepest = max(deepest, len(stack))
        if patience is not None:
            patience.chose()
        broken = not problem.decide(cell, state)


class AllDifferent:
    """At most one cell per group of states: a rule for `settle_quotas`.

    `group_of[state]` names the group of every state (all different states of
    one group count as the same thing, e.g. the orientations of one layout). A
    cell whose whole domain lies inside one group is decided to that group;
    two such cells in one group are a contradiction, and a decided group leaves
    the domain of every other cell. Unlike a `Quota` per group this is one pass
    over the cells.
    """

    def __init__(self, group_of):
        self.group_of = list(group_of)
        self.mask = {}
        for state, group in enumerate(self.group_of):
            self.mask[group] = self.mask.get(group, 0) | (1 << state)

    def apply(self, doms, changed, indices=None):
        owners, taken = {}, 0
        for idx, domain in enumerate(doms):
            if not domain:
                return False
            group = self.group_of[(domain & -domain).bit_length() - 1]
            if domain & ~self.mask[group] == 0:
                if group in owners:
                    return False
                owners[group] = idx
                taken |= self.mask[group]
        if not owners:
            return True
        own = {idx: self.mask[group] for group, idx in owners.items()}
        for idx, domain in enumerate(doms):
            new = domain & ~(taken & ~own.get(idx, 0))
            if new != domain:
                if not new:
                    return False
                doms[idx] = new
                changed.append(idx)
        return True


class GridProblem:
    """Bitmask domains on a `tilegen_csp.BitGrid` as a problem for `search`.

    grid    the BitGrid (adjacency and arc consistency)
    doms    list of domains, one int per cell; modified in place
    rng     anything with `random()` (a `random.Random` or numpy Generator) and
            `choice(list)`; `random.Random` is the one the socket solver uses
    rules   objects with `apply(doms, changed, indices)`: a `Quota`, an
            `AllDifferent`, or your own; they are applied after every
            propagation until nothing changes (`tilegen_csp.settle_quotas`)
    scopes  one cell-index list (or None for all cells) per rule

    `pick` takes an undecided cell of least domain size, ties broken by
    `rng.choice` over the cells in index order; `choose` draws a state of the
    domain uniformly.
    """

    def __init__(self, grid, doms, rng, rules=(), scopes=None):
        self.grid, self.doms, self.rng = grid, doms, rng
        self.rules = list(rules)
        self.scopes = list(scopes) if scopes is not None else [None] * len(self.rules)

    def settle(self, cells):
        """Propagate from the changed cells, then the rules. False on a contradiction."""
        if not self.grid.propagate(self.doms, deque(cells)):
            return False
        return not self.rules or settle_quotas(self.grid, self.rules, self.scopes, self.doms)

    def pick(self):
        sizes = [(domain.bit_count(), idx) for idx, domain in enumerate(self.doms) if domain & (domain - 1)]
        if not sizes:
            return None
        least = min(size for size, _ in sizes)
        return self.rng.choice([idx for size, idx in sizes if size == least])

    def choose(self, cell):
        states = list(iter_bits(self.doms[cell]))
        return states[int(self.rng.random() * len(states))] if states else None

    def snapshot(self):
        return list(self.doms)

    def restore(self, snapshot):
        self.doms[:] = snapshot

    def decide(self, cell, state):
        self.doms[cell] = 1 << state
        return self.settle([cell])

    def forbid(self, cell, state):
        narrowed = self.doms[cell] & ~(1 << state)
        if not narrowed:
            return False
        self.doms[cell] = narrowed
        return self.settle([cell])
