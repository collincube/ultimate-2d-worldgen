"""Lock-and-key gating: put locks on chokepoints of the route, each key on a
branch reachable before its lock, and prove the result is solvable.

The pass works on any undirected graph (`{node: [neighbours]}`), with builders
for the walkable cells of a raster (`grid_graph`) and for a compiled world's
places (`place_graph`):

1. start and goal are the two ends of a double-sweep distance field, so they
   sit far apart (`far_apart`);
2. a deliberately non-optimal route between them is drawn by a random descent
   of the goal's distance field at a temperature (`temperature_path`);
3. locks go on separators of start and goal (cut vertices whose removal
   disconnects them, found with Tarjan's low-link DFS in `separators`), spread
   along the route, so a lock really blocks the way;
4. lock i's key goes on the farthest side branch of the part of the map that
   is reachable from the start without crossing lock i (`plan_gates`), so every
   key is reachable before its lock and, by induction, after the earlier keys;
5. `check_solvable` replays the game: the reachable set grows as keys are
   collected until the goal is reached or nothing more opens. It is independent
   of the construction, so it also validates hand-made or edited layouts.

Idea-level port from MarkovJunior (MIT licence, github.com/mxgmn/MarkovJunior,
models/BasicKeys.xml and models/Keys.xml: far start and goal from a field, a
temperature-biased non-optimal path, locks and keys on it), whose rules adapt
the graph grammar of Joris Dormans, 'Engineering Emergence: Applied Theory for
Game Design' (PhD, University of Amsterdam, 2012); see also David Adams,
'Automatic Generation of Dungeons for Computer Games' (BSc, Sheffield, 2002).
The cut-vertex search is R. E. Tarjan's depth-first algorithm, 'Depth-first
search and linear graph algorithms', SIAM J. Computing 1(2), 1972. Rewritten
here with explicit graphs; no source text was carried over.

Opt-in: importing this module registers nothing. `register()` adds the world
feature 'nar.gates' (number 61, outside the 40 depth features, so the counts
the suites assert are untouched until a caller asks for it); `unregister()`
removes it. Randomness is named streams (`tf.rng_for(seed, 'keylock.path')`
and 'keylock.keys').
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

import tilegen_fields as tf
import tilegen_grammars as tg
import tilegen_world as tw


# ------------------------------------------------------------------- graphs

def grid_graph(passable):
    """4-connected graph of a boolean mask: nodes are (x, y) cells, neighbours
    in the order north, east, south, west."""
    passable = np.asarray(passable, dtype=bool)
    h, w = passable.shape
    adj = {}
    for y, x in zip(*np.nonzero(passable)):
        x, y = int(x), int(y)
        adj[(x, y)] = [(nx, ny) for nx, ny in ((x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y))
                       if 0 <= nx < w and 0 <= ny < h and passable[ny, nx]]
    return adj


def place_graph(world):
    """Graph of a compiled world's places: node ids joined by the physical
    adjacency topology, neighbours sorted."""
    top = world.topology("physical.adjacency", create=False)
    adj = {pid: [] for pid in world.place_order}
    if top is not None:
        for a, nbrs in top.edges.items():
            adj.setdefault(a, [])
            adj[a] = sorted(set(adj[a]) | set(nbrs))
    return adj


def bfs_distances(adj, sources, blocked=()):
    """Hop distance from the nearest of `sources`, skipping `blocked` nodes."""
    blocked = set(blocked)
    dist = {}
    queue = deque()
    for s in sources:
        if s not in blocked and s not in dist:
            dist[s] = 0
            queue.append(s)
    while queue:
        v = queue.popleft()
        for u in adj[v]:
            if u not in dist and u not in blocked:
                dist[u] = dist[v] + 1
                queue.append(u)
    return dist


def _farthest(dist):
    top = max(dist.values())
    return min(v for v, d in dist.items() if d == top)


def far_apart(adj, anchor=None):
    """Two nodes far apart in the component of `anchor` (default: the largest
    component), by a double sweep: the node farthest from the anchor, then the
    node farthest from that one. Exact on trees; ties go to the smallest node."""
    if not adj:
        raise ValueError("the graph has no nodes")
    if anchor is None:
        seen, best = set(), None
        for v in sorted(adj):
            if v in seen:
                continue
            comp = bfs_distances(adj, [v])
            seen.update(comp)
            if best is None or len(comp) > len(best[1]):
                best = (v, comp)
        anchor = best[0]
    a = _farthest(bfs_distances(adj, [anchor]))
    b = _farthest(bfs_distances(adj, [a]))
    return a, b


def _lowlink(adj, root):
    """Iterative depth-first search from `root`: entry and exit times, low-link
    values and the DFS tree parent of every reached node."""
    tin, tout, low, parent = {root: 0}, {}, {root: 0}, {root: None}
    clock = 1
    stack = [(root, iter(adj[root]))]
    while stack:
        v, it = stack[-1]
        advanced = False
        for u in it:
            if u not in tin:
                tin[u] = low[u] = clock
                clock += 1
                parent[u] = v
                stack.append((u, iter(adj[u])))
                advanced = True
                break
            if u != parent[v]:
                low[v] = min(low[v], tin[u])
        if advanced:
            continue
        stack.pop()
        tout[v] = clock - 1
        if stack:
            p = stack[-1][0]
            low[p] = min(low[p], low[v])
    return tin, tout, low, parent


def articulation_points(adj, root=None):
    """Cut vertices of the component of `root` (default: the smallest node)."""
    if not adj:
        return set()
    root = min(adj) if root is None else root
    tin, tout, low, parent = _lowlink(adj, root)
    children = {}
    for v, p in parent.items():
        if p is not None:
            children.setdefault(p, []).append(v)
    cuts = set()
    for v in tin:
        if v == root:
            if len(children.get(v, ())) > 1:
                cuts.add(v)
        elif any(low[c] >= tin[v] for c in children.get(v, ())):
            cuts.add(v)
    return cuts


def separators(adj, s, t):
    """Nodes other than s and t whose removal disconnects s from t, ordered by
    distance from s. One Tarjan pass: a node v separates them when t lies in
    the DFS subtree of a child c of v with low[c] >= tin[v]."""
    tin, tout, low, parent = _lowlink(adj, s)
    if t not in tin:
        return []
    found = []
    for c, v in parent.items():
        if v is None or v == s:
            continue
        if low[c] >= tin[v] and tin[c] <= tin[t] <= tout[c] and v != t:
            found.append(v)
    dist = bfs_distances(adj, [s])
    return sorted(set(found), key=lambda v: (dist[v], v))


def shortest_path(adj, s, t, blocked=()):
    """A fewest-hops path from s to t (smallest nodes first on ties), or []."""
    blocked = set(blocked)
    if s in blocked or t in blocked:
        return []
    parent = {s: None}
    queue = deque([s])
    while queue:
        v = queue.popleft()
        if v == t:
            break
        for u in adj[v]:
            if u not in parent and u not in blocked:
                parent[u] = v
                queue.append(u)
    if t not in parent:
        return []
    path = [t]
    while path[-1] != s:
        path.append(parent[path[-1]])
    return path[::-1]


def temperature_path(adj, s, t, rng, temperature=2.0):
    """A simple path from s to t that is deliberately not the shortest.

    A random descent of t's distance field: at each node the unvisited
    neighbours are weighted exp(-(d[n] - d[v]) / temperature), so the walk
    mostly closes in on the goal but sometimes steps away or sideways. At
    temperature <= 0 it is a shortest path. If the walk traps itself or runs
    long it finishes along a shortest path that avoids what it has visited
    (or, failing that, a plain shortest path).
    """
    dist = bfs_distances(adj, [t])
    if s not in dist:
        return []
    if temperature <= 0 or s == t:
        return shortest_path(adj, s, t)
    path, visited, v = [s], {s}, s
    budget = 6 * dist[s] + 24
    while v != t and budget > 0:
        budget -= 1
        options = [u for u in adj[v] if u not in visited and u in dist]
        if not options:
            break
        weights = np.array([math.exp(-(dist[u] - dist[v]) / temperature) for u in options])
        u = options[int(rng.choice(len(options), p=weights / weights.sum()))]
        path.append(u)
        visited.add(u)
        v = u
    if v == t:
        return path
    tail = shortest_path(adj, v, t, blocked=visited - {v})
    if tail:
        return path[:-1] + tail
    return shortest_path(adj, s, t)


# ------------------------------------------------------------- solvability

def check_solvable(adj, start, goal, locks, keys):
    """Replay the game and say whether the goal can be reached.

    `locks` maps a node to the id of the key that opens it and `keys` maps a
    key id to the node that holds it (a list or set of nodes if several copies exist; a tuple is one node). The
    reachable set from the start grows as keys are collected: a locked node is
    entered only once its key is held. Returns {'solvable', 'collected' (key
    ids in the order they were picked up), 'reach' (reachable node counts after
    each round), 'stuck' (locks never opened)}.
    """
    holders = {}
    for kid, where in keys.items():
        for node in (where if isinstance(where, (list, set, frozenset)) else [where]):
            holders.setdefault(node, []).append(kid)
    have, collected, reach_sizes = set(), [], []
    reached = set()
    while True:
        closed = {n for n, kid in locks.items() if kid not in have}
        reached = set(bfs_distances(adj, [start], blocked=closed)) if start not in closed else set()
        reach_sizes.append(len(reached))
        new = [kid for n in sorted(reached) for kid in holders.get(n, ()) if kid not in have]
        if not new:
            break
        for kid in new:
            if kid not in have:
                have.add(kid)
                collected.append(kid)
    stuck = sorted(n for n, kid in locks.items() if kid not in have)
    return {"solvable": goal in reached, "collected": collected, "reach": reach_sizes, "stuck": stuck}


# --------------------------------------------------------------- planning

def plan_gates(adj, locks=2, seed=0, temperature=2.0, start=None, goal=None):
    """Lock-and-key plan on a graph.

    Returns a dict: start, goal, path (the non-optimal route), chokepoints (all
    separators in order from the start), locks (list of {'id', 'node', 'key'}
    where 'key' is the node holding that lock's key, in route order),
    solvable, check (the `check_solvable` report) and reason (empty, or why
    fewer locks than asked for were placed).

    Locks sit on corridor-like chokepoints (two neighbours when there are any)
    spread along the route: lock i is the chokepoint nearest i / (locks + 1) of
    the way from start to goal, and no two locks are closer than
    route / (2 * (locks + 1)) hops (at least 4); if the chokepoints are too
    clustered for that, fewer locks are placed and the reason says so. Lock i's
    key goes on the side branch farthest from the route in the stretch between
    lock i - 1 and lock i (or anywhere before lock i if that stretch has no
    room), so the keys come in order where the map allows and each is reachable
    before its lock. Deterministic for a seed.
    """
    empty = {"start": None, "goal": None, "path": [], "chokepoints": [], "locks": [], "solvable": True,
             "check": None, "reason": "no nodes"}
    if not adj:
        return empty
    if start is None and goal is None:
        start, goal = far_apart(adj)
    elif goal is None:
        goal = _farthest(bfs_distances(adj, [start]))
    elif start is None:
        start = _farthest(bfs_distances(adj, [goal]))
    plan = dict(empty, start=start, goal=goal, reason="")
    if start == goal:
        plan.update(path=[start], reason="start and goal coincide")
        return plan
    path = temperature_path(adj, start, goal, tf.rng_for(seed, "keylock.path"), temperature)
    plan["path"] = path
    if not path:
        plan.update(solvable=False, reason="goal is not reachable from the start")
        return plan
    chokepoints = separators(adj, start, goal)
    plan["chokepoints"] = chokepoints
    wanted = max(0, int(locks))
    pool = [v for v in chokepoints if len(adj[v]) == 2] or list(chokepoints)
    dist_start = bfs_distances(adj, [start])
    route = dist_start[goal]
    min_gap = max(4, route // (2 * (wanted + 1)))      # hops between two locks along the route
    picks = []
    for i in range(wanted):
        target = route * (i + 1) / (wanted + 1)
        free = [(abs(dist_start[v] - target), v) for v in pool
                if all(abs(dist_start[v] - dist_start[q]) >= min_gap for q in picks)]
        if not free:
            break
        picks.append(min(free)[1])
    picks.sort(key=lambda v: (dist_start[v], v))
    if len(picks) < wanted:
        plan["reason"] = (f"only {len(picks)} chokepoint(s) at least {min_gap} hops apart between start "
                          f"and goal, wanted {wanted}")

    rng = tf.rng_for(seed, "keylock.keys")
    on_path = set(path)
    taken = set(picks) | {start, goal}
    regions = [set(bfs_distances(adj, [start], blocked=[lock])) for lock in picks]
    gates = []
    for i, lock in enumerate(picks):
        region = regions[i]
        stretch = region - regions[i - 1] - {picks[i - 1]} if i else region
        near_path = sorted(on_path & region)
        side = bfs_distances(adj, near_path, blocked=[lock]) if near_path else {}
        for scope in (stretch, region):
            candidates = [v for v in sorted(scope) if v not in taken]
            off_path = [v for v in candidates if v not in on_path]
            pool_k = off_path or candidates
            if pool_k:
                break
        else:
            note = f"no room for the key of lock {i + 1}"
            plan["reason"] = plan["reason"] + "; " + note if plan["reason"] else note
            continue
        depth = max(side.get(v, 0) for v in pool_k)
        best = [v for v in pool_k if side.get(v, 0) == depth]
        key_node = best[int(rng.integers(len(best)))]
        taken.add(key_node)
        gates.append({"id": len(gates) + 1, "node": lock, "key": key_node})
    plan["locks"] = gates
    report = check_solvable(adj, start, goal, {g["node"]: g["id"] for g in gates},
                            {g["id"]: g["key"] for g in gates})
    plan["check"] = report
    plan["solvable"] = report["solvable"]
    return plan


def plan_on_mask(passable, locks=2, seed=0, temperature=2.0):
    """`plan_gates` on the 4-connected cells of a boolean mask; nodes are (x, y)."""
    return plan_gates(grid_graph(passable), locks, seed, temperature)


def apply_plan(tm, plan, lock_overlay="locked", key_overlay="key"):
    """Mark a cell plan on a TileMap: lock cells get the existing 'locked'
    overlay and key cells a 'key' overlay. Floors and walls are not touched."""
    for gate in plan["locks"]:
        x, y = gate["node"]
        tm.overlay[y][x] = lock_overlay
        kx, ky = gate["key"]
        tm.overlay[ky][kx] = key_overlay
    return tm


# --------------------------------------------------------- the world feature

FEATURE_ID = "nar.gates"


def key_lock_gating(world):
    """World feature: plan locks and keys over the walkable cells and publish
    `artifacts['gates']`. Degrades to an empty, trivially solvable record when
    the map has no walkable route or no chokepoints. Knobs from the spec:
    'locks' (default 2) and 'lock_temperature' (default 2.0)."""
    passable = world.passable()
    plan = plan_on_mask(passable, int(world.spec.get("locks", 2)),
                        tf.seed_stream(world.seed, "keylock"), float(world.spec.get("lock_temperature", 2.0)))

    def where(cell):
        if cell is None:
            return None
        place = world.place_at(int(cell[0]), int(cell[1]))
        return {"cell": [int(cell[0]), int(cell[1])], "place": place.id if place else None}

    record = {
        "start": where(plan["start"]), "goal": where(plan["goal"]),
        "route_length": max(0, len(plan["path"]) - 1),
        "chokepoints": len(plan["chokepoints"]),
        "locks": [{"id": g["id"], "lock": where(g["node"]), "key": where(g["key"])} for g in plan["locks"]],
        "solvable": bool(plan["solvable"]),
        "unlock_order": list((plan["check"] or {}).get("collected", [])),
        "reason": plan["reason"],
    }
    world.artifacts["gates"] = record
    if not record["solvable"]:
        world.finding("error", "mission.unsolvable", "locks and keys leave the goal unreachable")
    return record


def register():
    """Add the 'nar.gates' feature (number 61) to the world registry. Idempotent."""
    if FEATURE_ID not in tw.FEATURES:
        tw.FEATURES[FEATURE_ID] = tw.Feature(
            FEATURE_ID, 61, "Lock and Key Gating", "Incidents & Narrative",
            ("places",), ("artifacts.gates",), key_lock_gating, "III")
    return FEATURE_ID


def unregister():
    """Remove the feature again, restoring the 40-feature registry."""
    tw.FEATURES.pop(FEATURE_ID, None)
