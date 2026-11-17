"""Equal-cost routes that keep their heading, for the legacy navigation profiles.

`tilegen_lowend_runtime.farthest_pair` walks the Dijkstra predecessor pointers
back from the goal. Among the many routes of the same cost that picks whichever
a pointer happened to record, which zigzags across open floor. This module
computes the field of cost-to-goal instead and walks from the start, always
stepping onto a neighbour that is still on an optimal route and, among those,
holding the previous heading while it can. The cost is identical to the
legacy route (same tile and overlay costs, same teleports); only the geometry
changes, with far fewer turns. The legacy runtime is not modified and its
routes and exported outputs are unchanged unless a caller opts in here.

Idea-level port from MarkovJunior (MIT licence, github.com/mxgmn/MarkovJunior):
the Path node's distance field from the goal set, descent walk and 'inertia'
tie-break (source/Path.cs); see also 'The Incredible Power of Dijkstra Maps'
(RogueBasin). Rewritten for this runtime's weighted four-neighbour grids; no
source text was carried over. When the walk is forced to turn it takes the
neighbour with the longest straight stretch of optimal route ahead (a
lookahead the original does not have, which lowers the turn count further).
"""
from __future__ import annotations

import heapq

import tilegen_lowend_runtime as rt

INF = 10 ** 12


def cost_to_goal(tilemap, profile_name, goal):
    """Cheapest cost from every cell to `goal` under a navigation profile.

    Returns (dist, passable) where dist[y][x] is INF for cells that cannot
    reach the goal. Entering a cell costs `rt.tile_cost` of that cell; a jump
    between two teleport nodes costs 1, exactly as `rt.shortest_paths` has it.
    """
    passable = rt.build_passable_grid(tilemap, profile_name)
    gx, gy = goal
    w, h = tilemap.w, tilemap.h
    dist = [[INF] * w for _ in range(h)]
    if not (0 <= gx < w and 0 <= gy < h and passable[gy][gx]):
        return dist, passable
    enter = [[rt.tile_cost(profile_name, tilemap.base[y][x], tilemap.overlay[y][x]) if passable[y][x] else INF
              for x in range(w)] for y in range(h)]
    teleports = rt.find_teleport_nodes(tilemap, passable)
    teleport_set = set(teleports)
    dist[gy][gx] = 0
    heap = [(0, gx, gy)]
    while heap:
        d, x, y = heapq.heappop(heap)
        if d != dist[y][x]:
            continue
        step = enter[y][x]                         # cost of stepping onto (x, y)
        for dx, dy in rt.TURN_DIRS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and passable[ny][nx] and d + step < dist[ny][nx]:
                dist[ny][nx] = d + step
                heapq.heappush(heap, (d + step, nx, ny))
        if (x, y) in teleport_set:
            for nx, ny in teleports:
                if (nx, ny) != (x, y) and d + 1 < dist[ny][nx]:
                    dist[ny][nx] = d + 1
                    heapq.heappush(heap, (d + 1, nx, ny))
    return dist, passable


def inertia_path(tilemap, profile_name, start, goal):
    """A cheapest route from `start` to `goal` as a list of [x, y], holding its
    heading while the straight step is still on an optimal route.

    Same cost as the legacy `rt.reconstruct_path` route, empty when the goal is
    unreachable, [start] when start == goal. Deterministic: when the heading is
    blocked the neighbour with the longest optimal straight run wins, then the
    order N, E, S, W.
    """
    start, goal = tuple(start), tuple(goal)
    dist, passable = cost_to_goal(tilemap, profile_name, goal)
    sx, sy = start
    w, h = tilemap.w, tilemap.h
    if not (0 <= sx < w and 0 <= sy < h) or dist[sy][sx] >= INF:
        return []
    enter = lambda x, y: rt.tile_cost(profile_name, tilemap.base[y][x], tilemap.overlay[y][x])
    teleports = rt.find_teleport_nodes(tilemap, passable)
    teleport_set = set(teleports)

    def descends(x, y, nx, ny):
        return (0 <= nx < w and 0 <= ny < h and passable[ny][nx]
                and dist[ny][nx] + enter(nx, ny) == dist[y][x])

    def run(x, y, k):
        """Steps that can be taken in direction k from (x, y) while staying optimal."""
        dx, dy = rt.TURN_DIRS[k]
        count = 0
        while descends(x, y, x + dx, y + dy):
            x, y, count = x + dx, y + dy, count + 1
        return count

    path = [[sx, sy]]
    x, y = sx, sy
    heading = None
    guard = w * h * 4 + len(teleports) + 8
    while (x, y) != goal and guard:
        guard -= 1
        options = [k for k in range(4) if descends(x, y, x + rt.TURN_DIRS[k][0], y + rt.TURN_DIRS[k][1])]
        if options:
            if heading in options:
                k = heading
            else:
                k = max(options, key=lambda j: (run(x, y, j), -j))
            heading = k
            x, y = x + rt.TURN_DIRS[k][0], y + rt.TURN_DIRS[k][1]
        elif (x, y) in teleport_set:
            jump = [(tx, ty) for tx, ty in teleports if (tx, ty) != (x, y) and dist[ty][tx] + 1 == dist[y][x]]
            if not jump:
                return []
            x, y = min(jump)
            heading = None
        else:
            return []
        path.append([x, y])
    return path if (x, y) == goal else []


def path_cost(tilemap, profile_name, path):
    """Total cost of a route in the runtime's terms: each walked step costs
    `tile_cost` of the cell entered, a teleport jump costs 1."""
    passable = rt.build_passable_grid(tilemap, profile_name)
    portals = set(rt.find_teleport_nodes(tilemap, passable))
    total = 0
    for (ax, ay), (bx, by) in zip(path, path[1:]):
        if abs(ax - bx) + abs(ay - by) == 1:
            total += rt.tile_cost(profile_name, tilemap.base[by][bx], tilemap.overlay[by][bx])
        elif (ax, ay) in portals and (bx, by) in portals:
            total += 1
        else:
            raise ValueError(f"({ax}, {ay}) -> ({bx}, {by}) is neither a step nor a teleport jump")
    return total


def count_turns(path):
    """Direction changes along a route (teleport jumps break the heading and are not counted)."""
    turns, previous = 0, None
    for (ax, ay), (bx, by) in zip(path, path[1:]):
        step = (bx - ax, by - ay)
        if abs(step[0]) + abs(step[1]) != 1:
            previous = None
            continue
        if previous is not None and step != previous:
            turns += 1
        previous = step
    return turns


def farthest_pair_inertia(tilemap, profile_name):
    """`rt.farthest_pair` with the route replaced by its inertia walk.

    Same start, goal, `path_cost` and reachability numbers; `path` is the new
    geometry and `path_steps` its length (equal-cost routes can differ in step
    count where tiles cost more than 1). None when the map has no walkable tile.
    """
    pair = rt.farthest_pair(tilemap, profile_name)
    if pair is None or not pair["success"]:
        return pair
    path = inertia_path(tilemap, profile_name, pair["start"], pair["goal"])
    return dict(pair, path=path, path_steps=max(0, len(path) - 1), success=bool(path))
