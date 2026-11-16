"""Opt-in bot playtest of a compiled World: walk it, score it, lint it.

A deterministic bot plays start -> goal on the 4-connected walkable mask,
fetching keys for locked cells (`artifacts['gates']` from tilegen_keylock) and
seeing a disc around each step. The design lint in tilegen_compiler only inspects the finished world;
this one plays it. Importing registers nothing; call `playtest(world)`,
`lint_playtest(world)` (adds findings) or `render_playtest(world, result, png)`.
Pure and deterministic: no RNG, so equal worlds give equal results.
"""
from __future__ import annotations

from collections import deque

import numpy as np

import tilegen_blueprint as tb

VISION = 5            # cells the bot sees around each step
_N4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _path(ok, blocked, src, dst):
    """Shortest 4-connected path src -> dst (inclusive) avoiding `blocked`, or None."""
    h, w = ok.shape
    prev = {src: None}
    q = deque([src])
    while q:
        c = q.popleft()
        if c == dst:
            out = []
            while c is not None:
                out.append(c)
                c = prev[c]
            return out[::-1]
        for dx, dy in _N4:
            n = (c[0] + dx, c[1] + dy)
            if 0 <= n[0] < w and 0 <= n[1] < h and ok[n[1]][n[0]] and n not in prev and n not in blocked:
                prev[n] = c
                q.append(n)
    return None


def _dead_ends(ok):
    """Walkable cells with exactly one walkable 4-neighbour, as a fraction of walkable."""
    p = np.pad(ok, 1)
    n = p[1:-1, 2:].astype(int) + p[1:-1, :-2] + p[2:, 1:-1] + p[:-2, 1:-1]
    return float(((n == 1) & ok).sum()) / max(1, int(ok.sum()))


def playtest(world, vision=VISION, max_steps=100000):
    """Play the world once. Returns a dict: solved, steps, shortest (ignoring
    locks), detour, keys, seen (fraction of walkable cells the bot saw),
    dead_end, fun (0-100), reason, trail ([(x, y)...])."""
    ok = world.passable()
    gates = world.artifacts.get('gates') or {}
    start = goal = None
    if gates.get('start') and gates.get('goal'):
        start, goal = tuple(gates['start']['cell']), tuple(gates['goal']['cell'])
    if start is None or not (ok[start[1]][start[0]] and ok[goal[1]][goal[0]]):
        start, goal = tb._bases(ok)
    res = {'solved': False, 'steps': 0, 'shortest': 0, 'detour': 0.0, 'keys': 0, 'locks': 0,
           'seen': 0.0, 'dead_end': _dead_ends(ok), 'fun': 0.0, 'reason': '', 'trail': []}
    if start is None or start == goal:
        res['reason'] = 'nothing to play'
        return res
    locks = {}      # lock cell -> key cell
    for g in gates.get('locks', []):
        if g.get('lock') and g.get('key'):
            locks[tuple(g['lock']['cell'])] = tuple(g['key']['cell'])
    res['locks'] = len(locks)
    free = _path(ok, set(), start, goal)
    if free is None:
        res['reason'] = 'goal unreachable from start'
        return res
    res['shortest'] = len(free) - 1
    pos, held, trail = start, set(), [start]
    while len(trail) < max_steps:
        closed = {c for c, k in locks.items() if k not in held}
        to_goal = _path(ok, closed, pos, goal)
        # nearest reachable key not yet held; the goal wins once nothing is left to fetch
        legs = [(len(p), k, p) for k in sorted(set(locks.values()) - held)
                if (p := _path(ok, closed, pos, k)) is not None]
        if legs:
            _, k, leg = min(legs, key=lambda t: (t[0], t[1]))
            held.add(k)
        elif to_goal is not None:
            leg = to_goal
        else:
            res['reason'] = 'stuck behind locks'
            break
        trail.extend(leg[1:])
        pos = leg[-1]
        if pos == goal:
            res['solved'] = True
            break
    res['trail'] = trail
    res['steps'] = len(trail) - 1
    res['keys'] = len(held)
    h, w = ok.shape
    ys, xs = np.mgrid[0:h, 0:w]
    seen = np.zeros_like(ok)
    for x, y in dict.fromkeys(trail):
        seen |= ((xs - x) ** 2 + (ys - y) ** 2) <= vision * vision
    res['seen'] = float((seen & ok).sum()) / max(1, int(ok.sum()))
    res['detour'] = res['steps'] / max(1, res['shortest'])
    res['fun'] = _fun(res)
    return res


def _fun(r):
    """0-100 heuristic: the bot should finish, see much of the map, take a
    non-trivial route (detour 1.3-3 is best) and meet some keys."""
    if not r['solved']:
        return 0.0
    route = max(0.0, 1.0 - abs(min(r['detour'], 6.0) - 2.0) / 4.0)
    keys = min(1.0, r['keys'] / 2.0)
    open_ = 1.0 - min(1.0, r['dead_end'] * 5.0)      # a few dead ends are fine, a maze of them is not
    return round(100 * (0.35 * r['seen'] + 0.30 * route + 0.15 * keys + 0.20 * open_), 1)


def lint_playtest(world, min_seen=0.25, **kw):
    """Run the bot and add findings to the world (opt-in; compile_recipe does not call it)."""
    r = playtest(world, **kw)
    world.artifacts['playtest'] = {k: v for k, v in r.items() if k != 'trail'}
    if not r['solved']:
        world.finding('error', 'playtest.unsolved', f"bot could not finish: {r['reason']}")
    elif r['seen'] < min_seen:
        world.finding('warn', 'playtest.shallow', f"bot saw only {r['seen']:.0%} of the walkable map")
    elif r['detour'] > 6:
        world.finding('warn', 'playtest.detour', f"route is {r['detour']:.1f}x the shortest path")
    return r


def render_playtest(world, result, path, px=8):
    """PNG of the map with the bot's trail (None result -> plain map), start green, goal red."""
    from PIL import Image, ImageDraw
    ok = world.passable()
    h, w = ok.shape
    img = Image.new('RGB', (w * px, h * px))
    d = ImageDraw.Draw(img)
    for y in range(h):
        for x in range(w):
            d.rectangle([x * px, y * px, x * px + px - 1, y * px + px - 1],
                        fill=(196, 188, 170) if ok[y][x] else (38, 40, 52))
    if result and result['trail']:
        t = result['trail']
        for i, (x, y) in enumerate(t):
            f = i / max(1, len(t) - 1)
            d.rectangle([x * px + 2, y * px + 2, x * px + px - 3, y * px + px - 3],
                        fill=(int(40 + 200 * f), int(120 * (1 - f)), int(220 * (1 - f)) + 30))
        for c, col in ((t[0], (30, 200, 60)), (t[-1], (230, 40, 40))):
            d.ellipse([c[0] * px - 2, c[1] * px - 2, c[0] * px + px + 1, c[1] * px + px + 1], fill=col)
    img.save(path)
    return path
