"""Opt-in export of a compiled World as an engine-agnostic level blueprint.

The blueprint is a small JSON level contract: grid dimensions, `tiles` rows
over digits 0-5 (0 void, 1 floor, 2 ravine/water, 3 bridge, 4 ally, 5 enemy),
bases, bridges, spawners, badGuys, quests, zones and locks. Importing registers
nothing and touches no generator or exporter; call `to_blueprint(world)` /
`write_blueprint`. Everything is derived from the world and named seed
streams, so equal worlds give equal bytes. `check_blueprint` applies the
reader's structural rules.
"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np

import tilegen_fields as tf

GENERATOR = "tilegen"
TILE_SIZE = 1
_CAPS = {"cells": 1_000_000, "entities": 5000, "quests": 200}
_RAVINE = ("water",)
_BRIDGE = ("bridge",)


def tile_rows(world):
    """Rows of digit strings: 0 void/wall, 1 floor, 2 water, 3 bridge."""
    ok = world.passable()
    rows = []
    for y in range(world.h):
        r = []
        for x in range(world.w):
            b = world.tilemap.base[y][x]
            r.append("3" if b in _BRIDGE else "2" if b in _RAVINE else "1" if ok[y][x] else "0")
        rows.append("".join(r))
    return rows


def _bfs(ok, src):
    h, w = ok.shape
    dist = {src: 0}
    q = deque([src])
    while q:
        x, y = q.popleft()
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < w and 0 <= ny < h and ok[ny][nx] and (nx, ny) not in dist:
                dist[(nx, ny)] = dist[(x, y)] + 1
                q.append((nx, ny))
    return dist


def _bases(ok):
    """(player, enemy) cells: first cell (row-major) of the largest 4-connected
    component, and the cell farthest from it. None, None when nothing walks."""
    seen, best, first = set(), {}, None
    for y, x in zip(*np.nonzero(ok)):
        if (int(x), int(y)) in seen:
            continue
        comp = _bfs(ok, (int(x), int(y)))
        seen.update(comp)
        if len(comp) > len(best):
            best = comp
            first = (int(x), int(y))
    if not best:
        return None, None
    far = max(sorted(best, key=lambda c: (c[1], c[0])), key=lambda c: best[c])
    return first, far


def to_blueprint(world, max_guards=24):
    ok = world.passable()
    rows = tile_rows(world)
    player, enemy = _bases(ok)
    cell = lambda c: {"gx": int(c[0]), "gz": int(c[1])}
    bp = {
        "generator": GENERATOR,
        "seed": int(world.seed),
        "biome": str(world.spec.get("template", "recipe")),
        "gridDimensions": {"width": int(world.w), "height": int(world.h), "tileSize": TILE_SIZE},
        "lodConfiguration": {"distanceMultiplier": 1, "totalLODObjects": 0},
        "playerBase": cell(player or (0, 0)),
        "enemyBases": [dict(id="eb0", name="Far base", **cell(enemy))] if enemy else [],
        "bridges": [], "spawners": [], "badGuys": [], "quests": [],
        "tiles": rows, "fords": [], "zones": [], "locks": [],
    }
    bseen = set()
    for y in range(world.h):
        for x in range(world.w):
            if rows[y][x] != "3" or (x, y) in bseen:
                continue
            run, q = [], deque([(x, y)])
            bseen.add((x, y))
            while q:
                cx, cy = q.popleft()
                run.append(cell((cx, cy)))
                for n in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    if 0 <= n[0] < world.w and 0 <= n[1] < world.h and rows[n[1]][n[0]] == "3" and n not in bseen:
                        bseen.add(n)
                        q.append(n)
            i = len(bp["bridges"])
            bp["bridges"].append({"id": f"br{i}", "name": f"Bridge {i}", **run[len(run) // 2],
                                  "tiles": run, "crossed": False})
    rng = np.random.default_rng(tf.seed_stream(world.seed, "blueprint"))
    guards = sorted(e["id"] for e in world.entities.values() if e.get("role") == "guard")
    for eid in guards[:max_guards]:
        pl = world.places.get(world.entities[eid].get("work"))
        if pl is None:
            continue
        ys, xs = np.nonzero(pl.cells & ok)
        if not len(xs):
            continue
        k = int(rng.integers(len(xs)))
        bp["badGuys"].append({"id": eid, "name": world.entities[eid].get("name", eid), "isBoss": False,
                              **cell((xs[k], ys[k])), "hp": 10, "alive": True})
    bp["quests"] = [{"id": "q-guards", "name": "Clear the guards", "target": max(1, len(bp["badGuys"])),
                     "progress": 0, "completed": False}]
    for pid in world.place_order[:200]:
        p = world.places[pid]
        bp["zones"].append({"id": pid, "kind": p.kind, "gx": int(round(p.centroid[0])),
                            "gz": int(round(p.centroid[1])), "area": p.area})
    gates = world.artifacts.get("gates")
    if gates:
        for g in gates.get("locks", []):
            if g.get("lock") and g.get("key"):
                (lx, ly), (kx, ky) = g["lock"]["cell"], g["key"]["cell"]
                bp["locks"].append({"id": str(g["id"]), "key": cell((kx, ky)), "cells": [cell((lx, ly))]})
    return bp


def check_blueprint(b):
    """Structural errors, mirroring checkBlueprint in the level generator."""
    e = []
    gd = b.get("gridDimensions") or {}
    W, H = gd.get("width"), gd.get("height")
    if not all(isinstance(v, int) and v >= 1 for v in (W, H)) or W * H > _CAPS["cells"]:
        return ["bad gridDimensions"]
    t = b.get("tiles")
    if not isinstance(t, list) or len(t) != H or any(
            not isinstance(r, str) or len(r) != W or set(r) - set("012345") for r in t):
        return ["tiles rows malformed"]
    inb = lambda o: isinstance(o, dict) and all(isinstance(o.get(k), int) for k in ("gx", "gz")) \
        and 0 <= o["gx"] < W and 0 <= o["gz"] < H
    ids = set()
    for name in ("enemyBases", "bridges", "badGuys", "spawners"):
        for o in b.get(name) or []:
            if not isinstance(o.get("id"), str):
                e.append(f"{name} entry without id")
            elif o["id"] in ids and name != "spawners":
                e.append(f"duplicate id {o['id']}")
            ids.add(o.get("id"))
            if not inb(o):
                e.append(f"{name} {o.get('id')} out of bounds")
            elif name == "badGuys" and t[o["gz"]][o["gx"]] == "0":
                e.append(f"badGuy {o['id']} on void")
            elif name == "bridges" and not all(inb(c) for c in o.get("tiles", [])):
                e.append(f"bridge {o['id']} tiles")
    if not inb(b.get("playerBase")):
        e.append("playerBase")
    q = set()
    for x in b.get("quests") or []:
        if not isinstance(x.get("id"), str) or x["id"] in q or not isinstance(x.get("target"), int) or x["target"] < 1:
            e.append("quest")
        q.add(x.get("id"))
    for L in b.get("locks") or []:
        if L.get("key") and not inb(L["key"]):
            e.append(f"lock {L.get('id')} key oob")
        e += [f"lock {L.get('id')} door oob" for c in L.get("cells", []) if not inb(c)]
    return e


def write_blueprint(world, path, **kw):
    bp = to_blueprint(world, **kw)
    errs = check_blueprint(bp)
    if errs:
        raise ValueError("invalid blueprint: " + "; ".join(errs[:3]))
    Path(path).write_text(json.dumps(bp, indent=2, sort_keys=True), encoding="utf-8")
    return bp


def load_blueprint(path):
    bp = json.loads(Path(path).read_text(encoding="utf-8"))
    errs = check_blueprint(bp)
    return (None, errs) if errs else (bp, [])
