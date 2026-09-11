import json
import math
import random
import zipfile
import heapq
from collections import Counter, deque
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

import tilegen_system as ts


MODULE_SIZE = 16
MODULE_SLOTS = 4

PASSABLE_TILES = {"floor", "platform", "bridge", "courtyard", "ladder_tile", "water", "rubble"}
BROAD_PASSABLE_TILES = {"floor", "platform", "bridge", "courtyard", "ladder_tile", "water", "rubble", "abyss", "lava"}
HARD_BLOCK_TILES = {"wall", "bookshelf"}

AGENT_PROFILES = {
    "npc": {
        "name": "Seeker NPC",
        "strategy": "weighted_shortest_path",
        "base_passable": {"floor", "platform", "bridge", "courtyard", "ladder_tile", "rubble"},
        "tile_costs": {
            "floor": 1,
            "platform": 1,
            "bridge": 1,
            "courtyard": 1,
            "ladder_tile": 1,
            "rubble": 2,
        },
        "overlay_costs": {"locked": 3, "challenge": 2, "rotate": 1, "shift": 1},
        "path_color": (255, 220, 120),
        "start_color": (120, 255, 120),
        "goal_color": (255, 120, 120),
    },
    "turtle": {
        "name": "Turtle Explorer",
        "strategy": "local_wall_follower",
        "base_passable": {"floor", "platform", "bridge", "courtyard", "ladder_tile", "rubble", "water"},
        "tile_costs": {
            "floor": 1,
            "platform": 1,
            "bridge": 1,
            "courtyard": 1,
            "ladder_tile": 1,
            "rubble": 2,
            "water": 2,
        },
        "overlay_costs": {"locked": 2, "challenge": 1, "rotate": 1, "shift": 1},
        "path_color": (120, 255, 200),
        "start_color": (120, 255, 120),
        "goal_color": (255, 120, 120),
    },
    "drone": {
        "name": "Survey Drone",
        "strategy": "weighted_shortest_path",
        "base_passable": {"floor", "platform", "bridge", "courtyard", "ladder_tile", "rubble", "water", "abyss", "lava"},
        "tile_costs": {
            "floor": 1,
            "platform": 1,
            "bridge": 1,
            "courtyard": 1,
            "ladder_tile": 1,
            "rubble": 2,
            "water": 1,
            "abyss": 2,
            "lava": 3,
        },
        "overlay_costs": {"locked": 1, "challenge": 1, "rotate": 1, "shift": 1},
        "path_color": (120, 220, 255),
        "start_color": (120, 255, 120),
        "goal_color": (255, 120, 120),
    },
}

WORLD_PRESETS = [
    {
        "name": "tiny_fast",
        "grid_w": 6,
        "grid_h": 6,
        "gap": 1,
        "unique_states": False,
        "seed": 11,
        "max_restarts": 24,
    },
    {
        "name": "balanced",
        "grid_w": 8,
        "grid_h": 8,
        "gap": 2,
        "unique_states": False,
        "seed": 23,
        "max_restarts": 32,
    },
    {
        "name": "all100_unique",
        "grid_w": 10,
        "grid_h": 10,
        "gap": 2,
        "unique_states": True,
        "seed": 37,
        "max_restarts": 128,
    },
]

DIRS = {
    "N": (0, -1),
    "E": (1, 0),
    "S": (0, 1),
    "W": (-1, 0),
}
OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}
TURN_DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # N E S W


def bit_count(mask: int) -> int:
    return mask.bit_count()


def iter_bits(mask: int):
    while mask:
        lsb = mask & -mask
        yield lsb.bit_length() - 1
        mask ^= lsb


def bitmask_to_slots(mask: int):
    return [idx for idx in range(MODULE_SLOTS) if (mask >> idx) & 1]


def ensure_base_output(base_root: Path):
    if not (base_root / "layouts").exists() or not (base_root / "renders").exists():
        ts.build_all(base_root)


def load_layout_catalog(base_root: Path):
    catalog_path = base_root / "catalog.json"
    if catalog_path.exists():
        catalog = {rec["id"]: rec for rec in json.loads(catalog_path.read_text(encoding="utf-8"))}
    else:
        catalog = {}

    records = []
    for layout_path in sorted((base_root / "layouts").glob("*.json")):
        payload = json.loads(layout_path.read_text(encoding="utf-8"))
        spec = next(spec for spec in ts.all_specs if spec["id"] == payload["id"])
        base = [[payload["base_legend"][v] for v in row] for row in payload["base"]]
        overlay = [[payload["overlay_legend"][v] for v in row] for row in payload["overlay"]]
        tilemap = ts.TileMap(payload["size"]["width"], payload["size"]["height"])
        for y in range(tilemap.h):
            for x in range(tilemap.w):
                tilemap.base[y][x] = base[y][x]
                tilemap.overlay[y][x] = overlay[y][x]
        rec = {
            "id": payload["id"],
            "slug": payload["slug"],
            "name": payload["name"],
            "template": payload["template"],
            "theme": payload.get("theme", ""),
            "spec": spec,
            "json_path": layout_path,
            "source_render": (base_root / catalog.get(payload["id"], {}).get("render_png", "")).resolve() if catalog.get(payload["id"], {}).get("render_png") else None,
            "base": base,
            "overlay": overlay,
            "tilemap": tilemap,
        }
        records.append(rec)
    return records


def resample_layers(base, overlay, out_size=MODULE_SIZE):
    h = len(base)
    w = len(base[0])
    out_base = [["wall" for _ in range(out_size)] for _ in range(out_size)]
    out_overlay = [["" for _ in range(out_size)] for _ in range(out_size)]
    for oy in range(out_size):
        y0 = int(oy * h / out_size)
        y1 = max(y0 + 1, int((oy + 1) * h / out_size))
        for ox in range(out_size):
            x0 = int(ox * w / out_size)
            x1 = max(x0 + 1, int((ox + 1) * w / out_size))
            base_counts = Counter()
            overlay_counts = Counter()
            for y in range(y0, min(y1, h)):
                for x in range(x0, min(x1, w)):
                    base_counts[base[y][x]] += 1
                    if overlay[y][x]:
                        overlay_counts[overlay[y][x]] += 1
            token = max(
                base_counts.items(),
                key=lambda kv: (kv[1], kv[0] in PASSABLE_TILES, kv[0] in BROAD_PASSABLE_TILES, kv[0]),
            )[0]
            out_base[oy][ox] = token
            if overlay_counts:
                out_overlay[oy][ox] = max(overlay_counts.items(), key=lambda kv: kv[1])[0]
    return out_base, out_overlay


def largest_component(mask):
    h = len(mask)
    w = len(mask[0])
    seen = [[False for _ in range(w)] for _ in range(h)]
    best = []
    for y in range(h):
        for x in range(w):
            if mask[y][x] and not seen[y][x]:
                comp = []
                dq = deque([(x, y)])
                seen[y][x] = True
                while dq:
                    cx, cy = dq.popleft()
                    comp.append((cx, cy))
                    for dx, dy in DIRS.values():
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < w and 0 <= ny < h and mask[ny][nx] and not seen[ny][nx]:
                            seen[ny][nx] = True
                            dq.append((nx, ny))
                if len(comp) > len(best):
                    best = comp
    out = [[False for _ in range(w)] for _ in range(h)]
    for x, y in best:
        out[y][x] = True
    return out, best


def side_slot_metrics(mask, side, slots=MODULE_SLOTS):
    h = len(mask)
    w = len(mask[0])
    entries = []
    if side in ("N", "S"):
        seg_len = w / slots
        for slot in range(slots):
            x0 = int(slot * seg_len)
            x1 = max(x0 + 1, int((slot + 1) * seg_len))
            best = None
            for depth in range(h):
                yy = depth if side == "N" else h - 1 - depth
                for x in range(x0, min(x1, w)):
                    if mask[yy][x]:
                        cand = (depth, abs((x0 + x1 - 1) / 2 - x), x, yy)
                        if best is None or cand < best:
                            best = cand
            if best is None:
                for depth in range(h):
                    yy = depth if side == "N" else h - 1 - depth
                    for x in range(w):
                        if mask[yy][x]:
                            cand = (depth, abs((seg_len * (slot + 0.5)) - x), x, yy)
                            if best is None or cand < best:
                                best = cand
            depth, _, ax, ay = best
            entries.append({"slot": slot, "cost": int(depth), "anchor": [int(ax), int(ay)]})
    else:
        seg_len = h / slots
        for slot in range(slots):
            y0 = int(slot * seg_len)
            y1 = max(y0 + 1, int((slot + 1) * seg_len))
            best = None
            for depth in range(w):
                xx = depth if side == "W" else w - 1 - depth
                for y in range(y0, min(y1, h)):
                    if mask[y][xx]:
                        cand = (depth, abs((y0 + y1 - 1) / 2 - y), xx, y)
                        if best is None or cand < best:
                            best = cand
            if best is None:
                for depth in range(w):
                    xx = depth if side == "W" else w - 1 - depth
                    for y in range(h):
                        if mask[y][xx]:
                            cand = (depth, abs((seg_len * (slot + 0.5)) - y), xx, y)
                            if best is None or cand < best:
                                best = cand
            depth, _, ax, ay = best
            entries.append({"slot": slot, "cost": int(depth), "anchor": [int(ax), int(ay)]})
    entries.sort(key=lambda e: (e["cost"], e["slot"]))
    return entries


def build_module_library(layout_records):
    modules = []
    for rec in layout_records:
        std_base, std_overlay = resample_layers(rec["base"], rec["overlay"], MODULE_SIZE)
        general_mask = [[std_base[y][x] in PASSABLE_TILES for x in range(MODULE_SIZE)] for y in range(MODULE_SIZE)]
        comp_mask, component = largest_component(general_mask)
        if not component:
            broad_mask = [[std_base[y][x] in BROAD_PASSABLE_TILES for x in range(MODULE_SIZE)] for y in range(MODULE_SIZE)]
            comp_mask, component = largest_component(broad_mask)
        side_metrics = {side: side_slot_metrics(comp_mask, side, MODULE_SLOTS) for side in "NESW"}
        side_masks = {}
        side_anchors = {}
        for side, entries in side_metrics.items():
            bitmask = 0
            anchor_map = {}
            for entry in entries:
                anchor_map[str(entry["slot"])] = entry["anchor"]
            for entry in entries[:2]:
                bitmask |= 1 << entry["slot"]
            side_masks[side] = bitmask
            side_anchors[side] = anchor_map
        walkable_count = sum(1 for row in comp_mask for v in row if v)
        module = {
            "state_index": len(modules),
            "layout_id": rec["id"],
            "slug": rec["slug"],
            "name": rec["name"],
            "template": rec["template"],
            "theme": rec["theme"],
            "source_render": str(rec["source_render"]) if rec["source_render"] else "",
            "source_layout_json": str(rec["json_path"]),
            "module_size": MODULE_SIZE,
            "base": std_base,
            "overlay": std_overlay,
            "largest_component_mask": [[1 if v else 0 for v in row] for row in comp_mask],
            "largest_component_tiles": walkable_count,
            "open_ratio": round(walkable_count / (MODULE_SIZE * MODULE_SIZE), 4),
            "side_masks": {side: int(mask) for side, mask in side_masks.items()},
            "side_slots": {side: bitmask_to_slots(mask) for side, mask in side_masks.items()},
            "side_anchor_map": side_anchors,
            "side_metrics": side_metrics,
            "compat_degree": 0,
        }
        modules.append(module)

    compat = {direction: [0 for _ in range(len(modules))] for direction in "NESW"}
    for i, mod_a in enumerate(modules):
        for j, mod_b in enumerate(modules):
            for direction in "NESW":
                if mod_a["side_masks"][direction] & mod_b["side_masks"][OPPOSITE[direction]]:
                    compat[direction][i] |= 1 << j
    for idx, module in enumerate(modules):
        degree = sum(bit_count(compat[direction][idx]) for direction in "NESW")
        module["compat_degree"] = int(degree)
    return modules, compat


def tile_cost(profile_name, tile_name, overlay_name=""):
    profile = AGENT_PROFILES[profile_name]
    cost = profile["tile_costs"].get(tile_name, 999999)
    cost += profile["overlay_costs"].get(overlay_name, 0)
    return cost


def build_passable_grid(tilemap: ts.TileMap, profile_name: str):
    profile = AGENT_PROFILES[profile_name]
    allowed = profile["base_passable"]
    grid = [[False for _ in range(tilemap.w)] for _ in range(tilemap.h)]
    for y in range(tilemap.h):
        for x in range(tilemap.w):
            base = tilemap.base[y][x]
            if base in allowed:
                grid[y][x] = True
    return grid


def find_teleport_nodes(tilemap: ts.TileMap, passable_grid=None):
    nodes = []
    for y in range(tilemap.h):
        for x in range(tilemap.w):
            if tilemap.overlay[y][x] == "teleport" and (passable_grid is None or passable_grid[y][x]):
                nodes.append((x, y))
    return nodes


def shortest_paths(tilemap: ts.TileMap, profile_name: str, start):
    passable = build_passable_grid(tilemap, profile_name)
    sx, sy = start
    if not (0 <= sx < tilemap.w and 0 <= sy < tilemap.h and passable[sy][sx]):
        return None, None, passable
    teleports = find_teleport_nodes(tilemap, passable)
    teleport_set = set(teleports)
    inf = 10**12
    dist = [[inf for _ in range(tilemap.w)] for _ in range(tilemap.h)]
    prev = {}
    pq = [(0, sx, sy)]
    dist[sy][sx] = 0
    while pq:
        cur_cost, x, y = heapq.heappop(pq)
        if cur_cost != dist[y][x]:
            continue
        for dx, dy in TURN_DIRS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < tilemap.w and 0 <= ny < tilemap.h and passable[ny][nx]:
                step_cost = tile_cost(profile_name, tilemap.base[ny][nx], tilemap.overlay[ny][nx])
                new_cost = cur_cost + step_cost
                if new_cost < dist[ny][nx]:
                    dist[ny][nx] = new_cost
                    prev[(nx, ny)] = (x, y)
                    heapq.heappush(pq, (new_cost, nx, ny))
        if (x, y) in teleport_set:
            for nx, ny in teleports:
                if (nx, ny) == (x, y):
                    continue
                new_cost = cur_cost + 1
                if new_cost < dist[ny][nx]:
                    dist[ny][nx] = new_cost
                    prev[(nx, ny)] = (x, y)
                    heapq.heappush(pq, (new_cost, nx, ny))
    return dist, prev, passable


def reconstruct_path(prev, start, goal):
    if start == goal:
        return [list(start)]
    cur = goal
    path = [list(goal)]
    while cur != start:
        cur = prev.get(cur)
        if cur is None:
            return []
        path.append(list(cur))
    path.reverse()
    return path


def farthest_pair(tilemap: ts.TileMap, profile_name: str):
    passable = build_passable_grid(tilemap, profile_name)
    walkable = [(x, y) for y in range(tilemap.h) for x in range(tilemap.w) if passable[y][x]]
    if not walkable:
        return None
    start = min(walkable, key=lambda p: (p[1], p[0]))
    dist1, _, _ = shortest_paths(tilemap, profile_name, start)
    reachable1 = [(dist1[y][x], (x, y)) for x, y in walkable if dist1[y][x] < 10**12]
    if not reachable1:
        return None
    a = max(reachable1, key=lambda kv: kv[0])[1]
    dist2, prev2, _ = shortest_paths(tilemap, profile_name, a)
    reachable2 = [(dist2[y][x], (x, y)) for x, y in walkable if dist2[y][x] < 10**12]
    if not reachable2:
        return None
    b = max(reachable2, key=lambda kv: kv[0])[1]
    path = reconstruct_path(prev2, a, b)
    walkable_reachable = sum(1 for x, y in walkable if dist2[y][x] < 10**12)
    return {
        "start": list(a),
        "goal": list(b),
        "path": path,
        "path_cost": int(dist2[b[1]][b[0]]),
        "path_steps": max(0, len(path) - 1),
        "reachable_tiles": int(walkable_reachable),
        "walkable_tiles": int(len(walkable)),
        "success": bool(path),
    }


def turtle_explore(tilemap: ts.TileMap, start=None, goal=None, max_steps=None, seed=0):
    profile_name = "turtle"
    passable = build_passable_grid(tilemap, profile_name)
    walkable = [(x, y) for y in range(tilemap.h) for x in range(tilemap.w) if passable[y][x]]
    if not walkable:
        return {
            "success": False,
            "start": None,
            "goal": None,
            "path": [],
            "steps": 0,
            "visited_tiles": 0,
            "coverage": 0.0,
        }
    if start is None or goal is None:
        farthest = farthest_pair(tilemap, profile_name)
        if farthest is None:
            start = list(min(walkable, key=lambda p: (p[1], p[0])))
            goal = list(start)
        else:
            start = farthest["start"]
            goal = farthest["goal"]
    sx, sy = start
    gx, gy = goal
    if max_steps is None:
        max_steps = min(max(64, len(walkable) * 4), 12000)
    rng = random.Random(seed)
    teleports = find_teleport_nodes(tilemap, passable)
    teleport_set = set(teleports)
    heading = 1  # east
    x, y = sx, sy
    visits = Counter()
    visits[(x, y)] = 1
    path = [[x, y]]
    arrived_via_teleport = False
    for _ in range(max_steps):
        if (x, y) == (gx, gy):
            break
        # Walk off an arrival portal before using another one. An isolated
        # portal has no walking exit, so it must still allow an onward jump.
        if (x, y) in teleport_set and len(teleports) > 1 and (
            not arrived_via_teleport or not any(
                0 <= x + dx < tilemap.w and 0 <= y + dy < tilemap.h
                and passable[y + dy][x + dx] for dx, dy in TURN_DIRS
            )
        ):
            options = [p for p in teleports if p != (x, y)]
            options.sort(key=lambda p: (visits[p], abs(p[0] - gx) + abs(p[1] - gy)))
            x, y = options[0]
            visits[(x, y)] += 1
            path.append([x, y])
            arrived_via_teleport = True
            continue
        ordering = [(heading - 1) % 4, heading, (heading + 1) % 4, (heading + 2) % 4]
        best = None
        for ndir in ordering:
            dx, dy = TURN_DIRS[ndir]
            nx, ny = x + dx, y + dy
            if 0 <= nx < tilemap.w and 0 <= ny < tilemap.h and passable[ny][nx]:
                score = (
                    visits[(nx, ny)],
                    0 if ndir == ordering[0] else 1 if ndir == ordering[1] else 2 if ndir == ordering[2] else 3,
                    abs(nx - gx) + abs(ny - gy),
                    rng.random(),
                )
                if best is None or score < best[0]:
                    best = (score, ndir, nx, ny)
        if best is None:
            break
        _, heading, x, y = best
        arrived_via_teleport = False
        visits[(x, y)] += 1
        path.append([x, y])
    return {
        "success": [x, y] == [gx, gy],
        "start": [sx, sy],
        "goal": [gx, gy],
        "path": path,
        "steps": max(0, len(path) - 1),
        "visited_tiles": int(len(visits)),
        "coverage": round(len(visits) / max(1, len(walkable)), 4),
    }


def render_path_overlay(tilemap: ts.TileMap, spec_stub: dict, path, start, goal, outpath: Path, color=(255, 220, 120), start_color=(120, 255, 120), goal_color=(255, 120, 120), max_map_px=360, header_text=None, grid_info=None):
    img = ts.render_map_image(tilemap, spec_stub, max_map_px=max_map_px)
    draw = ImageDraw.Draw(img)
    tile_px = max(4, min(64, max_map_px // max(tilemap.w, tilemap.h)))
    if tilemap.w <= 9:
        tile_px = max(tile_px, 24 if tilemap.w == 9 else 56)
    margin = 8
    if len(path) >= 2:
        pts = []
        for x, y in path:
            cx = margin + x * tile_px + tile_px // 2
            cy = margin + y * tile_px + tile_px // 2
            pts.append((cx, cy))
        draw.line(pts, fill=color, width=max(1, tile_px // 2))
    for point, fill in ((start, start_color), (goal, goal_color)):
        if point is None:
            continue
        px = margin + point[0] * tile_px + tile_px // 2
        py = margin + point[1] * tile_px + tile_px // 2
        radius = max(2, tile_px // 2)
        draw.ellipse([px - radius, py - radius, px + radius, py + radius], fill=fill, outline=(12, 12, 12))
    if header_text:
        draw.rectangle([6, 6, 6 + max(120, len(header_text) * 6), 18], fill=(12, 14, 18))
        draw.text((10, 8), header_text, fill=(235, 235, 240), font=ts.DEFAULT_FONT)
    if grid_info:
        cell_span = MODULE_SIZE + grid_info.get("gap", 0)
        for gx in range(1, grid_info["grid_w"]):
            x = margin + gx * cell_span * tile_px - (grid_info.get("gap", 0) * tile_px // 2)
            draw.line([x, margin, x, margin + tilemap.h * tile_px], fill=(80, 84, 92))
        for gy in range(1, grid_info["grid_h"]):
            y = margin + gy * cell_span * tile_px - (grid_info.get("gap", 0) * tile_px // 2)
            draw.line([margin, y, margin + tilemap.w * tile_px, y], fill=(80, 84, 92))
    img.save(outpath)


def build_contact_sheet(image_paths, labels, outpath, cols=5, cell_w=220, cell_h=240):
    rows = (len(image_paths) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), (8, 9, 14))
    for idx, (img_path, label) in enumerate(zip(image_paths, labels)):
        im = Image.open(img_path).convert("RGB")
        thumb = ImageOps.contain(im, (cell_w - 12, cell_h - 24))
        canvas = Image.new("RGB", (cell_w, cell_h), (18, 20, 24))
        canvas.paste(thumb, ((cell_w - thumb.width) // 2, 6))
        ImageDraw.Draw(canvas).text((8, cell_h - 18), label, fill=(230, 230, 230), font=ts.DEFAULT_FONT)
        sheet.paste(canvas, ((idx % cols) * cell_w, (idx // cols) * cell_h))
    sheet.save(outpath)


def build_navigation_catalog(layout_records, runtime_root: Path):
    agents_root = runtime_root / "agents"
    agents_root.mkdir(parents=True, exist_ok=True)
    catalog = []
    summary = {name: {"successes": 0, "failures": 0, "path_steps_total": 0, "layouts": 0} for name in AGENT_PROFILES}
    contact_sources = {name: [] for name in AGENT_PROFILES}
    contact_labels = {name: [] for name in AGENT_PROFILES}

    for rec in layout_records:
        entry = {
            "id": rec["id"],
            "name": rec["name"],
            "slug": rec["slug"],
            "template": rec["template"],
            "theme": rec["theme"],
            "size": {"width": rec["tilemap"].w, "height": rec["tilemap"].h},
            "agents": {},
        }
        spec_stub = {"id": rec["id"], "seed": rec["id"] * 13}
        for agent_name, profile in AGENT_PROFILES.items():
            agent_dir = agents_root / agent_name
            agent_dir.mkdir(exist_ok=True)
            if profile["strategy"] == "weighted_shortest_path":
                nav = farthest_pair(rec["tilemap"], agent_name)
                if nav is None:
                    nav = {
                        "success": False,
                        "start": None,
                        "goal": None,
                        "path": [],
                        "path_cost": 0,
                        "path_steps": 0,
                        "reachable_tiles": 0,
                        "walkable_tiles": 0,
                    }
            else:
                nav = turtle_explore(rec["tilemap"], seed=rec["id"] * 97)
                nav.setdefault("path_cost", nav["steps"])
                nav.setdefault("reachable_tiles", nav["visited_tiles"])
                nav.setdefault("walkable_tiles", nav["visited_tiles"])
                nav.setdefault("path_steps", nav["steps"])
            png_name = f"{rec['id']:03d}_{rec['slug']}.png"
            render_path = agent_dir / png_name
            start = nav["start"]
            goal = nav["goal"]
            render_path_overlay(
                rec["tilemap"],
                spec_stub,
                nav["path"],
                start,
                goal,
                render_path,
                color=profile["path_color"],
                start_color=profile["start_color"],
                goal_color=profile["goal_color"],
                max_map_px=360,
                header_text=agent_name,
            )
            nav_export = {k: v for k, v in nav.items()}
            nav_export["render_png"] = f"agents/{agent_name}/{png_name}"
            entry["agents"][agent_name] = nav_export
            summary[agent_name]["layouts"] += 1
            summary[agent_name]["path_steps_total"] += int(nav.get("path_steps", 0))
            if nav.get("success"):
                summary[agent_name]["successes"] += 1
            else:
                summary[agent_name]["failures"] += 1
            contact_sources[agent_name].append(render_path)
            label_suffix = "✓" if nav.get("success") else "✗"
            contact_labels[agent_name].append(f"{rec['id']:03d}{label_suffix}")
        catalog.append(entry)

    for agent_name in AGENT_PROFILES:
        build_contact_sheet(
            contact_sources[agent_name],
            contact_labels[agent_name],
            agents_root / f"contact_sheet_{agent_name}.png",
            cols=5,
        )
        avg_steps = summary[agent_name]["path_steps_total"] / max(1, summary[agent_name]["layouts"])
        summary[agent_name]["avg_path_steps"] = round(avg_steps, 2)
        summary[agent_name].pop("path_steps_total", None)

    return catalog, summary


def build_module_socket_sheet(modules, runtime_root: Path, cell_w=220, cell_h=240, cols=5):
    rows = (len(modules) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), (8, 9, 14))
    for idx, module in enumerate(modules):
        if module["source_render"] and Path(module["source_render"]).exists():
            base_im = Image.open(module["source_render"]).convert("RGB")
            thumb = ImageOps.contain(base_im, (cell_w - 18, cell_h - 44))
        else:
            temp_map = ts.TileMap(MODULE_SIZE, MODULE_SIZE)
            for y in range(MODULE_SIZE):
                for x in range(MODULE_SIZE):
                    temp_map.base[y][x] = module["base"][y][x]
                    temp_map.overlay[y][x] = module["overlay"][y][x]
            thumb = ImageOps.contain(ts.render_map_image(temp_map, {"id": module["layout_id"]}, max_map_px=180), (cell_w - 18, cell_h - 44))
        canvas = Image.new("RGB", (cell_w, cell_h), (18, 20, 24))
        ox = (cell_w - thumb.width) // 2
        oy = 8
        canvas.paste(thumb, (ox, oy))
        draw = ImageDraw.Draw(canvas)
        tx0 = ox
        ty0 = oy
        tx1 = ox + thumb.width - 1
        ty1 = oy + thumb.height - 1
        slot_w = thumb.width / MODULE_SLOTS
        slot_h = thumb.height / MODULE_SLOTS
        marker = (120, 220, 255)
        for slot in module["side_slots"]["N"]:
            x0 = int(tx0 + slot * slot_w + 2)
            x1 = int(tx0 + (slot + 1) * slot_w - 2)
            draw.rectangle([x0, ty0, x1, ty0 + 4], fill=marker)
        for slot in module["side_slots"]["S"]:
            x0 = int(tx0 + slot * slot_w + 2)
            x1 = int(tx0 + (slot + 1) * slot_w - 2)
            draw.rectangle([x0, ty1 - 4, x1, ty1], fill=marker)
        for slot in module["side_slots"]["W"]:
            y0 = int(ty0 + slot * slot_h + 2)
            y1 = int(ty0 + (slot + 1) * slot_h - 2)
            draw.rectangle([tx0, y0, tx0 + 4, y1], fill=marker)
        for slot in module["side_slots"]["E"]:
            y0 = int(ty0 + slot * slot_h + 2)
            y1 = int(ty0 + (slot + 1) * slot_h - 2)
            draw.rectangle([tx1 - 4, y0, tx1, y1], fill=marker)
        label = f"{module['layout_id']:03d}  {module['template']}"
        draw.text((8, cell_h - 18), label, fill=(230, 230, 230), font=ts.DEFAULT_FONT)
        sheet.paste(canvas, ((idx % cols) * cell_w, (idx // cols) * cell_h))
    outpath = runtime_root / "module_socket_sheet.png"
    sheet.save(outpath)
    return outpath


def choose_state_from_domain(domain_mask: int, modules, rng: random.Random, used_counts):
    states = list(iter_bits(domain_mask))
    if not states:
        return None
    weights = []
    for state in states:
        module = modules[state]
        degree = max(1, module["compat_degree"])
        diversity = 1 / (1 + used_counts.get(state, 0))
        weights.append(degree * diversity)
    total = sum(weights)
    r = rng.random() * total
    acc = 0.0
    for state, weight in zip(states, weights):
        acc += weight
        if acc >= r:
            return state
    return states[-1]


def solve_socket_wfc(modules, compat, grid_w, grid_h, seed=0, unique_states=False, max_restarts=64):
    rng = random.Random(seed)
    state_count = len(modules)
    all_states_mask = (1 << state_count) - 1
    cell_count = grid_w * grid_h

    for attempt in range(max_restarts):
        domains = [all_states_mask for _ in range(cell_count)]
        collapsed = [False for _ in range(cell_count)]
        used_mask = 0
        used_counts = Counter()

        def enqueue_unique_constraints(queue_init=None):
            dq = deque(queue_init or [])
            if not unique_states:
                return True, dq
            for idx in range(cell_count):
                if collapsed[idx]:
                    continue
                new_domain = domains[idx] & ~used_mask
                if new_domain != domains[idx]:
                    if new_domain == 0:
                        return False, None
                    domains[idx] = new_domain
                    dq.append(idx)
            return True, dq

        def propagate(queue):
            while queue:
                cell = queue.popleft()
                x = cell % grid_w
                y = cell // grid_w
                domain = domains[cell]
                for direction, (dx, dy) in DIRS.items():
                    nx, ny = x + dx, y + dy
                    if not (0 <= nx < grid_w and 0 <= ny < grid_h):
                        continue
                    nidx = ny * grid_w + nx
                    neighbor_old = domains[nidx]
                    allowed = 0
                    for state in iter_bits(domain):
                        allowed |= compat[direction][state]
                    neighbor_new = neighbor_old & allowed
                    if neighbor_new != neighbor_old:
                        if neighbor_new == 0:
                            return False
                        domains[nidx] = neighbor_new
                        queue.append(nidx)
            return True

        ok, queue = enqueue_unique_constraints([])
        if not ok or not propagate(queue):
            continue

        while True:
            pending = [(bit_count(domains[idx]), idx) for idx in range(cell_count) if not collapsed[idx]]
            if not pending:
                return {
                    "success": True,
                    "attempts": attempt + 1,
                    "placements": [domains[idx].bit_length() - 1 for idx in range(cell_count)],
                }
            non_singletons = [item for item in pending if item[0] > 1]
            if not non_singletons:
                if unique_states and len({domains[idx] for idx in range(cell_count)}) != cell_count:
                    break
                return {
                    "success": True,
                    "attempts": attempt + 1,
                    "placements": [domains[idx].bit_length() - 1 for idx in range(cell_count)],
                }
            min_entropy = min(item[0] for item in non_singletons)
            choices = [idx for count, idx in non_singletons if count == min_entropy]
            cell = rng.choice(choices)
            chosen_state = choose_state_from_domain(domains[cell], modules, rng, used_counts)
            if chosen_state is None:
                break
            domains[cell] = 1 << chosen_state
            collapsed[cell] = True
            used_counts[chosen_state] += 1
            if unique_states:
                used_mask |= 1 << chosen_state
            ok, queue = enqueue_unique_constraints([cell])
            if not ok or not propagate(queue):
                break

    return {"success": False, "attempts": max_restarts, "placements": []}


def choose_overlap_slot(module_a, side_a, module_b, side_b):
    overlap = module_a["side_masks"][side_a] & module_b["side_masks"][side_b]
    if overlap == 0:
        overlap = module_a["side_masks"][side_a] or module_b["side_masks"][side_b] or 1
    best = None
    for slot in range(MODULE_SLOTS):
        if not ((overlap >> slot) & 1):
            continue
        entry_a = next(entry for entry in module_a["side_metrics"][side_a] if entry["slot"] == slot)
        entry_b = next(entry for entry in module_b["side_metrics"][side_b] if entry["slot"] == slot)
        cand = (entry_a["cost"] + entry_b["cost"], abs(slot - (MODULE_SLOTS - 1) / 2), slot)
        if best is None or cand < best:
            best = cand
    return best[2] if best is not None else 0


def connector_tile_name(tile_a, tile_b):
    special = {tile_a, tile_b}
    if "abyss" in special or "lava" in special or "water" in special or "bridge" in special:
        return "bridge"
    if "platform" in special:
        return "platform"
    return "floor"


def carve_l(tilemap: ts.TileMap, a, b, tile_name="floor"):
    x0, y0 = a
    x1, y1 = b
    if x0 == x1 and y0 == y1:
        tilemap.set(x0, y0, base=tile_name)
        return
    if abs(x1 - x0) >= abs(y1 - y0):
        step = 1 if x1 >= x0 else -1
        for x in range(x0, x1 + step, step):
            tilemap.set(x, y0, base=tile_name)
        step = 1 if y1 >= y0 else -1
        for y in range(y0, y1 + step, step):
            tilemap.set(x1, y, base=tile_name)
    else:
        step = 1 if y1 >= y0 else -1
        for y in range(y0, y1 + step, step):
            tilemap.set(x0, y, base=tile_name)
        step = 1 if x1 >= x0 else -1
        for x in range(x0, x1 + step, step):
            tilemap.set(x, y1, base=tile_name)


def assemble_world(modules, placements, grid_w, grid_h, gap):
    world_w = grid_w * MODULE_SIZE + (grid_w - 1) * gap
    world_h = grid_h * MODULE_SIZE + (grid_h - 1) * gap
    tilemap = ts.TileMap(world_w, world_h, default_base="wall", default_overlay="")
    placement_records = []
    for gy in range(grid_h):
        for gx in range(grid_w):
            state = placements[gy * grid_w + gx]
            module = modules[state]
            ox = gx * (MODULE_SIZE + gap)
            oy = gy * (MODULE_SIZE + gap)
            for y in range(MODULE_SIZE):
                for x in range(MODULE_SIZE):
                    tilemap.base[oy + y][ox + x] = module["base"][y][x]
                    tilemap.overlay[oy + y][ox + x] = module["overlay"][y][x]
            placement_records.append(
                {
                    "grid": [gx, gy],
                    "state_index": state,
                    "layout_id": module["layout_id"],
                    "module_name": module["name"],
                    "offset": [ox, oy],
                }
            )

    connectors = []
    for gy in range(grid_h):
        for gx in range(grid_w):
            state_a = placements[gy * grid_w + gx]
            module_a = modules[state_a]
            ox_a = gx * (MODULE_SIZE + gap)
            oy_a = gy * (MODULE_SIZE + gap)
            if gx + 1 < grid_w:
                state_b = placements[gy * grid_w + gx + 1]
                module_b = modules[state_b]
                ox_b = (gx + 1) * (MODULE_SIZE + gap)
                oy_b = oy_a
                slot = choose_overlap_slot(module_a, "E", module_b, "W")
                entry_a = next(entry for entry in module_a["side_metrics"]["E"] if entry["slot"] == slot)
                entry_b = next(entry for entry in module_b["side_metrics"]["W"] if entry["slot"] == slot)
                ax, ay = entry_a["anchor"]
                bx, by = entry_b["anchor"]
                world_a = [ox_a + ax, oy_a + ay]
                world_b = [ox_b + bx, oy_b + by]
                seam_y = int(round((world_a[1] + world_b[1]) / 2))
                mid_a = [ox_a + MODULE_SIZE - 1, seam_y]
                mid_b = [ox_b, seam_y]
                connector_tile = connector_tile_name(module_a["base"][ay][ax], module_b["base"][by][bx])
                carve_l(tilemap, world_a, mid_a, connector_tile)
                carve_l(tilemap, mid_a, mid_b, connector_tile)
                carve_l(tilemap, mid_b, world_b, connector_tile)
                connectors.append({
                    "from": [gx, gy],
                    "to": [gx + 1, gy],
                    "slot": slot,
                    "tile": connector_tile,
                    "points": [world_a, mid_a, mid_b, world_b],
                })
            if gy + 1 < grid_h:
                state_b = placements[(gy + 1) * grid_w + gx]
                module_b = modules[state_b]
                ox_b = ox_a
                oy_b = (gy + 1) * (MODULE_SIZE + gap)
                slot = choose_overlap_slot(module_a, "S", module_b, "N")
                entry_a = next(entry for entry in module_a["side_metrics"]["S"] if entry["slot"] == slot)
                entry_b = next(entry for entry in module_b["side_metrics"]["N"] if entry["slot"] == slot)
                ax, ay = entry_a["anchor"]
                bx, by = entry_b["anchor"]
                world_a = [ox_a + ax, oy_a + ay]
                world_b = [ox_b + bx, oy_b + by]
                seam_x = int(round((world_a[0] + world_b[0]) / 2))
                mid_a = [seam_x, oy_a + MODULE_SIZE - 1]
                mid_b = [seam_x, oy_b]
                connector_tile = connector_tile_name(module_a["base"][ay][ax], module_b["base"][by][bx])
                carve_l(tilemap, world_a, mid_a, connector_tile)
                carve_l(tilemap, mid_a, mid_b, connector_tile)
                carve_l(tilemap, mid_b, world_b, connector_tile)
                connectors.append({
                    "from": [gx, gy],
                    "to": [gx, gy + 1],
                    "slot": slot,
                    "tile": connector_tile,
                    "points": [world_a, mid_a, mid_b, world_b],
                })
    return tilemap, placement_records, connectors


def build_worlds(modules, compat, runtime_root: Path):
    worlds_root = runtime_root / "worlds"
    worlds_root.mkdir(parents=True, exist_ok=True)
    world_records = []
    world_images = []
    world_labels = []

    for preset in WORLD_PRESETS:
        solved = solve_socket_wfc(
            modules,
            compat,
            grid_w=preset["grid_w"],
            grid_h=preset["grid_h"],
            seed=preset["seed"],
            unique_states=preset["unique_states"],
            max_restarts=preset["max_restarts"],
        )
        if not solved["success"]:
            raise RuntimeError(f"WFC failed for preset {preset['name']}")
        tilemap, placements, connectors = assemble_world(
            modules,
            solved["placements"],
            preset["grid_w"],
            preset["grid_h"],
            preset["gap"],
        )
        record = {
            "name": preset["name"],
            "grid_w": preset["grid_w"],
            "grid_h": preset["grid_h"],
            "module_size": MODULE_SIZE,
            "gap": preset["gap"],
            "unique_states": preset["unique_states"],
            "seed": preset["seed"],
            "attempts": solved["attempts"],
            "placements": placements,
            "connectors": connectors,
            "agents": {},
        }
        base_png = worlds_root / f"{preset['name']}_base.png"
        render_path_overlay(
            tilemap,
            {"id": 1000 + len(world_records), "seed": preset["seed"]},
            [],
            None,
            None,
            base_png,
            max_map_px=900,
            header_text=f"wfc:{preset['name']}",
            grid_info={"grid_w": preset["grid_w"], "grid_h": preset["grid_h"], "gap": preset["gap"]},
        )
        record["base_png"] = f"worlds/{base_png.name}"
        world_images.append(base_png)
        world_labels.append(preset["name"])

        for agent_name, profile in AGENT_PROFILES.items():
            if profile["strategy"] == "weighted_shortest_path":
                nav = farthest_pair(tilemap, agent_name)
                if nav is None:
                    nav = {
                        "success": False,
                        "start": None,
                        "goal": None,
                        "path": [],
                        "path_cost": 0,
                        "path_steps": 0,
                        "reachable_tiles": 0,
                        "walkable_tiles": 0,
                    }
            else:
                nav = turtle_explore(tilemap, seed=preset["seed"] * 17)
                nav.setdefault("path_cost", nav["steps"])
                nav.setdefault("reachable_tiles", nav["visited_tiles"])
                nav.setdefault("walkable_tiles", nav["visited_tiles"])
                nav.setdefault("path_steps", nav["steps"])
            agent_png = worlds_root / f"{preset['name']}_{agent_name}.png"
            render_path_overlay(
                tilemap,
                {"id": 1200 + len(world_records), "seed": preset["seed"]},
                nav["path"],
                nav.get("start"),
                nav.get("goal"),
                agent_png,
                color=profile["path_color"],
                start_color=profile["start_color"],
                goal_color=profile["goal_color"],
                max_map_px=900,
                header_text=f"wfc:{preset['name']}:{agent_name}",
                grid_info={"grid_w": preset["grid_w"], "grid_h": preset["grid_h"], "gap": preset["gap"]},
            )
            nav_export = {k: v for k, v in nav.items()}
            nav_export["render_png"] = f"worlds/{agent_png.name}"
            record["agents"][agent_name] = nav_export
        world_json = worlds_root / f"{preset['name']}.json"
        world_json.write_text(json.dumps(record, indent=2), encoding="utf-8")
        record["json_path"] = f"worlds/{world_json.name}"
        world_records.append(record)

    build_contact_sheet(world_images, world_labels, worlds_root / "contact_sheet_worlds.png", cols=3, cell_w=320, cell_h=260)
    return world_records


def build_viewer(modules, nav_catalog, nav_summary, world_records, runtime_root: Path):
    module_stats = {
        "count": len(modules),
        "avg_open_ratio": round(sum(m["open_ratio"] for m in modules) / max(1, len(modules)), 4),
        "avg_compat_degree": round(sum(m["compat_degree"] for m in modules) / max(1, len(modules)), 2),
    }
    embedded_modules = json.dumps([
        {
            "layout_id": m["layout_id"],
            "name": m["name"],
            "template": m["template"],
            "theme": m["theme"],
            "side_slots": m["side_slots"],
            "open_ratio": m["open_ratio"],
            "compat_degree": m["compat_degree"],
        }
        for m in modules
    ], ensure_ascii=False)
    embedded_nav_summary = json.dumps(nav_summary, ensure_ascii=False)
    embedded_worlds = json.dumps(world_records, ensure_ascii=False)
    html = f"""<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\"/>
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"/>
<title>Tilegen Low-End Runtime</title>
<style>
:root {{ --bg:#0d1016; --panel:#161b24; --line:#242d3b; --text:#e6ebf4; --muted:#96a4b8; --accent:#78b7ff; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; font:14px/1.45 system-ui, sans-serif; background:var(--bg); color:var(--text); }}
header {{ padding:22px 24px; border-bottom:1px solid var(--line); position:sticky; top:0; background:rgba(13,16,22,.96); backdrop-filter:blur(8px); z-index:10; }}
h1 {{ margin:0 0 8px; font-size:22px; }}
p {{ margin:0; color:var(--muted); }}
main {{ padding:20px 24px 40px; display:grid; gap:18px; }}
section {{ background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:16px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:14px; }}
.card {{ background:#111722; border:1px solid var(--line); border-radius:12px; padding:12px; }}
img {{ max-width:100%; border-radius:10px; border:1px solid #273243; display:block; background:#0b0f16; }}
.small {{ color:var(--muted); font-size:12px; }}
.badge {{ display:inline-block; padding:2px 8px; border:1px solid #334155; border-radius:999px; font-size:11px; color:#c8d8ee; margin-right:6px; margin-top:6px; }}
a {{ color:var(--accent); text-decoration:none; }}
pre {{ white-space:pre-wrap; color:#cfe1f5; background:#0f141d; padding:10px; border-radius:10px; border:1px solid var(--line); }}
</style>
</head>
<body>
<header>
  <h1>Low-End Runtime Extension</h1>
  <p>Socket-based WFC built from all 100 source layouts, plus turtle, NPC, and drone navigation over every layout and the assembled worlds.</p>
</header>
<main>
  <section>
    <h2>Runtime profile</h2>
    <div class=\"grid\">
      <div class=\"card\"><strong>Module states</strong><div class=\"small\">{module_stats['count']} layouts resampled to {MODULE_SIZE}×{MODULE_SIZE}</div></div>
      <div class=\"card\"><strong>Average open ratio</strong><div class=\"small\">{module_stats['avg_open_ratio']}</div></div>
      <div class=\"card\"><strong>Average compat degree</strong><div class=\"small\">{module_stats['avg_compat_degree']}</div></div>
      <div class=\"card\"><strong>WFC mode</strong><div class=\"small\">Entropy + bitset propagation + 4 socket slots per edge</div></div>
    </div>
    <div class=\"small\" style=\"margin-top:12px\">Module sockets are shown in the sheet below.</div>
    <div style=\"margin-top:12px\"><img src=\"module_socket_sheet.png\" alt=\"Module socket sheet\"></div>
  </section>
  <section>
    <h2>Agent summary</h2>
    <div id=\"agentSummary\" class=\"grid\"></div>
    <div class=\"grid\" style=\"margin-top:14px\">
      <div class=\"card\"><img src=\"agents/contact_sheet_npc.png\" alt=\"NPC contact sheet\"><div class=\"small\" style=\"margin-top:8px\">Seeker NPC</div></div>
      <div class=\"card\"><img src=\"agents/contact_sheet_turtle.png\" alt=\"Turtle contact sheet\"><div class=\"small\" style=\"margin-top:8px\">Turtle Explorer</div></div>
      <div class=\"card\"><img src=\"agents/contact_sheet_drone.png\" alt=\"Drone contact sheet\"><div class=\"small\" style=\"margin-top:8px\">Survey Drone</div></div>
    </div>
  </section>
  <section>
    <h2>World outputs</h2>
    <div id=\"worldGrid\" class=\"grid\"></div>
  </section>
  <section>
    <h2>Data files</h2>
    <pre>module_library.json\nnavigation_catalog.json\nruntime_summary.json\nworlds/*.json</pre>
  </section>
</main>
<script>
const modules = {embedded_modules};
const navSummary = {embedded_nav_summary};
const worlds = {embedded_worlds};
const agentSummary = document.getElementById('agentSummary');
agentSummary.innerHTML = Object.entries(navSummary).map(([key, value]) => `
  <div class=\"card\">
    <strong>${{key}}</strong>
    <div class=\"small\">successes: ${{value.successes}} / ${{value.layouts}}</div>
    <div class=\"small\">avg path steps: ${{value.avg_path_steps}}</div>
  </div>
`).join('');
const worldGrid = document.getElementById('worldGrid');
worldGrid.innerHTML = worlds.map(world => `
  <div class=\"card\">
    <strong>${{world.name}}</strong>
    <div class=\"small\">${{world.grid_w}}×${{world.grid_h}} modules · unique=${{world.unique_states}}</div>
    <div class=\"small\">attempts=${{world.attempts}}</div>
    <div style=\"margin-top:10px\"><img src=\"${{world.base_png}}\" alt=\"${{world.name}} base\"></div>
    <div class=\"small\" style=\"margin-top:8px\">
      <a href=\"${{world.base_png}}\">base</a> ·
      <a href=\"${{world.agents.npc.render_png}}\">npc</a> ·
      <a href=\"${{world.agents.turtle.render_png}}\">turtle</a> ·
      <a href=\"${{world.agents.drone.render_png}}\">drone</a> ·
      <a href=\"${{world.json_path}}\">json</a>
    </div>
  </div>
`).join('');
</script>
</body>
</html>
"""
    (runtime_root / "runtime_viewer.html").write_text(html, encoding="utf-8")


def build_runtime(base_root: Path, runtime_root: Path):
    ensure_base_output(base_root)
    runtime_root.mkdir(parents=True, exist_ok=True)
    layout_records = load_layout_catalog(base_root)
    modules, compat = build_module_library(layout_records)

    module_library_path = runtime_root / "module_library.json"
    module_library_path.write_text(json.dumps(modules, indent=2), encoding="utf-8")
    build_module_socket_sheet(modules, runtime_root)

    nav_catalog, nav_summary = build_navigation_catalog(layout_records, runtime_root)
    (runtime_root / "navigation_catalog.json").write_text(json.dumps(nav_catalog, indent=2), encoding="utf-8")
    (runtime_root / "agent_profiles.json").write_text(json.dumps({
        name: {
            "name": profile["name"],
            "strategy": profile["strategy"],
            "base_passable": sorted(profile["base_passable"]),
            "tile_costs": profile["tile_costs"],
            "overlay_costs": profile["overlay_costs"],
        }
        for name, profile in AGENT_PROFILES.items()
    }, indent=2), encoding="utf-8")

    world_records = build_worlds(modules, compat, runtime_root)
    runtime_summary = {
        "layout_count": len(layout_records),
        "module_count": len(modules),
        "module_size": MODULE_SIZE,
        "module_slots": MODULE_SLOTS,
        "agent_summary": nav_summary,
        "world_presets": [
            {
                "name": world["name"],
                "grid_w": world["grid_w"],
                "grid_h": world["grid_h"],
                "unique_states": world["unique_states"],
                "attempts": world["attempts"],
            }
            for world in world_records
        ],
    }
    (runtime_root / "runtime_summary.json").write_text(json.dumps(runtime_summary, indent=2), encoding="utf-8")
    build_viewer(modules, nav_catalog, nav_summary, world_records, runtime_root)

    readme = f"""# Low-End Runtime Extension

This package extends the base 100-layout tile generator with two new systems:

- agent navigation over every generated layout
- a low-end socket-WFC that composes the full 100-layout library into larger worlds

## Design goals

- data first: all runtime behavior is derived from exported tile data
- low-end friendly: small standardized modules, 4 edge sockets, bitset propagation, 4-neighbor movement
- no external art assets: all renders are procedural

## Files

- `module_library.json` — standardized 16×16 modules derived from all 100 source layouts
- `navigation_catalog.json` — per-layout agent results and render references
- `agent_profiles.json` — movement rules and costs
- `runtime_summary.json` — aggregate stats
- `module_socket_sheet.png` — visual summary of module sockets
- `agents/` — path overlays for each layout and agent type
- `worlds/` — WFC outputs and their navigation overlays
- `runtime_viewer.html` — local browser viewer

## Runtime model

Each of the 100 layouts becomes one WFC state.
Each state is downsampled to a 16×16 logic module.
Each side stores a 4-slot socket mask based on the nearest reachable edge anchors.
Adjacency is allowed when neighboring side masks overlap.

The `all100_unique` preset fills a 10×10 world using all 100 source layouts exactly once.

## Rebuild

```bash
python tilegen_lowend_runtime.py
```
"""
    (runtime_root / "README.md").write_text(readme, encoding="utf-8")
    return runtime_summary


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    base_root = script_dir / "tilegen_output"
    runtime_root = script_dir / "tilegen_runtime_output"
    summary = build_runtime(base_root, runtime_root)

    bundle_path = script_dir / "tilegen_full_stack_bundle.zip"
    if bundle_path.exists():
        bundle_path.unlink()
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in (script_dir / "tilegen_system.py", script_dir / "tilegen_lowend_runtime.py"):
            if path.exists():
                zf.write(path, arcname=path.name)
        for root in (base_root, runtime_root):
            for path in root.rglob("*"):
                if path.is_file():
                    zf.write(path, arcname=path.relative_to(script_dir))
    print(json.dumps(summary, indent=2))
    print(f"Bundle: {bundle_path}")
