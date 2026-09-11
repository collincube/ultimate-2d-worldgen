"""Runtime helpers for scheduling, culling, navigation, and chunk loading.

PathScheduler staggers route updates; ViewportCuller assigns distance tiers;
BakedNavigation caches flow fields; ChunkStreamer loads tiles in background
workers; ResolutionScaler adjusts tile size against a frame-time budget.
The headless `simulate` benchmark counts scheduled work over a fixed workload.

Inspired by the pathfinding and culling techniques in Blargis's optimisation
presentation. The implementations here operate on 2D tile maps.
"""
from __future__ import annotations

import heapq
import math
import queue
import threading
import time
from collections import defaultdict, deque

import numpy as np

import tilegen_fields as tf
import tilegen_grammars as tg


# ------------------------------------------------------- pathfinding budget

class PathScheduler:
    """Throttled, staggered path recomputation.

    Recomputing every agent's route every tick is the usual cost sink, and
    recomputing them all on the *same* tick is worse: it produces a periodic
    spike instead of a flat load. Each agent gets a deterministic phase offset
    so the work spreads evenly across the interval.
    """

    def __init__(self, interval=20, jitter=6, seed=0):
        self.interval = max(1, interval)
        self.jitter = jitter
        self.rng = np.random.default_rng(tf.seed_stream(seed, 'runtime.paths'))
        self.phase = {}
        self.recomputes = 0
        self.ticks = 0

    def register(self, agent_id):
        if agent_id not in self.phase:
            span = min(self.jitter, self.interval - 1)
            offset = int(self.rng.integers(-span, span + 1)) if span > 0 else 0
            self.phase[agent_id] = offset % self.interval
        return self.phase[agent_id]

    def due(self, agent_id, tick):
        """True when this agent should recompute on this tick."""
        self.register(agent_id)
        return (tick + self.phase[agent_id]) % self.interval == 0

    def tick(self, agent_ids, tick):
        self.ticks += 1
        due = [a for a in agent_ids if self.due(a, tick)]
        self.recomputes += len(due)
        return due

    def load_profile(self, agent_ids, ticks=120):
        """Count route updates scheduled on each tick."""
        return [len([a for a in agent_ids if self.due(a, t)]) for t in range(ticks)]


# -------------------------------------------------------------- LOD culling

LOD_ACTIVE, LOD_SIMULATED, LOD_FROZEN = 0, 1, 2
LOD_NAMES = {LOD_ACTIVE: 'active', LOD_SIMULATED: 'simulated', LOD_FROZEN: 'frozen'}


class ViewportCuller:
    """Distance rings that decide how much of an entity is updated.

    Inside the viewport entities are active; a surrounding ring receives
    reduced updates; distant entities are marked frozen. Callers decide which
    work to perform for each tier.
    """

    def __init__(self, view_w, view_h, simulate_margin=1.5, freeze_margin=3.0):
        self.view_w = view_w
        self.view_h = view_h
        self.simulate_margin = simulate_margin
        self.freeze_margin = freeze_margin

    def classify(self, camera, positions):
        """camera: (cx, cy). positions: (n, 2). Returns an LOD code per entity."""
        pos = np.asarray(positions, dtype=np.float32).reshape(-1, 2)
        dx = np.abs(pos[:, 0] - camera[0]) / max(self.view_w / 2.0, 1e-6)
        dy = np.abs(pos[:, 1] - camera[1]) / max(self.view_h / 2.0, 1e-6)
        d = np.maximum(dx, dy)
        lod = np.full(len(pos), LOD_FROZEN, dtype=np.int8)
        lod[d <= self.freeze_margin] = LOD_SIMULATED
        lod[d <= 1.0] = LOD_ACTIVE
        return lod

    def visible_bounds(self, camera, w, h):
        x0 = int(max(0, camera[0] - self.view_w / 2))
        y0 = int(max(0, camera[1] - self.view_h / 2))
        return (x0, y0,
                int(min(w, x0 + self.view_w)), int(min(h, y0 + self.view_h)))

    def counts(self, camera, positions):
        lod = self.classify(camera, positions)
        return {LOD_NAMES[k]: int((lod == k).sum())
                for k in (LOD_ACTIVE, LOD_SIMULATED, LOD_FROZEN)}


# ---------------------------------------------------------- baked navigation

class BakedNavigation:
    """Precomputed flow fields toward fixed goals.

    The analogue of baked lighting: pay once at load, then every agent heading
    for that goal reads a direction from a lookup instead of running its own
    search. Costs one integer field per goal and turns per-agent pathfinding
    into an array read.
    """

    def __init__(self, passable):
        self.passable = np.asarray(passable, dtype=bool)
        self.h, self.w = self.passable.shape
        self.fields = {}

    def bake(self, goal_id, goal_xy):
        """Dijkstra outward from the goal over 4-connected passable cells."""
        gx, gy = int(goal_xy[0]), int(goal_xy[1])
        INF = np.float32(np.inf)
        dist = np.full((self.h, self.w), INF, dtype=np.float32)
        if not (0 <= gx < self.w and 0 <= gy < self.h) or not self.passable[gy, gx]:
            self.fields[goal_id] = dist
            return dist
        dist[gy, gx] = 0
        frontier = deque([(gx, gy)])
        while frontier:
            x, y = frontier.popleft()
            d = dist[y, x] + 1
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if 0 <= nx < self.w and 0 <= ny < self.h and \
                        self.passable[ny, nx] and dist[ny, nx] > d:
                    dist[ny, nx] = d
                    frontier.append((nx, ny))
        self.fields[goal_id] = dist
        return dist

    def step(self, goal_id, pos):
        """One step downhill toward the goal, or None if there is no route."""
        dist = self.fields.get(goal_id)
        if dist is None:
            return None
        x, y = int(pos[0]), int(pos[1])
        if not (0 <= x < self.w and 0 <= y < self.h) or not np.isfinite(dist[y, x]):
            return None
        best, bestd = None, dist[y, x]
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < self.w and 0 <= ny < self.h and dist[ny, nx] < bestd:
                bestd = dist[ny, nx]
                best = (nx, ny)
        return best

    def route_length(self, goal_id, pos):
        dist = self.fields.get(goal_id)
        if dist is None:
            return math.inf
        x, y = int(pos[0]), int(pos[1])
        if not (0 <= x < self.w and 0 <= y < self.h):
            return math.inf
        return float(dist[y, x])


# -------------------------------------------------------------- chunk stream

class ChunkStreamer:
    """Background chunk prefetch, so map loading never blocks the frame loop.

    A worker thread materialises chunks around the camera while the main loop
    keeps running; `ready` only reports chunks that have finished. The point is
    the main thread never waits, exactly as with threaded level loading.
    """

    def __init__(self, tilemap, chunk=32, radius=1, workers=1):
        self.tilemap = tilemap
        self.chunk = chunk
        self.radius = radius
        self.cols = math.ceil(tilemap.w / chunk)
        self.rows = math.ceil(tilemap.h / chunk)
        self._ready = {}
        self._pending = set()
        self._queue = queue.Queue()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads = [threading.Thread(target=self._work, daemon=True)
                         for _ in range(max(1, workers))]
        for t in self._threads:
            t.start()
        self.loads = 0

    def _work(self):
        while not self._stop.is_set():
            try:
                key = self._queue.get(timeout=0.05)
            except queue.Empty:
                continue
            if key is None:
                break
            data = self._materialise(key)
            with self._lock:
                self._ready[key] = data
                self._pending.discard(key)
                self.loads += 1
            self._queue.task_done()

    def _materialise(self, key):
        cx, cy = key
        x0, y0 = cx * self.chunk, cy * self.chunk
        x1 = min(self.tilemap.w, x0 + self.chunk)
        y1 = min(self.tilemap.h, y0 + self.chunk)
        rows = [self.tilemap.base[y][x0:x1] for y in range(y0, y1)]
        return {'origin': (x0, y0), 'size': (x1 - x0, y1 - y0), 'rows': rows}

    def request_around(self, camera):
        cx = int(camera[0]) // self.chunk
        cy = int(camera[1]) // self.chunk
        wanted = set()
        for dy in range(-self.radius, self.radius + 1):
            for dx in range(-self.radius, self.radius + 1):
                x, y = cx + dx, cy + dy
                if 0 <= x < self.cols and 0 <= y < self.rows:
                    wanted.add((x, y))
        with self._lock:
            new = [k for k in wanted
                   if k not in self._ready and k not in self._pending]
            self._pending.update(new)
        for k in new:
            self._queue.put(k)
        return wanted

    def ready(self, keys=None):
        with self._lock:
            if keys is None:
                return dict(self._ready)
            return {k: self._ready[k] for k in keys if k in self._ready}

    def wait_idle(self, timeout=5.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if not self._pending:
                    return True
            time.sleep(0.01)
        return False

    def close(self):
        self._stop.set()
        for _ in self._threads:
            self._queue.put(None)
        for t in self._threads:
            t.join(timeout=1.0)


# ------------------------------------------------------- resolution scaling

class ResolutionScaler:
    """Adaptive tile size driven by measured frame cost.

    Drop the internal resolution when frames run long and recover it when they
    do not, with hysteresis so it does not oscillate on a noisy signal.
    """

    def __init__(self, base_tile_px=8, min_tile_px=2, target_ms=16.7,
                 window=12, tolerance=0.15):
        self.base = base_tile_px
        self.min = min_tile_px
        self.target = target_ms
        self.window = window
        self.tolerance = tolerance
        self.tile_px = base_tile_px
        self.samples = deque(maxlen=window)
        self.changes = 0

    def record(self, frame_ms):
        self.samples.append(frame_ms)
        if len(self.samples) < self.window:
            return self.tile_px
        avg = sum(self.samples) / len(self.samples)
        if avg > self.target * (1 + self.tolerance) and self.tile_px > self.min:
            self.tile_px = max(self.min, self.tile_px - 1)
            self.changes += 1
            self.samples.clear()
        elif avg < self.target * (1 - self.tolerance) and self.tile_px < self.base:
            self.tile_px = min(self.base, self.tile_px + 1)
            self.changes += 1
            self.samples.clear()
        return self.tile_px

    @property
    def scale(self):
        return self.tile_px / float(self.base)


# ------------------------------------------------------------- measurement

def simulate(tilemap, agents=200, ticks=180, interval=20, jitter=6,
             view=(48, 32), seed=0, goals=3):
    """Headless run comparing a naive loop against the budgeted one.

    Returns pathfinding, animation, and per-tick work counts for comparison.
    """
    passable = tg.passable_mask(tilemap)
    ys, xs = np.nonzero(passable)
    if len(xs) == 0:
        raise ValueError('map has no traversable cells')
    rng = np.random.default_rng(tf.seed_stream(seed, 'runtime.sim'))

    pick = rng.choice(len(xs), size=min(agents, len(xs)), replace=False)
    pos = np.stack([xs[pick], ys[pick]], axis=1).astype(np.float32)
    gpick = rng.choice(len(xs), size=min(goals, len(xs)), replace=False)

    baked = BakedNavigation(passable)
    t0 = time.perf_counter()
    for i, gi in enumerate(gpick):
        baked.bake(f'goal{i}', (xs[gi], ys[gi]))
    bake_ms = (time.perf_counter() - t0) * 1000.0

    culler = ViewportCuller(view[0], view[1])
    sched = PathScheduler(interval=interval, jitter=jitter, seed=seed)
    ids = [f'a{i}' for i in range(len(pos))]

    camera = np.array([tilemap.w / 2.0, tilemap.h / 2.0], dtype=np.float32)
    naive_paths = 0
    budget_paths = 0
    budget_anim = 0
    naive_anim = 0
    peak_naive = 0
    peak_budget = 0

    for t in range(ticks):
        camera[0] = tilemap.w / 2 + math.cos(t / 30.0) * tilemap.w / 4
        camera[1] = tilemap.h / 2 + math.sin(t / 37.0) * tilemap.h / 4
        lod = culler.classify(camera, pos)

        # naive: everyone paths and animates every tick
        naive_paths += len(pos)
        naive_anim += len(pos)
        peak_naive = max(peak_naive, len(pos))

        # budgeted: only non-frozen agents, and only on their own phase
        awake = [ids[i] for i in range(len(pos)) if lod[i] != LOD_FROZEN]
        due = sched.tick(awake, t)
        budget_paths += len(due)
        budget_anim += int((lod == LOD_ACTIVE).sum())
        peak_budget = max(peak_budget, len(due))

        for i in np.flatnonzero(lod != LOD_FROZEN):
            nxt = baked.step(f'goal{int(i) % len(gpick)}', pos[i])
            if nxt:
                pos[i] = nxt

    return {
        'agents': len(pos), 'ticks': ticks,
        'bake_ms': round(bake_ms, 2),
        'naive_path_calls': naive_paths,
        'budget_path_calls': budget_paths,
        'path_reduction': round(1 - budget_paths / max(naive_paths, 1), 4),
        'naive_animations': naive_anim,
        'budget_animations': budget_anim,
        'animation_reduction': round(1 - budget_anim / max(naive_anim, 1), 4),
        'peak_naive_per_tick': peak_naive,
        'peak_budget_per_tick': peak_budget,
        'spike_reduction': round(1 - peak_budget / max(peak_naive, 1), 4),
    }
