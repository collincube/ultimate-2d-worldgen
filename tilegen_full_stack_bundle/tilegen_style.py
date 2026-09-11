"""Style registry and neighbourhood-aware rendering.

tilegen_system.draw_tile is a pure function of (tile, seed, cx, cy): it never
receives the map, so a tile physically cannot know its neighbours and every
surface renders identically wherever it sits. It also reads one module-global
PALETTE, so the visual language is welded to the geometry.

This module separates the two. A StyleProfile is data describing a presentation
language; the renderer derives a NeighbourhoodSignature from the map and lets
the style react to it. The same TileMap can therefore render as dark procedural
tiles, a blueprint, an inked parchment chart or a topographic survey without any
change to the generators.

Styles own appearance only. They never touch traversal or simulation state.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field as dc_field

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import tilegen_system as ts

DEFAULT_FONT = ImageFont.load_default()


# ------------------------------------------------------------- surface roles

# Styles address roles, not tile names, so a new tile type does not require
# touching every style.
SURFACE_ROLE = {
    'wall': 'solid', 'bookshelf': 'solid',
    'floor': 'ground', 'platform': 'ground', 'ladder_tile': 'ground',
    'courtyard': 'verdure', 'rubble': 'debris', 'bridge': 'span',
    'water': 'liquid', 'lava': 'molten', 'abyss': 'void',
}
ROLE_ORDER = ['solid', 'ground', 'verdure', 'debris', 'span', 'liquid', 'molten', 'void']


def role_of(tile):
    return SURFACE_ROLE.get(tile, 'ground')


class NeighbourhoodSignature:
    """Per-cell context: what each cell touches, and which fields surround it.

    Computed once for the whole map with array shifts rather than per-cell
    lookups, so it costs about the same as a single pass over the grid.
    """

    __slots__ = ('tiles', 'roles', 'h', 'w', '_role_masks')

    def __init__(self, tm):
        self.tiles = np.array(tm.base, dtype=object)
        self.h, self.w = self.tiles.shape
        self.roles = np.vectorize(role_of, otypes=[object])(self.tiles)
        self._role_masks = {r: (self.roles == r) for r in ROLE_ORDER}

    def role(self, name):
        return self._role_masks.get(name, np.zeros((self.h, self.w), dtype=bool))

    def same_tile(self, direction):
        """True where the neighbour in `direction` holds the same tile."""
        t = self.tiles
        out = np.zeros((self.h, self.w), dtype=bool)
        if direction == 'N':
            out[1:, :] = t[1:, :] == t[:-1, :]
        elif direction == 'S':
            out[:-1, :] = t[:-1, :] == t[1:, :]
        elif direction == 'W':
            out[:, 1:] = t[:, 1:] == t[:, :-1]
        elif direction == 'E':
            out[:, :-1] = t[:, :-1] == t[:, 1:]
        return out

    def touches(self, mask, direction):
        """True where the neighbour in `direction` is inside `mask`.

        Out-of-bounds counts as False, so map edges read as closed.
        """
        out = np.zeros((self.h, self.w), dtype=bool)
        if direction == 'N':
            out[1:, :] = mask[:-1, :]
        elif direction == 'S':
            out[:-1, :] = mask[1:, :]
        elif direction == 'W':
            out[:, 1:] = mask[:, :-1]
        elif direction == 'E':
            out[:, :-1] = mask[:, 1:]
        return out

    def touches_any(self, mask):
        return (self.touches(mask, 'N') | self.touches(mask, 'S') |
                self.touches(mask, 'W') | self.touches(mask, 'E'))

    def exposed(self, role='solid'):
        """Cells of `role` that face something other than their own role -
        the edges that want a bevel, lintel or damage line."""
        m = self.role(role)
        return m & ~(self.touches(m, 'N') & self.touches(m, 'S') &
                     self.touches(m, 'W') & self.touches(m, 'E'))


# -------------------------------------------------------------- style profile

@dataclass
class StyleProfile:
    """Data describing a presentation language. No authoritative world state."""
    name: str
    version: str = '1'
    background: tuple = (18, 20, 24)
    colors: dict = dc_field(default_factory=dict)
    role_colors: dict = dc_field(default_factory=dict)
    tile_painter: object = None          # optional per-tile PIL painter
    texture_noise: int = 0               # per-cell colour jitter
    bevel: int = 0                       # bevel thickness in px on solid edges
    bevel_light: int = 26
    bevel_dark: int = 22
    outline: tuple = None                # colour of the solid/open seam
    outline_px: int = 1
    shoreline: tuple = None              # rim where liquid meets land
    shoreline_px: int = 1
    glow: dict = None                    # {'source','color','radius','strength'}
    hatch: dict = None                   # {'color','spacing','angle'}
    grid: tuple = None                   # faint cell grid colour
    contour: dict = None                 # {'field','levels','color'}
    field_shade: dict = None             # {'field','amount','warm'}
    span_shadow: tuple = None            # shadow under bridges over void
    overlay_color: tuple = (235, 235, 240)
    use_native_overlays: bool = False
    label_color: tuple = (220, 220, 225)
    label_bar: tuple = (24, 27, 33)
    border: tuple = (80, 84, 92)

    def color_for(self, tile):
        if tile in self.colors:
            return self.colors[tile]
        r = role_of(tile)
        if r in self.role_colors:
            return self.role_colors[r]
        return ts.PALETTE.get(tile, (128, 128, 128))


STYLES = {}


def register(style):
    STYLES[f"{style.name}.v{style.version}"] = style
    STYLES.setdefault(style.name, style)
    return style


# ------------------------------------------------------------------ profiles

register(StyleProfile(
    name='procedural_dark_tiles',
    background=(18, 20, 24),
    colors=dict(ts.PALETTE),
    tile_painter=ts.draw_tile,          # the original look, preserved exactly
    bevel=1, bevel_light=30, bevel_dark=26,
    shoreline=(120, 170, 210), shoreline_px=1,
    glow={'source': 'molten', 'color': (255, 150, 60), 'radius': 3.2, 'strength': 0.55},
    span_shadow=(8, 8, 12),
    use_native_overlays=True,
))

register(StyleProfile(
    name='architectural_blueprint',
    background=(16, 42, 92),
    role_colors={
        'solid': (16, 42, 92), 'ground': (22, 56, 116), 'verdure': (26, 66, 132),
        'debris': (20, 50, 104), 'span': (30, 74, 140),
        'liquid': (28, 78, 150), 'molten': (40, 70, 130), 'void': (11, 28, 64),
    },
    outline=(214, 232, 255), outline_px=1,
    hatch={'color': (86, 130, 200), 'spacing': 4, 'angle': 45},
    grid=(30, 62, 122),
    contour={'field': 'elevation', 'levels': 7, 'color': (74, 118, 190)},
    overlay_color=(226, 240, 255),
    label_color=(214, 232, 255), label_bar=(12, 32, 74), border=(120, 164, 226),
))

register(StyleProfile(
    name='parchment_ink',
    background=(226, 208, 172),
    role_colors={
        'solid': (196, 172, 132), 'ground': (232, 216, 183), 'verdure': (196, 202, 150),
        'debris': (208, 190, 158), 'span': (188, 154, 106),
        'liquid': (156, 182, 186), 'molten': (198, 132, 92), 'void': (150, 134, 108),
    },
    texture_noise=7,
    outline=(74, 56, 38), outline_px=1,
    hatch={'color': (150, 126, 92), 'spacing': 3, 'angle': 45},
    shoreline=(120, 146, 152), shoreline_px=1,
    contour={'field': 'elevation', 'levels': 6, 'color': (176, 152, 116)},
    overlay_color=(70, 52, 34),
    label_color=(70, 52, 34), label_bar=(208, 188, 150), border=(120, 96, 64),
))

register(StyleProfile(
    name='obsidian_gold',
    background=(10, 9, 12),
    role_colors={
        'solid': (24, 21, 27), 'ground': (48, 42, 52), 'verdure': (62, 74, 58),
        'debris': (40, 35, 42), 'span': (120, 92, 40),
        'liquid': (30, 52, 80), 'molten': (196, 96, 26), 'void': (6, 5, 8),
    },
    texture_noise=10,
    bevel=1, bevel_light=42, bevel_dark=14,
    outline=(198, 158, 74), outline_px=1,
    glow={'source': 'molten', 'color': (255, 168, 62), 'radius': 4.5, 'strength': 0.8},
    field_shade={'field': 'elevation', 'amount': 0.30},
    span_shadow=(4, 3, 5),
    overlay_color=(226, 190, 110),
    label_color=(214, 178, 96), label_bar=(18, 15, 20), border=(120, 96, 44),
))

register(StyleProfile(
    name='neon_circuit',
    background=(6, 8, 18),
    role_colors={
        'solid': (10, 14, 30), 'ground': (16, 26, 52), 'verdure': (18, 52, 48),
        'debris': (14, 20, 40), 'span': (40, 30, 70),
        'liquid': (12, 40, 78), 'molten': (120, 20, 70), 'void': (3, 4, 10),
    },
    outline=(0, 236, 255), outline_px=1,
    grid=(12, 26, 54),
    glow={'source': 'molten', 'color': (255, 40, 160), 'radius': 5.0, 'strength': 0.9},
    field_shade={'field': 'elevation', 'amount': 0.22},
    overlay_color=(0, 240, 255),
    label_color=(0, 236, 255), label_bar=(8, 12, 26), border=(0, 140, 170),
))

register(StyleProfile(
    name='satellite_topographic',
    background=(24, 32, 28),
    role_colors={
        'solid': (108, 100, 88), 'ground': (128, 142, 96), 'verdure': (78, 116, 66),
        'debris': (146, 134, 112), 'span': (150, 130, 96),
        'liquid': (46, 94, 140), 'molten': (176, 88, 40), 'void': (28, 30, 34),
    },
    texture_noise=9,
    shoreline=(206, 198, 154), shoreline_px=1,
    contour={'field': 'elevation', 'levels': 12, 'color': (60, 56, 44)},
    field_shade={'field': 'elevation', 'amount': 0.55, 'warm': True},
    overlay_color=(250, 250, 240),
    label_color=(236, 236, 224), label_bar=(20, 26, 22), border=(90, 96, 82),
))


# ------------------------------------------------------------------ rendering

def _px(base, t):
    """Scale an edge thickness with tile size so bevels and rims stay visible
    at 32px tiles without swamping 6px ones."""
    return max(1, int(round(base * t / 10.0)))


def _upscale(mask, t):
    return np.repeat(np.repeat(mask, t, axis=0), t, axis=1)


def _edge_template(t, side, k):
    """(t, t) boolean band along one side of a single cell."""
    tpl = np.zeros((t, t), dtype=bool)
    k = max(1, min(k, t))
    if side == 'N':
        tpl[:k, :] = True
    elif side == 'S':
        tpl[-k:, :] = True
    elif side == 'W':
        tpl[:, :k] = True
    elif side == 'E':
        tpl[:, -k:] = True
    return tpl


def _band(cell_mask, side, t, k, shape):
    """Pixel-space band on one side of every cell in cell_mask."""
    h, w = cell_mask.shape
    return _upscale(cell_mask, t) & np.tile(_edge_template(t, side, k), (h, w))[:shape[0], :shape[1]]


def _shade(rgb, arr, amount):
    """Multiply-shade an RGB image by a 0..1 array."""
    f = (1.0 + (arr - 0.5) * 2.0 * amount)[..., None]
    return np.clip(rgb.astype(np.float32) * f, 0, 255).astype(np.uint8)


def render_map(tm, spec=None, style='procedural_dark_tiles', tile_px=None,
               max_map_px=360, label=None, show_label=True):
    """Render a TileMap through a style profile, with neighbourhood context."""
    st = STYLES[style] if isinstance(style, str) else style
    spec = spec or {}
    h, w = tm.h, tm.w
    if tile_px is None:
        tile_px = max(4, min(64, max_map_px // max(w, h)))
        if w <= 9:
            tile_px = max(tile_px, 24 if w == 9 else 56)
    t = int(tile_px)
    seed = spec.get('id', 0) * 7919 + spec.get('seed', 0)

    sig = NeighbourhoodSignature(tm)
    fields = getattr(tm, 'fields', {}) or {}
    H, W = h * t, w * t

    # ---- base colours -------------------------------------------------
    if st.tile_painter is not None:
        img = Image.new('RGB', (W, H), st.background)
        d = ImageDraw.Draw(img)
        for y in range(h):
            row = tm.base[y]
            for x in range(w):
                st.tile_painter(d, x * t, y * t, t, row[x], seed, x, y)
        rgb = np.array(img, dtype=np.uint8)
    else:
        cell_rgb = np.zeros((h, w, 3), dtype=np.float32)
        for tile in np.unique(sig.tiles):
            cell_rgb[sig.tiles == tile] = st.color_for(tile)
        if st.texture_noise:
            jitter = np.zeros((h, w), dtype=np.float32)
            for cy in range(h):
                for cx in range(w):
                    jitter[cy, cx] = ts.noise(seed, cx, cy) - 0.5
            cell_rgb += (jitter * 2.0 * st.texture_noise)[..., None]
        cell_rgb = np.clip(cell_rgb, 0, 255).astype(np.uint8)
        rgb = _upscale(cell_rgb, t)

    # ---- field shading -------------------------------------------------
    if st.field_shade:
        fname = st.field_shade.get('field', 'elevation')
        if fname in fields:
            f = np.asarray(fields[fname], dtype=np.float32)
            f = (f - f.min()) / max(float(f.max() - f.min()), 1e-6)
            rgb = _shade(rgb, _upscale(f, t), st.field_shade.get('amount', 0.3))
            if st.field_shade.get('warm'):
                warm = _upscale(f, t)[..., None] * np.array([26, 8, -14], dtype=np.float32)
                rgb = np.clip(rgb.astype(np.float32) + warm, 0, 255).astype(np.uint8)

    solid = sig.role('solid')
    liquid = sig.role('liquid')
    molten = sig.role('molten')
    void = sig.role('void')
    span = sig.role('span')
    open_cells = ~solid

    # ---- bevels on exposed solid faces ---------------------------------
    if st.bevel:
        for side, delta in (('N', st.bevel_light), ('W', st.bevel_light),
                            ('S', -st.bevel_dark), ('E', -st.bevel_dark)):
            face = solid & sig.touches(open_cells, side)
            if face.any():
                band = _band(face, side, t, _px(st.bevel, t), (H, W))
                rgb[band] = np.clip(rgb[band].astype(np.int16) + delta, 0, 255).astype(np.uint8)

    # ---- shoreline where liquid meets land ------------------------------
    if st.shoreline is not None:
        land = ~liquid & ~void
        for side in ('N', 'S', 'W', 'E'):
            rim = liquid & sig.touches(land, side)
            if rim.any():
                band = _band(rim, side, t, _px(st.shoreline_px, t), (H, W))
                rgb[band] = (np.array(st.shoreline, dtype=np.float32) * 0.78 +
                             rgb[band].astype(np.float32) * 0.22).astype(np.uint8)

    # ---- heat spill from molten cells ------------------------------------
    if st.glow and molten.any():
        import tilegen_fields as tf
        src = {'molten': molten, 'liquid': liquid, 'void': void}.get(
            st.glow.get('source', 'molten'), molten)
        if src.any():
            dist = tf.distance_to_mask(src).a
            fall = np.clip(1.0 - dist / max(st.glow.get('radius', 4.0), 1e-6), 0, 1) ** 2
            fall = fall * st.glow.get('strength', 0.6)
            fall[src] = 0.0                     # light the surroundings, not the source
            add = _upscale(fall, t)[..., None] * np.array(st.glow['color'], dtype=np.float32)
            rgb = np.clip(rgb.astype(np.float32) + add, 0, 255).astype(np.uint8)

    # ---- shadow under spans crossing a void ------------------------------
    if st.span_shadow is not None and span.any():
        under = void & sig.touches(span, 'N')
        if under.any():
            band = _band(under, 'N', t, max(2, t // 3), (H, W))
            rgb[band] = (rgb[band].astype(np.float32) * 0.55 +
                         np.array(st.span_shadow, dtype=np.float32) * 0.45).astype(np.uint8)

    img = Image.fromarray(rgb, 'RGB')
    d = ImageDraw.Draw(img)

    # ---- hatching inside solids -----------------------------------------
    if st.hatch:
        sp = max(3, st.hatch.get('spacing', 4) * max(1, t // 3))
        col = st.hatch['color']
        step = sp
        for i in range(-H, W + H, step):
            d.line([(i, 0), (i + H, H)], fill=col, width=1)
        hatch_layer = np.array(img, dtype=np.uint8)
        keep = _upscale(solid, t)
        base = rgb.copy()
        base[keep] = hatch_layer[keep]
        img = Image.fromarray(base, 'RGB')
        d = ImageDraw.Draw(img)

    # ---- cell grid --------------------------------------------------------
    if st.grid and t >= 4:
        for x in range(0, W + 1, t):
            d.line([(x, 0), (x, H)], fill=st.grid)
        for y in range(0, H + 1, t):
            d.line([(0, y), (W, y)], fill=st.grid)

    # ---- structural outline on the solid/open seam ------------------------
    if st.outline is not None:
        arr = np.array(img, dtype=np.uint8)
        oc = np.array(st.outline, dtype=np.uint8)
        px = _px(st.outline_px, t) if t >= 14 else max(1, st.outline_px)
        for side in ('N', 'S', 'W', 'E'):
            face = solid & sig.touches(open_cells, side)
            if face.any():
                arr[_band(face, side, t, px, (H, W))] = oc
        img = Image.fromarray(arr, 'RGB')
        d = ImageDraw.Draw(img)

    # ---- contour lines from a continuous field ----------------------------
    if st.contour and st.contour.get('field', 'elevation') in fields:
        import tilegen_fields as tf
        f = tf.Field(np.asarray(fields[st.contour['field']], dtype=np.float32)).normalized()
        # resample to pixel resolution first: taking contours per cell and then
        # upscaling turns every isoline into a row of dashes
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        hi = tf.Field(f.sample((xx + 0.5) / t - 0.5, (yy + 0.5) / t - 0.5)).normalized()
        arr = np.array(img, dtype=np.uint8)
        cc = np.array(st.contour['color'], dtype=np.float32)
        n = max(2, st.contour.get('levels', 6))
        alpha = st.contour.get('alpha', 0.55)
        for i in range(1, n):
            cm = hi.contour_mask(i / n)
            arr[cm] = (arr[cm].astype(np.float32) * (1 - alpha) + cc * alpha).astype(np.uint8)
        img = Image.fromarray(arr, 'RGB')
        d = ImageDraw.Draw(img)

    # ---- overlays ---------------------------------------------------------
    for y in range(h):
        for x in range(w):
            o = tm.overlay[y][x]
            if not o:
                continue
            if st.use_native_overlays:
                ts.draw_overlay(d, x * t, y * t, t, o, seed, x, y)
            else:
                _draw_overlay_glyph(d, x * t, y * t, t, o, st.overlay_color)

    # ---- frame and label --------------------------------------------------
    if show_label:
        margin, label_h = 8, 20
        canvas = Image.new('RGB', (W + margin * 2, H + margin * 2 + label_h), st.background)
        canvas.paste(img, (margin, margin))
        cd = ImageDraw.Draw(canvas)
        cd.rectangle([margin - 2, margin - 2, margin + W + 1, margin + H + 1],
                     outline=st.border)
        cd.rectangle([0, canvas.height - label_h, canvas.width, canvas.height],
                     fill=st.label_bar)
        text = label if label is not None else f"{spec.get('id', 0):03d}"
        cd.text((8, canvas.height - label_h + 4), text, font=DEFAULT_FONT,
                fill=st.label_color)
        return canvas
    return img


def _draw_overlay_glyph(d, x0, y0, s, overlay, color):
    """Style-neutral overlay marks for profiles that do not use the native set."""
    mx, my = x0 + s // 2, y0 + s // 2
    q = max(1, s // 4)
    if overlay in ('door', 'locked'):
        d.rectangle([mx - q, my - q, mx + q, my + q], outline=color)
        if overlay == 'locked':
            d.line([mx - q, my - q, mx + q, my + q], fill=color)
    elif overlay in ('stair', 'ladder', 'elevator', 'shaft'):
        for i in range(3):
            yy = my - q + i * max(1, q)
            d.line([mx - q, yy, mx + q, yy], fill=color)
    elif overlay in ('teleport', 'rotate', 'shift'):
        d.ellipse([mx - q, my - q, mx + q, my + q], outline=color)
    elif overlay == 'pillar':
        d.ellipse([mx - q, my - q, mx + q, my + q], fill=color)
    elif overlay in ('hidden', 'deadend'):
        d.line([mx - q, my - q, mx + q, my + q], fill=color)
        d.line([mx - q, my + q, mx + q, my - q], fill=color)
    else:
        d.point((mx, my), fill=color)


def contact_sheet(images, labels, cols=5, cell_w=240, cell_h=262, bg=(12, 13, 16),
                  fg=(210, 212, 218)):
    """Grid overview of a set of renders."""
    rows = math.ceil(len(images) / cols)
    sheet = Image.new('RGB', (cols * cell_w, rows * cell_h), bg)
    d = ImageDraw.Draw(sheet)
    for i, (im, lab) in enumerate(zip(images, labels)):
        r, c = divmod(i, cols)
        thumb = im.copy()
        thumb.thumbnail((cell_w - 16, cell_h - 34))
        x = c * cell_w + (cell_w - thumb.width) // 2
        y = r * cell_h + 8
        sheet.paste(thumb, (x, y))
        d.text((c * cell_w + 8, r * cell_h + cell_h - 22), lab[:38],
               font=DEFAULT_FONT, fill=fg)
    return sheet
