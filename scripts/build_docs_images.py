"""Rebuild the docs/ illustrations that come straight from code.
usage: python scripts/build_docs_images.py [--only overworld overworlds autotiles] [--out docs]

Writes docs/overworld.webp (the README hero), docs/overworlds.webp (four climates
and shapes) and docs/autotiles.webp (generated autotile sheets and a blob-tiled
cave). The smooth overworld maps are lossy WebP at high quality (about a fifth of
the PNG size); the pixel-art autotiles are lossless WebP. Deterministic for a
given numpy/Pillow/libwebp build; runs from any directory. showcase.png,
worlds.png and layers.png are hand-assembled and not rebuilt here.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'tilegen_full_stack_bundle'
sys.path.insert(0, str(BUNDLE))

from PIL import Image  # noqa: E402

import tilegen_autotile as ta  # noqa: E402
import tilegen_grammars as tg  # noqa: E402
import tilegen_grammars2  # noqa: E402,F401  (registers cellular caves)
import tilegen_overworld as ow  # noqa: E402

BG = (14, 16, 20)


def _save(img, path, lossless=False, quality=92):
    """WebP: lossy at high quality for smooth maps, lossless for pixel art."""
    img = img.convert('RGB')
    if lossless:
        img.save(path, 'WEBP', lossless=True, quality=100, method=6)
    else:
        img.save(path, 'WEBP', quality=quality, method=6)
    print(f"  {path.relative_to(ROOT)}  {img.size[0]}x{img.size[1]}  {path.stat().st_size // 1024} KB")


def overworld(out):
    tm = ow.gen_overworld({'size': 160, 'seed': 5}, None)
    _save(ow.render_overworld(tm, upscale=5), out / 'overworld.webp')


def overworlds(out):
    variants = [
        ('Tropical archipelago', {'shape': 'archipelago', 'climate': 'tropical', 'seed': 11}),
        ('Highlands and a rain shadow', {'shape': 'highlands', 'seed': 5}),
        ('Peninsula, pole to pole, east wind', {'shape': 'peninsula', 'climate': 'pole_to_pole',
                                                'wind': 'east', 'seed': 4}),
        ('Polar realm, east wind', {'climate': 'polar', 'wind': 'east', 'seed': 2}),
    ]
    tiles = [ow.render_overworld(ow.gen_overworld({'size': 128, **spec}, None), upscale=3,
                                 legend=False, title=name) for name, spec in variants]
    gap = 12
    w, h = tiles[0].size
    canvas = Image.new('RGB', (2 * w + 3 * gap, 2 * h + 3 * gap), BG)
    for i, t in enumerate(tiles):
        canvas.paste(t, (gap + (i % 2) * (w + gap), gap + (i // 2) * (h + gap)))
    _save(canvas, out / 'overworlds.webp', quality=90)


def autotiles(out):
    sheets = ta.sheet_preview(scale=3, gap=12)
    cave = tg.GRAMMARS['ca_caves']({'id': 1, 'size': 26, 'seed': 5}, random.Random(5))
    renders = [ta.render_autotile(cave, skin, palette='classic', tile_px=10, scheme='blob', seed=3)
               for skin in ('cave', 'dungeon', 'shore')]
    gap = 12
    row_w = sum(r.width for r in renders) + gap * (len(renders) - 1)
    height = sheets.height + max(r.height for r in renders) + gap
    canvas = Image.new('RGB', (max(sheets.width, row_w + 2 * gap), height), BG)
    canvas.paste(sheets, ((canvas.width - sheets.width) // 2, 0))
    x = (canvas.width - row_w) // 2
    for r in renders:                       # pixel-exact: no resampling of pixel art
        canvas.paste(r, (x, sheets.height))
        x += r.width + gap
    _save(canvas, out / 'autotiles.webp', lossless=True)


JOBS = {'overworld': overworld, 'overworlds': overworlds, 'autotiles': autotiles}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--only', nargs='*', choices=sorted(JOBS), default=None)
    ap.add_argument('--out', default=str(ROOT / 'docs'))
    args = ap.parse_args(argv)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for name in args.only or JOBS:
        JOBS[name](out)


if __name__ == '__main__':
    main()
