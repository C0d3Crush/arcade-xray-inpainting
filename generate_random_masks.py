# -*- coding: utf-8 -*-
"""
generate_random_masks.py
────────────────────────────────────────────────────────────────────────────
Generates realistic vessel-like masks for ARCADE images by extracting real
vessel shapes from COCO annotations and randomly placing them — but NEVER
over real vessel locations annotated in the same COCO file.

This ensures the model learns to inpaint background, not reconstruct vessels.

Usage
─────
  python generate_random_masks.py \
      --annotations arcade/syntax/train/annotations/train.json \
      --images      arcade/syntax/train/images \
      --output      arcade/syntax/train/random_masks

  # Preview first mask
  python generate_random_masks.py ... --preview

Output
──────
  One mask per image, same filename, saved in --output directory.
  Masks are binary: 255 = to inpaint, 0 = background.
"""

import json
import argparse
import numpy as np
import cv2
from collections import defaultdict
from PIL import Image, ImageDraw
from pathlib import Path


STENOSIS_CATEGORY_NAME = 'stenosis'


def load_coco(ann_path):
    """
    Parse COCO JSON. Returns:
      shapes          — list of normalized [0,1] vessel polygon arrays (all images)
      filename_to_id  — dict mapping file_name → image_id
      id_to_info      — dict mapping image_id → {width, height, file_name}
      anns_by_image   — dict mapping image_id → list of annotation dicts
    """
    with open(ann_path) as f:
        coco = json.load(f)

    exclude_ids = {
        cat['id'] for cat in coco['categories']
        if cat['name'].lower() == STENOSIS_CATEGORY_NAME
    }

    id_to_info     = {img['id']: img for img in coco['images']}
    filename_to_id = {img['file_name']: img['id'] for img in coco['images']}

    anns_by_image = defaultdict(list)
    for ann in coco['annotations']:
        if ann['category_id'] not in exclude_ids:
            anns_by_image[ann['image_id']].append(ann)

    # Collect all vessel polygon shapes (normalized) as placement templates
    shapes = []
    for ann in coco['annotations']:
        if ann['category_id'] in exclude_ids:
            continue
        for poly in ann['segmentation']:
            pts = np.array(list(zip(poly[0::2], poly[1::2])), dtype=np.float32)
            if len(pts) >= 3:
                mn  = pts.min(axis=0)
                mx  = pts.max(axis=0)
                rng = mx - mn
                if rng[0] > 0 and rng[1] > 0:
                    shapes.append((pts - mn) / rng)

    print(f"  Loaded {len(shapes)} vessel shapes from {len(id_to_info)} images")
    return shapes, filename_to_id, id_to_info, anns_by_image


def make_vessel_mask(image_id, W, H, anns_by_image):
    """
    Rasterize real vessel annotations for one image into a binary mask.
    Returns uint8 array: 1 = real vessel (avoid), 0 = safe background.
    """
    pil = Image.new('L', (W, H), 0)
    draw = ImageDraw.Draw(pil)
    for ann in anns_by_image[image_id]:
        for poly in ann['segmentation']:
            xy = list(zip(poly[0::2], poly[1::2]))
            if len(xy) >= 3:
                draw.polygon(xy, fill=255)
    return (np.array(pil) > 0).astype(np.uint8)


def place_shape(draw, shape, W, H, rng, avoid_mask, max_tries=30):
    """
    Place one normalized vessel shape with random position, rotation, scale.
    The shape centre must NOT fall on a real vessel (avoid_mask == 1).
    Shape is skipped if no valid position is found within max_tries.
    """
    scale = rng.uniform(0.05, 0.4) * min(W, H)
    angle = rng.uniform(0, 2 * np.pi)
    rot   = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle),  np.cos(angle)]])
    pts_rot = (shape - 0.5) @ rot.T * scale

    for _ in range(max_tries):
        cx = rng.uniform(W * 0.1, W * 0.9)
        cy = rng.uniform(H * 0.1, H * 0.9)
        ix = int(np.clip(cx, 0, W - 1))
        iy = int(np.clip(cy, 0, H - 1))

        if avoid_mask[iy, ix] == 1:
            continue  # centre lands on a real vessel — retry

        pts = pts_rot.copy()
        pts[:, 0] += cx
        pts[:, 1] += cy
        xy = [(float(x), float(y)) for x, y in pts]
        if len(xy) >= 3:
            draw.polygon(xy, fill=255)
        return

    # No valid position found — skip this shape rather than place it unsafely


def generate_mask(W, H, shapes, avoid_mask, n_shapes=5, rng=None):
    """Generate one binary mask with vessel shapes placed in non-vessel regions."""
    if rng is None:
        rng = np.random.default_rng()

    mask = Image.new('L', (W, H), 0)
    draw = ImageDraw.Draw(mask)

    chosen = rng.choice(len(shapes), size=min(n_shapes, len(shapes)), replace=False)
    for idx in chosen:
        place_shape(draw, shapes[idx], W, H, rng, avoid_mask)

    return mask


def main():
    parser = argparse.ArgumentParser(
        description="Generate vessel-shaped masks that avoid real vessel regions"
    )
    parser.add_argument('--annotations', required=True,
                        help='COCO annotations JSON (used for shapes AND avoid regions)')
    parser.add_argument('--images',      required=True,
                        help='Directory of input images')
    parser.add_argument('--output',      required=True,
                        help='Output directory for generated masks')
    parser.add_argument('--n_shapes',    type=int, default=5,
                        help='Vessel shapes per mask (default: 5)')
    parser.add_argument('--seed',        type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--preview',     action='store_true',
                        help='Save only the first mask as preview and exit')
    args = parser.parse_args()

    print("Loading COCO annotations...")
    shapes, filename_to_id, id_to_info, anns_by_image = load_coco(args.annotations)

    if not shapes:
        print("ERROR: no vessel shapes found in annotations")
        return

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    images_path = Path(args.images)
    image_files = sorted(
        list(images_path.glob('*.png')) +
        list(images_path.glob('*.jpg')) +
        list(images_path.glob('*.jpeg'))
    )

    if not image_files:
        print(f"ERROR: no images found in {args.images}")
        return

    print(f"Generating masks for {len(image_files)} images → {args.output}")
    rng = np.random.default_rng(args.seed)

    for i, img_path in enumerate(image_files):
        image_id = filename_to_id.get(img_path.name)
        if image_id is None:
            print(f"  WARNING: {img_path.name} not in annotations, skipping")
            continue

        info = id_to_info[image_id]
        W, H = info['width'], info['height']

        avoid_mask = make_vessel_mask(image_id, W, H, anns_by_image)
        mask = generate_mask(W, H, shapes, avoid_mask,
                             n_shapes=args.n_shapes, rng=rng)

        if args.preview:
            mask.save('mask_preview.png')
            img_np = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            side   = Image.new('L', (W * 2, H), 0)
            side.paste(Image.fromarray(img_np), (0, 0))
            side.paste(mask, (W, 0))
            side.save('mask_preview_sidebyside.png')
            print("Saved: mask_preview.png + mask_preview_sidebyside.png")
            return

        mask.save(output_path / img_path.name)

        if (i + 1) % 100 == 0 or (i + 1) == len(image_files):
            print(f"  {i + 1}/{len(image_files)} done")

    print(f"\nDone. {len(image_files)} masks saved to {args.output}")


if __name__ == '__main__':
    main()
