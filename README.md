![Python 3.9](https://img.shields.io/badge/python-3.9-green.svg)

# CMT — ARCADE Grayscale Inpainting Fork

Based on Keunsoo Ko and Chang-Su Kim,
"Continuously Masked Transformer for Image Inpainting, ICCV, 2023"

This fork adapts CMT for **grayscale coronary angiography inpainting** using the [ARCADE dataset](https://arcade.grand-challenge.org/). The goal is to reconstruct realistic X-ray backgrounds by inpainting vessel regions, enabling synthetic image generation.

---

### Training on ARCADE
```bash
python train.py \
  --train_img arcade/syntax/train/images \
  --train_ann arcade/syntax/train/annotations/train.json \
  --val_img   arcade/syntax/val/images \
  --val_ann   arcade/syntax/val/annotations/val.json \
  --epochs 100 \
  --batch_size 4 \
  --input_size 256 \
  --device cpu
```

Checkpoints are saved to `checkpoints/`. The best model by validation PSNR is saved as `checkpoints/best.pth`.

**`--input_size` must be a power of 2 (e.g. 32, 64, 128, 256).** The network pyramid is built dynamically based on this value. Smaller sizes train faster but produce lower quality results.

**Smoke test** (verifies pipeline with a small subset of images):
```bash
python train.py \
  --train_img arcade/syntax/train/images \
  --train_ann arcade/syntax/train/annotations/train.json \
  --val_img   arcade/syntax/val/images \
  --val_ann   arcade/syntax/val/annotations/val.json \
  --smoke_test --smoke_size 20 --epochs 1 --batch_size 1 --device cpu
```

---

### Inference
```bash
python demo.py \
  --ckpt checkpoints/best.pth \
  --img_path ./samples/test_img \
  --mask_path ./samples/test_mask \
  --output_path ./samples/results \
  --device cpu
```

Images of any resolution are automatically resized to match the model's input size.
Use `--device cuda` for GPU.

---

### Changes from original

- Adapted pipeline to 1-channel grayscale input (from RGB)
- Dynamic encoder-decoder pyramid in `refine.py` — resolution controlled via `--input_size`
- Added `train.py` for training on ARCADE coronary angiography dataset
- Vessel masks generated automatically from COCO-format polygon annotations
- Added `--input_size` flag (power of 2, min 32) to control network resolution
- Added `--smoke_test` and `--smoke_size` flags for quick pipeline verification
- Added `--device` flag to `demo.py` for CPU/GPU selection
- Auto-resize input images to model resolution in `demo.py`
