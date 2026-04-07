```markdown
![Python 3.9](https://img.shields.io/badge/python-3.9-green.svg)
# CMT — ARCADE Grayscale Inpainting Fork
Based on Keunsoo Ko and Chang-Su Kim,
"Continuously Masked Transformer for Image Inpainting, ICCV, 2023"

This fork adapts CMT for **grayscale coronary angiography inpainting** using the [ARCADE dataset](https://arcade.grand-challenge.org/). The goal is to reconstruct realistic X-ray backgrounds by inpainting vessel regions, enabling synthetic image generation.

---

### Installation
```bash
git clone https://github.com/keunsoo-ko/CMT.git
cd CMT
python -m venv venv
source venv/bin/activate
pip install torch torchvision tqdm einops timm
pip install --only-binary=:all: opencv-python-headless
pip install "numpy<2"
```

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
  --device cpu
```

Checkpoints are saved to `checkpoints/`. The best model by validation PSNR is saved as `checkpoints/best.pth`.

**Smoke test** (verifies pipeline with 2 images, ~1 minute on CPU):
```bash
python train.py \
  --train_img arcade/syntax/train/images \
  --train_ann arcade/syntax/train/annotations/train.json \
  --val_img   arcade/syntax/val/images \
  --val_ann   arcade/syntax/val/annotations/val.json \
  --smoke_test --epochs 1 --batch_size 1 --device cpu
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

Images of any resolution are automatically resized to 256×256 for inference.
Use `--device cuda` for GPU.

---

### Changes from original
- Adapted pipeline to 1-channel grayscale input (from RGB)
- Added `train.py` for training on ARCADE coronary angiography dataset
- Vessel masks generated automatically from COCO-format polygon annotations
- Added `--device` flag to `demo.py` for CPU/GPU selection
- Added `--smoke_test` flag to `train.py` for quick pipeline verification
```
