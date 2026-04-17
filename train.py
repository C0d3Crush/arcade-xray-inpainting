# -*- coding: utf-8 -*-
import argparse, os, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
import cv2
from collections import defaultdict
from tqdm import tqdm
from network.network_pro import Inpaint
from utils import load_checkpoint, psnr
from skimage.metrics import structural_similarity as ssim_fn
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class ArcadeDataset(Dataset):
    """
    Loads grayscale coronary angiography images with vessel masks.

    Two modes:
    - mask_dir=None  : masks generated from COCO annotations (vessel polygons)
    - mask_dir=path  : masks loaded from folder (e.g. random_masks/)
                       → enables background inpainting training
    """
    STENOSIS_CATEGORY_ID = 26

    def __init__(self, img_dir, ann_path, image_size=256, mask_dir=None):
        self.img_dir    = img_dir
        self.image_size = image_size
        self.mask_dir   = mask_dir

        with open(ann_path) as f:
            coco = json.load(f)
        self.id_to_info = {img['id']: img for img in coco['images']}
        self.anns_by_image = defaultdict(list)
        for ann in coco['annotations']:
            if ann['category_id'] != self.STENOSIS_CATEGORY_ID:
                self.anns_by_image[ann['image_id']].append(ann)
        self.image_ids = [
            img_id for img_id in self.id_to_info
            if self.anns_by_image[img_id]
        ]

    def __len__(self):
        return len(self.image_ids)

    def _make_mask_from_annotations(self, image_id, W, H):
        """Rasterise vessel polygons into a binary mask (255 = vessel)."""
        mask = Image.new('L', (W, H), 0)
        draw = ImageDraw.Draw(mask)
        for ann in self.anns_by_image[image_id]:
            for poly in ann['segmentation']:
                xy = list(zip(poly[0::2], poly[1::2]))
                if len(xy) >= 3:
                    draw.polygon(xy, fill=255)
        return mask

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        info     = self.id_to_info[image_id]
        W, H     = info['width'], info['height']

        # Load image
        img_path = os.path.join(self.img_dir, info['file_name'])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        # Load or generate mask
        if self.mask_dir is not None:
            mask_path = os.path.join(self.mask_dir, info['file_name'])
            mask_img  = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_img is None:
                raise FileNotFoundError(f"Mask not found: {mask_path}")
            mask_np = mask_img.astype(np.float32) / 255.0
        else:
            mask_pil = self._make_mask_from_annotations(image_id, W, H)
            mask_np  = np.array(mask_pil, dtype=np.float32) / 255.0

        # Resize
        img  = cv2.resize(img,     (self.image_size, self.image_size),
                          interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask_np, (self.image_size, self.image_size),
                          interpolation=cv2.INTER_NEAREST)

        # Normalise image to [-1, 1]
        img_norm = (img.astype(np.float32) / 255.0) * 2.0 - 1.0
        img_t    = torch.from_numpy(img_norm).unsqueeze(0)
        mask_t   = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
        return img_t, mask_t


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
def ssim_loss(pred, target, window_size=11):
    """Differentiable SSIM loss (1 - SSIM) for grayscale images in [-1, 1]."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Gaussian window
    coords = torch.arange(window_size, dtype=pred.dtype, device=pred.device)
    coords -= window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    g = g / g.sum()
    window = g.unsqueeze(0) * g.unsqueeze(1)
    window = window.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

    pad = window_size // 2
    mu1 = torch.nn.functional.conv2d(pred,   window, padding=pad, groups=1)
    mu2 = torch.nn.functional.conv2d(target, window, padding=pad, groups=1)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = torch.nn.functional.conv2d(pred   * pred,   window, padding=pad) - mu1_sq
    sigma2_sq = torch.nn.functional.conv2d(target * target, window, padding=pad) - mu2_sq
    sigma12   = torch.nn.functional.conv2d(pred   * target, window, padding=pad) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return 1 - ssim_map.mean()


class InpaintingLoss(nn.Module):
    """L1 + SSIM loss on masked region + L1 background consistency."""
    def __init__(self, ssim_weight=0.5):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.ssim_weight = ssim_weight

    def forward(self, output, target, mask):
        loss_mask  = self.l1(output * mask,       target * mask)
        loss_valid = self.l1(output * (1 - mask), target * (1 - mask))
        loss_ssim  = ssim_loss(output, target)
        return loss_mask * 6.0 + loss_valid + self.ssim_weight * loss_ssim


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save({
        'epoch':      epoch,
        'state_dict': model.state_dict(),
        'optimizer':  optimizer.state_dict(),
        'loss':       loss,
    }, path)
    print(f"  Checkpoint saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train CMT on ARCADE grayscale X-rays")
    parser.add_argument('--train_img',   default='arcade/syntax/train/images')
    parser.add_argument('--train_ann',   default='arcade/syntax/train/annotations/train.json')
    parser.add_argument('--train_mask',  default=None,
                        help='Optional: folder with precomputed train masks (e.g. random_masks/). '
                             'If None, masks are generated from COCO annotations.')
    parser.add_argument('--val_img',     default='arcade/syntax/val/images')
    parser.add_argument('--val_ann',     default='arcade/syntax/val/annotations/val.json')
    parser.add_argument('--val_mask',    default=None,
                        help='Optional: folder with precomputed val masks.')
    parser.add_argument('--output_dir',  default='checkpoints')
    parser.add_argument('--ckpt',        default=None,
                        help='Resume from a CMT training checkpoint')
    parser.add_argument('--epochs',      type=int,   default=100)
    parser.add_argument('--batch_size',  type=int,   default=4)
    parser.add_argument('--lr',          type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int,   default=2)
    parser.add_argument('--save_every',  type=int,   default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--device',      default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--smoke_test',  action='store_true',
                        help='Run with a small subset to verify pipeline')
    parser.add_argument('--smoke_size',  type=int, default=2,
                        help='Number of images to use in smoke test')
    parser.add_argument('--input_size',  type=int, default=256,
                        help='Input image size (power of 2, min 32)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # ---- Datasets ----
    train_dataset = ArcadeDataset(args.train_img, args.train_ann, args.input_size,
                                  mask_dir=args.train_mask)
    val_dataset   = ArcadeDataset(args.val_img,   args.val_ann,   args.input_size,
                                  mask_dir=args.val_mask)

    if args.train_mask:
        print(f"  Using precomputed train masks from: {args.train_mask}")
    else:
        print(f"  Generating train masks from COCO annotations")

    if args.smoke_test:
        train_dataset.image_ids = train_dataset.image_ids[:args.smoke_size]
        val_dataset.image_ids   = val_dataset.image_ids[:max(1, args.smoke_size // 2)]

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers,
                              pin_memory=(args.device == 'cuda'))
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers)

    print(f"Train: {len(train_dataset)} images | Val: {len(val_dataset)} images")

    # ---- Model ----
    model = Inpaint(input_size=args.input_size).to(device)

    if args.ckpt and os.path.exists(args.ckpt):
        model = load_checkpoint(args.ckpt, model, device)
        print(f"  Resumed from checkpoint: {args.ckpt}")

    # ---- Optimiser & Loss ----
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = InpaintingLoss().to(device)

    # ---- Training loop ----
    best_val_psnr = 0.0
    log_path = os.path.join(args.output_dir, 'training_log.csv')
    drive_log_path = '/content/drive/MyDrive/training_log.csv' \
        if os.path.isdir('/content/drive/MyDrive') else None
    with open(log_path, 'w') as f:
        f.write('epoch,train_loss,val_psnr,val_ssim\n')
    if drive_log_path:
        with open(drive_log_path, 'w') as f:
            f.write('epoch,train_loss,val_psnr,val_ssim\n')

    for epoch in range(1, args.epochs + 1):
        # -- Train --
        model.train()
        train_loss = 0.0
        prog = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]")
        for img, mask in prog:
            img, mask = img.to(device), mask.to(device)
            optimizer.zero_grad()
            output = model(img, mask)
            loss   = criterion(output, img, mask)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            prog.set_postfix(loss=f"{loss.item():.4f}")
        train_loss /= len(train_loader)

        # -- Validate --
        model.eval()
        val_psnr = 0.0
        val_ssim = 0.0
        with torch.no_grad():
            for img, mask in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [val]"):
                img, mask = img.to(device), mask.to(device)
                output = model(img, mask)
                output = torch.clip(output, -1.0, 1.0)
                out_np = (output[:, 0].cpu().numpy() * 0.5 + 0.5) * 255.0
                gt_np  = (img[:, 0].cpu().numpy()    * 0.5 + 0.5) * 255.0
                for o, g in zip(out_np, gt_np):
                    val_psnr += psnr(o, g)
                    val_ssim += ssim_fn(o, g, data_range=255.0)
        val_psnr /= len(val_dataset)
        val_ssim /= len(val_dataset)

        scheduler.step()
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_psnr={val_psnr:.2f} dB | val_ssim={val_ssim:.4f}")

        with open(log_path, 'a') as f:
            f.write(f"{epoch},{train_loss:.4f},{val_psnr:.2f},{val_ssim:.4f}\n")
        if drive_log_path:
            with open(drive_log_path, 'a') as f:
                f.write(f"{epoch},{train_loss:.4f},{val_psnr:.2f},{val_ssim:.4f}\n")

        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            save_checkpoint(model, optimizer, epoch, train_loss,
                            os.path.join(args.output_dir, 'best.pth'))

        if epoch % args.save_every == 0:
            save_checkpoint(model, optimizer, epoch, train_loss,
                            os.path.join(args.output_dir, f'epoch_{epoch:03d}.pth'))

    print(f"\nTraining complete. Best val PSNR: {best_val_psnr:.2f} dB")
    print(f"Checkpoints in: {args.output_dir}/")


if __name__ == '__main__':
    main()
