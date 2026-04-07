# -*- coding: utf-8 -*-
import argparse, os, json, glob
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

import warnings
warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ArcadeDataset(Dataset):
    """
    Loads grayscale coronary angiography images and generates vessel masks
    from COCO-format polygon annotations (all categories except stenosis).
    """
    STENOSIS_CATEGORY_ID = 26

    def __init__(self, img_dir, ann_path, image_size=256):
        self.img_dir    = img_dir
        self.image_size = image_size

        with open(ann_path) as f:
            coco = json.load(f)

        self.id_to_info = {img['id']: img for img in coco['images']}

        # Group vessel annotations (exclude stenosis) by image_id
        self.anns_by_image = defaultdict(list)
        for ann in coco['annotations']:
            if ann['category_id'] != self.STENOSIS_CATEGORY_ID:
                self.anns_by_image[ann['image_id']].append(ann)

        # Only keep images that have at least one vessel annotation
        self.image_ids = [
            img_id for img_id in self.id_to_info
            if self.anns_by_image[img_id]
        ]

    def __len__(self):
        return len(self.image_ids)

    def _make_mask(self, image_id, W, H):
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

        # Load image as grayscale
        img_path = os.path.join(self.img_dir, info['file_name'])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        # Build vessel mask
        mask_pil = self._make_mask(image_id, W, H)
        mask_np  = np.array(mask_pil, dtype=np.float32) / 255.0

        # Resize both to model input size
        img  = cv2.resize(img,  (self.image_size, self.image_size),
                          interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask_np, (self.image_size, self.image_size),
                          interpolation=cv2.INTER_NEAREST)

        # Normalise image to [-1, 1]
        img_norm = (img.astype(np.float32) / 255.0) * 2.0 - 1.0

        # Tensors: (1, H, W)
        img_t  = torch.from_numpy(img_norm).unsqueeze(0)
        mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)

        return img_t, mask_t


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class InpaintingLoss(nn.Module):
    """Simple L1 loss on the masked (vessel) region only."""
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, output, target, mask):
        # Penalise only inside the mask (where vessels were)
        loss_mask  = self.l1(output * mask, target * mask)
        # Also penalise outside the mask (background consistency)
        loss_valid = self.l1(output * (1 - mask), target * (1 - mask))
        return loss_mask * 6.0 + loss_valid


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

    parser.add_argument('--train_img',  default='arcade/syntax/train/images')
    parser.add_argument('--train_ann',  default='arcade/syntax/train/annotations/train.json')
    parser.add_argument('--val_img',    default='arcade/syntax/val/images')
    parser.add_argument('--val_ann',    default='arcade/syntax/val/annotations/val.json')
    parser.add_argument('--output_dir', default='checkpoints')
    parser.add_argument('--ckpt',       default=None,  help='Resume from checkpoint')
    parser.add_argument('--epochs',     type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--num_workers',type=int, default=2)
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--device',     default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--smoke_test', action='store_true',
                    help='Run with 2 train / 1 val image to verify pipeline')
    parser.add_argument('--smoke_size', type=int, default=2,
                    help='Number of images to use in smoke test')
    parser.add_argument('--input_size', type=int, default=256,
                    help='Input image size (must be power of 2, min 32)')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # ---- Datasets ----
    train_dataset = ArcadeDataset(args.train_img, args.train_ann, args.input_size)
    val_dataset   = ArcadeDataset(args.val_img,   args.val_ann,   args.input_size)
    
    # Smoke test — remove these two lines for real training
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
        print(f"Resumed from {args.ckpt}")

    # ---- Optimiser & Loss ----
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    criterion = InpaintingLoss().to(device)

    # ---- Training loop ----
    best_val_psnr = 0.0

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
        with torch.no_grad():
            for img, mask in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [val]"):
                img, mask = img.to(device), mask.to(device)
                output = model(img, mask)
                output = torch.clip(output, -1.0, 1.0)

                # PSNR on masked region only (convert to [0,255])
                out_np = (output[:, 0].cpu().numpy() * 0.5 + 0.5) * 255.0
                gt_np  = (img[:, 0].cpu().numpy()    * 0.5 + 0.5) * 255.0
                for o, g in zip(out_np, gt_np):
                    val_psnr += psnr(o, g)

        val_psnr /= len(val_dataset)

        scheduler.step()

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_psnr={val_psnr:.2f} dB")

        # Save best
        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            save_checkpoint(model, optimizer, epoch, train_loss,
                            os.path.join(args.output_dir, 'best.pth'))

        # Save periodic
        if epoch % args.save_every == 0:
            save_checkpoint(model, optimizer, epoch, train_loss,
                            os.path.join(args.output_dir, f'epoch_{epoch:03d}.pth'))

    print(f"\nTraining complete. Best val PSNR: {best_val_psnr:.2f} dB")
    print(f"Checkpoints saved in: {args.output_dir}/")


if __name__ == '__main__':
    main()
