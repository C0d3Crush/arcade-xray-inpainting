import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.swin_transformer import SwinTransformer


def conv_block(in_c, out_c):
    """Conv + GELU — no normalization to avoid issues with small spatial sizes."""
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1),
        nn.GELU()
    )


class Refine(nn.Module):
    """
    U-Net style refine network using timm SwinTransformer as encoder.
    Variable input size — must be a power of 2, minimum 64.

    Architecture:
        in_conv: (B, in_c+1, H, W) -> (B, 32, H, W)
        swin:    encodes down to H//(4*2^(num_stages-1)) spatial size
        decoder: upsamples back to (B, 1, H, W) with skip connections
    """
    def __init__(self, in_c, input_size=256):
        super().__init__()
        assert input_size >= 64 and (input_size & (input_size - 1)) == 0, \
            "input_size must be a power of 2 and >= 64"

        self.input_size = input_size

        # num_stages: ensure minimum spatial size after swin >= 4x4
        # spatial after swin = input_size // (4 * 2^(num_stages-1))
        # we want this >= 4, so num_stages <= log2(input_size/16)
        self.num_stages = max(2, int(math.log2(input_size)) - 4)

        # ---- Initial conv ----
        self.in_conv = conv_block(in_c + 1, 32)

        # ---- Swin Encoder ----
        depths    = [2] * self.num_stages
        num_heads = [max(1, 2**i) for i in range(self.num_stages)]

        self.swin = SwinTransformer(
            img_size=input_size,
            patch_size=4,
            in_chans=32,
            num_classes=0,
            embed_dim=64,
            depths=depths,
            num_heads=num_heads,
            window_size=min(8, input_size // 4),
            strict_img_size=False,
            always_partition=False,
            global_pool='',
        )

        # Swin output channels per stage: 64, 128, 256, ...
        self.enc_channels = [64 * (2**i) for i in range(self.num_stages)]

        # ---- Decoder ----
        self.dec_convs = nn.ModuleList()

        in_ch = self.enc_channels[-1]
        for i in range(self.num_stages - 1, 0, -1):
            skip_ch = self.enc_channels[i - 1]
            self.dec_convs.append(conv_block(in_ch + skip_ch, skip_ch))
            in_ch = skip_ch

        # Final merge with in_conv output (32ch)
        self.dec_convs.append(conv_block(in_ch + 32, 32))

        # ---- Output ----
        self.out_conv = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Tanh()
        )

    def _swin_features(self, x):
        """Run swin encoder, return list of (B, C, H, W) per stage."""
        feat = self.swin.patch_embed(x)  # (B, H//4, W//4, C) in newer timm

        if hasattr(self.swin, 'absolute_pos_embed') and self.swin.absolute_pos_embed is not None:
            feat = feat + self.swin.absolute_pos_embed

        if hasattr(self.swin, 'pos_drop'):
            feat = self.swin.pos_drop(feat)
        elif hasattr(self.swin, 'drop_after_pos'):
            feat = self.swin.drop_after_pos(feat)

        stage_feats = []
        for layer in self.swin.layers:
            feat = layer(feat)
            # timm >= 0.9 returns (B, H, W, C), older returns (B, L, C)
            if feat.dim() == 4:
                spatial = feat.permute(0, 3, 1, 2)  # (B, C, H, W)
            else:
                B_, L, C_ = feat.shape
                h = int(math.isqrt(L))
                spatial = feat.reshape(B_, h, h, C_).permute(0, 3, 1, 2)
            stage_feats.append(spatial)

        return stage_feats

    def forward(self, x):
        B, C, H, W = x.shape

        # Encoder
        x0 = self.in_conv(x)                   # (B, 32, H, W)
        stage_feats = self._swin_features(x0)  # list of (B, C_i, H_i, W_i)

        # Decoder with skip connections
        d = stage_feats[-1]

        for i, conv in enumerate(self.dec_convs[:-1]):
            skip = stage_feats[-(i + 2)]
            d = F.interpolate(d, size=skip.shape[2:], mode='bilinear', align_corners=False)
            d = conv(torch.cat([d, skip], dim=1))

        # Upsample to full resolution and merge with initial conv
        d = F.interpolate(d, size=(H, W), mode='bilinear', align_corners=False)
        d = self.dec_convs[-1](torch.cat([d, x0], dim=1))

        return self.out_conv(d)
