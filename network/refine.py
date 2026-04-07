from network.swin import *
import torch.nn as nn
import torch
import math


class Refine(nn.Module):
    def __init__(self, in_c, input_size=256):
        super(Refine, self).__init__()
        assert input_size >= 4 and (input_size & (input_size - 1)) == 0, \
            "input_size must be a power of 2 and >= 4"

        dim = 32
        self.input_size = input_size

        # ------------------------------------------------------------------
        # Compute pyramid levels dynamically
        # We downsample from input_size → patch_size (input_size//4) via
        # PatchEmbed, then halve until we reach 4, then go to 1.
        #
        # Levels after PatchEmbed: input_size//4, //8, //16, ..., 4, 1
        # ------------------------------------------------------------------
        patch_size  = 4
        after_patch = input_size // patch_size   # e.g. 256//4 = 64

        # Build list of resolutions from after_patch down to 4, then 1
        resolutions = []
        r = after_patch
        while r > 4:
            resolutions.append(r)
            r = r // 2
        resolutions.append(4)   # last strided layer always ends at 4
        # resolutions = [64, 32, 16, 8, 4] for input_size=256

        num_heads_list = []
        for res in resolutions:
            # num_heads scales with depth: smaller resolution → more heads
            level = int(math.log2(after_patch // res))
            num_heads_list.append(max(1, 2 ** level))

        # dim multiplier at each level: 1, 2, 4, 8, 16, ...
        dim_mults = [2 ** i for i in range(len(resolutions) + 1)]
        # dim_mults[0] = 1  (after PatchEmbed)
        # dim_mults[1] = 2  (after first BasicLayer)
        # ...

        # window sizes — clamp to resolution
        def win(res):
            return min(4, res)

        # ------------------------------------------------------------------
        # Encoder
        # ------------------------------------------------------------------
        encoder_blocks = []

        # Block 0: Conv2d (input_size × input_size)
        encoder_blocks.append(
            nn.Sequential(
                nn.Conv2d(in_c + 1, dim // 2, kernel_size=7, stride=1, padding=3),
                nn.GELU()
            )
        )

        # Block 1: PatchEmbed (input_size → after_patch)
        encoder_blocks.append(
            PatchEmbed(
                img_size=(input_size, input_size),
                patch_size=patch_size,
                in_chans=dim // 2,
                embed_dim=dim,
                norm_layer=nn.LayerNorm
            )
        )

        # Blocks 2..N-1: strided BasicLayers
        for i, res in enumerate(resolutions[:-1]):   # all except last (4)
            next_res = res // 2
            d = dim * dim_mults[i]
            h = num_heads_list[i]
            encoder_blocks.append(
                BasicLayer(
                    dim=d,
                    input_resolution=(res, res),
                    num_heads=h,
                    depth=2,
                    stride=2,
                    window_size=win(res)
                )
            )

        # Last encoder block: BasicLayer at res=4 → AvgPool → 1×1
        last_dim = dim * dim_mults[len(resolutions) - 1]
        last_heads = num_heads_list[-1]
        encoder_blocks.append(
            nn.Sequential(
                BasicLayer(
                    dim=last_dim,
                    input_resolution=(4, 4),
                    num_heads=last_heads,
                    depth=2,
                    stride=None,
                    window_size=2
                ),
                AvgPool(last_dim),
                nn.Linear(last_dim, last_dim),
                nn.ReLU()
            )
        )

        self.encoder_blocks = nn.ModuleList(encoder_blocks)

        # ------------------------------------------------------------------
        # Decoder
        # ------------------------------------------------------------------
        # Mirror of encoder. encoder outputs (from bottom to top):
        #   1×1  (bottleneck, dim=last_dim)
        #   4×4  (dim=last_dim)
        #   8×8  (dim=last_dim//2)
        #   ...
        #   after_patch×after_patch (dim=dim)
        #   input_size×input_size (dim=dim//2, before PatchEmbed)

        decoder_blocks = []

        # Dec block 0: bottleneck 1×1 → 4×4
        decoder_blocks.append(
            nn.Sequential(
                nn.Linear(last_dim, last_dim),
                nn.ReLU(),
                UpSample((1, 1), last_dim, last_dim, 4)
            )
        )

        # Dec block 1: 4×4 (with skip) → 8×8
        # skip from encoder level len(resolutions)-1 has dim=last_dim
        decoder_blocks.append(
            nn.Sequential(
                BasicLayer(
                    dim=last_dim * 2,
                    input_resolution=(4, 4),
                    num_heads=last_heads,
                    depth=2,
                    stride=None,
                    window_size=2
                ),
                UpSample((4, 4), last_dim * 2, last_dim, 2)
            )
        )

        # Dec blocks 2..N-1: upsample through resolutions
        # We go from res=8 up to res=after_patch
        up_resolutions = []
        r = 8
        while r <= after_patch:
            up_resolutions.append(r)
            r *= 2
        # up_resolutions = [8, 16, 32, 64] for input_size=256

        for j, res in enumerate(up_resolutions):
            # encoder skip at this resolution has dim = dim * dim_mults[len(resolutions)-2-j]
            enc_idx = len(resolutions) - 2 - j
            enc_dim = dim * dim_mults[max(0, enc_idx)]
            cur_dim = last_dim + enc_dim   # concat of decoder + skip

            h = max(1, last_heads // (2 ** (j + 1)))
            next_dim = last_dim // (2 ** (j + 1))
            next_dim = max(dim, next_dim)

            decoder_blocks.append(
                nn.Sequential(
                    BasicLayer(
                        dim=cur_dim,
                        input_resolution=(res, res),
                        num_heads=max(1, h),
                        depth=2 if res <= 16 else 2,
                        stride=None,
                        window_size=win(res)
                    ),
                    UpSample((res, res), cur_dim, enc_dim // 2, 2)
                )
            )

        # Final dec block: after_patch×after_patch → input_size×input_size
        # This mirrors the PatchEmbed (patch_size=4 upscale)
        final_enc_dim = dim  # encoder block 1 output dim
        final_skip_dim = dim // 2  # encoder block 0 output (conv)
        final_cur_dim = final_enc_dim + final_skip_dim  # but skip from block 1

        decoder_blocks.append(
            nn.Sequential(
                BasicLayer(
                    dim=final_cur_dim,
                    input_resolution=(after_patch, after_patch),
                    num_heads=1,
                    depth=2,
                    stride=None,
                    window_size=win(after_patch)
                ),
                UpSample((after_patch, after_patch), final_cur_dim, dim // 2, patch_size)
            )
        )

        # Output block: input_size×input_size → 1 channel
        decoder_blocks.append(
            nn.Sequential(
                BasicLayer(
                    dim=dim,
                    input_resolution=(input_size, input_size),
                    num_heads=1,
                    depth=2,
                    stride=None,
                    window_size=min(8, input_size)
                ),
                Mlp(dim, out_features=1),
                nn.Tanh()
            )
        )

        self.decoder_blocks = nn.ModuleList(decoder_blocks)

    def forward(self, x):
        B, C, H, W = x.shape
        feats = []
        for f in self.encoder_blocks:
            x = f(x)
            feats.append(x)
        feats.pop()  # remove bottleneck, keep rest as skips

        for i, f in enumerate(self.decoder_blocks):
            if i == 0:
                x = f(x)
            else:
                feat = feats[-1]
                if len(feat.shape) > 3:
                    feat = feat.view(feat.size(0), feat.size(1), -1).permute(0, 2, 1)
                x = f(torch.cat((x, feat), -1))
                feats.pop()

        outputs = x.reshape(B, H, W, 1).permute(0, 3, 1, 2)
        return outputs
