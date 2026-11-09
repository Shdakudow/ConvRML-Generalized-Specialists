"""Shared ConvNeXtV2-Base encoder and feature-pyramid (FPN) residual decoder.

Measurements enter as RGB (N,3,H,W) in [0,1]. Model forwards replicate-pad
right/bottom to multiples of 32, then crop outputs to H/W. The encoder extracts
four resolutions; the decoder combines them and predicts an RGB correction.
The correction is added to the original measurement, producing unbounded output.
"""

import torch
from torch import nn
from torch.nn import functional as F
import timm


class Refine(nn.Module):
    """Two conv/normalization/activation stages preserving (N,256,H,W)."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.GroupNorm(8, 256),
            nn.GELU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.GroupNorm(8, 256),
            nn.GELU(),
        )

    def forward(self, x):
        return self.layers(x)


class ConvNeXtFPN(nn.Module):
    """Common encoder/decoder with one or more RGB residual output heads.

    Output shape is (N,3*output_heads,H,W). Im2im requests three RGB heads;
    the other methods use one. Losses and uncertainty logic live in models/.
    """

    def __init__(self, *, output_heads=1):
        super().__init__()
        if not isinstance(output_heads, int) or output_heads < 1:
            raise ValueError("output_heads must be a positive integer")
        self.output_heads = output_heads
        self.encoder = timm.create_model(
            "convnextv2_base.fcmae_ft_in22k_in1k", pretrained=False, features_only=True
        )
        # Despite the architecture preset's name, pretrained=False starts from
        # random weights. Each 1x1 lateral projection maps its stage to 256 channels.
        self.lateral = nn.ModuleList(
            [nn.Conv2d(c, 256, 1) for c in [128, 256, 512, 1024]]
        )
        self.refine = nn.ModuleList([Refine() for _ in range(4)])
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), Refine()
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), Refine()
        )
        self.head = nn.Conv2d(256, 3 * output_heads, 3, padding=1)
        # Normalize only the encoder input. The residual addition still uses the
        # original [0,1] measurement; these constants are not output normalization.
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    def features(self, x):
        """Return decoder features (N,256,H/2,W/2) from RGB (N,3,H,W).

        This internal method expects stride-aligned input from a model forward.
        Encoder stages have 128/256/512/1024 channels at 1/4,1/8,1/16,1/32 size.
        FPN merges coarse features with finer ones, then up1 doubles resolution.
        """
        encoder_features = self.encoder((x - self.mean) / self.std)
        decoder_features = self.refine[3](self.lateral[3](encoder_features[3]))
        # Start at the coarsest stage, then merge each finer encoder resolution.
        for stage_index in [2, 1, 0]:
            decoder_features = self.refine[stage_index](
                F.interpolate(
                    decoder_features,
                    size=encoder_features[stage_index].shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                + self.lateral[stage_index](encoder_features[stage_index])
            )
        return self.up1(decoder_features)

    def reconstruct(self, x, features):
        """Add each predicted RGB correction to x; pack heads along channels."""
        residuals = self.head(self.up2(features))
        if self.output_heads == 1:
            return x + residuals
        residuals = residuals.reshape(len(x), self.output_heads, 3, *x.shape[-2:])
        return (residuals + x[:, None]).flatten(1, 2)

    def forward(self, x):
        """Map RGB (N,3,H,W) to unbounded (N,3*output_heads,H,W), with size retained."""
        height, width = x.shape[-2:]
        if height % 32 or width % 32:
            x = F.pad(x, (0, (-width) % 32, 0, (-height) % 32), mode="replicate")
        return self.reconstruct(x, self.features(x))[..., :height, :width]
