"""Full-network MC dropout for RGB measurements of variable size.

Measurements/targets: floating (N,3,H,W) in [0,1]. File format and intensity
conversion belong to the loader. The inherited backbone forward replicate-pads
to multiples of 32 and crops each stochastic prediction back to H/W.
Existing PLD runs use 256x256 images; that is an experiment setting.
Predictions are unbounded normalized intensities.
Difference from Im2im: one MSE-trained RGB output; uncertainty comes from repeated
stochastic forwards, not endpoint heads. Dropout follows the MLP in all 36 encoder
blocks and GELU in all 12 decoder stages. Default p=0.05, inference uses 30 passes.
The unwrapped state dictionary matches the trained ``mc_005`` checkpoints.
"""

from contextlib import contextmanager
import torch
from torch import nn
from timm.models.convnext import ConvNeXtBlock

from convrml.backbone import ConvNeXtFPN
from convrml.intervals import gaussian_interval_from_mean_variance


@contextmanager
def dropout_inference(model):
    """Enable dropout for prediction and restore every module's mode afterward."""
    modes = {module: module.training for module in model.modules()}
    try:
        model.eval()
        for module in model.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                module.train()
        yield
    finally:
        for module, was_training in modes.items():
            module.training = was_training


class MCDropout(ConvNeXtFPN):
    """Elementwise dropout in all 36 encoder blocks and 12 decoder conv stages."""

    def __init__(self, p=0.05):
        super().__init__()
        if not 0 < p < 1:
            raise ValueError("Expected 0 < dropout probability < 1")
        self.encoder_dropout = nn.ModuleDict()
        self.dropout_locations = []
        self._dropout_handles = []
        for name, block in list(self.encoder.named_modules()):
            if isinstance(block, ConvNeXtBlock):
                key = name.replace(".", "__")
                self.encoder_dropout[key] = nn.Dropout(p)
                self._dropout_handles.append(
                    block.mlp.register_forward_hook(self._dropout_hook(key))
                )
                self.dropout_locations.append(
                    "encoder." + name + ".mlp (before residual addition)"
                )
        for name, block in [(f"refine.{i}", b) for i, b in enumerate(self.refine)] + [
            ("up1.1", self.up1[1]),
            ("up2.1", self.up2[1]),
        ]:
            layers = []
            for layer in block.layers:
                layers.append(layer)
                if isinstance(layer, nn.GELU):
                    layers.append(nn.Dropout(p))
                    self.dropout_locations.append(
                        name + ".after_activation_" + str(len(layers) - 1)
                    )
            block.layers = nn.Sequential(*layers)
        if len(self.encoder_dropout) != 36 or len(self.dropout_locations) != 48:
            raise ValueError("Expected 36 encoder and 12 decoder dropout sites")

    def _dropout_hook(self, key):
        def hook(module, inputs, output):
            return self.encoder_dropout[key](output)

        return hook

    def loss(self, y, target):
        """Scalar MSE over all RGB values; model.train() enables training dropout."""
        return (self(y).float() - target).square().mean()

    @torch.no_grad()
    def bounds(self, y, samples=30, alpha=0.1):
        """Return (mean, lower, upper), each unbounded (N,3,H,W).

        Dropout stays active during sampling. The variance measures model spread;
        no observation-noise precision term is added. Native bounds approximate
        this spread as Gaussian; RCPS calibration is a separate step.
        """
        if samples < 2:
            raise ValueError("At least two stochastic predictions are required")
        # Welford's update avoids retaining all stochastic images in memory.
        mean = squared_deviations = None
        with dropout_inference(self):
            for sample_index in range(samples):
                sample = self(y).float()
                if mean is None:
                    mean, squared_deviations = sample, torch.zeros_like(sample)
                else:
                    delta = sample - mean
                    mean = mean + delta / (sample_index + 1)
                    squared_deviations = squared_deviations + delta * (sample - mean)
        variance = squared_deviations / samples
        lower, upper = gaussian_interval_from_mean_variance(mean, variance, alpha)
        return mean, lower, upper
