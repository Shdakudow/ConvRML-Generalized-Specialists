# Upstream methods and attribution

This repository provides a ConvNeXt/FPN adaptation and comparison. It does not
claim authorship of the original reconstruction or uncertainty methods.

| Component | Reference |
| --- | --- |
| ConvRML and paired lensless data | [Kabuli et al., ConvRML](https://doi.org/10.1364/OE.608614), [original code](https://github.com/lakabuli/ConvRML) |
| QUTCC conditioning and quantile training | [Ye et al., QUTCC](https://arxiv.org/abs/2507.14760v2), [reference code](https://github.com/Shdakudow/code-for-quant) |
| Im2im uncertainty formulation | [Angelopoulos et al., Image-to-Image Regression](https://arxiv.org/abs/2202.05265) |
| Deep ensemble baseline | [Lakshminarayanan, Pritzel and Blundell](https://proceedings.neurips.cc/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html) |
| MC dropout baseline | [Gal and Ghahramani](https://proceedings.mlr.press/v48/gal16.html) |
| LPIPS perceptual metric | [Zhang et al., official implementation](https://github.com/richzhang/PerceptualSimilarity), [TorchMetrics implementation](https://lightning.ai/docs/torchmetrics/stable/image/learned_perceptual_image_patch_similarity.html) |

The upstream BSD 3-Clause [LICENSE](LICENSE), including its copyright notice, is
retained. The original ConvRML citation is preserved separately in
[upstream_convrml_CITATION.cff](archive/provenance/upstream_convrml_CITATION.cff).
The root [CITATION.cff](CITATION.cff) identifies this software comparison and lists
upstream references, without assigning a manuscript DOI or inventing paper authors.

QUTCC here uses ConvNeXt/FPN instead of its original U-Net. Im2im is end-to-end;
ensemble members use Gaussian NLL; MC dropout uses MSE and full-network dropout.
These adaptations and calibration distinctions are documented in
[the model comparison](archive/publication_support/docs/MODEL_COMPARISON.md). Source captures and equivalence
records remain available in [archive/provenance](archive/provenance/).
