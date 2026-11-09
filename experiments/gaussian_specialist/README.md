# Gaussian-only specialists

Train Im2im and QUTCC for 50 epochs using the same shared trainer and model
losses. Every measurement receives fresh Gaussian noise with SD 0.01 on the
[0,1] scale, followed by clipping. Targets remain unchanged.

For a controlled comparison, supply the generalized run's **untrained epoch-zero
weights** for the same method. The trainer checks the zero-step metadata and
strictly loads the tensors; it does not load trained generalized weights.
Recorded hashes in [protocol.json](protocol.json) identify the historical
comparison's initial states. New matched runs should use the same reference
file for each method in both studies.

```bash
python experiments/generalized_inputs/train.py \
  --protocol experiments/gaussian_specialist/protocol.json \
  --method qutcc --manifest /path/to/split.json \
  --initial-random-state /path/to/generalized/qutcc/initial_random.pt \
  --output runs/gaussian/qutcc --epochs 50 --device cuda
```

Repeat for Im2im using its own reference and output folder. Adam starts fresh.
The first 50 epochs' learning-rate schedule, batch size, seed and validation
panel match the generalized study. Resume complete epochs with `--resume`
and omit `--initial-random-state`. Calibration and test evaluation are separate.

A specialist receives roughly ten times as many Gaussian examples as the mixture
at the same epoch count; report that exposure difference when comparing scores.
Historical launchers and stronger deployment identity checks are preserved in
[the snapshot](https://github.com/Shdakudow/ConvRML-Uncertainty/tree/f00b699876f348c84947f935dc223810ece03cca/archive/research_prototype_20261102/).

## Calibration and testing

`evaluate.py` loads the completed epoch-50 specialist with frozen weights. It
fits a nominal 90% interval on all 3,000 Gaussian-corrupted calibration scenes,
then evaluates the separate 2,000 Gaussian-corrupted test scenes. Gaussian
noise has SD 0.01; per-scene seeds match the original Exp 4 comparison exactly.

```bash
python experiments/gaussian_specialist/evaluate.py \
  --method qutcc \
  --comparison-source /path/to/frozen/experiment4/source \
  --training-root /path/to/gaussian-specialists \
  --manifest /path/to/split.json \
  --output evaluations/gaussian/qutcc
```

The frozen comparison source supplies the original deterministic inputs and
empirical calibration functions. Its Python files are hashed in `request.json`.
This is the demo's empirical coverage matching procedure, not an RCPS
certificate. QUTCC calibration failures remain explicitly recorded; the
evaluator never labels an unattainable target as successful calibration.

`report.json` includes PSNR, SSIM, MAE, interval coverage/width/crossing, and
uncertainty-versus-error Pearson/Spearman correlations. A spatial overlap score
measures how often the most uncertain 10% of pixels are also among the largest
10% reconstruction errors; independent selections have an expected overlap of
10%. `examples/` contains the first 50 scenes in manifest order, showing original
input, noisy input, target, reconstruction, uncertainty and absolute error.
Heatmaps use the same fixed [0,1] intensity-width scale. Reports retain raw and
clipped metrics separately. Calibration is committed before test images open.
