# Generalized input training

Im2im and QUTCC start from random weights and reuse the shared model losses.
Each epoch visits the training pairs once. The online sampler chooses clean or
one of nine modifications with 10% probability each per image. Exposure uses
0.5 or 1.5 with equal probability within its share. Targets stay unchanged;
modified datasets are not stored. See [input modifications](../../input_modifications/README.md).

```bash
python experiments/generalized_inputs/train.py \
  --method qutcc --manifest /path/to/split.json \
  --output runs/generalized/qutcc --epochs 50 --device cuda
```

Repeat with `--method im2im` and a separate output directory. The default
[protocol](protocol.json) specifies Adam, effective batch 4, one-image
microbatches and FP32. Its full horizon is 100 epochs:

- Epochs 1–9: linear warmup from 1e-5 to 1e-4.
- Epochs 10–50: cosine decay toward 1e-9.
- Epochs 51–100: a separate cosine phase from 1e-5 to 1e-6, preserving Adam moments.

`--epochs 50` stops at 50 without changing the schedule. Continue by using the
same command with `--epochs 100 --resume`. Resume checks settings/split identity
and restores optimizer and augmentation/quantile RNG state after the last
complete epoch. Interrupted partial epochs are repeated.

After each epoch, the same 100 validation scenes are tested under all ten
conditions (1,000 predictions). Scene selection and noise draws are fixed.
Logs include per-condition PSNR, SSIM, MAE and training condition counts.
Calibration and test pairs are not opened by this trainer.

Outputs: `config.json`, `initial_random.pt` (untrained weights), `history.json`,
`latest.pt` (full epoch recovery), and `checkpoints/epoch_*.pt` every ten epochs.
`--smoke` performs at most two training batches and one small validation panel;
use a separate output directory for smoke checks.

Read `run_training` for setup, epochs and saving; `train_one_epoch` for input
modification, loss and Adam; `ValidationPanel`/`validate` for held-out metrics.

Historical launchers, mid-epoch recovery, extension migrations and exact frozen
sources are in the [snapshot](https://github.com/Shdakudow/ConvRML-Uncertainty/tree/f00b699876f348c84947f935dc223810ece03cca/archive/research_prototype_20261102/).
Resume existing cluster deployments with their original engine. Published launch
records in `reproducibility/` describe those frozen runs, rather than new runs
of the simplified trainer.
