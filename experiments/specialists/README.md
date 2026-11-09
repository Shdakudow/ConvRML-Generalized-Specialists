# Individual corruption specialists

Use the shared generalized-input trainer to train one corruption family at a
time. Each family has a QUTCC and Im2im run for 50 epochs. Every training input
uses the selected family; targets are unchanged. Modifications are generated
on demand rather than storing additional datasets.

Available families are Poisson, Poisson + Gaussian, exposure, dead pixels,
uneven pixel response (`fixed_pattern`), blur, resolution loss, and JPEG.
Parameter values come from `input_modifications/presets.py`. Exposure randomly
uses global gain 0.50 or 1.50 independently per image. Sensor masks and patterns
are also resampled independently per image.

The original Gaussian specialist's optimizer, 50-epoch learning-rate schedule,
data order, and 100-scene × ten-condition validation panel are preserved. Each
method loads the same **untrained epoch-zero weights** as the original study,
verified using its original tensor hash, and starts with a fresh Adam optimizer.
No trained Gaussian or generalized weights initialize a specialist.

`run.py` verifies the split, initialization, and frozen source, then calls the
shared trainer. `task.py` requires a successful GPU smoke check before starting
the full run in a separate folder. Smoke weights never initialize full training.
`submit.py --conditions ...` records an explicit, duplicate-safe Slurm array,
running at most two tasks concurrently by default. On scheduler time limits,
`run.sbatch` requeues the task; the trainer restores the last complete epoch,
including optimizer and random-generator states. Partial epochs are replayed.

Full results live at `results/CONDITION/METHOD/`; smoke outputs are separate.
Recovery state is saved in `latest.pt` every epoch, and weight milestones in
`checkpoints/epoch_010.pt` through `epoch_050.pt`. Configurations, input counts,
per-condition validation histories, source identity, and completion receipts
are saved beside the weights. Calibration and test sets are not used in training.
