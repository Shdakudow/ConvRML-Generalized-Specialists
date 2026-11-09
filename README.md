# ConvRML: generalized models and corruption specialists

Compare **QUTCC and Im2im** trained on a mixture of measurement conditions with
models trained on one condition. Ground-truth images stay unchanged.

The four-model baseline comparison remains in
[ConvRML-Uncertainty](https://github.com/Shdakudow/ConvRML-Uncertainty).
This repository contains the separate generalized/specialist studies.

## Install

Use Python 3.9+ for training (the frozen evaluation environment uses Python 3.11).
Install PyTorch/torchvision for your hardware, then:

```bash
python -m pip install -r requirements.txt -c reproducibility/constraints-tested.txt
python experiments/generalized_inputs/train.py --help
```

## What to read

| Folder | Purpose |
| --- | --- |
| `experiments/generalized_inputs/` | Mixture training, up to 100 epochs |
| `experiments/gaussian_specialist/` | Gaussian-only 50-epoch protocol and evaluation |
| `experiments/specialists/` | Eight other 50-epoch specialist families; optional Slurm submission |
| `experiments/category_calibration/` | Separate calibration of an existing generalized model for each condition |
| `input_modifications/` | Noise, exposure, blur, missing pixels, resolution and JPEG changes |
| `dataset.py` | Paired measurement/target image loading |
| `models/`, `convrml/`, `configs/` | Shared runtime snapshot from the baseline repository |
| `reproducibility/` | Historical launch records, environment and migration hashes |
| `tests/` | Small CPU checks for data, corruptions, training and resume |

The runtime snapshot keeps the original import paths and includes the shared
model package; these studies train only Im2im and QUTCC. It is intentionally
independent of future baseline-repository updates. Python files moved from the
baseline are unchanged; `reproducibility/migration.json` records their hashes
and original commit. `source_commit.txt` identifies that upstream snapshot.

## Data

Images and weights are external, not stored in this repository. The recorded
MIRFLICKR split has 17,999 training, 2,000 validation, 3,000 calibration and 2,000
test pairs. A split file lists image paths:

```json
{"roles": {"train": [{"id": "scene_a", "measurement": {"path": "measurements/a.png"}, "target": {"path": "targets/a.png"}}]}}
```

Include the other split lists in the same file. Paths may be absolute or relative
to that file. Prepared RGB images are converted from uint8 to float32 / 255;
the models handle padding and cropping. The specialist launchers reproduce the
recorded split and require its hash and matched untrained initialization files.

## Train

```bash
python experiments/generalized_inputs/train.py \
  --method qutcc --manifest /path/to/split.json \
  --output runs/generalized/qutcc --epochs 100 --device cuda
```

Repeat with `--method im2im` and a separate output directory. The mixture samples
clean inputs or one of nine modifications per image. Each family has probability
0.1; exposure uses 0.5 or 1.5 gain within its share. Targets never change.

For a Gaussian specialist, use the same trainer with
`--protocol experiments/gaussian_specialist/protocol.json`, `--epochs 50`,
and `--initial-random-state /path/to/matched/initial_random.pt`.
See the [full Gaussian instructions](experiments/gaussian_specialist/README.md).

Other specialists use the matched, untrained initial weights from
`INITIAL_ROOT/METHOD/initial_random.pt`:

```bash
python experiments/specialists/run.py \
  --condition poisson --method qutcc \
  --manifest /path/to/split.json --initial-root /path/to/initial-root \
  --root runs/specialists
```

This launcher requires CUDA and the original matched initialization/split.
See [specialist details](experiments/specialists/README.md) for the eight families.
The optional `run.sbatch` contains Cornell-specific paths/account settings;
adapt those for another cluster. Do not use this relocated snapshot to resume
existing deployed jobs: those retain their frozen sources and checkpoints.

## Calibration and evaluation

Training never uses calibration or test targets. Gaussian and category-specific
evaluators require the frozen Exp4 comparison source via `--comparison-source`.
That source is an external deployment artifact, not bundled here. They reuse its
deterministic corruptions and empirical calibration to reproduce the demo;
this is **not an RCPS confidence guarantee**. See
[category-calibration instructions](experiments/category_calibration/README.md).

Category-specific fits compare with the generalized model's global fit using the
same checkpoint and test measurements. Calibration targets 90%; test coverage is
measured independently and can differ. A failed fit remains a reported failure.

## Verify

```bash
python -m unittest discover -s tests -v
```

No datasets, checkpoints or active cluster jobs were moved during repository
separation. Historical source stays in the original repository's archive.
