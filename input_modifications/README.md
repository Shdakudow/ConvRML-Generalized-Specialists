# Raw-input modifications for MIRFLICKR / DiffuserCam

Every method has its own implementation file. All methods modify **the recorded
measurement only**. The paired lensed scene target stays unchanged. Functions
accept finite floating-point tensors shaped `(B, C, H, W)` in `[0, 1]`, return
the same shape/device/dtype, and never edit the input in place.

The existing MIRFLICKR inputs are prepared 8-bit RGB exports, loaded by dividing
by 255. They are not untouched native 12-/16-bit sensor arrays. Synthetic noise
is additional to noise and processing already present in these recordings.

| Method file | Default | Interpretation |
| --- | --- | --- |
| [gaussian.py](gaussian.py) | SD `0.01` | Additive signal-independent noise |
| [poisson.py](poisson.py) | Count scale `1330` | Signal-dependent shot noise; SD is `sqrt(y/1330)` |
| [poisson_gaussian.py](poisson_gaussian.py) | SD `0.01`, count scale `1330` | Simultaneous shot + read noise; clip once after both |
| [exposure.py](exposure.py) | Gains `0.50` and `1.50` in separate datasets | Brightness/gain mismatch on existing measurements |
| [dead_pixels.py](dead_pixels.py) | `10%` of spatial locations | Persistent missing RGB locations; strong failure stress test |
| [fixed_pattern.py](fixed_pattern.py) | Gain SD `1%`, offset SD `0.003` | Persistent per-channel pixel gain and offset variation |
| [gaussian_blur.py](gaussian_blur.py) | Sigma `0.5` pixels, `5×5` kernel | Gaussian measurement-plane filtering |
| [resolution.py](resolution.py) | Averaging factor `64` | Bin, bilinearly restore, then crop to the original dimensions |
| [jpeg.py](jpeg.py) | Quality `50`, subsampling `0` / 4:4:4 | One lossy JPEG round-trip without chroma downsampling |

The count scale `1330` gives about `0.01` **RMS added noise** for a dataset mean
of `0.133`, estimated from the 50 saved clean demo measurements. It is not
constant per-pixel SD or a calibrated photon count. At other intensities the
Poisson SD changes. Combined Gaussian + Poisson noise has approximately `0.0141`
RMS at that reference mean. All these values precede clipping and PNG quantization;
the preparation report measures the actual saved-image changes.

Exposure scaling also scales existing noise; it is not a fresh acquisition with
new exposure-dependent shot/read statistics. Gaussian filtering does not simulate
a changed diffuser PSF. JPEG tests a compressed-input pipeline. These assumptions
must remain explicit when interpreting reconstruction quality or uncertainty.

## Use one method

```python
import torch
from input_modifications.gaussian import gaussian_noise
from input_modifications.dead_pixels import sample_dead_pixel_mask, apply_dead_pixels

# measurement: (B, 3, H, W), float32 in [0,1]. Leave scene_target unchanged.
rng = torch.Generator(device=measurement.device).manual_seed(42)
noisy_measurement = gaussian_noise(measurement, sigma=0.01, generator=rng)

# Sample a sensor once; reuse its mask on every frame from that sensor.
dead_mask = sample_dead_pixel_mask(measurement, drop_rate=0.10, generator=rng)
defective_measurement = apply_dead_pixels(measurement, dead_mask)
```

## Prepare independent datasets

### Train without saving modified copies

[online.py](online.py) samples **one condition per image**: 10% clean and
10% for each of the nine method families. It uses the settings in the table above.
Exposure is one family: its share splits equally between global gains 0.50 and
1.50 (5% each overall). Other strengths stay fixed in this first sampler.
Probabilities are approximate frequencies, not exact quotas in every batch.

This module is an online input transform. The existing Dataset/DataLoader reads
the original measurement/target pairs; the transform modifies only the measurement
batch before the model forward pass. Model inference, loss computation, and
optimizer updates remain in the training loop. No training loop is implemented
in this module.

```python
from input_modifications.online import sample_training_modification

# Create once, outside the epoch/batch loop; do not reset it on every batch.
rng = torch.Generator().manual_seed(42)
for measurement, target in train_loader:  # CPU float BCHW tensors in [0,1]
    model_input, conditions = sample_training_modification(
        measurement, generator=rng
    )
    # Feed model_input to the model; use the original target for its loss.
```

Call this only on training measurements. Both models can use the same protocol.
Inputs must have one or three channels and height/width at least 64. No image
copies are saved, and targets are never passed to the sampler. Each image samples
an independent simulated sensor mask/pattern; use the individual helpers to share
patterns within a sensor's frames. Tensor transforms keep floating point values;
JPEG includes codec quantization. This does not round every transform to PNG.
For CPU data-loader workers, initialize a separate generator per worker from its
PyTorch worker seed; a copied generator would otherwise repeat streams. Save the
generator state in a training checkpoint to resume its sequence.

### Save independent datasets when needed

```bash
python -m input_modifications.prepare_dataset \
  --manifest /path/to/original/split.json \
  --output /path/to/new/raw-input-modifications \
  --workers 8
```

`split.json` must be the original paired-path manifest accepted by
`dataset.ImagePairs`, containing `roles.train`, `roles.val`, `roles.calibration`,
and `roles.test`. Relative source paths resolve against that manifest. Original
file/pixel hashes are checked when present and always recorded in the outputs.

The full MIRFLICKR split has 17,999 training, 2,000 validation, 3,000 calibration,
and 2,000 test pairs: **24,999 pairs**. Ten independent modifications generate
**249,990 PNG measurements**, with eleven separate manifests including a clean
reference that simply points to the original files. Methods are not randomly
mixed within the same dataset. Only `poisson_gaussian` deliberately combines two
noise processes.

```text
raw-input-modifications/
  protocol.json                 settings, source/code hashes, software versions
  progress.json                 completed source pairs and generated file count
  READY.json                    final verification and measured input changes
  preview.jpg                   one source example per split, all variants
  sensor/                       persistent dead mask and gain/offset arrays
  clean/manifest.json           references originals; no copies or changes
  gaussian/manifest.json        same IDs/targets, generated measurement paths
    measurements/train/*.png
    measurements/val/*.png
    measurements/calibration/*.png
    measurements/test/*.png
  poisson/                      separate manifest and measurement files
  poisson_gaussian/
  exposure_050/
  exposure_150/
  dead_pixels/
  fixed_pattern/
  gaussian_blur/
  resolution/
  jpeg/
```

Every generated PNG is decoded again and compared with its intended pixels.
Each manifest records source/output file and pixel hashes, unchanged target
paths/hashes, parameters, deterministic named seeds, and the input distortion.
IDs/order remain identical across variants. Output paths are kept separate from
all source-image folders. Sensor patterns are shared across **all splits and
frames**; Gaussian/Poisson draws use separate reproducible per-image streams.

Factor 64 applies to each spatial axis: 270×480 is padded on the bottom/right to
320×512, pooled to just 5×8 bins, interpolated to 320×512, then cropped back to
270×480. The large replicated border influences the edge bins; this is an
extreme binning-and-resampling stress test.
The original bottom two rows are included. JPEG is decoded once, then stored as
PNG to prevent an accidental second lossy encode.

Use `--limit-per-split 2` with a separate output directory for a small smoke run;
every smoke manifest is explicitly marked. Use `--variants gaussian poisson`
to prepare a subset. An interrupted build can be resumed with identical code,
source, seed, settings and software. Existing outputs must reproduce exactly;
changed settings require a new directory.

Generated images remain outside Git. This command prepares and verifies data;
it does not train models or recalibrate checkpoints. The historical retired
`raw_corrupted` dataset fingerprint is **not** the identity of these new datasets.
Any subsequent training/evaluation needs an explicitly recorded new protocol.

The [earlier MIRFLICKR preparation report](../reproducibility/raw_input_modifications_20261030.json)
records all 249,990 generated measurements, the independent audit, exact settings,
source/code identities, dataset paths, and measured changes after PNG quantization.
That immutable run used JPEG 85, factor 4, and gains 0.75/1.25. The current
defaults above are the stronger follow-up settings; use a new output directory
when materializing them.

The [stronger-setting reconstruction pilot](../reproducibility/stronger_input_reconstruction_20261030.json)
evaluates JPEG 50, resolution factor 8, and global exposure gains 0.50/1.50 on the
same first 50 held-out images, using the existing clean QUTCC/Im2im checkpoints
and unchanged clean calibration. It records inference results, not generalized
training results.

The [factor-64 reconstruction pilot](../reproducibility/resolution64_reconstruction_20261030.json)
uses the same 50 scenes and frozen models/calibration. At only 5×8 averaged bins,
mean PSNR falls to 13.97 dB (QUTCC) and 13.93 dB (Im2im), with empirical interval
coverage of 28.02% and 25.30%. These are pilot inference results; the full dataset
has not been regenerated at factor 64.

## Verification

```bash
python -m unittest tests.test_input_modifications -v
```

Checks cover noise statistics, seeded reproducibility, persistent defects,
unmodified originals/targets, factor-four/eight/64 edge retention, JPEG formats, separate
loader-compatible manifests, and identical resumption with a different worker count.

[Generalized-model training design](GENERALIZATION.md) explains joint sampling,
parameter conditioning, and validation. The
[training runner](../experiments/generalized_inputs/README.md) trains the agreed
online mixture. The optional parameter-conditioning proposal remains unimplemented.

The current plan completes 50 epochs, then continues through epoch 100 with a
separate lower-rate phase, preserving the trained state and all original targets.
