# Category-specific calibration of generalized models

Calibrate frozen epoch-100 QUTCC and Im2im weights separately for each of 11
conditions (ten training families, with exposure at two gains). Use 3,000
calibration scenes and 2,000 disjoint test scenes per condition. Do not retrain.

```bash
python experiments/category_calibration/evaluate.py \
  --method qutcc --condition gaussian \
  --comparison-source /path/to/frozen/experiment4/source \
  --reference /path/to/original/experiment4/data.json \
  --manifest /path/to/split.json --output outputs/gaussian/qutcc
```

Requires CUDA, the frozen comparison source, the original demo data.json, and its
referenced epoch-100 checkpoint files. The reference supplies checkpoint hashes,
test-input hashes and global-calibration metrics; it never selects a calibration
parameter. Paths in this historical reference must point to the matching local
checkpoints. The calibration algorithm is empirical equal-tail q search for
QUTCC and asymmetric scale fitting for Im2im, not RCPS.

Each output folder records calibration.json, report.json, per-image metrics,
example PNGs and a READY.json completion receipt. Reports compare coverage and
width with global calibration and verify that inputs and reconstruction metrics
match the original comparison. Existing completed folders are reused only when
the request matches. Cluster submission and local download monitors remain
operational scripts on Unicorn/the original workstation, not portable commands.
