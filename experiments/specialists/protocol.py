"""Matched 50-epoch configurations for the remaining input specialists."""

import json
from pathlib import Path

from input_modifications.online import CONDITIONS
from input_modifications.presets import PRESETS

REPOSITORY = Path(__file__).resolve().parents[2]
SPECIALISTS = tuple(name for name in CONDITIONS if name not in ("clean", "gaussian"))


def make_protocol(condition):
    if condition not in SPECIALISTS:
        raise ValueError(f"Not a remaining specialist condition: {condition}")
    source = REPOSITORY / "experiments/gaussian_specialist/protocol.json"
    settings = json.loads(source.read_text())
    settings.update(
        protocol=f"raw_online_{condition}_scratch50_20261104_v1",
        training_condition=condition,
        training_distribution={condition: 1.0},
        primary_validation_condition=condition,
        modification_parameters=(
            {"global_gains": [0.5, 1.5], "probabilities": [0.5, 0.5]}
            if condition == "exposure"
            else dict(PRESETS[condition])
        ),
        modification_sampling="independent per image; generated on demand",
    )
    settings.pop("gaussian_sigma")
    return settings
