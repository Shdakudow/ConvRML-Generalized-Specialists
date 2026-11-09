"""Train one specialist from matched untrained weights, or resume its full run."""

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY))
from convrml.protocol import sha256, write_json
from experiments.generalized_inputs import train as training
from experiments.specialists.protocol import SPECIALISTS, make_protocol


def weights_hash(state):
    """Use the original Gaussian comparison's tensor identity, before renaming."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def immutable_json(path, value):
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(f"Run configuration changed: {path.name}")
    write_json(path, value)


def source_identity(commit):
    """Hash the active model package and trainer, including this launcher."""
    return dict(
        commit=commit,
        files={
            str(path.relative_to(REPOSITORY)): sha256(path)
            for folder in (
                "models",
                "convrml",
                "input_modifications",
                "experiments/generalized_inputs",
                "experiments/specialists",
            )
            for path in sorted((REPOSITORY / folder).rglob("*.py"))
        },
        dataset_sha256=sha256(REPOSITORY / "dataset.py"),
        torch=torch.__version__,
        cuda=torch.version.cuda,
    )


def main(args):
    settings = make_protocol(args.condition)
    if sha256(args.manifest) != settings["manifest_sha256"]:
        raise ValueError("Not the original matched MIRFLICKR split")
    output = (
        args.root
        / ("smoke" if args.smoke else "results")
        / args.condition
        / args.method
    )
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    immutable_json(output / "protocol.json", settings)
    commit_path = REPOSITORY / "source_commit.txt"
    source = source_identity(commit_path.read_text().strip())
    immutable_json(output / "source.json", source)
    latest = output / "latest.pt"
    initial_path = args.initial_root / args.method / "initial_random.pt"
    initialization = output / "initialization.json"
    if not latest.exists():
        initial = torch.load(initial_path, map_location="cpu", weights_only=False)
        if (
            initial.get("epoch"),
            initial.get("trained_steps"),
            initial.get("method"),
            initial.get("pretrained"),
        ) != (0, 0, args.method, False):
            raise ValueError("Expected untrained epoch-zero weights")
        digest = weights_hash(initial["model"])
        if digest != settings["matched_initial_weights_sha256"][args.method]:
            raise ValueError("Initial tensors do not match the original comparison")
        immutable_json(
            initialization,
            dict(
                initialization="from scratch; matched untrained epoch-zero weights",
                method=args.method,
                trained_steps=0,
                pretrained=False,
                reference=str(initial_path),
                reference_file_sha256=sha256(initial_path),
                initial_weights_sha256=digest,
            ),
        )
        del initial
    elif not initialization.exists():
        raise ValueError("Recovery checkpoint has no initial-weight receipt")
    command = [
        "--method",
        args.method,
        "--manifest",
        str(args.manifest),
        "--output",
        str(output),
        "--device",
        "cuda",
        "--workers",
        "2",
    ]
    if latest.exists():
        command += ["--resume"]
    else:
        command += ["--initial-random-state", str(initial_path)]
    if args.smoke:
        command += ["--smoke"]
    else:
        smoke = args.root / "smoke" / args.condition / args.method / "SMOKE.json"
        checked = json.loads(smoke.read_text())
        if (
            not checked["passed"]
            or checked["source"] != source
            or checked["settings"] != settings
        ):
            raise ValueError(
                "A passing smoke check with this exact source/settings is required"
            )
    training.SETTINGS.clear()
    training.SETTINGS.update(settings)
    started = time.monotonic()
    write_json(
        output / "progress.json",
        dict(
            stage="training",
            condition=args.condition,
            method=args.method,
            target_epochs=50,
            resumed=latest.exists(),
        ),
    )
    training.run_training(training.arguments(command))
    saved = torch.load(latest, map_location="cpu", weights_only=False)
    expected_epochs = 1 if args.smoke else 50
    expected_count = 8 if args.smoke else 17999
    if saved["global_step"] != (2 if args.smoke else 225000):
        raise ValueError("Wrong optimizer-step budget")
    if saved["epoch"] != expected_epochs or [
        row["epoch"] for row in saved["history"]
    ] != list(range(1, expected_epochs + 1)):
        raise ValueError("Training did not complete the expected epochs")
    if any(
        row["condition_counts"] != {args.condition: expected_count}
        for row in saved["history"]
    ):
        raise ValueError("Training used the wrong input distribution")
    for row in saved["history"]:
        if row["validation"]["count"] != (20 if args.smoke else 1000):
            raise ValueError("Validation panel changed")
    if args.smoke:
        initial = torch.load(
            output / "initial_random.pt", map_location="cpu", weights_only=False
        )
        if all(
            torch.equal(initial["model"][key], tensor)
            for key, tensor in saved["model"].items()
        ):
            raise ValueError("Smoke optimizer did not update model weights")
    result = dict(
        passed=True,
        condition=args.condition,
        method=args.method,
        epochs=expected_epochs,
        global_step=saved["global_step"],
        settings=settings,
        source=source,
        seconds=time.monotonic() - started,
        gpu=torch.cuda.get_device_name(),
        primary_validation=saved["history"][-1]["validation"]["conditions"][
            args.condition
        ],
    )
    write_json(output / ("SMOKE.json" if args.smoke else "COMPLETE.json"), result)
    write_json(
        output / "progress.json",
        dict(stage="smoke_complete" if args.smoke else "complete", **result),
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=SPECIALISTS, required=True)
    parser.add_argument("--method", choices=("qutcc", "im2im"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--initial-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    main(parser.parse_args())
