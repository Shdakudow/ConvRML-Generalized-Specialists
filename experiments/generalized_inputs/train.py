"""Train Im2im or QUTCC on clean/corrupted measurements, with epoch-boundary resume.

The model/loss definitions are shared with root train.py. protocol.json records
this study's input distribution and two-phase learning-rate schedule. Historical
Slurm/requeue engines are preserved in archive/research_prototype_20261102/.
"""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
import time

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchmetrics.functional.image import structural_similarity_index_measure as ssim

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY))
from convrml.checkpoints import load_model_weights, save_checkpoint
from convrml.protocol import audit_manifest, write_json
from dataset import ImagePairs
from input_modifications.gaussian import gaussian_noise
from input_modifications.online import (
    CONDITIONS,
    apply_condition,
    sample_training_modification,
)
from models import Im2im, QUTCC

SETTINGS_PATH = Path(__file__).with_name("protocol.json")
SETTINGS = json.loads(SETTINGS_PATH.read_text())


def learning_rate(step, total_steps):
    """Keep the first 50 epochs intact, then use a separate lower-rate phase."""
    if "base_epochs" in SETTINGS:
        boundary = total_steps * SETTINGS["base_epochs"] / SETTINGS["epochs"]
        if step >= boundary:
            fraction = min((step - boundary) / (total_steps - boundary), 1.0)
            cosine = (1 + math.cos(math.pi * fraction)) / 2
            return (
                SETTINGS["continuation_final_lr"]
                + (
                    SETTINGS["continuation_initial_lr"]
                    - SETTINGS["continuation_final_lr"]
                )
                * cosine
            )
        total_steps = boundary
    fraction = min(step / total_steps, 1.0)
    warmup = SETTINGS["warmup_fraction"]
    if fraction <= warmup:
        return SETTINGS["initial_lr"] + (fraction / warmup) * (
            SETTINGS["peak_lr"] - SETTINGS["initial_lr"]
        )
    cosine = (1 + math.cos(math.pi * (fraction - warmup) / (1 - warmup))) / 2
    return SETTINGS["final_lr"] + (SETTINGS["peak_lr"] - SETTINGS["final_lr"]) * cosine


class ValidationPanel(Dataset):
    """The same held-out scenes under every condition, with fixed random draws."""

    def __init__(self, original, count):
        self.original = original
        self.indices = torch.randperm(
            len(original),
            generator=torch.Generator().manual_seed(SETTINGS["validation_seed"]),
        )[:count].tolist()

    def __len__(self):
        return len(CONDITIONS) * len(self.indices)

    def __getitem__(self, index):
        condition_index, scene_index = divmod(index, len(self.indices))
        source_index = self.indices[scene_index]
        measurement, target = self.original[source_index]
        generator = torch.Generator().manual_seed(
            SETTINGS["validation_seed"] + 10000 * condition_index + source_index
        )
        modified = apply_condition(
            measurement.unsqueeze(0), CONDITIONS[condition_index], generator
        )[0]
        return modified, target, condition_index


def modify_training_batch(measurement, condition, generator):
    """Apply the chosen family independently per image, or sample the mixture."""
    if condition == "mixture":
        return sample_training_modification(measurement, generator=generator)
    if condition == "gaussian":
        # Preserve the original Gaussian specialist's random-number stream.
        modified = gaussian_noise(
            measurement, sigma=SETTINGS["gaussian_sigma"], generator=generator
        )
    elif condition in CONDITIONS:
        # Exposure gains and sensor masks/patterns must not be shared by a batch.
        modified = torch.cat(
            [
                apply_condition(image.unsqueeze(0), condition, generator)
                for image in measurement
            ]
        )
    else:
        raise ValueError(f"Unknown training condition: {condition}")
    return modified, [condition] * len(measurement)


def train_one_epoch(
    model,
    optimizer,
    loader,
    device,
    augmentation_rng,
    quantile_rng,
    global_step,
    total_steps,
    smoke=False,
):
    """Change measurements only, accumulate one batch's gradients, then update Adam."""
    model.train()
    loss_sum = 0.0
    image_count = 0
    condition_counts = Counter()
    for index, batch in enumerate(loader):
        if smoke and index == 2:
            break
        measurement = batch[:, 0]
        condition = SETTINGS.get("training_condition", "mixture")
        modified, labels = modify_training_batch(
            measurement, condition, augmentation_rng
        )
        lr = learning_rate(global_step, total_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        batch_loss = 0.0
        for sample_index in range(len(batch)):
            y = modified[sample_index : sample_index + 1].to(device, non_blocking=True)
            target = batch[sample_index : sample_index + 1, 1].to(
                device, non_blocking=True
            )
            if isinstance(model, QUTCC):
                loss = model.loss(y, target, generator=quantile_rng)
            else:
                loss = model.loss(y, target)
            loss = loss / len(batch)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            batch_loss += float(loss.detach())
        optimizer.step()
        global_step += 1
        loss_sum += batch_loss * len(batch)
        image_count += len(batch)
        condition_counts.update(labels)
    return dict(
        loss=loss_sum / image_count,
        lr=lr,
        global_step=global_step,
        condition_counts=dict(condition_counts),
    )


@torch.inference_mode()
def validate(model, panel, device, workers=2):
    """Measure each condition on the same held-out scenes and fixed noise draws."""
    model.eval()
    totals = {name: Counter() for name in CONDITIONS}
    for measurement, target, condition in DataLoader(
        panel, batch_size=1, num_workers=workers, pin_memory=True
    ):
        target = target.to(device, non_blocking=True)
        prediction = model.center(measurement.to(device, non_blocking=True))
        if not torch.isfinite(prediction).all():
            raise ValueError("Nonfinite validation prediction")
        prediction = prediction.clamp(0, 1)
        row = totals[CONDITIONS[condition.item()]]
        row["psnr"] += float(
            -10 * (prediction - target).square().mean().clamp_min(1e-12).log10()
        )
        row["ssim"] += float(ssim(prediction, target, data_range=1.0))
        row["mae"] += float((prediction - target).abs().mean())
        row["count"] += 1
    conditions = {
        name: dict(
            count=row["count"],
            **{key: row[key] / row["count"] for key in ("psnr", "ssim", "mae")},
        )
        for name, row in totals.items()
    }
    return dict(
        conditions=conditions,
        macro_psnr=sum(row["psnr"] for row in conditions.values()) / len(CONDITIONS),
        count=len(panel),
        clipped=True,
        interval_calibration_performed=False,
    )


def run_training(args):
    """Load data, train/validate complete epochs and save standard recovery state."""
    torch.set_num_threads(2)
    torch.manual_seed(SETTINGS["seed"])
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    latest = output / "latest.pt"
    if latest.exists() and not args.resume:
        raise FileExistsError("Use --resume or a new output folder")
    target_epoch = SETTINGS["epochs"] if args.epochs is None else args.epochs
    if args.smoke:
        target_epoch = 1
    if not 1 <= target_epoch <= SETTINGS["epochs"] or args.workers < 0:
        raise ValueError("Invalid epoch or worker count")
    data = audit_manifest(args.manifest)
    training = ImagePairs(args.manifest, "train")
    validation = ImagePairs(args.manifest, "val")
    panel = ValidationPanel(
        validation, 2 if args.smoke else SETTINGS["validation_per_condition"]
    )
    model = QUTCC() if args.method == "qutcc" else Im2im()
    if args.resume and args.initial_random_state:
        raise ValueError("Choose --resume or --initial-random-state")
    if SETTINGS.get("matched_initial_weights_sha256") and not (
        args.resume or args.initial_random_state
    ):
        raise ValueError("Matched comparison requires --initial-random-state")
    if args.initial_random_state and not args.resume:
        initial = torch.load(
            args.initial_random_state, map_location="cpu", weights_only=False
        )
        if (
            initial.get("epoch") != 0
            or initial.get("trained_steps") != 0
            or initial.get("method") != args.method
            or initial.get("pretrained") is not False
        ):
            raise ValueError(
                "Matched initialization requires untrained epoch-zero weights"
            )
        load_model_weights(model, initial["model"])
    if not args.resume:
        save_checkpoint(
            output / "initial_random.pt",
            dict(
                epoch=0,
                trained_steps=0,
                pretrained=False,
                method=args.method,
                model=model.state_dict(),
            ),
        )
    device = torch.device(args.device)
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=SETTINGS["initial_lr"], betas=tuple(SETTINGS["betas"])
    )
    augmentation_rng = torch.Generator().manual_seed(SETTINGS["seed"] + 1000)
    quantile_rng = torch.Generator(device=device).manual_seed(SETTINGS["seed"] + 2000)
    config = dict(SETTINGS, method=args.method, dataset="raw_clean", member=None)
    completed_epoch = global_step = 0
    history = []
    if args.resume:
        state = torch.load(latest, map_location="cpu", weights_only=False)
        if (
            state["config"] != config
            or state["data"] != data
            or state["smoke"] != args.smoke
        ):
            raise ValueError("Resume settings or dataset changed")
        load_model_weights(model, state["model"])
        optimizer.load_state_dict(state["optimizer"])
        augmentation_rng.set_state(state["augmentation_rng"])
        quantile_rng.set_state(state["quantile_rng"])
        torch.set_rng_state(state["cpu_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"])
        completed_epoch, global_step, history = (
            state["epoch"],
            state["global_step"],
            state["history"],
        )
    if completed_epoch > target_epoch:
        raise ValueError("Target precedes the saved epoch")
    write_json(output / "config.json", dict(config=config, data=data, smoke=args.smoke))
    total_steps = SETTINGS["epochs"] * math.ceil(
        len(training) / SETTINGS["effective_batch"]
    )
    for epoch in range(completed_epoch + 1, target_epoch + 1):
        order = torch.randperm(
            len(training),
            generator=torch.Generator().manual_seed(SETTINGS["seed"] + epoch),
        ).tolist()
        loader = DataLoader(
            Subset(training, order),
            batch_size=SETTINGS["effective_batch"],
            num_workers=args.workers,
            pin_memory=True,
            generator=torch.Generator().manual_seed(SETTINGS["seed"] + epoch + 10000),
        )
        started = time.monotonic()
        stats = train_one_epoch(
            model,
            optimizer,
            loader,
            device,
            augmentation_rng,
            quantile_rng,
            global_step,
            total_steps,
            args.smoke,
        )
        global_step = stats["global_step"]
        measured = validate(model, panel, device, args.workers)
        history.append(
            dict(
                epoch=epoch,
                training_loss=stats["loss"],
                learning_rate=stats["lr"],
                condition_counts=stats["condition_counts"],
                validation=measured,
                training_and_validation_seconds=time.monotonic() - started,
            )
        )
        state = dict(
            config=config,
            data=data,
            epoch=epoch,
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            global_step=global_step,
            history=history,
            augmentation_rng=augmentation_rng.get_state(),
            quantile_rng=quantile_rng.get_state(),
            cpu_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state() if device.type == "cuda" else None,
            smoke=args.smoke,
        )
        save_checkpoint(latest, state)
        if (
            epoch % SETTINGS["milestone_every_epochs"] == 0
            or epoch == SETTINGS["epochs"]
        ):
            save_checkpoint(
                output / "checkpoints" / f"epoch_{epoch:03d}.pt",
                dict(
                    config=config,
                    epoch=epoch,
                    model=state["model"],
                    history=history,
                    smoke=args.smoke,
                ),
            )
        write_json(output / "history.json", history)
        print(
            f"Epoch {epoch}/{target_epoch}: loss={stats['loss']:.6g}, validation PSNR={measured['macro_psnr']:.4f}",
            flush=True,
        )
    return latest


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=SETTINGS_PATH)
    parser.add_argument("--method", choices=["qutcc", "im2im"], required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial-random-state", type=Path)
    parser.add_argument(
        "--epochs", type=int, help="Absolute stopping epoch; full schedule stays fixed."
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = arguments()
    SETTINGS.clear()
    SETTINGS.update(json.loads(args.protocol.read_text()))
    run_training(args)
