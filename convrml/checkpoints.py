"""Save weights, resume complete epochs, and load the selected model strictly."""

import hashlib
from collections import OrderedDict
from pathlib import Path

import torch

from .protocol import make_model, profile, settings, write_json


def read_checkpoint(path):
    """Hash and deserialize the same open file, even during atomic replacement."""
    with Path(path).open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
        stream.seek(0)
        checkpoint = torch.load(stream, map_location="cpu", weights_only=False)
    if (
        not isinstance(checkpoint, dict)
        or "model" not in checkpoint
        or "epoch" not in checkpoint
    ):
        raise ValueError("Expected a checkpoint with model state and completed epoch")
    return checkpoint, digest.hexdigest()


def identity(checkpoint):
    """Normalize recorded names from portable and historical checkpoints.

    Legacy ``ensemble_0`` identifies a member; ``mc_005`` identifies p=0.05.
    Historical ``camera=raw`` maps to the clean/corrupted DLMD dataset selectors.
    These translations affect metadata only; model tensors load separately.
    """
    meta = checkpoint.get("run") or checkpoint.get("config") or checkpoint
    arm = meta.get("arm", meta.get("method"))
    if isinstance(arm, str) and arm.startswith("ensemble_"):
        method, member = "ensemble", int(arm.split("_")[1])
    elif isinstance(arm, str) and arm.startswith("mc_"):
        if arm != "mc_005":
            raise ValueError("The published MC comparison uses p=0.05")
        method, member = "mc_dropout", None
    else:
        method, member = arm, meta.get("member")
    dataset = meta.get("dataset")
    if dataset is None:
        camera = meta.get("camera")
        if camera == "raw":
            dataset = (
                "raw_corrupted"
                if "corrupt" in meta.get("protocol", "")
                else "raw_clean"
            )
        else:
            dataset = camera
    return dict(
        method=method,
        dataset=dataset,
        member=member,
        epoch=checkpoint["epoch"],
        seeds=meta.get("seeds", {}),
        smoke=checkpoint.get("smoke", False),
        protocol=meta.get("protocol", meta.get("schema", checkpoint.get("schema"))),
    )


def validate_identity(checkpoint, run, *, smoke=False):
    """Reject a checkpoint for the wrong method, dataset or ensemble member."""
    info = identity(checkpoint)
    for key in ("method", "dataset", "member"):
        if info[key] != run[key]:
            raise ValueError(f"Checkpoint {key} does not match this evaluation")
    if info["smoke"] and not smoke:
        raise ValueError("Use --smoke to evaluate a smoke checkpoint")
    return info


def load_model(path, run, device, *, smoke=False, factory=None):
    checkpoint, digest = read_checkpoint(path)
    info = validate_identity(checkpoint, run, smoke=smoke)
    model = (factory or make_model)(run)
    load_model_weights(model, checkpoint["model"])
    model.to(device).eval()
    return model, dict(info, file=Path(path).name, sha256=digest)


def load_model_weights(model, state):
    """Strictly load weights, translating legacy Im2im names for any dataset.

    Im2im now always saves backbone.*. Older base.* and net.* dictionaries are
    renamed in memory; the original checkpoint and its provenance are untouched.
    Other model formats pass directly to PyTorch's strict loader.
    """
    from models import Im2im

    if isinstance(model, Im2im):
        roots = {key.split(".", 1)[0] for key in state}
        legacy = roots.intersection(("base", "net"))
        if legacy:
            if len(roots) != 1:
                raise ValueError("Ambiguous Im2im checkpoint: mixed parameter prefixes")
            prefix = next(iter(legacy))
            renamed = OrderedDict(
                ("backbone." + key[len(prefix) + 1 :], value)
                for key, value in state.items()
            )
            if hasattr(state, "_metadata"):
                renamed._metadata = {
                    (
                        "backbone" + key[len(prefix) :]
                        if key == prefix or key.startswith(prefix + ".")
                        else key
                    ): value
                    for key, value in state._metadata.items()
                }
            state = renamed
    return model.load_state_dict(state, strict=True)


def save_checkpoint(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def resume_training(
    path, model, optimizer, expected, *, device, quantile_rng, data_rng
):
    """Restore ordinary epoch-boundary state after checking settings and split IDs."""
    checkpoint, _ = read_checkpoint(path)
    if any(checkpoint[key] != expected[key] for key in ("run", "data", "smoke")):
        raise ValueError("Resume settings or dataset changed")
    load_model_weights(model, checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    torch.set_rng_state(checkpoint["rng"])
    if device.type == "cuda":
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    quantile_rng.set_state(checkpoint["quantile_rng"])
    data_rng.set_state(checkpoint["data_rng"])
    return checkpoint


def initialize_from_clean(model, path, method, *, smoke=False):
    """Initialize corruption adaptation from the same method's clean DLMD epoch 30."""
    checkpoint, digest = read_checkpoint(path)
    info = validate_identity(checkpoint, profile(3, "raw_clean", method), smoke=smoke)
    if (info["method"], info["dataset"], info["epoch"]) != (method, "raw_clean", 30):
        raise ValueError("Adaptation parent must be this method's clean raw epoch 30")
    load_model_weights(model, checkpoint["model"])
    return dict(sha256=digest, **info)


def model_checkpoint(checkpoint):
    """Select milestone fields from full state, retaining weight/data provenance.

    Optimizer moments, batch progress and RNG tensors belong only in latest.pt;
    milestone exports contain enough identity information for strict inference.
    """
    keys = (
        "schema",
        "run",
        "data",
        "model",
        "epoch",
        "history",
        "smoke",
        "parent",
        "source_sha256",
        "environment",
    )
    return {key: checkpoint[key] for key in keys if key in checkpoint}


def save_training_progress(
    output, checkpoint, optimizer, device, quantile_rng, data_rng
):
    """Save recovery state each epoch, milestone weights, and readable history.

    ``checkpoint`` supplies model weights, epoch/history, global step and run
    identity. latest.pt also needs Adam moments and every RNG stream for exact
    resume; milestone files retain only model/evaluation fields.
    """
    full = dict(
        checkpoint,
        optimizer=optimizer.state_dict(),
        rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        quantile_rng=quantile_rng.get_state(),
        data_rng=data_rng.get_state(),
    )
    save_checkpoint(output / "latest.pt", full)
    if checkpoint["epoch"] % settings()["checkpoint_interval"] == 0:
        destination = output / "checkpoints" / f"epoch_{checkpoint['epoch']:03d}.pt"
        if destination.exists():
            raise FileExistsError(destination)
        save_checkpoint(destination, model_checkpoint(full))
    write_json(output / "history.json", checkpoint["history"])


def load_predictor(paths, run, device, *, smoke=False, model_loader=None):
    """Load one model or aggregate independent, same-epoch ensemble members."""
    loaded, provenance = [], []
    for index, path in enumerate(paths):
        member = index if run["method"] == "ensemble" else None
        member_run = profile(run["experiment"], run["dataset"], run["method"], member)
        model, info = (model_loader or load_model)(
            path, member_run, device, smoke=smoke
        )
        loaded.append(model)
        provenance.append(info)
    if len({row["epoch"] for row in provenance}) != 1:
        raise ValueError("Ensemble members must use the same completed epoch")
    if run["method"] != "ensemble":
        return loaded[0], provenance

    for field in ("initialization", "data"):
        if len({row["seeds"].get(field) for row in provenance}) != len(provenance):
            raise ValueError(
                "Ensemble members need distinct initialization and data-order seeds"
            )
    from models import DeepEnsemble

    model = (
        DeepEnsemble(loaded, base_interval_alpha=settings()["alpha"]).to(device).eval()
    )
    return model, provenance
