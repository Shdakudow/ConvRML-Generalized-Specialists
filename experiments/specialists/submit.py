"""Submit explicitly selected specialist families once, with bounded GPU use."""

import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY))
from convrml.protocol import sha256, write_json
from experiments.specialists.protocol import SPECIALISTS, make_protocol


def main(args):
    if not 1 <= args.max_parallel <= 4 or len(set(args.conditions)) != len(
        args.conditions
    ):
        raise ValueError("Choose unique conditions and one to four active GPUs")
    settings = make_protocol(args.conditions[0])
    if sha256(args.manifest) != settings["manifest_sha256"]:
        raise ValueError("Wrong original split")
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "logs").mkdir(exist_ok=True)
    tasks = [
        dict(index=i, condition=condition, method=method)
        for i, (condition, method) in enumerate(
            (c, m) for c in args.conditions for m in ("qutcc", "im2im")
        )
    ]
    plan = dict(
        root=str(args.root.resolve()),
        manifest=str(args.manifest.resolve()),
        initial_root=str(args.initial_root.resolve()),
        epochs=50,
        conditions=args.conditions,
        max_parallel=args.max_parallel,
        source_commit=(REPOSITORY / "source_commit.txt").read_text().strip(),
        tasks=tasks,
    )
    with (args.root / "submission.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        existing = args.root / "submission.json"
        if existing.exists():
            saved = json.loads(existing.read_text())
            if saved["plan"] != plan:
                raise ValueError("A different specialist submission already exists")
            print(json.dumps(saved))
            return
        for condition in args.conditions:
            write_json(
                args.root / "protocols" / f"{condition}.json", make_protocol(condition)
            )
        write_json(args.root / "tasks.json", plan)
        command = [
            "/usr/local/slurm/current/bin/sbatch",
            "--parsable",
            "--job-name=specialist50",
            f"--array=0-{len(tasks)-1}%{args.max_parallel}",
            f"--chdir={REPOSITORY}",
            f"--output={args.root}/logs/%A_%a.out",
            f"--error={args.root}/logs/%A_%a.err",
            str(Path(__file__).with_name("run.sbatch")),
            str(Path(__file__).with_name("task.py")),
            "--plan",
            str(args.root / "tasks.json"),
        ]
        job = subprocess.check_output(command, text=True).strip().split(";")[0]
        if not job.isdigit():
            raise ValueError("Unrecognized Slurm submission response")
        receipt = dict(job=job, plan=plan, command=command)
        write_json(existing, receipt)
        print(json.dumps(receipt))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", nargs="+", choices=SPECIALISTS, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--initial-root", type=Path, required=True)
    parser.add_argument("--max-parallel", type=int, default=2)
    main(parser.parse_args())
