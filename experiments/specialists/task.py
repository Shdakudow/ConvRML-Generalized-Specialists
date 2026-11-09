"""Run the array member's GPU smoke check, followed by its fresh 50-epoch run."""

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.specialists.run import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    task = plan["tasks"][int(os.environ["SLURM_ARRAY_TASK_ID"])]
    run = argparse.Namespace(
        condition=task["condition"],
        method=task["method"],
        root=Path(plan["root"]),
        manifest=Path(plan["manifest"]),
        initial_root=Path(plan["initial_root"]),
        smoke=True,
    )
    smoke = run.root / "smoke" / run.condition / run.method / "SMOKE.json"
    if not smoke.exists():
        main(run)
    run.smoke = False
    main(run)
