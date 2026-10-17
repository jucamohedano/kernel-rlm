"""Validate experiment records or prepare inputs locally; never launch compute."""

import argparse
import json
from pathlib import Path

from . import prepare_run, validate_registry


def main() -> None:
    environment = Path(__file__).resolve().parents[1]
    project = environment.parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("validate", "prepare"))
    parser.add_argument("--experiment", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    result = (
        validate_registry(project, environment)
        if args.action == "validate"
        else prepare_run(
            project,
            environment,
            args.experiment,
            args.run_id,
            pilot=args.pilot,
            allow_dirty=args.allow_dirty,
        )
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
