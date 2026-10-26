"""Explicit command-line operations; imports do not launch calculations."""

import argparse
import json
import os
from pathlib import Path

for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ.setdefault(variable, "1")


def main():
    from .cases import CASES, Case, get_case

    parser = argparse.ArgumentParser(
        prog="ghost-lattice",
        description="Polycrystal LPM calculations and independent reference generation",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="List configured cases without computation")
    for name in ("run", "fem"):
        command = sub.add_parser(
            name, help="Generate one particle or FEM loading history"
        )
        command.add_argument("case", choices=sorted(CASES))
        command.add_argument("--config", type=Path, help="Optional complete Case JSON")
        command.add_argument("--output", type=Path, required=True)
        command.add_argument(
            "--data-root",
            type=Path,
            help="Parent of generated inputs/; default: outputs",
        )
        command.add_argument("--seed", type=int)
        if name == "run":
            command.add_argument(
                "--method", choices=("proposed", "merged"), default="proposed"
            )
            command.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
            command.add_argument(
                "--assembly", choices=("owner_force", "energy", "energy_gfc")
            )
        else:
            command.add_argument("--level", type=int, default=0)
    comp = sub.add_parser(
        "compare", help="Compare fresh particle and FEM checkpoint files"
    )
    comp.add_argument("particles", type=Path)
    comp.add_argument("fem", type=Path)
    comp.add_argument("--quantity", choices=("alpha", "ges", "strain"), default="alpha")
    comp.add_argument("--load", type=float)
    comp.add_argument(
        "--reverse", action="store_true", help="Use the return-leg history checkpoint"
    )
    comp.add_argument("--output", type=Path)
    prep = sub.add_parser(
        "prepare-experiment", help="Process downloaded EBSD and HR-DIC observations"
    )
    prep.add_argument("specimen", choices=("CG", "FG"))
    prep.add_argument("--raw", type=Path, required=True)
    prep.add_argument("--output", type=Path, default=Path("outputs/inputs"))
    args = parser.parse_args()
    if args.command == "list":
        print(
            json.dumps({name: case.as_dict() for name, case in CASES.items()}, indent=2)
        )
    elif args.command in ("run", "fem"):
        case = get_case(args.case, args.seed)
        if args.config:
            value = json.loads(args.config.read_text(encoding="utf-8"))
            value["checkpoints"] = tuple(value["checkpoints"])
            if args.seed is not None:
                value["seed"] = args.seed
            case = Case(**value)
        if args.command == "run":
            from .solver import run

            run(
                case,
                args.output,
                args.method,
                args.backend,
                args.data_root,
                assembly=args.assembly,
            )
        else:
            from .fem import run

            run(case, args.output, args.level, args.data_root)
    elif args.command == "compare":
        from .comparison import compare

        print(
            json.dumps(
                compare(
                    args.particles,
                    args.fem,
                    args.quantity,
                    args.load,
                    args.reverse,
                    args.output,
                ),
                indent=2,
            )
        )
    elif args.command == "prepare-experiment":
        from .experiment import prepare

        print(json.dumps(prepare(args.raw, args.output, args.specimen), indent=2))


if __name__ == "__main__":
    main()
