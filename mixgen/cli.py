"""mixgen CLI: ``python -m mixgen PLAN.yaml [--root DIR] [--dry-run] ...``."""
from __future__ import annotations

import argparse
import json
import sys

from mixgen import __version__
from mixgen.plan import PlanError, load_plan
from mixgen.stages import DEFAULT_STAGES, STAGES, parse_stages, run


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="mixgen",
        description="Plan-driven generation -> QA -> mastering -> DJ-mix assembly.")
    ap.add_argument("plan", help="plan YAML (see examples/lightworld-relay.plan.yaml)")
    ap.add_argument("--root", help="base dir for every path in the plan (default: /music)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what each stage would do; never generate or call a backend")
    ap.add_argument("--backend", choices=("acestep", "fake"),
                    help="generation backend (default: plan 'backend', else acestep)")
    ap.add_argument("--stages", default=",".join(DEFAULT_STAGES),
                    help=f"comma list of stages to run (default: {','.join(DEFAULT_STAGES)}; "
                         f"'stems' is opt-in, see {','.join(STAGES)})")
    ap.add_argument("--force", action="store_true", help="rerun stages even when outputs exist")
    ap.add_argument("--json", action="store_true", help="print the run report as JSON")
    ap.add_argument("--version", action="version", version=f"mixgen {__version__}")
    args = ap.parse_args(argv)
    try:
        plan = load_plan(args.plan, args.root)
        stages = parse_stages(args.stages)
    except PlanError as exc:
        print(f"mixgen: {exc}", file=sys.stderr)
        return 2
    try:
        report = run(plan, stages, backend_name=args.backend, force=args.force,
                     dry_run_mode=args.dry_run,
                     log=lambda m: print(m, file=sys.stdout if args.dry_run else sys.stderr))
    except PlanError as exc:
        print(f"mixgen: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("mixgen: interrupted", file=sys.stderr)
        return 130
    if args.json:
        print(json.dumps(report, indent=2))
    elif not args.dry_run:
        print(f"mixgen: done ({', '.join(stages)})")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
