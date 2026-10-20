"""Single entry point: python -m lsa.train --config configs/<name>.yaml --out <dir> [--set k=v ...]"""
from __future__ import annotations

import argparse

from lsa.utils import load_config


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Train an lsa model (dispatches on config.phase)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True, help="output dir (local or Google Drive); resumes if it has a checkpoint")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="dotted config overrides")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    phase = cfg.get("phase")
    if phase == 1:
        from lsa.train import phase1

        phase1.run(cfg, args.out)
    else:
        raise SystemExit(f"unknown phase {phase!r} in {args.config}")


if __name__ == "__main__":
    main()
