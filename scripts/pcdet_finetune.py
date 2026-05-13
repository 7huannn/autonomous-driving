#!/usr/bin/env python3
"""Project-local OpenPCDet finetune wrapper with runtime import shims."""

from __future__ import annotations

import argparse
import runpy
import sys
import types
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OpenPCDet train.py with project-local compatibility shims")
    parser.add_argument("--openpcdet-repo", type=Path, default=Path("../repos/OpenPCDet"), help="Path to OpenPCDet repo")
    parser.add_argument("--cfg-file", type=Path, required=True, help="Training config YAML")
    parser.add_argument("--pretrained-model", type=Path, default=Path("data/checkpoints/pointpillar_7728.pth"), help="Initial checkpoint")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size per GPU")
    parser.add_argument("--workers", type=int, default=2, help="Dataloader workers")
    parser.add_argument("--extra-tag", default="carla_probe_finetune", help="OpenPCDet extra_tag")
    parser.add_argument("--set", dest="set_cfgs", nargs=argparse.REMAINDER, default=None, help="Extra cfg overrides")
    return parser.parse_args()


def inject_stubs() -> None:
    mod_name = "repos.OpenPCDet.pcdet.datasets.argo2.argo2_dataset"
    if mod_name not in sys.modules:
        mod = types.ModuleType(mod_name)
        mod.Argo2Dataset = type("Argo2Dataset", (object,), {})
        sys.modules[mod_name] = mod

    mod_name = "repos.OpenPCDet.pcdet.models.backbones_3d.dsvt"
    if mod_name not in sys.modules:
        mod = types.ModuleType(mod_name)
        mod.DSVT = type("DSVT", (object,), {})
        sys.modules[mod_name] = mod


def main() -> int:
    args = parse_args()
    project_root = Path.cwd().resolve()
    autonomous_root = project_root.parent.resolve()
    if str(autonomous_root) not in sys.path:
        sys.path.insert(0, str(autonomous_root))

    openpcdet_repo = args.openpcdet_repo.resolve()
    if not openpcdet_repo.exists():
        raise FileNotFoundError(f"OpenPCDet repo not found: {openpcdet_repo}")

    cfg_file = args.cfg_file.resolve()
    if not cfg_file.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_file}")

    pretrained = args.pretrained_model.resolve()
    if not pretrained.exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained}")

    inject_stubs()

    train_py = openpcdet_repo / "tools" / "train.py"
    tools_dir = openpcdet_repo / "tools"
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    old_argv = sys.argv[:]
    sys.argv = [
        str(train_py),
        "--cfg_file", str(cfg_file),
        "--pretrained_model", str(pretrained),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--workers", str(args.workers),
        "--extra_tag", str(args.extra_tag),
        "--launcher", "none",
    ]
    if args.set_cfgs:
        sys.argv.extend(["--set", *args.set_cfgs])

    try:
        runpy.run_module("repos.OpenPCDet.tools.train", run_name="__main__")
    finally:
        sys.argv = old_argv
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
