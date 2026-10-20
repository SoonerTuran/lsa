"""Config, seeding, device and checkpoint helpers."""
from __future__ import annotations

import contextlib
import copy
import os
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def amp_dtype(device: torch.device, enabled: bool) -> torch.dtype | None:
    """Autocast dtype: bf16 on Ampere+ GPUs, fp16 on older GPUs (T4), None on CPU/disabled."""
    if not enabled or device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16


def device_name(device: torch.device) -> str:
    return torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"


def apply_overrides(cfg: dict, overrides: Iterable[str]) -> dict:
    """Apply dotted ``key.sub=value`` overrides; values are parsed as YAML scalars/lists."""
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"override {item!r} must look like key.sub=value")
        node = cfg
        *parents, leaf = key.strip().split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(raw)
    return cfg


def load_config(path: str | os.PathLike, overrides: Iterable[str] = ()) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg = apply_overrides(cfg, overrides)
    cfg.setdefault("name", Path(path).stem)
    return cfg


def save_yaml(obj, path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def atomic_save(obj, path: str | os.PathLike) -> None:
    """torch.save via a temp file + rename, so an interrupted save never corrupts `path`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def rng_state() -> dict:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available() and len(state["cuda"]) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(state["cuda"])


def autocast(device: torch.device, dtype: torch.dtype | None):
    """torch.autocast for `dtype`, or a no-op context when dtype is None."""
    return contextlib.nullcontext() if dtype is None else torch.autocast(device.type, dtype=dtype)
