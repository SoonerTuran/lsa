"""Grid evaluation, marching cubes and SDF slice images for analytic or learned SDFs.

An "SDF callable" maps torch points (N,3) -> sdf (N,); it may live on any device.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
import trimesh
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from skimage import measure
from torch import Tensor

SDFFn = Callable[[Tensor], Tensor]


def grid_points(res: int, bound: float = 1.0, device=None, dtype=torch.float32) -> Tensor:
    """(res^3, 3) points of a regular grid over [-bound, bound]^3, 'ij' order (x slowest)."""
    ax = torch.linspace(-bound, bound, res, device=device, dtype=dtype)
    return torch.stack(torch.meshgrid(ax, ax, ax, indexing="ij"), dim=-1).reshape(-1, 3)


@torch.no_grad()
def eval_points(sdf_fn: SDFFn, pts: Tensor, chunk: int = 2**18) -> np.ndarray:
    return torch.cat([sdf_fn(c).float().cpu() for c in pts.split(chunk)]).numpy()


def eval_grid(sdf_fn: SDFFn, res: int = 64, bound: float = 1.0, device=None, chunk: int = 2**18) -> np.ndarray:
    """SDF on a res^3 grid over [-bound,bound]^3 -> float32 array (res,res,res) indexed [x,y,z]."""
    return eval_points(sdf_fn, grid_points(res, bound, device), chunk).reshape(res, res, res)


def extract_mesh(grid: np.ndarray, bound: float = 1.0, level: float = 0.0) -> trimesh.Trimesh | None:
    """Marching cubes of a grid from :func:`eval_grid`; None if there is no zero crossing."""
    if not (grid.min() < level < grid.max()):
        return None
    h = 2.0 * bound / (grid.shape[0] - 1)
    # default gradient_direction ('descent') yields outward winding for SDFs (volume > 0)
    verts, faces, normals, _ = measure.marching_cubes(grid, level=level, spacing=(h, h, h))
    return trimesh.Trimesh(verts - bound, faces, vertex_normals=normals, process=False)


def mesh_from_sdf(sdf_fn: SDFFn, res: int = 64, bound: float = 1.0, device=None) -> trimesh.Trimesh | None:
    return extract_mesh(eval_grid(sdf_fn, res, bound, device), bound)


def save_ply(sdf_fn: SDFFn, path: str | os.PathLike, res: int = 64, bound: float = 1.0, device=None):
    """Marching-cubes mesh of the zero level set saved as .ply. Returns the mesh, or None
    (and writes nothing) if the SDF has no surface inside the grid."""
    mesh = mesh_from_sdf(sdf_fn, res, bound, device)
    if mesh is None:
        print(f"warning: no zero level set, skipping {path}")
        return None
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(path))
    return mesh


_PLANES = (("xy (z=0)", 2, (0, 1)), ("xz (y=0)", 1, (0, 2)), ("yz (x=0)", 0, (1, 2)))


def plane_slice(sdf_fn: SDFFn, normal_axis: int, res: int = 128, bound: float = 1.0, device=None) -> np.ndarray:
    """SDF on the plane through the origin orthogonal to `normal_axis`; array [u, v] over the
    two remaining axes in increasing order."""
    ax = torch.linspace(-bound, bound, res, device=device)
    u, v = torch.meshgrid(ax, ax, indexing="ij")
    pts = torch.zeros(res, res, 3, device=device)
    uv_axes = [a for a in range(3) if a != normal_axis]
    pts[..., uv_axes[0]], pts[..., uv_axes[1]] = u, v
    return eval_points(sdf_fn, pts.reshape(-1, 3)).reshape(res, res)


def slice_png(
    sdf_fns: SDFFn | Sequence[SDFFn],
    path: str | os.PathLike,
    labels: Sequence[str] | None = None,
    res: int = 128,
    bound: float = 1.0,
    vmax: float = 0.5,
    title: str | None = None,
    device=None,
) -> None:
    """Save 3 orthogonal slices (xy, xz, yz through the origin) of one or more SDFs, one row
    per SDF, coloured by value (red inside, blue outside) with the zero contour in black."""
    fns = [sdf_fns] if callable(sdf_fns) else list(sdf_fns)
    labels = list(labels) if labels is not None else [""] * len(fns)
    fig = Figure(figsize=(9, 3 * len(fns)), dpi=100)
    FigureCanvasAgg(fig)
    axes = fig.subplots(len(fns), 3, squeeze=False)
    extent = (-bound, bound, -bound, bound)
    ax_names = "xyz"
    for r, fn in enumerate(fns):
        for c, (name, n_axis, (a0, a1)) in enumerate(_PLANES):
            img = plane_slice(fn, n_axis, res, bound, device).T  # rows = v, cols = u
            ax = axes[r, c]
            ax.imshow(img, origin="lower", extent=extent, cmap="RdBu", vmin=-vmax, vmax=vmax)
            if img.min() < 0 < img.max():
                ax.contour(img, levels=[0.0], colors="k", linewidths=1.2, extent=extent)
            ax.set_xlabel(ax_names[a0])
            ax.set_ylabel(ax_names[a1])
            ax.set_title(f"{labels[r]} {name}".strip(), fontsize=9)
            ax.set_aspect("equal")
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path))
