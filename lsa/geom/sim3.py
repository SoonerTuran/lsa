"""Batched Sim(3) similarity transforms and the 6D rotation parameterisation.

A Sim3 element T = (R, t, s) acts on points by ``T(x) = s R x + t``.
``A.compose(B)`` is A∘B (apply B first).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class Sim3:
    R: Tensor  # (B,3,3) rotation
    t: Tensor  # (B,3) translation
    s: Tensor  # (B,) positive scale

    @property
    def batch_size(self) -> int:
        return self.R.shape[0]

    @staticmethod
    def identity(B: int, device=None, dtype=torch.float32) -> "Sim3":
        R = torch.eye(3, device=device, dtype=dtype).expand(B, 3, 3).clone()
        return Sim3(R, torch.zeros(B, 3, device=device, dtype=dtype), torch.ones(B, device=device, dtype=dtype))

    def compose(self, other: "Sim3") -> "Sim3":
        """self ∘ other: x -> self(other(x))."""
        R = self.R @ other.R
        t = self.s[:, None] * (self.R @ other.t[..., None])[..., 0] + self.t
        return Sim3(R, t, self.s * other.s)

    def inverse(self) -> "Sim3":
        Rt = self.R.transpose(1, 2)
        s_inv = 1.0 / self.s
        t = -s_inv[:, None] * (Rt @ self.t[..., None])[..., 0]
        return Sim3(Rt, t, s_inv)

    def apply(self, points: Tensor) -> Tensor:
        """points (B,N,3) -> s R x + t, (B,N,3)."""
        return self.s[:, None, None] * torch.einsum("bij,bnj->bni", self.R, points) + self.t[:, None, :]

    def matrix(self) -> Tensor:
        """Homogeneous (B,4,4) matrix."""
        M = torch.zeros(self.batch_size, 4, 4, device=self.R.device, dtype=self.R.dtype)
        M[:, :3, :3] = self.s[:, None, None] * self.R
        M[:, :3, 3] = self.t
        M[:, 3, 3] = 1.0
        return M

    def to(self, *args, **kwargs) -> "Sim3":
        return Sim3(self.R.to(*args, **kwargs), self.t.to(*args, **kwargs), self.s.to(*args, **kwargs))

    @staticmethod
    def random(
        B: int,
        generator: torch.Generator | None = None,
        device=None,
        dtype=torch.float32,
        t_range: float = 0.5,
        s_range: tuple[float, float] = (0.5, 1.25),
    ) -> "Sim3":
        """Uniform random rotation, t ~ U[-t_range,t_range]^3, s ~ U[s_range]."""
        R = random_rotation(B, generator=generator, dtype=dtype)
        t = (torch.rand(B, 3, generator=generator, dtype=dtype) * 2 - 1) * t_range
        lo, hi = s_range
        s = lo + (hi - lo) * torch.rand(B, generator=generator, dtype=dtype)
        return Sim3(R, t, s).to(device)


def quaternion_to_matrix(q: Tensor) -> Tensor:
    """Unit quaternions (...,4) in (w,x,y,z) order -> rotation matrices (...,3,3)."""
    w, x, y, z = F.normalize(q, dim=-1).unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def random_rotation(B: int, generator: torch.Generator | None = None, dtype=torch.float32) -> Tensor:
    """Haar-uniform rotations via normalised Gaussian quaternions, (B,3,3)."""
    return quaternion_to_matrix(torch.randn(B, 4, generator=generator, dtype=dtype))


def rot6d_to_matrix(d6: Tensor) -> Tensor:
    """Zhou et al. 2019 continuous 6D representation (...,6) -> rotation (...,3,3).

    Gram-Schmidt on the two 3-vectors; they become the first two COLUMNS of R.
    """
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def matrix_to_rot6d(R: Tensor) -> Tensor:
    """Rotation (...,3,3) -> 6D (first two columns), inverse of rot6d_to_matrix."""
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)
