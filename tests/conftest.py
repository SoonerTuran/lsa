"""Shared test setup.

Phase-0 requirement -> test:
  known distances (sphere, box face/edge/corner, ...)   test_primitives.py::test_*_known_distances
  eikonal |grad| = 1 away from medial axes              test_primitives.py::test_eikonal_away_from_medial_axes
  transform_sdf == transform points then evaluate (s)   test_compose.py, test_sim3.py::test_transform_sdf_with_sim3
  Sim3 compose/inverse round trip, inv(T)(T(x)) = x     test_sim3.py::test_inverse_roundtrip_points, ...
  6D rotation orthonormal with det +1                   test_sim3.py::test_rot6d_outputs_are_rotations
  union == min                                          test_compose.py::test_union_is_min
  sampled GT matches re-evaluation                      test_sampling.py::test_sampled_gt_matches_reevaluation
  marching cubes of a sphere recovers radius (2%)       test_viz.py::test_marching_cubes_recovers_sphere_radius
"""
import random

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def _seed_everything():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    yield
