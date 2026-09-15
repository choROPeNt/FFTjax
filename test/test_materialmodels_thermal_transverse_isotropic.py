"""
Standalone test for materialmodels.thermal.transverse_isotropic.
ThermalConductivityTransverseIsotropic.

Five checks
-----------
1. Reference frame (default fiber_dir, i.e. Z): conductivity_tensor() ==
   diag(k_T, k_T, k_L) exactly.
2. Rotation to a coordinate axis (X, Y): the diagonal entries permute
   exactly as expected -- k_L always lands on the fibre axis, k_T on the
   other two.
3. Isotropic degeneracy: k_L == k_T must reproduce
   materialmodels.thermal.isotropic.ThermalConductivityIsotropic's
   conductivity tensor exactly, for any fiber_dir -- transverse isotropy is
   a strict generalization of isotropy, same relationship the elastic
   TransverseIsotropic has to LinearElasticIsotropic (see
   test/test_materialmodels_tensors.py's own check 3).
4. Rotational invariance about its own axis (the defining property of
   transverse isotropy), and that an off-axis rotation actually changes the
   tensor -- same pair of checks test_materialmodels_tensors.py runs for
   the elastic model.
5. Per-voxel orientation field: conductivity_field_oriented on a two-voxel
   field with different directions must match two independent
   single-direction conductivity_tensor_rotated calls.

Usage
-----
    python -m pytest test/test_materialmodels_thermal_transverse_isotropic.py
"""

import sys
sys.path.insert(0, "src")

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)
import jax.numpy as jnp
import numpy as np

from materialmodels.thermal.isotropic import ThermalConductivityIsotropic
from materialmodels.thermal.transverse_isotropic import ThermalConductivityTransverseIsotropic

k_L, k_T = 5.0, 1.0
trans = ThermalConductivityTransverseIsotropic(k_L=k_L, k_T=k_T, name="carbon fibre")

# ── [1] reference frame ──────────────────────────────────────────────────────
K_ref = np.asarray(trans.conductivity_tensor())
assert np.allclose(K_ref, np.diag([k_T, k_T, k_L])), K_ref
print(f"[1] reference frame: K = diag({K_ref[0,0]:.3g}, {K_ref[1,1]:.3g}, {K_ref[2,2]:.3g})")
print("[1] PASSED")

# ── [2] rotation to a coordinate axis ────────────────────────────────────────
K_x = np.asarray(trans.conductivity_tensor_rotated(jnp.array([1.0, 0.0, 0.0])))
K_y = np.asarray(trans.conductivity_tensor_rotated(jnp.array([0.0, 1.0, 0.0])))
assert np.allclose(K_x, np.diag([k_L, k_T, k_T]), atol=1e-10), K_x
assert np.allclose(K_y, np.diag([k_T, k_L, k_T]), atol=1e-10), K_y
print(f"[2] fibre along X -> diag{tuple(np.diag(K_x).round(3))}; "
      f"along Y -> diag{tuple(np.diag(K_y).round(3))}")
print("[2] PASSED")

# ── [3] isotropic degeneracy ──────────────────────────────────────────────────
k = 2.5
degenerate = ThermalConductivityTransverseIsotropic(k_L=k, k_T=k, fiber_dir=[1.0, 1.0, 0.0])
iso_ref = ThermalConductivityIsotropic(k=k)
err = float(jnp.max(jnp.abs(degenerate.conductivity_tensor() - iso_ref.conductivity_tensor())))
print(f"[3] isotropic degeneracy (k_L=k_T={k}, off-axis fiber_dir): "
      f"max|K_trans - K_iso| = {err:.3e}")
assert err < 1e-10, f"isotropic-limit mismatch: {err:.3e}"
print("[3] PASSED")

# ── [4] rotational invariance about own axis ─────────────────────────────────
C_rot = trans.conductivity_tensor_rotated(jnp.array([0.0, 0.0, 1.0]))
err = float(jnp.max(jnp.abs(C_rot - K_ref)))
print(f"[4] rotation about own axis: max|K_rotated - K_ref| = {err:.3e}")
assert err < 1e-10, f"rotation about the fibre's own axis must be exact: {err:.3e}"

err_offaxis = float(jnp.max(jnp.abs(K_x - K_ref)))
assert err_offaxis > 1.0, "rotating the fibre 90 degrees must change the conductivity tensor"
print(f"    off-axis rotation changes K as expected: max diff = {err_offaxis:.3e}")
print("[4] PASSED")

# ── [5] per-voxel orientation field ──────────────────────────────────────────
orientations = jnp.array([[0.0, 1.0], [0.0, 0.0], [1.0, 0.0]])   # voxel 0: Z, voxel 1: X
K_field = np.asarray(trans.conductivity_field_oriented(orientations))
assert K_field.shape == (3, 3, 2)
assert np.allclose(K_field[..., 0], K_ref, atol=1e-10)
assert np.allclose(K_field[..., 1], K_x, atol=1e-10)
print("[5] per-voxel field matches independent single-direction calls")
print("[5] PASSED")

print("\ntest_materialmodels_thermal_transverse_isotropic: all checks passed")
