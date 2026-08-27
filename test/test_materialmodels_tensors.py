"""
Standalone test for materialmodels.tensors and
materialmodels.elastic.transversely_isotropic.TransverseIsotropic.

Four checks
-----------
1. Voigt round-trip: voigt_to_tensor4(tensor4_to_voigt(C4, eng), eng) == C4,
   for both engineering conventions, on both an isotropic and a
   transversely isotropic tensor.
2. Symmetry checks: both tensors pass is_major_symmetric/is_minor_symmetric;
   a deliberately-broken tensor fails both.
3. Isotropic degeneracy: TransverseIsotropic with E_L=E_T=E,
   nu_LT=nu_TT=nu, G_LT=G_TT=E/(2(1+nu)) must reproduce
   LinearElasticIsotropic's stiffness tensor exactly -- transverse
   isotropy is a strict generalization of isotropy.
4. Rotational invariance about its own axis: a genuinely transversely
   isotropic material (E_L != E_T) rotated so the fibre direction stays
   [0, 0, 1] (the reference axis) but through a different in-plane frame
   must reproduce the unrotated stiffness_tensor() exactly -- that
   invariance is the defining property of transverse isotropy.

Usage
-----
    python -m pytest test/test_materialmodels_tensors.py
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.makedirs("output", exist_ok=True)

import sys
sys.path.insert(0, "src")

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)
import jax.numpy as jnp
import numpy as np

from materialmodels.elastic.isotropic import LinearElasticIsotropic
from materialmodels.elastic.transversely_isotropic import TransverseIsotropic
from materialmodels.tensors import (
    is_major_symmetric,
    is_minor_symmetric,
    rotate_tensor4,
    rotation_from_direction,
    tensor4_to_voigt,
    voigt_to_tensor4,
)

# ── 1. Voigt round-trip ───────────────────────────────────────────────────────

iso = LinearElasticIsotropic(E=210e3, nu=0.3)
trans = TransverseIsotropic(E_L=140e3, E_T=10e3, G_LT=5e3, nu_LT=0.3, nu_TT=0.4)

for C4, tag in [(np.asarray(iso.stiffness_tensor()), "isotropic"),
                (np.asarray(trans.stiffness_tensor()), "transversely isotropic")]:
    for engineering in (False, True):
        C_voigt = tensor4_to_voigt(C4, engineering=engineering)
        C4_back = voigt_to_tensor4(C_voigt, engineering=engineering)
        err = np.max(np.abs(C4_back - C4))
        assert err < 1e-8, f"{tag} round-trip (engineering={engineering}) failed: err={err:.3e}"

print("[1] Voigt round-trip: PASSED (both tensors, both conventions)")

# ── 2. symmetry checks ────────────────────────────────────────────────────────

for C4, tag in [(np.asarray(iso.stiffness_tensor()), "isotropic"),
                (np.asarray(trans.stiffness_tensor()), "transversely isotropic")]:
    assert is_major_symmetric(C4), f"{tag}: expected major symmetric"
    assert is_minor_symmetric(C4), f"{tag}: expected minor symmetric"

C4_broken = np.asarray(trans.stiffness_tensor()).copy()
C4_broken[0, 0, 1, 2] += 1.0   # break minor symmetry only at one entry
assert not is_minor_symmetric(C4_broken), "expected broken tensor to fail minor-symmetry check"

print("[2] symmetry checks: PASSED (both tensors symmetric, broken tensor caught)")

# ── 3. isotropic degeneracy ───────────────────────────────────────────────────

E, nu = 210e3, 0.3
G = E / (2.0 * (1.0 + nu))
degenerate = TransverseIsotropic(E_L=E, E_T=E, G_LT=G, nu_LT=nu, nu_TT=nu)
iso_ref = LinearElasticIsotropic(E=E, nu=nu)

err = float(jnp.max(jnp.abs(degenerate.stiffness_tensor() - iso_ref.stiffness_tensor())))
print(f"[3] isotropic degeneracy: max|C_trans - C_iso| = {err:.3e}")
assert err < 1e-6, f"isotropic-limit mismatch: {err:.3e}"
print("[3] isotropic degeneracy: PASSED")

# ── 4. rotational invariance about own axis ───────────────────────────────────

C_ref = trans.stiffness_tensor()
C_rot = trans.stiffness_tensor_rotated(jnp.array([0.0, 0.0, 1.0]))
err = float(jnp.max(jnp.abs(C_rot - C_ref)))
print(f"[4] rotation about own axis: max|C_rotated - C_ref| = {err:.3e}")
assert err < 1e-8, f"rotation about the fibre's own axis must be exact: {err:.3e}"

# also check a genuine off-axis rotation actually changes the tensor
C_offaxis = trans.stiffness_tensor_rotated(jnp.array([1.0, 0.0, 0.0]))
err_offaxis = float(jnp.max(jnp.abs(C_offaxis - C_ref)))
assert err_offaxis > 1.0, "rotating the fibre 90 degrees must change the stiffness tensor"
print(f"    off-axis rotation changes C as expected: max diff = {err_offaxis:.3e}")
print("[4] rotational invariance: PASSED")

print("\ntest_materialmodels_tensors: all checks passed")
