"""
Standalone test for materialmodels.inelastic.hardening -- the pluggable
isotropic hardening laws both plasticity models now take.

The module's whole claim is that a hardening law is swappable without
touching a plasticity model, because every return mapping here reduces to
one scalar equation

    A - B*dlam = sigma_y(alpha_prev + dlam)                            [*]

so these checks are mostly about [*]: that each law solves it, that the
solution really satisfies it, and that a law the module has never seen can
be dropped in and work.

Eight checks
------------
1. The tabulated curve itself: exact at the nodes, linearly interpolated
   between them, and PERFECTLY PLASTIC past the last one (the plateau
   convention -- a voxel running off the end of calibrated data must not
   harden into extrapolated territory).
2. solve_return actually solves [*]: the residual vanishes at every plastic
   state, including ones whose root lands in a later segment than the one
   alpha_prev starts in -- the segment-crossing case the closed form exists
   to get right -- and dlam is exactly 0 at every elastic state.
3. Linear-table equivalence: a table whose points lie on a straight line IS
   LinearHardening below its last node. Two independent code paths (closed
   form vs segment search) computing the same thing, which is the strongest
   check available without an external reference.
4. LinearHardening reproduces the closed form the plasticity models used to
   inline, BIT-FOR-BIT -- the guarantee that routing them through this
   module changed no existing result. (The end-to-end version of this is
   test_materialmodels_drucker_prager's checks 1 and 7.)
5. The extension point is real: a Voce law defined HERE, implementing only
   sigma_y/dsigma_y and inheriting the generic solver, must solve [*] to
   machine precision and drive both J2Plasticity and DruckerPrager
   unmodified. This is what "pluggable" has to mean to be worth the split.
6. Autodiff and batching: sigma_y/solve_return survive vmap + jacfwd with no
   NaN, and d(dlam)/dA matches a finite difference within a segment. The
   models get their consistent tangent by differentiating straight through
   these, so a NaN here is a NaN in the global Newton.
7. Every documented validation error actually raises -- in particular a
   TRANSPOSED table, which is otherwise silent (a stress where a strain
   belongs is still a number).
8. The config path: build_hardening/resolve_hardening, including the
   mutually-exclusive H-vs-hardening spellings.

Usage
-----
    python test/test_materialmodels_hardening.py
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
sys.path.insert(0, "src")

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)
import jax
import jax.numpy as jnp
import numpy as np

from materialmodels.inelastic.hardening import (
    IsotropicHardening, LinearHardening, PiecewiseLinearHardening,
    build_hardening, resolve_hardening,
)
from materialmodels.inelastic.plasticity_drucker_prager import DruckerPrager
from materialmodels.inelastic.plasticity_j2 import J2Plasticity

SY0, H_LIN = 56.1, 1.0e3
TABLE = [[0.000, 56.1], [0.020, 70.0], [0.100, 78.0]]
E, NU = 3.76e3, 0.39
MU = E / (2.0 * (1.0 + NU))
B_J2 = 3.0 * MU

pw = PiecewiseLinearHardening(TABLE)
lin = LinearHardening(SY0, H_LIN)


# ── [1] the tabulated curve, and the plateau ────────────────────────────────
H0 = (70.0 - 56.1) / 0.02      # 695
H1 = (78.0 - 70.0) / 0.08      # 100
for al, want in [(0.000, 56.1), (0.010, 56.1 + H0 * 0.010), (0.020, 70.0),
                 (0.060, 70.0 + H1 * 0.040), (0.100, 78.0),
                 (0.500, 78.0), (5.000, 78.0)]:
    got = float(pw.sigma_y(jnp.asarray(al)))
    assert abs(got - want) < 1e-12, (al, got, want)
for al, want in [(0.005, H0), (0.050, H1), (0.300, 0.0)]:
    got = float(pw.dsigma_y(jnp.asarray(al)))
    assert abs(got - want) < 1e-9, (al, got, want)
# The plateau is the whole point of the convention: past the last node the
# curve must be FLAT, not extrapolated along the last slope.
assert float(pw.sigma_y(jnp.asarray(9.9))) == float(pw.sigma_y(jnp.asarray(0.1)))
print(f"[1] tabulated curve: nodes exact, slopes {H0:g}/{H1:g}, plateau at "
      f"{float(pw.sigma_y(jnp.asarray(9.9))):g} past alpha=0.1")
print("[1] PASSED")


# ── [2] solve_return solves [*], across and within segments ─────────────────
def residual(law, A, B, alpha_prev, dlam):
    return A - B * dlam - float(law.sigma_y(jnp.asarray(alpha_prev + dlam)))


worst_res, n_plastic, n_elastic, n_crossed = 0.0, 0, 0, 0
for A in np.linspace(40.0, 140.0, 41):
    for alpha_prev in [0.0, 0.005, 0.019, 0.02, 0.05, 0.099, 0.15, 0.40]:
        dlam = float(pw.solve_return(jnp.asarray(A), jnp.asarray(B_J2),
                                     jnp.asarray(alpha_prev)))
        assert np.isfinite(dlam) and dlam >= 0.0, (A, alpha_prev, dlam)
        f_trial = A - float(pw.sigma_y(jnp.asarray(alpha_prev)))
        if f_trial > 0.0:
            n_plastic += 1
            worst_res = max(worst_res, abs(residual(pw, A, B_J2, alpha_prev, dlam)))
            # did the root leave the segment alpha_prev started in?
            seg0 = int(pw._segment_index(jnp.asarray(alpha_prev)))
            seg1 = int(pw._segment_index(jnp.asarray(alpha_prev + dlam)))
            n_crossed += seg1 != seg0
        else:
            n_elastic += 1
            # Elastic must be EXACTLY zero, not merely small -- the models
            # rely on it to collapse to the elastic identity with no branch.
            assert dlam == 0.0, (A, alpha_prev, dlam)
assert worst_res < 1e-11, worst_res
assert n_crossed > 0, "no segment-crossing state in the sweep -- check 2 is not testing what it claims"
print(f"[2] {n_plastic} plastic ({n_crossed} crossing a breakpoint), {n_elastic} elastic: "
      f"max|residual of [*]| = {worst_res:.2e}, elastic dlam exactly 0")
print("[2] PASSED")


# ── [3] a straight-line table IS LinearHardening (below its last node) ──────
lin_tab = PiecewiseLinearHardening([[0.0, SY0], [0.05, SY0 + H_LIN * 0.05],
                                    [0.30, SY0 + H_LIN * 0.30]])
worst_sy, worst_dl = 0.0, 0.0
for A in np.linspace(40.0, 200.0, 33):
    for alpha_prev in [0.0, 0.01, 0.04, 0.12, 0.25]:
        a_j, a_l = jnp.asarray(alpha_prev), jnp.asarray(A)
        worst_sy = max(worst_sy, abs(float(lin_tab.sigma_y(a_j)) - float(lin.sigma_y(a_j))))
        d_tab = float(lin_tab.solve_return(a_l, jnp.asarray(B_J2), a_j))
        d_lin = float(lin.solve_return(a_l, jnp.asarray(B_J2), a_j))
        if alpha_prev + d_tab <= 0.30:      # below the plateau, they must agree
            worst_dl = max(worst_dl, abs(d_tab - d_lin))
assert worst_sy < 1e-12 and worst_dl < 1e-12, (worst_sy, worst_dl)
print(f"[3] straight-line table vs LinearHardening: max|d sigma_y|={worst_sy:.2e}  "
      f"max|d dlam|={worst_dl:.2e}")
print("[3] PASSED")


# ── [4] LinearHardening is bit-for-bit the formula it replaced ──────────────
worst_bits = 0
for A in np.linspace(40.0, 200.0, 33):
    for alpha_prev in [0.0, 0.01, 0.04, 0.12, 0.25]:
        for B in (B_J2, 3.0 * MU + 9.0 * 1.0e3 * 0.13 * 0.05, 0.0):
            got = float(lin.solve_return(jnp.asarray(A), jnp.asarray(B),
                                         jnp.asarray(alpha_prev)))
            # exactly the expression plasticity_j2/plasticity_drucker_prager
            # used to inline, same grouping of the same floats
            ref = float(jnp.maximum(A - (SY0 + H_LIN * alpha_prev), 0.0) / (B + H_LIN))
            worst_bits += got != ref
assert worst_bits == 0, f"{worst_bits} states differ from the inlined closed form"
print(f"[4] LinearHardening vs the inlined closed form: {worst_bits} differing states "
      "out of 495 (bit-for-bit)")
print("[4] PASSED")


# ── [5] the extension point: a law the module has never seen ────────────────
class Voce(IsotropicHardening):
    """sigma_y = sigma_y0 + Q*(1 - exp(-b*alpha)) -- smooth, saturating.

    Defined in the TEST, not the module: it implements only sigma_y and
    dsigma_y and inherits solve_return, which is exactly what a future
    production law would do. It deliberately does not override
    affine_segments, so it is not usable with a_tip > 0 (see check 7).
    """

    def __init__(self, sigma_y0, Q, b):
        self.sigma_y0, self.Q, self.b = float(sigma_y0), float(Q), float(b)

    def sigma_y(self, alpha):
        return self.sigma_y0 + self.Q * (1.0 - jnp.exp(-self.b * alpha))

    def dsigma_y(self, alpha):
        return self.Q * self.b * jnp.exp(-self.b * alpha)


voce = Voce(SY0, 22.0, 15.0)
worst_v = 0.0
for A in np.linspace(40.0, 140.0, 41):
    for alpha_prev in [0.0, 0.01, 0.05, 0.20]:
        dlam = float(voce.solve_return(jnp.asarray(A), jnp.asarray(B_J2),
                                       jnp.asarray(alpha_prev)))
        assert np.isfinite(dlam) and dlam >= 0.0
        if A - float(voce.sigma_y(jnp.asarray(alpha_prev))) > 0.0:
            worst_v = max(worst_v, abs(residual(voce, A, B_J2, alpha_prev, dlam)))
        else:
            assert dlam == 0.0
assert worst_v < 1e-10, worst_v

# ...and it drives both models with no change to either.
eps = jnp.array([[0.02, 0.004, 0.0], [0.004, -0.006, 0.001], [0.0, 0.001, -0.006]])
z33, z0 = jnp.zeros((3, 3)), jnp.asarray(0.0)
for model in (J2Plasticity(E=E, nu=NU, hardening=voce),
              DruckerPrager(E=E, nu=NU, hardening=voce, a_f=0.13, a_g=0.05)):
    sig, C, (ep, al) = model.stress_and_tangent(eps, z33, z0)
    assert np.all(np.isfinite(sig)) and np.all(np.isfinite(C)) and float(al) > 0.0, model
    # the returned state must sit ON the Voce yield surface
    p = float(jnp.trace(sig)) / 3.0
    s = sig - p * jnp.eye(3)
    q = float(jnp.sqrt(1.5 * jnp.sum(s ** 2)))
    a_f = getattr(model, "a_f", 0.0)
    f = q + 3.0 * a_f * p - float(voce.sigma_y(al))
    assert abs(f) < 1e-9 * SY0, (model, f)
print(f"[5] unseen Voce law: max|residual of [*]| = {worst_v:.2e}; drives J2Plasticity "
      "and DruckerPrager unmodified, both returning onto its surface")
print("[5] PASSED")


# ── [6] vmap + jacfwd ───────────────────────────────────────────────────────
A_batch = jnp.asarray(np.linspace(40.0, 160.0, 64))
ap_batch = jnp.asarray(np.linspace(0.0, 0.30, 64))
for law, tag in ((pw, "piecewise_linear"), (voce, "voce")):
    dl = jax.jit(jax.vmap(lambda a, p: law.solve_return(a, B_J2, p)))(A_batch, ap_batch)
    assert np.all(np.isfinite(dl)) and np.all(np.asarray(dl) >= 0.0), tag
    g = jax.jit(jax.vmap(jax.grad(lambda a, p: law.solve_return(a, B_J2, p))))(A_batch, ap_batch)
    assert np.all(np.isfinite(g)), tag

# d(dlam)/dA inside one segment: 1/(B + H_seg) analytically
h = 1e-6
for alpha_prev, A0, H_seg in [(0.005, 90.0, H0), (0.05, 90.0, H1), (0.20, 90.0, 0.0)]:
    g = float(jax.grad(lambda a: pw.solve_return(a, jnp.asarray(B_J2), jnp.asarray(alpha_prev)))(
        jnp.asarray(A0)))
    fd = (float(pw.solve_return(jnp.asarray(A0 + h), jnp.asarray(B_J2), jnp.asarray(alpha_prev)))
          - float(pw.solve_return(jnp.asarray(A0 - h), jnp.asarray(B_J2), jnp.asarray(alpha_prev)))) / (2 * h)
    exact = 1.0 / (B_J2 + H_seg)
    assert abs(g - fd) < 1e-6 * abs(exact) and abs(g - exact) < 1e-9, (alpha_prev, g, fd, exact)
print("[6] vmap+jit+grad finite on 64 states for both laws; d(dlam)/dA matches "
      "1/(B+H_seg) on all three segments")
print("[6] PASSED")


# ── [7] validation ──────────────────────────────────────────────────────────
def raises(fn, needle):
    try:
        fn()
    except ValueError as exc:
        assert needle in str(exc), f"wrong message for {needle!r}: {exc}"
        return True
    raise AssertionError(f"expected a ValueError mentioning {needle!r}")


# The transposed table is the dangerous one: (yield_stress, plastic_strain)
# instead of (plastic_strain, yield_stress) is silently a valid-looking table.
raises(lambda: PiecewiseLinearHardening([[56.1, 0.0], [70.0, 0.02]]), "plastic strain")
raises(lambda: PiecewiseLinearHardening([[0.0, 56.1]]), "at least 2")
raises(lambda: PiecewiseLinearHardening([[0.0, 56.1], [0.02, 50.0]]), "non-decreasing")
raises(lambda: PiecewiseLinearHardening([[0.0, 56.1], [0.02, 70.0], [0.01, 75.0]]), "strictly increasing")
raises(lambda: PiecewiseLinearHardening([[0.0, 0.0], [0.02, 70.0]]), "> 0")
raises(lambda: LinearHardening(SY0, -1.0), "softening")
raises(lambda: J2Plasticity(E=E, nu=NU, sigma_y0=SY0, H=H_LIN, hardening=pw), "mutually exclusive")
raises(lambda: J2Plasticity(E=E, nu=NU), "needs a hardening law")
raises(lambda: J2Plasticity(E=E, nu=NU, sigma_y0=1.0, hardening=pw), "contradicts")
raises(lambda: DruckerPrager(E=E, nu=NU, hardening=pw), "a_f is required")
# a smooth law has no affine segments, so it cannot use the tip-rounded branch
raises(lambda: DruckerPrager(E=E, nu=NU, hardening=voce, a_f=0.13, a_tip=5.0),
       "not piecewise affine")
# ...but the tabulated one can
DruckerPrager(E=E, nu=NU, hardening=pw, a_f=0.13, a_g=0.05, a_tip=0.1 * SY0)
print("[7] 11 validation errors raise as documented (transposed table, softening, "
      "H+hardening, missing a_f, smooth law with a_tip, ...)")
print("[7] PASSED")


# ── [8] the config path ─────────────────────────────────────────────────────
b_pw = build_hardening({"law": "piecewise_linear", "table": TABLE})
assert float(b_pw.sigma_y(jnp.asarray(0.05))) == float(pw.sigma_y(jnp.asarray(0.05)))
b_lin = build_hardening({"law": "linear", "sigma_y0": SY0, "H": "1.0e3"})  # YAML str exponent
assert isinstance(b_lin, LinearHardening) and b_lin.H == H_LIN
raises(lambda: build_hardening({"law": "voce", "Q": 1.0}), "unknown law")
raises(lambda: build_hardening({"law": "piecewise_linear"}), "requires a 'table'")
raises(lambda: build_hardening({"law": "piecewise_linear", "table": TABLE, "H": 1.0}), "unexpected key")
raises(lambda: build_hardening(TABLE), "'law'")
assert isinstance(resolve_hardening(None, SY0, H_LIN), LinearHardening)
assert resolve_hardening(pw, None, None) is pw
assert resolve_hardening(pw, SY0, None) is pw          # consistent sigma_y0 is allowed
print("[8] build_hardening/resolve_hardening: both laws, YAML string exponents, "
      "and 4 config errors")
print("[8] PASSED")

print("\ntest_materialmodels_hardening: all checks passed")
