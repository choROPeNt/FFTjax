"""
Standalone test for materialmodels.inelastic.plasticity_drucker_prager
.DruckerPrager -- the pressure-sensitive drop-in replacement for
J2Plasticity.

Six checks
----------
1. J2 degeneracy: with a_f = a_g = 0 the Drucker-Prager cone IS the von
   Mises cylinder, so stress, tangent AND updated state must reproduce
   J2Plasticity's to machine precision over random strains and random
   prior states. The strongest available correctness check, same argument
   as test_problems_mechanics_nonlinear's [1]: it compares against an
   independently-verified model rather than trusting the new code alone.
2. Uniaxial yield asymmetry: the surface implies sigma_t = sigma_y0/(1+a_f)
   in tension and sigma_c = sigma_y0/(1-a_f) in compression -- the property
   a_f is actually calibrated from, so it's what a config author gets wrong
   first.
3. Consistency: every plastically-returned state must sit ON the yield
   surface, f = q + 3*a_f*p - (sigma_y0 + H*alpha) = 0, cone and apex
   branches alike, with no NaN in stress, tangent or state.
4. Apex return: a strongly hydrostatic-tensile trial state must return to
   the cone's vertex exactly (q = 0, f = 0) with a finite tangent, for
   associated, weakly-dilatant AND zero-dilatancy flow -- the a_g = 0 case
   being the one whose apex formula would divide by zero if parametrized
   by the volumetric plastic strain instead of by dalpha (see stress()).
5. Consistent tangent: jacfwd's tangent must match a central finite
   difference of stress() on both branches; and its major symmetry must
   hold for associated flow (a_g = a_f) and genuinely FAIL for
   non-associated flow -- the documented, physically-correct behaviour,
   asserted so it can't silently regress into a symmetrized approximation.
6. Drop-in through the real solver: assemble_local_update +
   solve_displacement_based_nonlinear must run a DruckerPrager matrix with
   no change to either (they duck-type on stress_and_tangent_field), and
   macroscopic tension must yield strictly more voxels than compression of
   the same magnitude -- the pressure sensitivity surviving homogenization,
   which is the whole point of the model over J2.

Usage
-----
    python -m pytest test/test_materialmodels_drucker_prager.py
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
sys.path.insert(0, "src")

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)
import jax
import jax.numpy as jnp
import numpy as np

from generation.rve import make_square_composite_rve
from materialmodels.assembly import assemble_local_update
from materialmodels.elastic.isotropic import LinearElasticIsotropic
from materialmodels.inelastic.plasticity_drucker_prager import DruckerPrager
from materialmodels.inelastic.plasticity_j2 import J2Plasticity
from operators.green import build_freq_grid
from problems.mechanics import solve_displacement_based_nonlinear

E, NU, SY, H = 3.76e3, 0.39, 50.0, 1.0e3
rng = np.random.default_rng(0)


def _q_p(sigma):
    """von Mises equivalent stress and mean stress of a (..., 3, 3) stress."""
    p = jnp.trace(sigma, axis1=-2, axis2=-1) / 3.0
    s = sigma - p[..., None, None] * jnp.eye(3)
    return jnp.sqrt(1.5 * jnp.sum(s ** 2, axis=(-2, -1))), p


# ── 1. J2 degeneracy ─────────────────────────────────────────────────────────

j2  = J2Plasticity(E=E, nu=NU, sigma_y0=SY, H=H)
dp0 = DruckerPrager(E=E, nu=NU, sigma_y0=SY, H=H, a_f=0.0, a_g=0.0)

N = 300
eps_r   = jnp.array(rng.normal(scale=0.02, size=(N, 3, 3)))
eps_p_r = jnp.array(rng.normal(scale=0.005, size=(N, 3, 3)))
eps_p_r = 0.5 * (eps_p_r + eps_p_r.transpose(0, 2, 1))
alpha_r = jnp.abs(jnp.array(rng.normal(scale=0.01, size=N)))

s_j2, C_j2, (ep_j2, al_j2) = jax.jit(jax.vmap(j2.stress_and_tangent))(eps_r, eps_p_r, alpha_r)
s_dp, C_dp, (ep_dp, al_dp) = jax.jit(jax.vmap(dp0.stress_and_tangent))(eps_r, eps_p_r, alpha_r)

d_sigma = float(jnp.max(jnp.abs(s_j2 - s_dp)))
d_C     = float(jnp.max(jnp.abs(C_j2 - C_dp)))
d_eps_p = float(jnp.max(jnp.abs(ep_j2 - ep_dp)))
d_alpha = float(jnp.max(jnp.abs(al_j2 - al_dp)))
print(f"[1] J2 degeneracy over {N} random states: max|dsigma|={d_sigma:.3e}  max|dC|={d_C:.3e}  "
      f"max|deps_p|={d_eps_p:.3e}  max|dalpha|={d_alpha:.3e}")
assert float(jnp.max(al_j2)) > 0.0, "degeneracy check never engaged the plastic branch"
assert d_sigma < 1e-9 * float(jnp.max(jnp.abs(s_j2)))
assert d_C     < 1e-9 * float(jnp.max(jnp.abs(C_j2)))
assert d_eps_p < 1e-12 and d_alpha < 1e-12
print("[1] PASSED")

# ── 2. uniaxial tension/compression yield asymmetry ─────────────────────────

A_F = 0.25
dp_pp = DruckerPrager(E=E, nu=NU, sigma_y0=SY, H=0.0, a_f=A_F, a_g=A_F)  # perfectly plastic


def _uniaxial_yield(model, sign, n_scan=40001, e_max=0.05):
    """Last stress before the plastic branch engages along uniaxial stress-free-lateral strain."""
    scan = jax.jit(jax.vmap(lambda exx: model.stress(
        jnp.diag(jnp.array([exx, -NU * exx, -NU * exx])), jnp.zeros((3, 3)), jnp.array(0.0),
    )))
    xs = jnp.array(np.linspace(0.0, sign * e_max, n_scan))
    sigma, (_, alpha) = scan(xs)
    i = int(np.argmax(np.array(alpha) > 1e-14)) - 1
    assert i > 0, "uniaxial scan never yielded -- widen e_max"
    return float(sigma[i, 0, 0])


sig_t, sig_c = _uniaxial_yield(dp_pp, +1), _uniaxial_yield(dp_pp, -1)
exact_t, exact_c = SY / (1.0 + A_F), -SY / (1.0 - A_F)
print(f"[2] uniaxial yield (a_f={A_F}): tension {sig_t:+.4f} vs exact {exact_t:+.4f}   "
      f"compression {sig_c:+.4f} vs exact {exact_c:+.4f}")
# tolerance is the scan resolution itself, not the model's: the sampled point
# is the last ELASTIC one before yield, so it undershoots by up to one step.
assert abs(sig_t - exact_t) < 1e-2 and abs(sig_c - exact_c) < 1e-2
print("[2] PASSED")

# ── 3/4. consistency, finiteness, apex return ───────────────────────────────

dp = DruckerPrager(E=E, nu=NU, sigma_y0=SY, H=H, a_f=A_F, a_g=0.05)

M = 2000
eps_big = jnp.array(rng.normal(scale=0.05, size=(M, 3, 3)))
sigma_m, C_m, (eps_p_m, alpha_m) = jax.jit(jax.vmap(dp.stress_and_tangent))(
    eps_big, jnp.zeros((M, 3, 3)), jnp.zeros(M)
)
q_m, p_m = _q_p(sigma_m)
f_m = q_m + 3.0 * dp.a_f * p_m - (dp.sigma_y0 + dp.H * alpha_m)
plastic = np.array(alpha_m) > 1e-14
apex    = np.array(q_m) < 1e-6
f_scale = float(dp.sigma_y0)

n_cone, n_apex = int((plastic & ~apex).sum()), int((plastic & apex).sum())
worst_f = float(np.max(np.abs(np.array(f_m)[plastic])))
print(f"[3] {int(plastic.sum())}/{M} plastic ({n_cone} cone, {n_apex} apex): "
      f"max|f| after return = {worst_f:.3e} (sigma_y0 = {f_scale:g})")
assert n_cone > 0 and n_apex > 0, "random sweep did not exercise both return branches"
assert worst_f < 1e-9 * f_scale, "returned state does not lie on the yield surface"
assert not bool(jnp.any(jnp.isnan(sigma_m))), "NaN in stress"
assert not bool(jnp.any(jnp.isnan(C_m))), "NaN in tangent"
assert not bool(jnp.any(jnp.isnan(eps_p_m))), "NaN in plastic strain"
print("[3] PASSED")

# the virgin state (eps = eps_p = 0) is the exact point where d/deps sqrt(s:s)
# is singular -- it must give the plain elastic tangent, not NaN.
s_v, C_v, _ = dp.stress_and_tangent(jnp.zeros((3, 3)), jnp.zeros((3, 3)), jnp.array(0.0))
assert bool(jnp.all(s_v == 0.0)) and bool(jnp.all(jnp.isfinite(C_v)))
assert float(jnp.max(jnp.abs(C_v - dp.stiffness_tensor()))) == 0.0, \
    "virgin-state tangent is not the elastic stiffness"

eps_hyd = jnp.eye(3) * 0.05  # far past the cone vertex in hydrostatic tension
for a_g in (A_F, 0.05, 0.0):
    m = DruckerPrager(E=E, nu=NU, sigma_y0=SY, H=H, a_f=A_F, a_g=a_g)
    s_a, C_a, (_, al_a) = m.stress_and_tangent(eps_hyd, jnp.zeros((3, 3)), jnp.array(0.0))
    q_a, p_a = _q_p(s_a)
    f_a = float(q_a) + 3.0 * A_F * float(p_a) - (SY + H * float(al_a))
    print(f"[4] apex return a_g={a_g:.2f}: q={float(q_a):.3e}  p={float(p_a):.4f}  "
          f"alpha={float(al_a):.4e}  f={f_a:+.3e}  tangent finite={bool(jnp.all(jnp.isfinite(C_a)))}")
    assert float(q_a) < 1e-9, "apex return left a nonzero deviatoric stress"
    assert abs(f_a) < 1e-9 * f_scale
    assert float(al_a) > 0.0
    assert bool(jnp.all(jnp.isfinite(C_a))), "apex tangent is not finite"
print("[4] PASSED")

# ── 5. consistent tangent vs finite differences, and its major symmetry ─────

def _tangent_report(model, n_samples=60, h=1e-7):
    worst = {"cone": (0.0, 0), "apex": (0.0, 0)}
    worst_asym = 0.0
    for _ in range(n_samples):
        e = jnp.array(rng.normal(scale=0.03, size=(3, 3)))
        e = 0.5 * (e + e.T)
        sigma, C, (_, alpha) = model.stress_and_tangent(e, jnp.zeros((3, 3)), jnp.array(0.0))
        if float(alpha) <= 1e-14:
            continue  # elastic: tangent is trivially the elastic stiffness
        q, _ = _q_p(sigma)
        key = "apex" if float(q) < 1e-6 else "cone"
        scale = float(jnp.max(jnp.abs(C)))
        err = 0.0
        for i in range(3):
            for j in range(3):
                d = np.zeros((3, 3)); d[i, j] += 0.5; d[j, i] += 0.5
                d = jnp.array(d)
                sp, _ = model.stress(e + h * d, jnp.zeros((3, 3)), jnp.array(0.0))
                sm, _ = model.stress(e - h * d, jnp.zeros((3, 3)), jnp.array(0.0))
                err = max(err, float(jnp.max(jnp.abs(
                    (sp - sm) / (2.0 * h) - jnp.einsum("ijkl,kl->ij", C, d)
                ))))
        prev, cnt = worst[key]
        worst[key] = (max(prev, err / scale), cnt + 1)
        worst_asym = max(worst_asym, float(jnp.max(jnp.abs(C - C.transpose(2, 3, 0, 1)))) / scale)
    return worst, worst_asym


for a_g, associated in ((A_F, True), (0.05, False)):
    m = DruckerPrager(E=E, nu=NU, sigma_y0=SY, H=H, a_f=A_F, a_g=a_g)
    worst, asym = _tangent_report(m)
    (e_cone, n_c), (e_apex, n_a) = worst["cone"], worst["apex"]
    print(f"[5] a_g={a_g:.2f} ({'associated' if associated else 'non-associated'}): "
          f"FD-vs-jacfwd rel err cone({n_c})={e_cone:.2e} apex({n_a})={e_apex:.2e}   "
          f"rel major asymmetry={asym:.2e}")
    assert n_c > 0 and n_a > 0, "tangent check did not exercise both branches"
    assert e_cone < 1e-6 and e_apex < 1e-6, "consistent tangent disagrees with finite differences"
    if associated:
        assert asym < 1e-12, "associated flow must give a major-symmetric tangent"
    else:
        assert asym > 1e-3, ("non-associated flow must give a NON-symmetric tangent -- a symmetric "
                             "one here means the dilatancy term stopped differing from the friction term")
print("[5] PASSED")

# ── 6. drop-in through assemble_local_update + the nonlinear solver ─────────

phase_np, n, L, _ = make_square_composite_rve(phi=0.5, r_fiber=0.005, dx=0.0002, N_min=16, nz=1)
Nv = int(np.prod(n))
phase = jnp.array(phase_np.reshape(-1))
xi_flat = build_freq_grid(n, L)

materials = [
    DruckerPrager(E=E, nu=NU, sigma_y0=56.1, H=H, a_f=0.13, a_g=0.05, name="epoxy-DP"),
    LinearElasticIsotropic(E=70.0e3, nu=0.20, name="glass"),
]
local_update, state0 = assemble_local_update(materials, phase)

n_plastic = {}
for label, sign in (("tension", +1.0), ("compression", -1.0)):
    eps_bar = jnp.zeros((3, 3)).at[0, 0].set(sign * 0.02)
    _, sigma, (_, alpha), converged, n_iter = solve_displacement_based_nonlinear(
        n, xi_flat, eps_bar, local_update, state0,
        toler_lin=1e-8, maxiter_lin=2000, toler_nr=1e-8, maxiter_nr=50,
    )
    n_plastic[label] = int(jnp.sum((alpha > 1e-12) & (phase == 0)))
    print(f"[6] macroscopic {label:11s}: converged={converged}  n_iter={n_iter:2d}  "
          f"plastic matrix voxels={n_plastic[label]}/{int(jnp.sum(phase == 0))}  "
          f"sigma_11_avg={float(jnp.mean(sigma[0, 0])):+.3f} MPa")
    assert converged, f"Newton did not converge under macroscopic {label}"
    assert not bool(jnp.any(jnp.isnan(sigma)))

assert n_plastic["tension"] > n_plastic["compression"], (
    "Drucker-Prager showed no tension/compression asymmetry after homogenization -- "
    f"tension {n_plastic['tension']}, compression {n_plastic['compression']}"
)
print("[6] PASSED")

print("\ntest_materialmodels_drucker_prager: all checks passed")
