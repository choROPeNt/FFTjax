"""
Standalone test for materialmodels.phasefield.drucker_prager
.PhaseFieldDruckerPrager -- Drucker-Prager plasticity coupled to AT2
phase-field damage through the effective stress -- and for the assembly and
driver wiring that carries it
(materialmodels.assembly.assemble_inelastic_pff_local_update,
problems.fracture's "newton" staggered path).

Ten checks
----------
1. d = 0 degeneracy: g(0) = 1, so stress, tangent AND updated plastic state
   must reproduce DruckerPrager's. The state must match BIT-FOR-BIT (the
   return mapping is called through super(), unmodified); sigma and C only to
   float64 epsilon, because g(0) = (1-k)*1 + k does not round to exactly 1.0
   and the p-split regroups s + p*I into two additions. The strongest check
   available, same argument as test_materialmodels_drucker_prager's [1]: it
   compares against an independently-verified model.
2. Elastic degeneracy: below yield, at ANY d, stress and tangent must
   reproduce PhaseFieldIsotropic's -- the two formulas coincide
   algebraically, so a sign or split error cannot hide. A second independent
   reference model, this one for the damage half.
3. Damage does not move the yield surface. Recompute sigma_eff = C:(eps -
   eps_p) from the returned state and require f = 0 for every d in [0, 1).
   This is the effective-stress design decision, asserted rather than
   assumed.
4. Compression survives cracking. A fully cracked voxel must carry
   hydrostatic COMPRESSION undegraded and hydrostatic TENSION not at all --
   the Amor structure at the stress level, and what stops a crack
   interpenetrating.
5. Consistent tangent vs central finite differences, across elastic / cone /
   apex states and a range of d; plus the major-symmetry behaviour inherited
   from DruckerPrager (symmetric at a_g = a_f, genuinely not otherwise).
6. Plastic work: W_p accumulated over one coarse increment must match a
   fine-increment integration of the same path, and must be monotone
   non-decreasing. Checked for NON-associated flow, where the closed form
   int sigma_y dalpha would be wrong.
7. Driving force: drive - psi_pos must equal beta_p*W_p exactly, and psi_pos
   must demonstrably be built on the ELASTIC strain (assert it differs from
   the total-strain split once plastic, or the check proves nothing).
8. Through the real driver: problems.fracture.solve_fracture on a seeded
   cube must auto-select the newton path, converge, crack, and SOFTEN; and
   macroscopic tension must damage more than compression of the same
   magnitude -- the pressure sensitivity surviving into fracture, which is
   the point of this model over a brittle matrix.
9. Guard rails: lippmann_schwinger raises, solve_fracture_fixed raises, and
   both other assembly combinators raise rather than silently driving this
   material undegraded.
10. The Newton warm start across increments (problems.fracture's eps_init):
   restarting a mixed-BC sweep from its own converged strain must pass the
   first Newton residual check, and solve_fracture_incremental must thread
   the previous increment's strain far enough past the peak that a cold
   first sweep would fail.

Usage
-----
    python test/test_materialmodels_phasefield_drucker_prager.py
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
sys.path.insert(0, "src")

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)
import jax
import jax.numpy as jnp
import numpy as np

from materialmodels.assembly import (assemble_inelastic_pff_local_update,
                                     assemble_local_update,
                                     assemble_pff_local_update)
from materialmodels.elastic.isotropic import LinearElasticIsotropic
from materialmodels.inelastic.plasticity_drucker_prager import DruckerPrager
from materialmodels.phasefield.drucker_prager import PhaseFieldDruckerPrager
from materialmodels.phasefield.isotropic import PhaseFieldIsotropic
from materialmodels.phasefield.splits import amor_split
from problems.fracture import solve_fracture, solve_fracture_fixed, solve_fracture_incremental

E, NU, SY, H = 3.76e3, 0.39, 56.1, 1.0e3
A_F, A_G, GC = 0.13, 0.05, 6.4e-4
rng = np.random.default_rng(0)


def _sym(a):
    return 0.5 * (a + jnp.swapaxes(a, -1, -2))


def _q_p(sigma):
    """von Mises equivalent stress and mean stress of a (..., 3, 3) stress."""
    p = jnp.trace(sigma, axis1=-2, axis2=-1) / 3.0
    s = sigma - p[..., None, None] * jnp.eye(3)
    return jnp.sqrt(1.5 * jnp.sum(s ** 2, axis=(-2, -1))), p


def _states(N, scale=0.02, ep_scale=0.004, al_max=0.06):
    eps = _sym(jnp.array(rng.normal(scale=scale, size=(N, 3, 3))))
    ep = _sym(jnp.array(rng.normal(scale=ep_scale, size=(N, 3, 3))))
    al = jnp.array(rng.uniform(0.0, al_max, size=N))
    return eps, ep, al


pf = PhaseFieldDruckerPrager(E=E, nu=NU, Gc=GC, sigma_y0=SY, H=H,
                             a_f=A_F, a_g=A_G, name="dp matrix")
dp = DruckerPrager(E=E, nu=NU, sigma_y0=SY, H=H, a_f=A_F, a_g=A_G)


# ── 1. d = 0 collapses onto DruckerPrager ───────────────────────────────────
N1 = 400
eps1, ep1, al1 = _states(N1)
s_pf, C_pf, (epn_pf, aln_pf, wpn_pf) = jax.vmap(pf.stress_and_tangent)(
    eps1, jnp.zeros(N1), ep1, al1, jnp.zeros(N1))
s_dp, C_dp, (epn_dp, aln_dp) = jax.vmap(dp.stress_and_tangent)(eps1, ep1, al1)

fin1 = jnp.isfinite(s_pf).all(axis=(1, 2)) & jnp.isfinite(s_dp).all(axis=(1, 2))
assert int(jnp.sum(fin1)) > N1 // 2, "too many unreachable states to test anything"


def _masked_max(x, mask):
    return float(jnp.max(jnp.abs(jnp.where(
        mask.reshape((-1,) + (1,) * (x.ndim - 1)), x, 0.0))))


d_sig1 = _masked_max(s_pf - s_dp, fin1)
d_C1 = _masked_max(C_pf - C_dp, fin1)
rel_sig1 = d_sig1 / max(_masked_max(s_dp, fin1), 1e-30)
rel_C1 = d_C1 / max(_masked_max(C_dp, fin1), 1e-30)
d_ep1 = _masked_max(epn_pf - epn_dp, fin1)
d_al1 = _masked_max(aln_pf - aln_dp, fin1)
print(f"[1] d=0 vs DruckerPrager over {int(jnp.sum(fin1))} states: "
      f"sigma rel={rel_sig1:.2e}  C rel={rel_C1:.2e}  "
      f"eps_p abs={d_ep1:.2e}  alpha abs={d_al1:.2e}")
# The STATE must be bit-for-bit: it comes straight out of the untouched
# return mapping, with no degraded arithmetic applied to it at all.
assert d_ep1 == 0.0 and d_al1 == 0.0, (d_ep1, d_al1)
assert rel_sig1 < 1e-14 and rel_C1 < 1e-14, (rel_sig1, rel_C1)
assert float(jnp.max(wpn_pf)) > 0.0, "no plastic work accumulated -- check 1 is elastic"
print("[1] PASSED")


# ── 2. the elastic limit IS PhaseFieldIsotropic ─────────────────────────────
# Independent of check 1: that one fixes d and varies plasticity, this one
# fixes plasticity (off) and varies d. Between them, both halves of the
# coupling are pinned to an already-verified model.
N2 = 400
pfi = PhaseFieldIsotropic(E=E, nu=NU, Gc=GC)
eps2 = _sym(jnp.array(rng.normal(scale=8e-4, size=(N2, 3, 3))))
d2 = jnp.array(rng.uniform(0.0, 1.0, size=N2))
s2, C2, (ep2, al2, wp2) = jax.vmap(pf.stress_and_tangent)(
    eps2, d2, jnp.zeros((N2, 3, 3)), jnp.zeros(N2), jnp.zeros(N2))
s2r, C2r, _psi2 = jax.vmap(pfi.stress_and_tangent)(eps2, d2)
assert float(jnp.max(al2)) == 0.0, "check 2's states are not elastic"
rel_s2 = float(jnp.max(jnp.abs(s2 - s2r))) / float(jnp.max(jnp.abs(s2r)))
rel_C2 = float(jnp.max(jnp.abs(C2 - C2r))) / float(jnp.max(jnp.abs(C2r)))
print(f"[2] elastic, d ~ U(0,1), vs PhaseFieldIsotropic: "
      f"sigma rel={rel_s2:.2e}  C rel={rel_C2:.2e}")
assert rel_s2 < 1e-13 and rel_C2 < 1e-13, (rel_s2, rel_C2)
print("[2] PASSED")


# ── 3. damage must NOT move the yield surface ───────────────────────────────
worst_f3, n_pl3 = 0.0, 0
for d_val in (0.0, 0.2, 0.5, 0.8, 0.99):
    s3, C3, (ep3, al3, wp3) = jax.vmap(pf.stress_and_tangent)(
        eps1, jnp.full(N1, d_val), ep1, al1, jnp.zeros(N1))
    # The effective stress is recoverable from the state alone: the return
    # mapping is consistent, so sigma_eff = C : (eps - eps_p) exactly.
    eps_e = eps1 - ep3
    tr = jnp.trace(eps_e, axis1=1, axis2=2)
    sig_eff = (pf.lam * tr[:, None, None] * jnp.eye(3) + 2.0 * pf.mu * eps_e)
    q_e, p_e = _q_p(sig_eff)
    f3 = q_e + 3.0 * A_F * p_e - pf.hardening.sigma_y(al3)
    plastic3 = jnp.isfinite(q_e) & (al3 > al1 + 1e-12)
    n_pl3 += int(jnp.sum(plastic3))
    worst_f3 = max(worst_f3, float(jnp.max(jnp.abs(jnp.where(plastic3, f3, 0.0)))))
    assert bool(jnp.all(jnp.isfinite(jnp.where(
        fin1[:, None, None, None, None], C3, 0.0)))), f"non-finite tangent at d={d_val}"
print(f"[3] effective stress on the yield surface over d in [0, 0.99]: "
      f"{n_pl3} plastic states, max|f|={worst_f3:.2e} (sigma_y0={SY:g})")
assert worst_f3 < 1e-8 * SY, worst_f3
print("[3] PASSED")


# ── 4. a crack carries compression, not tension ─────────────────────────────
zero3, zero0 = jnp.zeros((3, 3)), jnp.asarray(0.0)
eps_c = -3e-3 * jnp.eye(3)
p_c_int = float(jnp.trace(pf.stress(eps_c, jnp.asarray(0.0), zero3, zero0, zero0)[0])) / 3.0
p_c_brk = float(jnp.trace(pf.stress(eps_c, jnp.asarray(1.0), zero3, zero0, zero0)[0])) / 3.0
p_t_int = float(jnp.trace(pf.stress(-eps_c, jnp.asarray(0.0), zero3, zero0, zero0)[0])) / 3.0
p_t_brk = float(jnp.trace(pf.stress(-eps_c, jnp.asarray(1.0), zero3, zero0, zero0)[0])) / 3.0
print(f"[4] hydrostatic compression p: intact {p_c_int:+.4f} -> cracked {p_c_brk:+.4f} "
      f"(ratio {p_c_brk / p_c_int:.6f})")
print(f"    hydrostatic tension     p: intact {p_t_int:+.4f} -> cracked {p_t_brk:+.4f} "
      f"(ratio {p_t_brk / p_t_int:.2e})")
assert abs(p_c_brk / p_c_int - 1.0) < 1e-9, "compression was degraded"
assert p_t_brk / p_t_int < 1e-5, "tension was not degraded"
print("[4] PASSED")


# ── 5. consistent tangent vs finite differences ─────────────────────────────
# Directional-derivative form, the same convention (and the same h) as
# test_materialmodels_drucker_prager's check 5: compare (sp - sm)/(2h)
# against C : d for each perturbation direction d, rather than assembling a
# component-wise C_fd, which needs a factor-of-2 convention for the
# symmetrized off-diagonal entries that is easy to get wrong.
def _fd_err(model, eps, d_dmg, ep, al, wp, C, h=1e-7):
    err = 0.0
    for i in range(3):
        for j in range(3):
            dd = np.zeros((3, 3))
            dd[i, j] += 0.5
            dd[j, i] += 0.5
            dd = jnp.array(dd)
            sp = model.stress(eps + h * dd, d_dmg, ep, al, wp)[0]
            sm = model.stress(eps - h * dd, d_dmg, ep, al, wp)[0]
            err = max(err, float(jnp.max(jnp.abs(
                (sp - sm) / (2.0 * h) - jnp.einsum("ijkl,kl->ij", C, dd)))))
    return err


worst5 = 0.0
n5 = 0
n5_plastic = 0
for d_val in (0.0, 0.4, 0.9):
    for i in range(24):
        eps_i, ep_i, al_i = eps1[i], ep1[i], al1[i]
        s_i, C_i, (_, al_new_i, _) = pf.stress_and_tangent(
            eps_i, jnp.asarray(d_val), ep_i, al_i, jnp.asarray(0.0))
        if not bool(jnp.all(jnp.isfinite(s_i))):
            continue
        err = _fd_err(pf, eps_i, jnp.asarray(d_val), ep_i, al_i,
                      jnp.asarray(0.0), C_i)
        scale = max(float(jnp.max(jnp.abs(C_i))), 1e-30)
        worst5 = max(worst5, err / scale)
        n5 += 1
        if float(al_new_i) > float(al_i) + 1e-14:
            n5_plastic += 1
print(f"[5] tangent vs central FD over {n5} states x 3 damage levels "
      f"({n5_plastic} of them plastic): max rel err = {worst5:.2e}")
assert n5 > 40, f"only {n5} FD comparisons"
assert n5_plastic > 10, (
    f"only {n5_plastic} plastic states -- check 5 would be testing the elastic "
    "tangent, which is trivial"
)
assert worst5 < 1e-5, worst5

# Major symmetry. DruckerPrager's tangent is symmetric for ASSOCIATED flow
# and genuinely not otherwise. Degradation breaks that: once a voxel is both
# damaged AND plastic the tangent is asymmetric even at a_g = a_f, because the
# Amor stress-level operator g*P_+ + P_- commutes with the ISOTROPIC elastic
# tangent but not with the elastoplastic one's s⊗s structure. Asserted here
# with its d-dependence, since the magnitude (up to ~30%) is what decides
# whether CG on this operator is trustworthy -- see the model's docstring.
pf_assoc = PhaseFieldDruckerPrager(E=E, nu=NU, Gc=GC, sigma_y0=SY, H=H,
                                   a_f=A_F, a_g=A_F)


def _asym(model, d_val, plastic_only=True, n=40):
    worst, cnt = 0.0, 0
    for i in range(n):
        _, C_i, (_, al_new, _) = model.stress_and_tangent(
            eps1[i], jnp.asarray(d_val), ep1[i], al1[i], jnp.asarray(0.0))
        if not bool(jnp.all(jnp.isfinite(C_i))):
            continue
        if plastic_only and float(al_new) <= float(al1[i]) + 1e-14:
            continue
        a = float(jnp.max(jnp.abs(C_i - jnp.einsum("ijkl->klij", C_i))))
        worst = max(worst, a / max(float(jnp.max(jnp.abs(C_i))), 1e-30))
        cnt += 1
    return worst, cnt


print("    major asymmetry of the tangent, associated flow (a_g = a_f):")
asym_by_d = {}
for d_val in (0.0, 0.1, 0.5, 0.9):
    a5, c5 = _asym(pf_assoc, d_val)
    asym_by_d[d_val] = a5
    print(f"      d={d_val:<4g} asym={a5:.3e}  ({c5} plastic states)")
assert c5 > 10, f"only {c5} plastic states -- the asymmetry scan is vacuous"
# Undamaged: DruckerPrager's own behaviour, so associated flow is symmetric.
assert asym_by_d[0.0] < 1e-12, (
    f"associated flow at d=0 is not symmetric ({asym_by_d[0.0]:.2e}) -- that is "
    "DruckerPrager's tangent unchanged, so something in the coupling is wrong"
)
# Damaged: asymmetric, and monotonically more so.
assert asym_by_d[0.9] > 1e-2, (
    f"damage did not break the symmetry ({asym_by_d[0.9]:.2e}) -- either the "
    "degradation is not reaching the tangent, or something symmetrized it"
)
assert asym_by_d[0.1] < asym_by_d[0.5] < asym_by_d[0.9], (
    f"asymmetry is not monotone in d: {asym_by_d}"
)
# The elastic limit must stay EXACTLY symmetric at any d -- there it is a true
# energy Hessian (PhaseFieldIsotropic), so this isolates the cause to the
# interaction of degradation with the PLASTIC tangent, not to degradation alone.
worst_el5 = 0.0
for d_val in (0.0, 0.5, 0.9):
    for i in range(30):
        _, C_el, _ = pf_assoc.stress_and_tangent(
            eps2[i], jnp.asarray(d_val), jnp.zeros((3, 3)), jnp.asarray(0.0),
            jnp.asarray(0.0))
        sc = max(float(jnp.max(jnp.abs(C_el))), 1e-30)
        worst_el5 = max(worst_el5, float(jnp.max(jnp.abs(
            C_el - jnp.einsum("ijkl->klij", C_el)))) / sc)
print(f"      elastic states, any d: asym={worst_el5:.3e} (a true energy Hessian)")
assert worst_el5 < 1e-12, worst_el5
print("[5] PASSED")


# ── 6. accumulated plastic work ─────────────────────────────────────────────
# One coarse increment against many fine ones along the SAME strain path.
# Non-associated flow on purpose: sigma_eff : deps_p is the only correct
# integrand there, and int sigma_y dalpha would disagree.
def _walk(model, eps_end, n_steps, d_val=0.0):
    ep, al, wp = jnp.zeros((3, 3)), jnp.asarray(0.0), jnp.asarray(0.0)
    for k in range(1, n_steps + 1):
        _, (ep, al, wp) = model.stress(
            eps_end * (k / n_steps), jnp.asarray(d_val), ep, al, wp)
    return ep, al, wp


eps_end6 = jnp.array([[0.02, 0.004, 0.0], [0.004, -0.006, 0.0], [0.0, 0.0, -0.006]])
_, al_c, wp_c = _walk(pf, eps_end6, 4)
_, al_f, wp_f = _walk(pf, eps_end6, 400)
print(f"[6] W_p, 4 steps vs 400 along one path: {float(wp_c):.5f} vs {float(wp_f):.5f} "
      f"(rel {abs(float(wp_c - wp_f)) / float(wp_f):.2e}); alpha {float(al_c):.5f} "
      f"vs {float(al_f):.5f}")
assert float(wp_f) > 0.0, "no plastic work -- check 6 path is elastic"
assert abs(float(wp_c - wp_f)) / float(wp_f) < 0.05, "W_p is not path-convergent"

wp_seq = []
ep6, al6, wp6 = jnp.zeros((3, 3)), jnp.asarray(0.0), jnp.asarray(0.0)
for k in range(1, 41):
    _, (ep6, al6, wp6) = pf.stress(eps_end6 * (k / 40.0), jnp.asarray(0.3), ep6, al6, wp6)
    wp_seq.append(float(wp6))
assert all(b >= a - 1e-15 for a, b in zip(wp_seq, wp_seq[1:])), "W_p decreased"
print(f"    monotone over 40 increments, final W_p={wp_seq[-1]:.5f}")
print("[6] PASSED")


# ── 7. the driving force is built as documented ─────────────────────────────
BETA = 3.0
pf_b = PhaseFieldDruckerPrager(E=E, nu=NU, Gc=GC, sigma_y0=SY, H=H,
                               a_f=A_F, a_g=A_G, beta_p=BETA)
N7 = 200
eps7, ep7, al7 = _states(N7)
_, (ep7n, al7n, wp7n) = jax.vmap(pf_b.stress)(
    eps7, jnp.full(N7, 0.4), ep7, al7, jnp.zeros(N7))
drive7, psi7 = jax.vmap(pf_b.driving_force)(eps7, ep7n, wp7n)
gap7 = float(jnp.max(jnp.abs((drive7 - psi7) - BETA * wp7n)))
print(f"[7] drive - psi_pos vs beta_p*W_p: max|diff|={gap7:.2e}")
assert gap7 < 1e-10, gap7

# psi_pos must use the ELASTIC strain. Compare against the total-strain split:
# they have to DIFFER, or this check proves nothing about which was used.
psi_tot, _ = jax.vmap(amor_split, in_axes=(0, None, None))(eps7, pf_b.lam, pf_b.mu)
psi_el, _ = jax.vmap(amor_split, in_axes=(0, None, None))(eps7 - ep7n, pf_b.lam, pf_b.mu)
pl7 = jnp.isfinite(psi7) & (al7n > al7 + 1e-12)
d_el = float(jnp.max(jnp.abs(jnp.where(pl7, psi7 - psi_el, 0.0))))
d_tot = float(jnp.max(jnp.abs(jnp.where(pl7, psi7 - psi_tot, 0.0))))
print(f"    psi_pos vs elastic-strain split={d_el:.2e}, vs total-strain split={d_tot:.2e}")
assert d_el < 1e-10, "psi_pos is NOT the elastic-strain split"
assert d_tot > 1e-3, (
    f"the elastic- and total-strain splits differ by only {d_tot:.2e} on these "
    "states, so check 7 cannot tell them apart"
)
print("[7] PASSED")


# ── 8. through the real driver ──────────────────────────────────────────────
n8 = (16, 16, 16)
L8 = (1.0, 1.0, 1.0)
Nv8 = int(np.prod(n8))
l0_8 = 4.0 * L8[0] / n8[0]
mat8 = PhaseFieldDruckerPrager(E=E, nu=NU, Gc=2.0e-2, sigma_y0=SY, H=H,
                               a_f=A_F, a_g=A_G, name="dp")
phase8 = jnp.zeros(Nv8, dtype=int)
_idx = np.indices(n8)
_c = np.array([ni / 2 for ni in n8])
_dist = np.linalg.norm(
    np.stack([_idx[i] + 0.5 - _c[i] for i in range(3)]).reshape(3, -1), axis=0)
d_seed = jnp.array((_dist <= 2.0).astype(float) * 0.9)
H_seed = jnp.zeros(Nv8)

_SOLVE_KW = dict(formulation="displacement", toler_lin=1e-8, maxiter_cg=600,
                 toler_helm=1e-6, maxiter_helm=300, eta=0.0, d_thres=0.95,
                 toler_st_abs=1e-4, toler_st_rel=1e-5, maxiter_st=30,
                 toler_nr=1e-7, maxiter_nr=40)


def _pull(eps11, n_steps=6):
    """Load-step to eps11 and return the (sigma_11, d_max, alpha_max) history."""
    # eps_init threaded exactly as solve_fracture_incremental does it. Without
    # it the last increments here start Newton cold, which past the peak can
    # fail or not depending on round-off -- see check 10.
    d, Hh, st, eps = d_seed, H_seed, None, None
    hist = []
    for k in range(1, n_steps + 1):
        eb = jnp.zeros((3, 3)).at[0, 0].set(eps11 * k / n_steps)
        sol = solve_fracture(n8, L8, phase8, [mat8], eb, l0_8, None, d, Hh,
                             state_init=st, eps_init=eps, **_SOLVE_KW)
        d, Hh, st, eps = sol.d, sol.H, (sol.eps_p, sol.alpha, sol.W_p), sol.eps
        hist.append((float(jnp.mean(sol.sigma[0, 0])), float(jnp.max(sol.d)),
                     float(jnp.max(sol.alpha)), bool(sol.converged_mech), sol.iter_nr))
        assert sol.eps_p is not None, "newton path was not selected"
    return hist


hist_t = _pull(+0.012)
print("[8] tension, load-stepped (newton path auto-selected):")
for k, (s, dm, am, cm, it) in enumerate(hist_t, 1):
    print(f"      step {k}: sigma_11={s:+9.4f}  d_max={dm:.4f}  alpha_max={am:.3e}  "
          f"mech={cm}  nr={it}")
assert all(cm for *_, cm, _ in hist_t), "a tension increment's Newton solve failed"
sig_t = [s for s, *_ in hist_t]
assert max(sig_t) > 1.0, f"no load carried at all: {sig_t}"
assert sig_t[-1] < 0.5 * max(sig_t), (
    f"stress never softened (peak {max(sig_t):.3f}, final {sig_t[-1]:.3f}) -- "
    "nothing cracked, so check 8 is not testing fracture"
)
assert all(b >= a - 1e-9 for a, b in zip([d for _, d, *_ in hist_t],
                                         [d for _, d, *_ in hist_t][1:])), \
    "damage decreased"
d_max_t = hist_t[-1][1]

# The pressure sensitivity must survive into fracture: the same |eps_11| in
# compression must crack LESS. That is the whole reason for Drucker-Prager
# here rather than J2 or a brittle matrix.
hist_c = _pull(-0.012)
d_max_c = hist_c[-1][1]
print(f"    d_max after tension={d_max_t:.4f}  after compression={d_max_c:.4f}")
assert d_max_t > d_max_c + 1e-3, (
    f"compression damaged as much as tension ({d_max_c:.4f} vs {d_max_t:.4f}) -- "
    "the tension/compression split is not reaching the damage solve"
)
print("[8] PASSED")


# ── 9. guard rails ──────────────────────────────────────────────────────────
def _raises(fn, needle, what):
    try:
        fn()
    except (ValueError, TypeError) as exc:
        assert needle in str(exc), f"wrong message for {what}: {exc}"
        return
    raise AssertionError(f"{what}: expected a raise mentioning {needle!r}")


eb9 = jnp.zeros((3, 3)).at[0, 0].set(0.01)
_raises(lambda: solve_fracture(n8, L8, phase8, [mat8], eb9, l0_8, None,
                               d_seed, H_seed, formulation="lippmann_schwinger"),
        "cannot be used with a stateful", "lippmann_schwinger + coupled material")
_raises(lambda: solve_fracture_fixed(n8, L8, phase8, [mat8], eb9, l0_8, None,
                                     d_seed, H_seed, formulation="displacement"),
        "cannot take a stateful", "solve_fracture_fixed + coupled material")

eps9 = _sym(jnp.array(rng.normal(scale=0.01, size=(3, 3, Nv8)).transpose(2, 0, 1))
            ).transpose(1, 2, 0)
d9 = jnp.full(Nv8, 0.3)
_raises(lambda: assemble_pff_local_update([mat8], phase8)(eps9, d9),
        "assemble_inelastic_pff_local_update",
        "assemble_pff_local_update + coupled material")
_raises(lambda: assemble_local_update([mat8], phase8)[0](
            eps9, (jnp.zeros((3, 3, Nv8)), jnp.zeros(Nv8))),
        "assemble_inelastic_pff_local_update",
        "assemble_local_update + coupled material")
_raises(lambda: PhaseFieldDruckerPrager(E=E, nu=NU, Gc=GC, sigma_y0=SY, H=H,
                                        a_f=A_F, beta_p=-1.0),
        "beta_p", "negative beta_p")

# And the mixed list the combinator is meant to support must still work.
mats9 = [mat8, PhaseFieldIsotropic(E=20e3, nu=0.2, Gc=1e-3),
         LinearElasticIsotropic(E=70e3, nu=0.2, k_res=1.0, Gc=1e-2)]
phase9 = jnp.array(([0] * (Nv8 // 2) + [1] * (Nv8 // 4)
                    + [2] * (Nv8 - Nv8 // 2 - Nv8 // 4)))
lu9, df9, st9 = assemble_inelastic_pff_local_update(mats9, phase9)
sig9, C9, (ep9, al9, wp9) = lu9(eps9, d9, st9)
drive9, psi9 = df9(eps9, (ep9, al9, wp9))
assert bool(jnp.all(jnp.isfinite(sig9))) and bool(jnp.all(jnp.isfinite(C9)))
assert bool(jnp.all(jnp.isfinite(drive9)))
# Only the coupled phase may have advanced plastic state.
other = (phase9 != 0)
assert float(jnp.max(jnp.where(other, al9, 0.0))) == 0.0, \
    "a non-plastic phase's state was advanced"
assert float(jnp.max(jnp.where(phase9 == 0, al9, 0.0))) > 0.0, \
    "the plastic phase never yielded -- the mixed-list check is vacuous"
print("[9] guard rails raise as documented, and a mixed "
      "coupled/brittle/elastic list still assembles")
print("[9] PASSED")


# ── 10. the Newton warm start across increments (eps_init) ─────────────────
# (a) Exactness, under MIXED BC, where the split matters: eps_init is taken
# apart into a zero-mean fluctuation plus the mean strain on stress-controlled
# directions. Restarting one sweep from its own converged strain must then
# satisfy Newton at the very first residual check -- a dropped lateral
# contraction or a fluctuation with the wrong mean would not.
ctrl10 = ((0, 0, 0), (0, 1, 0), (0, 0, 1))      # pull 11, 22/33 traction-free
kw10 = dict(_SOLVE_KW, control=ctrl10)
d10, H10, st10, eps10 = d_seed, H_seed, None, None
for k in (1, 2):
    eb = jnp.zeros((3, 3)).at[0, 0].set(0.002 * k)
    s10 = solve_fracture(n8, L8, phase8, [mat8], eb, l0_8, None, d10, H10,
                         state_init=st10, eps_init=eps10, **kw10)
    assert bool(s10.converged), f"mixed-BC setup increment {k} did not converge"
    d10, H10, st10, eps10 = s10.d, s10.H, (s10.eps_p, s10.alpha, s10.W_p), s10.eps
eb10 = jnp.zeros((3, 3)).at[0, 0].set(0.006)
kw10_one = dict(kw10, maxiter_st=1)
one = solve_fracture(n8, L8, phase8, [mat8], eb10, l0_8, None, d10, H10,
                     state_init=st10, eps_init=eps10, **kw10_one)
again = solve_fracture(n8, L8, phase8, [mat8], eb10, l0_8, None, d10, H10,
                       state_init=st10, eps_init=one.eps, **kw10_one)
lat = float(jnp.mean(one.eps[1, 1]))
print(f"[10] mixed BC, one sweep: nr={one.iter_nr} (alpha_max="
      f"{float(jnp.max(one.alpha)):.3e}, eps_22={lat:+.3e}); "
      f"restarted from its own strain: nr={again.iter_nr}")
assert bool(one.converged_mech), "the reference sweep's Newton failed"
assert float(jnp.max(one.alpha)) > 0.0 and lat < 0.0, \
    "no plastic flow or no lateral contraction -- (a) would not test the split"
assert again.iter_nr == 1, (
    f"restart from a converged strain took {again.iter_nr} Newton iterations "
    "-- eps_init is not being split back into the state it came from")

# (b) solve_fracture_incremental must thread it. Pulled this far past the
# peak, a first sweep started from the uniform eps_bar fails Newton, and
# stepping="fixed" raises on the first increment that does not converge.
res10 = solve_fracture_incremental(
    n8, L8, phase8, [mat8], jnp.zeros((3, 3)).at[0, 0].set(0.016), l0_8, None,
    d_seed, H_seed, stepping="fixed", dt_step=1.0 / 8, **_SOLVE_KW)
sig10 = [float(jnp.mean(r.solution.sigma[0, 0])) for r in res10]
print("[10] solve_fracture_incremental, 8 fixed steps to eps_11=0.016: "
      f"sigma_11 = {' '.join(f'{s:.3g}' for s in sig10)}")
assert all(bool(r.solution.converged_mech) for r in res10)
assert all(s > 0.0 for s in sig10), \
    f"negative stress under applied tension -- a non-equilibrium state: {sig10}"
print("[10] PASSED")

print("\ntest_materialmodels_phasefield_drucker_prager: all checks passed")
