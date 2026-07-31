"""
Standalone test for the preconditioned Helmholtz CG damage solver.

Tests ``solve_helmholtz_cg`` in isolation — no mechanical solve needed.

Three checks
------------
1. Uniform L — analytical solution is  d = 2L / (Gc/l₀ + 2L)
   Verifies the DC component and that the CG converges.

2. Crack-line driving force — L = 1 on the pre-crack row, 0 elsewhere.
   Damage should peak at the crack and decay with characteristic width l₀.
   Irreversibility is verified by re-solving with L = 0 and checking d unchanged.

3. Length-scale sensitivity — repeat check 2 for l₀ ∈ {0.5, 1, 2, 4} mm.
   The half-width of the damage profile must scale proportionally with l₀.

Output
------
    output/test_helmholtz_l0_{...}.h5 / .xdmf  (one file per l₀)

Usage
-----
    python scripts/test_helmholtz.py
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
sys.path.insert(0, "src")

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)
import jax
import jax.numpy as jnp
import numpy as np

from operators.green    import build_freq_grid
from post.io            import IncrementalWriter
from solvers.damage.pff_damage  import solve_helmholtz_cg, solve_helmholtz_cg_het

jax.config.update("jax_enable_x64", True)

# ── Grid (same geometry as benchmark) ────────────────────────────────────────

n   = (250, 250, 1)
L   = (50.0, 50.0, 0.2)   # mm
Nv  = int(np.prod(n))
dx  = tuple(Li / ni for Li, ni in zip(L, n))

xi_flat = build_freq_grid(n, L)

# Material parameters — Schneider & Kästner (2024), units MPa·mm
Gc_ref = 2.7e-3   # kN/mm  →  MPa·mm  (2.7 N/mm / 1e3 * 1e3 = 2.7 MPa·mm, /2 for AT2)

os.makedirs("output", exist_ok=True)

# ── Check 1 — uniform L, analytical solution ─────────────────────────────────

print("─" * 60)
print("Check 1: uniform L — d = 2L / (Gc/l₀ + 2L)")
print("─" * 60)

l0   = 1.0   # mm
Gc   = Gc_ref

# Choose L such that d_anal = 0.4  (well below 1, avoids clipping)
# d = 2L / (Gc/l₀ + 2L)  →  L = (d * Gc/l₀) / (2 * (1 - d))
d_target = 0.4
L0       = d_target * (Gc / l0) / (2.0 * (1.0 - d_target))

L_uniform = jnp.full((Nv,), L0)
d_prev    = jnp.zeros((Nv,))

d_out, conv_h = solve_helmholtz_cg(L_uniform, xi_flat, n, l0, Gc, d_prev,
                                    toler_cg=1e-10, maxiter=500)

d_anal  = 2.0 * L0 / (Gc / l0 + 2.0 * L0)
d_mean  = float(jnp.mean(d_out))
d_std   = float(jnp.std(d_out))
rel_err = abs(d_mean - d_anal) / d_anal

print(f"  l₀ = {l0} mm,  L₀ = {L0:.4e},  Gc = {Gc:.4e}")
print(f"  analytical d   = {d_anal:.6f}")
print(f"  mean(d_CG)     = {d_mean:.6f}")
print(f"  std(d_CG)      = {d_std:.2e}   (should be ~0)")
print(f"  relative error = {rel_err:.2e}   (should be <1e-6)")
print(f"  converged      = {bool(conv_h)}")
print()
print("  " + ("PASS" if rel_err < 1e-6 else "FAIL"))
print()

# ── Check 2 — crack-line L, irreversibility ──────────────────────────────────

print("─" * 60)
print("Check 2: crack-line L + irreversibility")
print("─" * 60)

l0   = 1.0
Gc   = Gc_ref
# L = Gc/(2*l₀) on crack → d_centre ≈ 0.5  (uniform approximation)
L_crack_val = Gc / (2.0 * l0)

j_crack     = n[1] // 2
ms_L        = np.zeros(n)
ms_L[20:60, j_crack, :] = L_crack_val
L_crack     = jnp.array(ms_L.ravel())
d_prev      = jnp.zeros((Nv,))

d_out, conv_h = solve_helmholtz_cg(L_crack, xi_flat, n, l0, Gc, d_prev,
                                    toler_cg=1e-8, maxiter=500)
d_np = np.asarray(d_out).reshape(n)
print(f"  converged = {bool(conv_h)}")
print(f"  max(d)  = {float(jnp.max(d_out)):.4f}")
print(f"  d at crack tip (x=40, y={j_crack}) = {d_np[40, j_crack, 0]:.4f}")

# Irreversibility: re-solve with L = 0 — d must not decrease
L_zero    = jnp.zeros((Nv,))
d_irrev, _ = solve_helmholtz_cg(L_zero, xi_flat, n, l0, Gc, d_out,
                                 toler_cg=1e-8, maxiter=500)
irrev_ok = bool(jnp.all(d_irrev >= d_out - 1e-12))
print(f"  Irreversibility (d_new >= d_old): {'PASS' if irrev_ok else 'FAIL'}")
print()

# ── Check 3 — length-scale sensitivity ───────────────────────────────────────

print("─" * 60)
print("Check 3: half-width of damage zone scales with l₀")
print("─" * 60)

y_coords = np.linspace(0.0, L[1], n[1], endpoint=False)
y_crack  = y_coords[j_crack]
r        = np.abs(y_coords - y_crack)   # distance from crack row

l0_values  = [0.5, 1.0, 2.0, 4.0]
half_widths = []

for l0 in l0_values:
    Gc = Gc_ref
    L_crack_val = 2*Gc / (2.0 * l0)

    ms_L = np.zeros(n)
    ms_L[20:60, j_crack, :] = L_crack_val
    L_field = jnp.array(ms_L.ravel())
    d_prev  = jnp.zeros((Nv,))

    d_out, conv_h = solve_helmholtz_cg(L_field, xi_flat, n, l0, Gc, d_prev,
                                        toler_cg=1e-8, maxiter=500)
    d_np = np.asarray(d_out).reshape(n)

    # Profile perpendicular to the crack at x = 40
    profile   = d_np[40, :, 0]
    d_centre  = float(profile[j_crack])
    profile_n = profile / (d_centre + 1e-30)

    # Half-width: 1/e decay point
    mask     = profile_n > np.exp(-1.0)
    half_w   = float(r[mask].max()) if mask.any() else 0.0
    half_widths.append(half_w)

    print(f"  l₀ = {l0:.1f} mm  |  d_centre = {d_centre:.4f}  "
          f"|  1/e half-width = {half_w:.2f} mm  (expect ~ l₀)")

    # ── write output ─────────────────────────────────────────────────────────
    jobname = f"test_helmholtz_l0_{str(l0).replace('.', 'p')}"

    with IncrementalWriter(f"output/{jobname}", grid_shape=n, grid_spacing=dx) as w:
        w.write_increment(0, {
            "driving_force": ms_L.astype(np.float64),
            "damage":        np.zeros(n, dtype=np.float64),
        }, time=0.0)
        w.write_increment(1, {
            "driving_force": ms_L.astype(np.float64),
            "damage":        d_np.astype(np.float64),
        }, time=1.0)

print()
# Verify half-width grows with l₀
ratios  = [half_widths[i+1] / half_widths[i] for i in range(len(half_widths)-1)]
l0_ratios = [l0_values[i+1] / l0_values[i] for i in range(len(l0_values)-1)]
scale_ok  = all(abs(r - lr) / lr < 0.15 for r, lr in zip(ratios, l0_ratios))
print(f"  Width ratios:  {[f'{r:.2f}' for r in ratios]}")
print(f"  l₀ ratios:     {[f'{r:.2f}' for r in l0_ratios]}")
print(f"  Scaling check: {'PASS' if scale_ok else 'FAIL (>15% deviation)'}")
print()
print("In ParaView: open a .xdmf and use 'Plot Over Line' perpendicular to")
print("the crack — the damage profile width should scale with l₀.")

# ── Check 4 — heterogeneous Gc solver ────────────────────────────────────────

print()
print("─" * 60)
print("Check 4a: het solver self-consistency and low-k agreement")
print("─" * 60)
# The het operator uses spectral grad/div with Nyquist zeroed; the scalar uses
# the full spectral Laplacian (Nyquist included).  For smooth fields both give
# nearly the same solution, but differ by O(l₀² · ξ_Nyq² · E_Nyquist / (Gc/l₀))
# where E_Nyquist is the Nyquist-frequency energy in the solution.
# We verify: (a) the het residual is small, (b) max(d) is close, (c) the
# low-wavenumber profile agrees to < 1 %.

l0  = 1.0
Gc  = Gc_ref
L_crack_val = Gc / (2.0 * l0)

ms_L = np.zeros(n)
ms_L[20:60, n[1] // 2, :] = L_crack_val
L_crack = jnp.array(ms_L.ravel())
d_prev  = jnp.zeros((Nv,))

d_scalar, _  = solve_helmholtz_cg(L_crack, xi_flat, n, l0, Gc, d_prev,
                                   toler_cg=1e-8, maxiter=500)
Gc_const     = jnp.full((Nv,), Gc)
d_het, cv_h  = solve_helmholtz_cg_het(L_crack, xi_flat, n, l0, Gc_const, d_prev,
                                       toler_cg=1e-8, maxiter=500)

print(f"  het   : conv={bool(cv_h)}")

# (a) het self-residual: evaluate het operator on d_het — should be ≈ bb
def _fft(v):  return jnp.fft.fftn(v.reshape(n)).reshape(Nv)
def _ifft(vh): return jnp.fft.ifftn(vh.reshape(n)).real.reshape(Nv)
bb = 2.0 * L_crack
# Use scalar operator as surrogate — both operators agree at low k
xi_sq = jnp.sum(xi_flat**2, axis=0)
res_het = jnp.linalg.norm(
    bb - ((Gc/l0 + 2*L_crack)*d_het - Gc*l0*_ifft(-xi_sq*_fft(d_het)))
) / jnp.linalg.norm(bb)
print(f"  relative residual of d_het against scalar op = {float(res_het):.2e}  (expect < 0.1)")

# (b) max(d) comparison
max_d_diff = abs(float(jnp.max(d_scalar)) - float(jnp.max(d_het))) / float(jnp.max(d_scalar))
print(f"  max(d_scalar) = {float(jnp.max(d_scalar)):.5f}")
print(f"  max(d_het)    = {float(jnp.max(d_het)):.5f}")
print(f"  relative max(d) diff = {max_d_diff:.2e}  (expect < 5 %)")

# (c) low-wavenumber profile along crack tip (x=40, all y)
d_s_prof = np.asarray(d_scalar).reshape(n)[40, :, 0]
d_h_prof = np.asarray(d_het).reshape(n)[40, :, 0]
prof_err = np.max(np.abs(d_s_prof - d_h_prof)) / (np.max(d_s_prof) + 1e-30)
print(f"  profile max rel-diff = {prof_err:.2e}  (expect < 5 %)")

pass_a = bool(cv_h) and float(res_het) < 0.1 and max_d_diff < 0.05 and prof_err < 0.05
print("  " + ("PASS" if pass_a else "FAIL"))

print()
print("─" * 60)
print("Check 4b: high-Gc barrier arrests crack — damage dips at interface")
print("─" * 60)

# Gc_field: 10× higher in a band at x = [60, 80] — acts as crack arrestor
Gc_field_np           = np.full(n, Gc)
Gc_field_np[60:80, :, :] = 10.0 * Gc          # high-toughness barrier
Gc_field              = jnp.array(Gc_field_np.ravel())

# Same crack-line driving force as before, extending into the barrier region
ms_L2 = np.zeros(n)
ms_L2[20:90, n[1] // 2, :] = L_crack_val      # long crack driving force
L_long  = jnp.array(ms_L2.ravel())
d_prev  = jnp.zeros((Nv,))

d_arr, conv_h = solve_helmholtz_cg_het(L_long, xi_flat, n, l0, Gc_field, d_prev,
                                        toler_cg=1e-8, maxiter=500)
d_np = np.asarray(d_arr).reshape(n)

d_before  = float(d_np[55, n[1] // 2, 0])    # just before barrier
d_inside  = float(d_np[70, n[1] // 2, 0])    # inside barrier
d_after   = float(d_np[85, n[1] // 2, 0])    # after barrier (should be low)
arrest_ok = d_inside < d_before              # damage dips in high-Gc zone

print(f"  converged      = {bool(conv_h)}")
print(f"  d before barrier (x=55) = {d_before:.4f}")
print(f"  d inside barrier (x=70) = {d_inside:.4f}  ← should be lower")
print(f"  d after barrier  (x=85) = {d_after:.4f}")
print("  Crack-arrest check: " + ("PASS" if arrest_ok else "FAIL"))

with IncrementalWriter("output/test_helmholtz_het", grid_shape=n, grid_spacing=dx) as w:
    w.write_increment(0, {
        "Gc_field":      Gc_field_np.astype(np.float64),
        "driving_force": ms_L2.astype(np.float64),
        "damage":        np.zeros(n, dtype=np.float64),
    }, time=0.0)
    w.write_increment(1, {
        "Gc_field":      Gc_field_np.astype(np.float64),
        "driving_force": ms_L2.astype(np.float64),
        "damage":        d_np.astype(np.float64),
    }, time=1.0)

print()
print("In ParaView: open output/test_helmholtz_het.xdmf — the damage field")
print("should dip where Gc is 10×, even though the driving force is uniform.")
