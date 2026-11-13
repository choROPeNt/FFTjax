"""
Standalone test for checkpoint/restart of a phase-field fracture run:
problems.fracture.solve_fracture_incremental's write_checkpoint (writer side)
and state_init / eps_init / t_start / first_step (restart side), and the
reader between them, utils.io.reader.read_fracture_checkpoint.

Four checks
-----------
1. Reader layout, on fields that are NOT symmetric in space, on a non-cubic
   grid: random d, H, eps, sigma, eps_p, alpha, W_p written the way
   solve_fracture_incremental writes them (field_to_grid -> to_voigt ->
   IncrementalWriter, which stores the axes z-y-x) must read back
   bit-for-bit in the solver layout. Not vacuous: one voxel is checked by
   hand-computed index (z-y-x on disk, Voigt slot 3 = 12, slot 5 = 23), and
   a reading that skips the axis reversal gets different numbers. A file
   still held open by its writer (another process: a run in progress) must
   be readable too.
2. A checkpointed run -- the production Drucker-Prager card (beta_p 3,
   sigma_crit, Gc/l0 = 4) with a pre-cracked seed, uniaxial stress through
   free surfaces, eta > 0, NGMRES, automatic stepping with growing steps:
   every increment carries history, eps_p, time and dt, and increment k read
   back from disk is the run's own in-memory state of increment k,
   bit-for-bit. Increment k must be plastic, softening, hold voxels above
   d_thres and an H that differs from psi+ (history the strain cannot give
   back), or check 3 would not test the state.
3. Restart from increment k, from the checkpoint on disk: increment k+1 --
   and the one after it, reached with the grown step -- must match the
   original run to round-off in d, H, sigma, eps, eps_p, alpha, W_p and t,
   with the step numbers continuing at k+1; and the restart's own output
   file must agree with the original's. Control: the same restart with H
   rebuilt from the written psi+ -- all a restart could do before the
   history was written -- must be the same increment and clearly different.
4. Guard rails: t_start > 0 with stepping "fixed"/"single", write_checkpoint
   without a writer, t_start = 1, a missing increment, an increment written
   without the checkpoint and a grid mismatch all raise.

Usage
-----
    python test/test_problems_fracture_restart.py      (CPU: ~1 min)
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
sys.path.insert(0, "src")

import subprocess
import tempfile
import time
import warnings
from pathlib import Path

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)
import h5py
import jax.numpy as jnp
import numpy as np

from materialmodels.phasefield.drucker_prager import PhaseFieldDruckerPrager
from post.fields import field_to_grid, to_voigt
from problems.fracture import solve_fracture_incremental
from problems.incremental import solve_automatic
from problems.utils.loadcases import free_surface_control
from utils.io.reader import read_fracture_checkpoint, read_increment
from utils.io.xdmf_writer import IncrementalWriter

T_START = time.perf_counter()
_TMP = tempfile.TemporaryDirectory(prefix="fftjax_restart_")
TMP = Path(_TMP.name)


def _sym(a):
    return 0.5 * (a + np.swapaxes(a, 0, 1))


def _raises(fn, exc, needle, what):
    try:
        fn()
    except exc as e:
        assert needle in str(e), f"{what}: wrong message: {e}"
        return
    raise AssertionError(f"{what}: expected {exc.__name__} mentioning {needle!r}")


# ── 1. reader layout on a non-symmetric field, non-cubic grid ──────────────
n1 = (5, 6, 7)
Nv1 = int(np.prod(n1))
rng = np.random.default_rng(0)
truth = {
    "d": rng.random(Nv1), "H": 10.0 * rng.random(Nv1),
    "eps": _sym(rng.normal(size=(3, 3, Nv1))), "sigma": _sym(rng.normal(size=(3, 3, Nv1))),
    "eps_p": _sym(rng.normal(size=(3, 3, Nv1))),
    "alpha": rng.random(Nv1), "W_p": rng.random(Nv1),
}
# Exactly the conversions of solve_fracture_incremental's _on_increment.
fields1 = {
    "phase":        (rng.random(Nv1) > 0.5).reshape(n1).astype(np.float32),
    "strain":       to_voigt(field_to_grid(truth["eps"], n1)).astype(np.float64),
    "stress":       to_voigt(field_to_grid(truth["sigma"], n1)).astype(np.float64),
    "damage":       truth["d"].reshape(n1),
    "history":      truth["H"].reshape(n1),
    "strain_p":     truth["alpha"].reshape(n1),
    "plastic_work": truth["W_p"].reshape(n1),
    "eps_p":        to_voigt(field_to_grid(truth["eps_p"], n1)).astype(np.float64),
}
with IncrementalWriter(str(TMP / "layout"), grid_shape=n1, grid_length=(0.5, 0.6, 0.7)) as w:
    w.write_increment(3, fields1, time=0.4375, attrs={"dt": 0.125})
    # an increment written WITHOUT the checkpoint, for check 4
    w.write_increment(4, {k: v for k, v in fields1.items() if k != "history"}, time=0.5)

ck1 = read_fracture_checkpoint(TMP / "layout.h5", 3, n1)
for key, ref in truth.items():
    assert ck1[key].shape == ref.shape and np.array_equal(ck1[key], ref), \
        f"{key} does not round-trip through the checkpoint"
for got, ref in zip(ck1["state"], (truth["eps_p"], truth["alpha"], truth["W_p"])):
    assert np.array_equal(got, ref)
assert (ck1["step"], ck1["t"], ck1["dt"], ck1["n"]) == (3, 0.4375, 0.125, n1)
assert np.array_equal(ck1["phase"], fields1["phase"].ravel())
fields1_back, attrs1 = read_increment(TMP / "layout.h5", 3)
assert all(np.array_equal(fields1_back[k], v) for k, v in fields1.items())
assert attrs1 == {"time": 0.4375, "dt": 0.125}

# By hand, independent of both the writer's and the reader's code: on disk
# the axes are (z, y, x) and the Voigt slots [11, 22, 33, 12, 13, 23].
ix, iy, iz = 4, 1, 6
v = (ix * n1[1] + iy) * n1[2] + iz
with h5py.File(TMP / "layout.h5", "r") as f:
    raw_d = f["increment_000003/damage"][...]
    raw_eps = f["increment_000003/strain"][...]
assert raw_d.shape == n1[::-1] and raw_eps.shape == (*n1[::-1], 6)
assert raw_d[iz, iy, ix] == truth["d"][v] == ck1["d"][v]
assert raw_eps[iz, iy, ix, 3] == truth["eps"][0, 1, v] == ck1["eps"][1, 0, v]
assert raw_eps[iz, iy, ix, 5] == truth["eps"][1, 2, v] == ck1["eps"][2, 1, v]
assert not np.array_equal(raw_d.ravel(), truth["d"]), \
    "skipping the axis reversal gives the same field -- the check is vacuous"

# A run still in progress keeps its file open, and HDF5 locks it: the reader
# must fall back to an unlocked read (with a warning) rather than fail.
holder = subprocess.Popen(
    [sys.executable, "-c",
     "import h5py, sys, time; f = h5py.File(sys.argv[1], 'a'); "
     "print('open', flush=True); time.sleep(60)", str(TMP / "layout.h5")],
    stdout=subprocess.PIPE, text=True)
try:
    assert holder.stdout is not None and holder.stdout.readline().strip() == "open"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ck1_locked = read_fracture_checkpoint(TMP / "layout.h5", 3, n1)
    assert np.array_equal(ck1_locked["H"], truth["H"])
    lock_note = ("locked by another process: read unlocked, with a warning"
                 if any("locked" in str(c.message) for c in caught)
                 else "no lock taken on this filesystem")
finally:
    holder.kill()
    holder.wait()
print(f"[1] n={n1}: d, H, eps, sigma, eps_p, alpha, W_p, t, dt round-trip bit-for-bit "
      f"(z-y-x on disk, Voigt 12 in slot 3, 23 in slot 5); open file: {lock_note}")
print("[1] PASSED")


# ── 2. a checkpointed run ───────────────────────────────────────────────────
n = (8, 10, 12)                     # non-cubic: an axis mix-up cannot pass
h = 0.1
L = tuple(ni * h for ni in n)
Nv = int(np.prod(n))
l0 = 4.0 * h
# The production card of sim/sim/14_root_cause (Gc/l0 = 4 N/mm^2).
mat = PhaseFieldDruckerPrager(E=1400.0, nu=0.39, sigma_y0=17.0, H=250.0, a_f=0.13, a_g=0.05,
                              a_tip=5.61, Gc=4.0 * l0, sigma_crit=22.0, beta_p=3.0, k_res=1e-6)
phase = jnp.zeros(Nv, dtype=int)
idx = np.indices(n)
_c = np.array([3.0, 4.0, 5.0])      # off-centre
seed = np.linalg.norm(np.stack([idx[i] + 0.5 - _c[i] for i in range(3)]).reshape(3, -1),
                      axis=0) <= 1.6
# A pre-crack above d_thres with a large history: hybrid irreversibility keeps
# H there at max(H, D), so H carries memory no strain field can give back.
d0 = jnp.asarray(seed * 0.96)
H0 = jnp.asarray(seed * 20.0)
eps_bar = jnp.zeros((3, 3)).at[0, 0].set(0.03)
D_THRES, DT_MAX, F_INC = 0.95, 0.35, 1.5
KW = dict(stepping="automatic", formulation="displacement",
          control=free_surface_control(0, 0), stress_goal=jnp.zeros((3, 3)),
          toler_lin=1e-6, maxiter_cg=600, toler_helm=1e-6, maxiter_helm=300,
          eta=0.05, d_thres=D_THRES, toler_st_abs=1e-3, toler_st_rel=1e-4, maxiter_st=60,
          toler_nr=1e-5, maxiter_nr=40, accelerator="ngmres", accel_depth=5, d_cap=0.975,
          dt_min=1e-3, dt_max=DT_MAX, factor_inc=F_INC, factor_dec=0.5,
          max_cutbacks=4, max_steps=50)

t0 = time.perf_counter()
with IncrementalWriter(str(TMP / "run_a"), grid_shape=n, grid_length=L) as w:
    res_a = solve_fracture_incremental(n, L, phase, [mat], eps_bar, l0, None, d0, H0,
                                       dt_init=0.15, writer=w, write_checkpoint=True, **KW)
wall_a = time.perf_counter() - t0
assert [r.step for r in res_a] == list(range(1, len(res_a) + 1)) and len(res_a) >= 4, \
    f"expected >= 4 increments, got steps {[r.step for r in res_a]}"
with h5py.File(TMP / "run_a.h5", "r") as f:
    for r in res_a:
        grp = f[f"increment_{r.step:06d}"]
        assert "history" in grp and "eps_p" in grp, f"increment {r.step}: no checkpoint fields"
        assert grp.attrs["time"] == r.t and grp.attrs["dt"] == r.dt, f"increment {r.step}: time/dt"

K = 2
sol_k = res_a[K - 1].solution
ck = read_fracture_checkpoint(TMP / "run_a.h5", K, n)
mem_k = {"d": sol_k.d, "H": sol_k.H, "eps": sol_k.eps, "sigma": sol_k.sigma,
         "eps_p": sol_k.eps_p, "alpha": sol_k.alpha, "W_p": sol_k.W_p}
for key, ref in mem_k.items():
    assert np.array_equal(ck[key], np.asarray(ref)), \
        f"checkpoint {key} != the run's in-memory state at increment {K}"
assert ck["t"] == res_a[K - 1].t and ck["dt"] == res_a[K - 1].dt
asym_eps = float(jnp.max(jnp.abs(sol_k.eps - jnp.swapaxes(sol_k.eps, 0, 1))))
asym_ep = float(jnp.max(jnp.abs(sol_k.eps_p - jnp.swapaxes(sol_k.eps_p, 0, 1))))
sig11 = [float(jnp.mean(r.solution.sigma[0, 0])) for r in res_a]
d_k = np.asarray(sol_k.d)
H_mem = float(jnp.max(jnp.abs(sol_k.H - sol_k.psi_pos)))
print(f"[2] {len(res_a)} increments in {wall_a:.0f}s, t = "
      f"{' '.join(f'{r.t:.4f}' for r in res_a)}, sigma_11 = {' '.join(f'{s:.2f}' for s in sig11)}")
print(f"[2] increment {K}: alpha_max={float(jnp.max(sol_k.alpha)):.3e}  "
      f"n(d>=d_thres)={int((d_k >= D_THRES).sum())}  max|H - psi+|={H_mem:.2e}  "
      f"max|eps - eps^T|={asym_eps:.1e}  max|eps_p - eps_p^T|={asym_ep:.1e}")
print(f"[2] checkpoint of increment {K} read back == in-memory state, bit-for-bit")
assert float(jnp.max(sol_k.alpha)) > 0.0, "increment k is not plastic -- check 3 would not test eps_p"
assert (d_k >= D_THRES).any(), "no voxel above d_thres -- H would carry no history"
assert H_mem > 1.0, "H == psi+ everywhere -- H is recomputable, check 3 would not test it"
assert sig11[K] < sig11[K - 1], "no softening in increment k+1 -- not a fracture increment"
assert asym_eps == 0.0 and asym_ep == 0.0, \
    "eps/eps_p not symmetric to the bit -- the Voigt checkpoint would not be exact"
print("[2] PASSED")


# ── 3. restart from increment K ─────────────────────────────────────────────
dt_restart = min(ck["dt"] * F_INC, DT_MAX)   # the step the original tried next
t0 = time.perf_counter()
with IncrementalWriter(str(TMP / "run_b"), grid_shape=n, grid_length=L) as w:
    res_b = solve_fracture_incremental(
        n, L, phase, [mat], eps_bar, l0, None,
        jnp.asarray(ck["d"]), jnp.asarray(ck["H"]),
        state_init=ck["state"], eps_init=ck["eps"], t_start=ck["t"], first_step=K + 1,
        dt_init=dt_restart, writer=w, write_checkpoint=True, **KW)
wall_b = time.perf_counter() - t0
assert [r.step for r in res_b] == [r.step for r in res_a[K:]], \
    f"restart steps {[r.step for r in res_b]} != original {[r.step for r in res_a[K:]]}"

FIELDS = ("d", "H", "sigma", "eps", "eps_p", "alpha", "W_p")
print(f"[3] restart from increment {K} (t={ck['t']:.4f}, first dt={dt_restart:.4f}) "
      f"in {wall_b:.0f}s; max |restart - original| (abs / relative to max|original|):")
worst = 0.0
for ra, rb in zip(res_a[K:], res_b):
    a, b = ra.solution, rb.solution
    assert rb.t == ra.t and rb.dt == ra.dt, f"increment {ra.step}: t/dt differ"
    assert b.iter_staggered == a.iter_staggered, f"increment {ra.step}: different sweep count"
    cols = []
    for key in FIELDS:
        x, y = np.asarray(getattr(a, key)), np.asarray(getattr(b, key))
        diff = float(np.max(np.abs(x - y)))
        rel = diff / max(float(np.max(np.abs(x))), 1e-300)
        worst = max(worst, rel)
        cols.append(f"{key} {diff:.1e}/{rel:.1e}")
    print(f"      increment {ra.step} (t={ra.t:.4f}): " + "  ".join(cols))
assert worst <= 1e-10, f"restart differs from the original beyond round-off: {worst:.2e}"

# Control: the best a restart could do from the output as written before
# checkpoints existed -- H rebuilt as the written driving force
# (strain_energy_pos, psi+), everything else exact. It must be the same
# increment (same t, converged) and visibly different, or the history field
# would not be needed and the comparison above would prove nothing about it.
ctrl = []
H_old = read_increment(TMP / "run_a.h5", K)[0]["strain_energy_pos"].ravel()
try:
    solve_fracture_incremental(
        n, L, phase, [mat], eps_bar, l0, None, jnp.asarray(ck["d"]), jnp.asarray(H_old),
        state_init=ck["state"], eps_init=ck["eps"], t_start=ck["t"], first_step=K + 1,
        dt_init=dt_restart, on_increment=lambda r, _w: ctrl.append(r),
        **dict(KW, max_steps=1))
except RuntimeError as e:          # max_steps=1: stop after increment K+1
    assert "max_steps" in str(e), e
a, c = res_a[K].solution, ctrl[0].solution
assert ctrl[0].t == res_a[K].t and bool(c.converged), "the control is not the same increment"
sens = {key: float(np.max(np.abs(np.asarray(getattr(a, key)) - np.asarray(getattr(c, key)))))
        / float(np.max(np.abs(np.asarray(getattr(a, key))))) for key in FIELDS}
print(f"      control, H rebuilt from psi+ (no history on disk): increment {K + 1} relative "
      "differences " + "  ".join(f"{k} {v:.1e}" for k, v in sens.items()))
assert sens["d"] > 1e-3, "restarting without the history changed nothing -- H is not tested"

# The restart's own output agrees with the original's, increment for increment.
for r in res_b:
    fa, aa = read_increment(TMP / "run_a.h5", r.step)
    fb, ab = read_increment(TMP / "run_b.h5", r.step)
    assert aa == ab, f"increment {r.step}: attributes {aa} != {ab}"
    assert fa.keys() == fb.keys()
    for key in fa:
        scale = max(float(np.max(np.abs(fa[key]))), 1e-300)
        assert float(np.max(np.abs(fa[key] - fb[key]))) <= 1e-10 * scale, \
            f"increment {r.step}: written {key} differs"
print(f"[3] steps continue at {K + 1}; worst relative difference {worst:.1e}; the restart's "
      f"output file matches the original's for increments {[r.step for r in res_b]}")
print("[3] PASSED")


# ── 4. guard rails ──────────────────────────────────────────────────────────
_args = (n, L, phase, [mat], eps_bar, l0, None, d0, H0)
_raises(lambda: solve_fracture_incremental(*_args, stepping="fixed", dt_step=0.5, t_start=0.5,
                                           formulation="displacement"),
        ValueError, "needs stepping='automatic'", "t_start > 0 with stepping='fixed'")
_raises(lambda: solve_fracture_incremental(*_args, stepping="single", first_step=3,
                                           formulation="displacement"),
        ValueError, "needs stepping='automatic'", "first_step != 1 with stepping='single'")
_raises(lambda: solve_fracture_incremental(*_args, stepping="automatic", write_checkpoint=True,
                                           formulation="displacement"),
        ValueError, "needs a writer", "write_checkpoint without a writer")
_raises(lambda: solve_automatic(lambda t: None, t_start=1.0),
        ValueError, "t_start must be in [0, 1)", "t_start = 1")
_raises(lambda: read_fracture_checkpoint(TMP / "layout.h5", 9, n1),
        KeyError, "available: [3, 4]", "a missing increment")
_raises(lambda: read_fracture_checkpoint(TMP / "layout.h5", 4, n1),
        KeyError, "write_checkpoint", "an increment written without the checkpoint")
_raises(lambda: read_fracture_checkpoint(TMP / "layout.h5", 3, n1[::-1]),
        ValueError, "not the expected", "a grid mismatch")
print("[4] restart without automatic stepping, a checkpoint without a writer, t_start = 1, "
      "and reading a missing / non-checkpoint increment or the wrong grid all raise")
print("[4] PASSED")

print(f"\ntest_problems_fracture_restart: all checks passed ({time.perf_counter() - T_START:.0f}s)")
