# Findings: phase-field fracture with a Drucker-Prager matrix

Running log of what has been established about the DP + AT2 phase-field
fracture solver (`problems.fracture`, `problems.mechanics`), with the evidence
behind each point. Newest first. Raw logs and configs live under `sim/sim/`
(a symlink, not in git); each entry names its study folder.

Settings that are pinned rather than free (eta, `maxiter_st`, `d_cap`, the
shear-free BC) are in `CLAUDE.md`, not repeated here.

---

## 2026-11-13 — BD-Z 60^3 with the three solver settings (`sim/sim/15_code_review/`)

Readout of the two full runs F25 left pending: `bi_z_fix2` (`toler_lin: 1e-3`,
`maxiter_cg: 4000`, `newton_line_search: 4`) and `bi_z_fix3` (the same +
`toler_st_abs: 3e-3`). Both reached 5.00 %. Tables: `CODE_REVIEW.md`, T3.

### F27. The three settings remove every mechanics failure in a full run

| | `bi_z_fix` (F24) | `bi_z_fix2` | `bi_z_fix3` |
|---|---|---|---|
| sweeps with a failed Newton | 97 of 761 | **0** of 1,727 | **0** of 670 |
| CG corrections with relres >= 1 | 1,532 of 10,785 | **0** of 4,994 | **0** of 2,301 |
| Newton its per sweep, mean / max | 15.0 / 30 | 3.9 / 11 | 4.4 / 10 |
| time per sweep for t > 0.53, GPU alone | 9.8 s | 9.8 s | (GPU shared) |

Rejected attempts included. Up to the peak the response is unchanged: `fix2`
matches `bi_z_fix` in every printed digit through t = 0.42 (25.24 MPa at
2.10 %). A sweep costs what it did, with the mechanics converged in it. The
settings are pinned in `CLAUDE.md` for the Newton path.

### F28. What a run now pays for is rejected snap attempts

Every rejection in the two runs is the staggered loop at `maxiter_st: 200` with
Newton converged in every sweep (I14):

| | `bi_z_fix2` | `bi_z_fix3` |
|---|---|---|
| rejected attempts | 5 | 1 |
| sweeps in them / in accepted increments | 1,000 / 727 | 200 / 470 |
| wall time in them | 6 h 49 of 9 h 11 | 2 h 10 of 6 h 12 |

(The GPU was shared, so the wall times do not compare with `bi_z_fix`'s
2 h 32.) In `fix2` the rejected attempts had dt 0.04, 0.02, 0.015, 0.0169 and
0.0190 and came no closer than err_abs 1.1e-3 to 7.3e-3; the snap increments it
accepted had dt 0.0075–0.0142 (10–76 sweeps), and one of dt 0.0214 took 152.
Two of the five rejections are the stepper growing dt by 1.5 right after an
accepted snap increment, two more are its descent from `dt_max` (0.04, then
0.02). A crack running at fixed strain is re-run from scratch after each one —
the cutback regime `CLAUDE.md` describes for a low `maxiter_st`, reached here
at 200.

### F29. The snap interval of the curve depends on the step sequence

sigma_zz in MPa (strain in brackets where the runs have no common point):

| eps_zz | `bi_z_fix` | `bi_z_fix2` | `bi_z_fix3` |
|---|---|---|---|
| ~2.30 % | 17.45 (2.316) | 22.32 (2.299) | 16.54 (2.300) |
| ~2.50 % | 16.29 (2.442) | 9.92 (2.525) | 8.04 (2.500) |
| ~2.9 % | 7.29 (2.832) | 7.73 (2.885) | 7.01 (2.900) |
| 5.00 % | 4.72 | 4.77 | 4.76 |

Peak and 5 % agree (within 0.05 MPa, 0.2 % of the peak, at 5 %); inside the
snap the runs differ by up to 5.8 MPa at equal strain. `fix2` crossed it in
steps of dt 0.0075–0.021, `fix3` in 0.02–0.04, so the pair does not separate
the tolerance from the step size: **it does not decide `toler_st_abs`**. Between
2.1 and 2.7 % the curve is not a converged result in any of the three runs.
Which of the two effects it is (the tolerance, or eta/dt changing with every
cutback) is not established.

### Open (2026-11-13)

- **Cut the cost of rejected snap attempts before the tangled 152^3 run** (F28;
  by voxel count a sweep there costs at least 16x a 60^3 one). Test by replay from the
  `bi_z_fix2` checkpoints (`tools/replay.py`), not by full runs. First
  candidates: the stepper (a larger cut after an attempt that ran out of
  sweeps; no growth after an increment that needed many), and
  `toler_st_abs: 3e-3`.
- **`toler_st_abs` 1e-3 vs 3e-3**: both tolerances on the same fixed steps
  through the snap, restarted from the same checkpoint (F29).
- If that is not enough: the candidates of the 2026-11-12 list below
  (accelerator that can step back, increment-referenced bounds F16–F17, coupled
  solve F18 — its verdict was measured with the indefinite preconditioner).
- Then T4, tangled 152^3; decide immune vs breakable fibres first (F5).
- Unchanged from 2026-11-12: CG matvec at half the cost (`CODE_REVIEW.md` R11),
  which kink Newton cycles across (R10), toler_nr not dimensionless (F8), eta
  units (`CLAUDE.md`).

---

## 2026-11-12 — code review: the CG preconditioner was indefinite (`sim/sim/15_code_review/`)

Full list of code issues, evidence and status: `CODE_REVIEW.md` (R1–R12,
N1–N3). The model is unchanged in everything below: linear hardening
17 + 250 alpha, no plastic freeze, production card.

### F21. The CG preconditioner of every free-surface run was indefinite (bug, fixed)

The bordered mixed-BC operator is negative semidefinite on both blocks; its
preconditioner was `+K0^-1` on the displacement block and `(-Nv C0)^-1` on the
macroscopic-strain block. Measured: `x.Mx` = +24 / -8.6e-9 on the two blocks.
CG still converged, with `r.Mr` changing sign in 334–1,083 of its iterations.
Fixed in `solvers.elliptic.vector.mixed_bc.macro_block_preconditioner`
(`+Nv C0`); pure strain BC was never affected. Same Newton correction system,
24^3, CG iterations to a relative residual:

| state | before: to 1e-2 / 1e-7 / relres at 500 its | fixed |
|---|---|---|
| elastic, undamaged | 14 / 50 / – | 13 / 43 / – |
| t = 0.57, plastic, no capped voxel | 208 / 826 / 4.2e-5 | 86 / **275** / converged |
| t = 0.61, 64 capped voxels | 612 / > 3000 / **2.4e-2** | 242 / 1259 / **5.2e-4** |

Small on an elastic composite, large on a plastic tangent, which is why every
test passed with it.

### F22. With it fixed, the 40^3 pass criterion is met by the production model

| | 24^3 before / fixed | 40^3 before / fixed |
|---|---|---|
| CG corrections with relres >= 1 | 8 / **0** | 383 / **0** |
| Newton its after the first capped voxel, max | 10 / 6 | 27 / **10** (cap increment), 7 after |
| CG meeting 1e-7 before the cap | 33 of 147 / all | 28 of 104 / all |
| sweeps to 5 %, sigma_zz at 5 % | 176, 2.840 / 176, 2.840 | 348, 2.203 / 344, 2.579 |
| wall to 5 % | 1,618 s / 460 s | 4,517 s / 677 s |

("before" = the 2026-11-08 logs, GPU shared by ~5 jobs; "fixed" shared by 2–3.)
The response is the same: at 24^3 sigma_zz, sweeps and n_cap agree in every one
of the 26 increments to all printed digits. At 40^3 they agree through 4.25 %
(22 increments); in the last four the sweep counts differ (12, 6, 7, 5 against
11, 8, 9, 6) and sigma_zz separates, to 2.579 against 2.203 MPa at 5 % (2.3 % of
the peak). Which is closer to a tight-tolerance answer is not established; the
accepted state of this formulation depends on the iteration path at
toler_st_abs 1e-3 (F17).
The criterion of `14_root_cause/README.md` section 6 (no relres >= 1, Newton
<= 10 at 40^3) was missed narrowly by `linexp` and clearly by linear + freeze
(F15). It is met here without either.

### F23. Corrections to F2, F4, F11, F14, F15

- **The non-symmetric tangent does not slow CG** (F2 cause 2's successor in
  F11/I6). With the old preconditioner the symmetric part of the tangent needed
  half the iterations (363 vs 826); with the fixed one they are equal (280 vs
  275 before the cap, 1,234 vs 1,259 with 64 capped voxels).
- **"Divergent corrections" (relres >= 1) are CG stopped at its cap** far from
  converged, on a residual curve that is spiky, not a tangent that defeats CG.
  They vanish at 24^3/40^3 with the fix, and at 60^3 with more iterations (F25).
- **F14's attribution** (divergent part = plastic band, non-convergent part =
  d_cap contrast) does not survive: stopping plastic flow changes the CG count
  by ~10 % (1,115 vs 1,259), the damage field alone accounts for 956, and
  plasticity alone for 245.
- **F4** (maxiter_cg 3000, GMRES, SPD tangent, line search) and **F15's
  solver columns** were all measured with the indefinite preconditioner. The
  physics columns of F15 stand (linexp delays the cap).
- **The model decision of F15 is no longer needed for the solver.** Whether
  the effective-state runaway (F9) should be removed on physical grounds is a
  separate question; alpha still reaches 2.9 at 40^3 and 4.4 in BD-Z.

### F24. BD-Z at 60^3 runs to 5 %

`small_bi_z.yaml` (unchanged model and settings) on the fixed code: 5.00 % in
2 h 32 min, 29 increments, 761 sweeps, 3 rejected attempts; peak 25.24 MPa at
2.10 % as before, 22.1 MPa at 2.18 %, 7.8 at 2.63 %, 4.7 at 5 %. The old stall
increment t = 0.435 (193+ sweeps over 4.7 h, never accepted) took 49 sweeps and
14.5 min. Earlier: 2.10 % (baseline), 2.47 % (GMRES, 7.7 h), 2.56 % (guard,
7.7 h).

It is not clean: 97 sweeps with a failed Newton, CG at relres ~0.5 after the
snap, and three rejected attempts (47 of the 152 min): two for diverging
Newton, one of 200 sweeps in which Newton converged every time and the damage
did not (I14).

### F25. The two remaining mechanics failures at 60^3, and the settings that remove them

Replays of single BD-Z increments from checkpoints (`tools/replay.py`, 2–7 min
each; first 14 sweeps):

| increment | solver settings | sweeps with failed Newton | Newton its | time |
|---|---|---|---|---|
| t 0.42 -> 0.435 (old stall) | production: toler_lin 1e-7, maxiter_cg 500 | 4 | 196 | 267 s |
| | maxiter_cg 2000 | 0 | 77 | 265 s |
| | **toler_lin 1e-3, maxiter_cg 4000** | **0** | **69** | **148 s** |
| t 0.41 -> 0.45 (rejected) | production | 7, then rejected | 256 | 198 s |
| | maxiter_cg 2000 | 3 | 161 | 407 s |
| | newton_line_search 4 | 7 | 257 | 203 s |
| | **maxiter_cg 2000 + newton_line_search 4** | **0** | **105** | 237 s |

1. **The CG cap.** After the snap a solve to 1e-3 needs ~3,000 iterations; 500
   leave relres 0.3–2 and Newton takes 20–30 poor steps or fails. A looser
   tolerance with a high cap is cheaper where CG is easy and sufficient where
   it is hard; the damage path is unchanged (err_abs identical to 4 digits).
2. **Newton cycling.** In the rejected attempt the Newton residual settles
   into a period-2 cycle (4.5 <-> 3.9) with *accurate* linear solves (relres
   2e-3): a kink, not CG. One step halving breaks it. This is the pattern of
   `14_root_cause/README.md` 2.4. The line search needs good directions to be
   of use, which is why F3 found it useless.

Best-iterate CG (`linear_solver: cg_best`) and the line search alone do not
help.

3. **What is left is the damage loop, and it does not converge here.**
   Replayed in full with accurate mechanics (maxiter_cg 2000; Newton converged
   in all 200 sweeps), the old stall increment reaches err_abs 1.009e-3 at
   sweep 59, then a second growth burst starts (2.5e-3 at sweep 100, 3.6e-2 at
   140–150) and it ends unconverged at sweep 200 with 8.4e-3, 679 capped voxels
   and sigma_zz 20.16 MPa. The production-settings run accepted it at sweep 49
   on a dip to 9.4e-4, with 232 capped voxels and 22.06 MPa: not a converged
   state, the crack was still running at fixed strain (F8's caveat). At 3e-3
   the accurate replay would accept at sweep 56. A robust Newton removes the
   accidental early exits and the accidental acceptances alike.

Full BD-Z runs with these settings (`bi_z_fix2`, and `bi_z_fix3` with
toler_st_abs 3e-3 on top): F27–F29.

### F26. Speed: the mechanics solve is compiled once (F20 fixed)

The compiled Newton pieces are built once per run and take the damage field as
an argument (`build_newton_kernels`, `prepare_fracture`); the redundant local
updates around each Newton solve are gone. 24^3, same GPU load: 6.9 -> 2.6 s
per sweep before the cap from this alone, 1.1 s with F21. Results identical to
every printed digit (33 sweeps at the cap, sigma_zz 10.909, 176 sweeps to 5 %).
Plain plasticity (`solve_inelastic.py` example, CPU): 186 -> 136 s.

### Tried and ruled out (T0, corrected preconditioner)

| idea | result (CG iterations to 1e-2 / 1e-7, capped 24^3 state; reference 242 / 1259) |
|---|---|
| rotated / hexahedral gradient in the Newton solver | 679 / > 3000 (better before the cap: 215 vs 275 to 1e-7) |
| preconditioner scaled by the degradation, `g^-1/2 K0^-1 g^-1/2` | 1925 / > 3000 |

### Open (2026-11-12)

- Read out `bi_z_fix2` and `bi_z_fix3`: done 2026-11-13 (F27–F29). The three
  settings are pinned; the pair does not decide `toler_st_abs` (F29). The
  current list is "Open (2026-11-13)" above.
- Then T4: tangled 152^3, on the fixed code with those settings. Decide immune
  vs breakable fibres first (F5).
- **The staggered loop is now the problem to solve** (I14, F14): 33 sweeps at
  24^3 before and after, and at 60^3 a crack running at fixed strain that 200
  sweeps do not finish (F25 point 3). Candidates on the table: more sweeps or a
  looser tolerance (F8), a viscosity that actually limits the advance per
  increment (I5: eta/dt 1–3 MPa against H 1e2–1e3), dissipation-based step
  control (I10), the constrained/increment-referenced formulation (F16–F17),
  a coupled solve (F18, worth retrying now that the mechanics block solves).
- CG matvec at half the cost (CODE_REVIEW R11, 2.15x measured): the largest
  remaining speed factor that needs no new method.
- Which kink Newton cycles across (CODE_REVIEW R10): not diagnosed.
- toler_nr still not dimensionless (F8); eta units (CLAUDE.md).

---

## 2026-11-08 evening — case agents A–E (`sim/sim/14_root_cause/results_{A..E}.md`)

### F15. Linear-then-exponential hardening (`linexp`) stops the band runaway; linear-then-ideal makes it worse

Production law 17 + 250 alpha up to alpha_t = 7.5 %, then `tail: ideal`
(`linideal`) or `tail: exponential`, b = 30, slope-continuous (`linexp`).
Reproducer, `--level newton` (results_A):

| run | cap | sweeps to 5 % | Newton max after cap | CG relres >= 1 after cap | band alpha / W_p at end |
|---|---|---|---|---|---|
| 24^3 linear | 3.05 % | 176 | 10 | 8 | 2.2 / 398 MPa |
| 24^3 linideal | 2.45 % | stopped 2.65 % | 16 | 15 | 11.6 / 134 |
| 24^3 linexp (± frz) | 4.25 % | 106 | 8 | 0 | 0.22 / 19 |
| 40^3 linear | 1.85 % | 348 | 27 | 383 | 2.85 / 542 |
| 40^3 linear + frz | 1.85 % | 162 | 14 | 14 | 0.40 / 20 |
| 40^3 linexp | 2.25 % | 141 | 11 | 1 (1.1) | 0.24 / 49 |
| 40^3 linexp + frz | 2.25 % | 139 | 10 | 2 (1.1) | 0.19 / 13 |

- `linideal` fails before the first capped voxel: a plastic band forms at
  1.65 % (alpha 0.34 -> 5.5 at d_max 0.85), Newton needs 13–16 iterations.
  Ideal plasticity at the apex is the worst case for the tangent (as Voce, F14).
- `linexp` misses the 40^3 pass criterion (no relres >= 1, Newton <= 10) only
  narrowly; what is left tracks n_cap (Newton 9 -> 11 past ~1,370 capped
  voxels), i.e. the d_cap stiffness contrast (I6), not the plastic band.
  The freeze on top adds little.
- **It changes the physics after initiation**: pre-peak response unchanged
  (peak +0.1–0.4 %), but the cap comes later (24^3: 3.05 -> 4.25 %; 40^3:
  1.85 -> 2.25 %) and the softening branch sits 1–6 MPa higher, because
  W_p stays at 12–49 MPa instead of 300–540 and beta_p W_p drives less damage.
  The linear law + freeze leaves the response unchanged up to the cap.

### F16. The cap-jump increment has two solutions; clipping the bounds picks the cracked one

From the stored 24^3 state t = 0.57, one increment to 0.61 (results_B, results_C):

- Production (clip, per-sweep floor and H): 33 sweeps, 56 capped voxels,
  sigma_zz 10.91 MPa. At a tight tolerance (1e-6) 58 sweeps, 64 capped, 10.68.
- Bounds solved (`bsolve`, `inc`): **no** capped voxel, d_max 0.70,
  sigma_zz 14.73, in 6–9 sweeps at 1e-3.
- The switch is **clip vs solve**: even plain Picard with clip runs away to 64
  capped voxels; with the bounds solved, both iteration paths stay at
  d_max 0.70. eta/dt (same), the H reference (d < d_thres, so H = D) and the
  floor (bsolve and inc agree to 1.2e-2) are not the cause.
- But the increment-referenced equations **also have a cracked solution**:
  started from production's answer, `inc` converges to 56 capped voxels,
  sigma_zz 10.42. Two exact fixed points at the same load; the start picks one.
- Production's own answer is not an exact solution of those equations (KKT
  residual 0.226, 2,092 voxels > 1e-2; its capped voxels hold H > D up to
  3.6x, results_C), and its 1e-3 answer is 6.1e-2 in d from its tight answer.

### F17. Increment-referenced irreversibility is path-independent but not cheaper

24^3 reproducer and the t = 0.57 increment (results_B):

| | sweeps at the cap | sweeps to 5 % | CG relres >= 1 after cap | NGMRES vs Picard, max\|dd\| (tight tol.) |
|---|---|---|---|---|
| production | 33 | 176 | 4 | 2.3e-2 (fails) |
| bsolve | 27 | 193 | 51 | 5.0e-4 |
| inc, NGMRES | 50 | 350 | 69 | **2.7e-6 (passes)** |
| inc, Picard | 43 | 292 | 50 | |

- `inc` meets the path-independence criterion; production does not. The
  production tolerance 1e-3 hides the difference (2.4e-3 between inc paths).
- Neither variant saves sweeps (I14 stands). `inc` with the step-back NGMRES
  is round-off sensitive at the cap: one run was rejected and stopped, the
  identical rerun accepted after 50 sweeps. `inc` with plain Picard had no
  rejection.
- Solving the bounds lowers the reproducer's pre-cap sigma_zz by 8–20 %. This
  is an artefact of its one-voxel seeded flaw, which clip keeps as a thin slab
  (sum d 211) and the solved bounds spread into a real damage profile
  (sum d 1071). Configs starting from d0 = 0 should not see it.
- No src/ defect found in the new options.

### F18. Coupled (u, d) Newton–Krylov: not converging, and nothing to gain on this increment

`tools/coupled_newton.py` (Jacobian verified to 1e-10) did not converge in 40
iterations in any variant (results_C). Two causes: clipping ~1,900 voxels at
the lower bound turns the Newton step into an ascent direction (22 of 40 line
searches at 1/128), fixed by a consistent active set (`--pdas`); and GMRES with
the block-Jacobi Fourier preconditioner stagnates on the mechanics block
(true u residual ~0.3–1.0 where production CG reaches 1e-7). With both fixes it
reaches the inc answer in ~6 iterations, then stalls at r_u 1.7e-2, at ~3,600
GMRES iterations per Newton step. Open: one retest with a properly solved
mechanics block, on an increment where the inc formulation itself jumps.

### F19. Checkpoint/restart works (I12)

Restart from increment k reproduces k+1 and every later increment bit-for-bit
(results_D; patch `14_root_cause/patches/D_I12_restart.patch`, test
`test/test_problems_fracture_restart.py`). The strain field must be in the
checkpoint: without it Newton diverged at the restart step. The step size is
part of the state (stored as a `dt` attribute).

### F20. Every staggered sweep recompiles the mechanics solve

The first Newton iteration of each sweep takes 3.5 s (linear) / 22 s (linexp)
at 24^3, later ones 0.3–0.6 s: 44 % / 83 % of sweep time (results_A). Cause:
`fracture.py` builds a new `_nl` closure per sweep that captures the current
d array, and `solve_displacement_based_nonlinear` defines its `@jit` functions
inside the call, so each sweep traces and compiles afresh with d baked in as a
constant. Hoisting the jitted functions and passing d as an argument should
make 24^3 runs ~2x (linear) to ~5x (linexp) faster. Not fixed yet.

### Regression (results_E)

All 35 test files pass on the working tree (CPU, distributed ones on 4 host
devices); the one failure on HEAD (drucker_prager check 12) is fixed (F13).
After the restart patch: `test_problems_fracture_restart.py` (new, 4/4) and
`test_problems_incremental.py` pass in the main tree.

---

## 2026-11-08 — root cause: the effective state of capped voxels runs away (`sim/sim/14_root_cause/`)

Long form, issue list (I1–I13), test ladder and tools: `14_root_cause/README.md`.
Everything below is measured on the written 13_small_rve fields or with the
24^3 reproducer `14_root_cause/tools/repro_fibre.py` (6 min to the cap).

### F9. At the first capped voxel the effective DP state runs away — every case

The transmitted stress has to pass through g(d_cap) = 6.3e-4, so the effective
stress needed is ~1e3x larger. The effective DP material supplies it by
hardening without limit: linear sigma_y = 17 + 250 alpha, sitting at the apex
under hydrostatic tension. In the increment where d first reaches the cap,
across matrix voxels: alpha_max 0.2–0.5 -> 1.9–3.3, W_p max 5–28 -> 335–992 MPa,
psi+ max 3–11 -> 362–1065 MPa (uni_x, uni_y, bi_x, bi_z, tri_x alike). The band's
driving force is 100–250x Gc/l0 (median) and keeps growing: uni_x at 5 % has
alpha_max 7.7, W_p 3,700 MPa. One BD-Z band voxel has p_eff = 1,830 MPa.

### F10. 97–99 % of the wall time comes after the first capped voxel

Before it: <= 7 sweeps per increment in every run. After it: up to 37–108.
Post-cap sweeps track the largest single-voxel damage jump per increment
(Spearman 0.54), not total crack growth (0.03) or newly capped voxels (0.02):
the cost is local runaways at the band edge. The edge driving force scales like
1/g^2 of the voxel's own damage, so the local Picard gain is far above 1.

### F11. Correction to F2: the post-peak tangent is SPD and elliptic, but kappa ~ 1e8

BD-Z increment 19, all 26,944 damaged matrix voxels: **none** is in hydrostatic
compression, and none has an indefinite symmetric part, fails strong
ellipticity, or has det(acoustic) <= 0. F2's cause 2 is therefore out, but its
cause 1 (the d_cap stiffness contrast, ~1e5) stands: it alone keeps CG from
converging in 500 iterations (F14). The plastic band makes things worse: at the
apex the deviatoric block collapses (smallest eigenvalue 2.7e-3 g mu ~ 9e-4 MPa),
and the non-associated flow makes the tangent 13–17 % asymmetric. That is what
turns non-convergence into divergent corrections. Every F4 remedy acted
downstream of these. BD-Z's "snap-through" (F3/F6) is not a severance either:
sigma_zz falls gradually 25.2 -> 13.9 MPa over 2.15 -> 2.6 %, with capped voxels
going 149 -> 1,557.

### F12. Damage bounds are imposed by clipping, not solved for

The floor (`scalar.py`) and d_cap (`fracture.py`) clip an unconstrained Helmholtz
solution. Once a bound is active (18k–48k voxels per post-peak BD-Z solve), that
is not the minimiser of the damage subproblem: off by up to 0.12 in d, with 650
voxels off by > 1e-2, and it systematically under-predicts damage. A
primal-dual active-set solve matches L-BFGS-B to 1.5e-6 in 4–11 reduced CG
solves. In the reproducer it does **not** reduce the sweep count: this is a
correctness fix, not a convergence fix.

### F13. Bug I7 (tabulated law + a_tip): two mechanisms, both fixed (2026-11-08 evening)

Both in the tip-rounded return's fast path for piecewise-affine laws
(`plasticity_drucker_prager.py`):

1. **Segment-pick tie.** With alpha_prev on the plateau and the trial state on
   the surface (f_trial ~ 1e-14: re-returning a converged state), the
   breakpoint test compared two equal numbers. Round-off picked the segment
   *below* alpha_prev, whose extrapolation read the state as elastic, and the
   q-solve returned q > q_trial: stress scaled up (up to 1e16 relative), alpha
   unchanged. ~0.5 % of random states (1164/200k on CPU, 978/200k on GPU) for
   linear-to-7.5 %-then-ideal. Fix: breakpoints at or below alpha_prev always
   count.
2. **Bisecting away from a converged root.** The fixed-count (8) safeguarded
   Newton rejected its own converged step: under jit, XLA computes the residual
   in separate fusions with different rounding, so the bracket was not updated
   and the step landed a few ulps outside it. The loop then bisected a wide
   bracket and could not get back. Eager mode never does this. It was check 12's
   failure (one finite-difference state at f = 0.04; the tangent itself was
   right). It also hits the **linear** law with a_tip on the test card
   (f = 0.026). On the production card: 0 of 200k random states and 0 of ~180k
   evaluations of the stored 24^3 state, before and after, so production runs
   were not affected in practice. Fix: every safeguarded Newton loop (affine
   q-solve, general dlam-solve, generic `solve_return`) returns its best iterate
   (smallest |residual|), not its last.

`test_materialmodels_drucker_prager.py` check 16 (re-return idempotency for
linear / table + ideal tail / table + exponential tail, sharp and rounded tip,
jit + vmap) fails on the old code and passes now; check 12 passes.

### F14. Two separate problems: mechanics breaks (plasticity), jumps are slow (AT2 + Picard)

24^3 reproducer (`14_root_cause/README.md` section 5):

- **Mechanics, two parts.** At the cap increment CG reaches a geometric-mean
  relres of 4e-2 (pre-cap 2.5e-6). That **non-convergence** is the d_cap
  stiffness contrast (F2 cause 1, still correct): it is the same with plastic
  flow stopped near the cap (2.5e-2). The **divergent** corrections
  (relres >= 1: 4 of 196) come from the plastic band's tangent: 0 of 157 with
  plastic flow stopped. Divergence is what stops Newton at 60^3 (~40 % of
  corrections in BD-Z); at 24^3 Newton survives on inexact steps (7–10
  iterations). Next check: a 40^3 reproducer.
- **Sweep cost.** The cap increment costs 30–45 sweeps whatever is changed:
  base 33, plastic flow stopped near the cap 31, damage bounds solved
  (PDAS) 30, **4x smaller steps 45** (and 172 instead of 49 sweeps to get
  there). The band-edge Picard gain is 8H/(Gc/l0 + 2H) = 2.7–4, so smaller
  steps help only below the viscous time scale (~1e-4 of the load path).
- Stopping plastic flow near the cap removes the plastic part of the runaway
  (alpha 0.45 vs 1.89, W_p 22 vs 323 MPa) and cuts total sweeps to 5 % by 22 %
  (138 vs 176, 0 rejections). Taking W_p from the transmitted stress changes
  the physics instead: that cell does not crack by 5 %.

---

## 2026-11-07 — small-RVE convergence study (`sim/sim/13_small_rve/`)

**Why.** The 152^3 tangled RVE (DP + fracture, `sim/fracture_tangled_pp.yaml`)
stalled at t = 0.27 (~1.35% strain). Before more 152^3 runs: does the same
setup converge on the 60^3 cells (`12_mini_eta/ReVE_num_fibers=4_{uni,bi,tri}.vti`,
phi ~0.26), one morphology at a time?

**Setup.** `13_small_rve/make_configs.py`: the tangled config with only input,
load direction, BC (`free_surfaces: true`) and output changed, plus
**breakable fibres** (`phasefield_isotropic`, Gc 2.76712, sigma_c ~1.72 GPa).
PP DP matrix (beta_p 3, sigma_crit 22), eta 0.05 / dt_max 0.04 / 5% target
(0.2% strain per step), `maxiter_cg` 500, `maxiter_st` 200, `d_cap` 0.975,
NGMRES. uni fibres along x; bi fibres in the xy plane.

### F1. Results so far (default solver: CG, consistent tangent)

| run | load | strain | rejected | peak MPa @ strain | wall h* |
|---|---|---|---|---|---|
| uni_y | transverse | **5.00% (done)** | 0 | 17.10 @ 1.25% | 3.6 |
| uni_x | longitudinal | **5.00% (done)** | 0 | 74.96 @ 1.05% | 6.1 |
| bi_x | in-plane | **5.00% (done)** | 0 | 33.78 @ 1.85% | 7.5 |
| bi_z | out-of-plane | **stalled at 2.15%** | 2 | 25.24 @ 2.10% | – |
| tri_x | isotropic | **5.00% (done)** | 0 | 18.72 @ 1.85% | 6.9 |

\* 2-6 jobs shared the GPU throughout, with Newton-level logging; absolute
times are pessimistic. The September mini sweep (same physics: eta 0.025 at a
10% target = eta 0.05 at 5%) took 4.8 h to 5% on uni_y with 3 jobs sharing.

**UD and BD in-plane converge; BD out-of-plane stalls** (F3) at 60^3, ~10 min
per attempt instead of hours. It is NOT the same stall as tangled (F7).

### F2. Once d_cap is reached, the Newton system is not SPD and CG fails

Achieved CG relres per Newton iteration is now logged (`cg_res=` at
`--log-level newton`). Before d_cap, CG meets toler_lin 1e-7. Once d_max hits
d_cap it essentially never does, and 20-45% of the returned corrections have
relres >= 1 (worse than no correction). Two causes, both measured:

1. **Contrast.** Matrix at d_cap (E 1400 x ~6e-4 ~ 0.9 MPa) next to glass
   (70 GPa): ~1e5. With the reference-medium (Fourier) preconditioner CG needs
   ~1000-3000 iterations (toy, 24^3 linear elastic, fibre + crack band).
2. **Not SPD.** Per-voxel tangent of the PP card (Mandel 6x6, symmetric part):

   | state | a_g 0.05 (as run) | a_g = a_f 0.13 |
   |---|---|---|
   | intact or cracked, tension | 7% asymmetric, SPD | symmetric, SPD |
   | cracked (d 0.975), compression | 4% asym., min eig -5.5 | 11% asym., min eig -34 |

   Non-associated flow makes every plastic voxel asymmetric. A cracked voxel
   under compression is asymmetric AND indefinite even for associated flow:
   the stress-level Amor split degrades the deviatoric/tensile stress after the
   return, so the plastic vol-dev coupling is degraded on one side only.
   A run with a_g = a_f failed the same way (56/221 corrections relres >= 1).

On UD, Newton survives this and the runs converge. On BD-Z it does not (F3).

### F3. The stall mechanism (BD-Z; NOT the tangled one, see F7)

Crack snaps through at t ~ 0.43 (d_max 0.57 -> 0.975 within one increment).
dt 0.04 and 0.02: Newton near-misses, then diverges (r 4-9) -> divergence
guard -> cutback. At dt 0.015 (t = 0.435): Newton fails every sweep at
r = 1e-4..3e-3, err_abs plateaus at ~1.2e-2. **Neither guard fires**: r < 1
(divergence guard silent), err_abs > toler (damage-converged stall guard
silent). ~85 s/sweep x up to maxiter_st 200 -> ~4 h per attempt. (The tangled
run sat 17 h in a state that looked similar but is not: F7.) Over ~390 Newton
iterations there, CG returned
relres >= 1 in 151 and 1e-2..1 in 237: Newton has no usable direction.
`newton_line_search: 4` prevents the blow-up but stalls the same way
(252/377 relres >= 1; it only shrinks the bad steps).

### F4. What did NOT help (all tested; code reverted unless noted)

| change | result |
|---|---|
| `maxiter_cg` 3000 (immune fibres) | Newton converges at the critical increment, but **same sweep count** (42 vs 43) at 2-3x per sweep |
| Fourier-Jacobi preconditioner (D^-1/2 K0^-1 D^-1/2, exact operator diagonal) | **worse** than plain Fourier everywhere: with a spectral gradient the derivative kernel vanishes at the origin, so a voxel's diagonal carries its neighbours' stiffness, not its own. Real runs: CG failed 30/30 on the first, undamaged increment. Published results are for FE gradients. |
| per-voxel SPD tangent (sym. part, eigenvalues floored) for the linear solve | CG behaves, but modified Newton stalls at d_cap: 3 (uni_x) and 1 (uni_y) rejections |
| GMRES instead of CG, on UD | removes relres >= 1 corrections, but **same staggered sweeps** at every d_cap increment, ~1.8x cost per Newton iteration. Not needed on UD. (On BD-Z it may matter: F6.) |
| no-progress guard (F6) | cuts BD-Z attempts at ~30 sweeps, but puts the run into a cutback spiral and would have rejected ~40 increments that converged (12_mini_eta, bi_x) |
| smaller dt to avoid the BD-Z snap-through (F6) | the crack snaps through within one increment at every dt tried (0.04 down to 0.0056) |

Patches, configs and logs: `13_small_rve/superseded_solver_variants/`
(guard + GMRES: `noprogress_guard_and_gmres_2026-11-08.patch`).

### F5. Other findings

- **Damage-immune vs breakable fibres** (uni_x): immune fibres reach d_cap
  earlier (1.05% vs 1.45%) and need ~2x the sweeps per increment, but also had
  0 rejections up to 1.45%. Breakable fibres did not "fix" UD-X; UD-X never
  failed.
- **`sim/fracture_tangled_pp.yaml` has damage-IMMUNE fibres** (`isotropic_elastic`,
  `k_res: 1.0`), although commit 5461b8c says "breakable fibres". The stalled
  tangled run used immune fibres.
- **JAX `gmres` bug**: its restart loop compares the preconditioned residual
  ||M r|| with tol * ||b|| (unpreconditioned). On a 0.1 mm cell
  ||M b|| = 8e-4 < 1.5e-2, so it returned x0 = 0 without iterating. Worked
  around by left-preconditioning ourselves (`M(A x) = M b`, no `M=`).
- The slow part at d_cap on UD is the **damage fixed point** (err_abs plateau
  10-40 sweeps per increment while the crack grows at fixed strain), not the
  linear solve.

### F6. BD-Z gets past the snap-through, given enough sweeps; the guard hurts (2026-11-08)

Runs `small_bi_z_guard` (CG + no-progress guard) and `small_bi_z_gmres`
(GMRES + guard), launched 2026-11-07 23:10, ran overnight. **Both passed the
snap-through.** The guard is reverted; GMRES stays as an opt-in config key
(`linear_solver: gmres`, default cg). Guard patch:
`superseded_solver_variants/noprogress_guard_and_gmres_2026-11-08.patch`.

| run (2026-11-08 06:55) | strain | rejected | sweeps | Newton its | wall h |
|---|---|---|---|---|---|
| bi_z (CG, baseline, stopped 23:13) | 2.10% | 2 | 112 | 1645 | 1.1 |
| bi_z_guard (CG + guard) | 2.56% | 14 | 772 | 13721 | 7.7 |
| bi_z_gmres (GMRES + guard) | 2.47% | 6 | 385 | 8217 | 7.7 |

**The snap-through increment converges, slowly.** Every attempt past t ~ 0.42
snaps the crack through within the increment (d_max ~0.6 -> 0.975 in 10-12
sweeps) at every dt tried (0.04 down to 0.0056); cutbacks do not avoid it.
After the snap Newton misses for 10-30 sweeps, then recovers: CG+guard
accepted t = 0.4294 (dt 0.0056) after 32 sweeps, GMRES accepted t = 0.435
(dt 0.015) after 47. The baseline stall (F3) was stopped at sweep 36 and
may have converged too.

**GMRES vs CG on t = 0.435 (dt 0.015, same start state).** Identical damage
path (err_abs to 3 digits) through sweep ~28. From sweep ~23, GMRES gets
Newton to 1-8x toler_nr (r 1e-5..8e-5; converged at sweeps 29, 46, 47 ->
accepted); CG stays at 10-300x (r 1e-4..3e-3) in both CG runs, through
sweep 36 (baseline) / 30 (guard). On the earlier dt 0.04 / 0.02 attempts
GMRES stagnated at relres 0.6-0.98 and gained nothing; it never returned
relres >= 1 (CG: ~40%). Cost ~2x CG per sweep. **Clean A/B running**
(no guard): `small_bi_z_cg_ng` vs `small_bi_z_gmres_ng`, launched 06:57.

**The no-progress guard puts runs in a cutback spiral.** After t ~ 0.50 it
fired on nearly every increment of bi_z_guard (dt down to 0.0007; t 0.4998 ->
0.5117 in 2 h). Replayed (`13_small_rve/replay_guard.py`, 4 variants) over all
52 sweep logs in `sim/sim/`: as tested (10 failed-Newton sweeps without a 10%
err_abs drop) it would have rejected ~40 increments that were accepted after
30-150 sweeps (12_mini_eta incl. the pinned-eta bi/tri runs, uni_x t = 0.85,
bi_x t = 0.65). That is the low-`maxiter_st` regime `CLAUDE.md` warns about.
Its reference also only moved down, so it fired during the post-snap
recovery (reference = the pre-snap err_abs). Variants (re-anchor while Newton
converges; count only misses with r > 10 toler_nr; N = 5) all still reject
some converging increments. For a fast diagnostic, set a low `maxiter_st`
in the diagnostic config instead.

### F7. Long increments are slow fixed points, not stalls

Newton-failure residuals per attempt (`13_small_rve/newton_fail_stats.py`):

| attempt | outcome | failed sweeps | r of failures (median, max) | failures with r > 10 toler |
|---|---|---|---|---|
| tangled t = 0.27 | stopped after 52 sweeps (17 h) | 25/52 | 2.7e-5, 8.1e-5 | 0% |
| bi_x t = 0.65 | **accepted after 41 sweeps** | 22/41 | 5.3e-5, 3.0e-4 | 14% |
| uni_x t = 0.53 | **accepted after 32 sweeps** | 21/32 | 1.4e-4, 9.0e-2 | 52% |
| bi_z t = 0.435, CG | stopped after 36 sweeps (F3) | 27/36 | 9.2e-4, 2.8e-2 | 81% |
| bi_z t = 0.435, GMRES | **accepted after 47 sweeps** | | | |
| bi_z t = 0.45, 0.43 | diverged | 7-10/14-21 | 1e-3..1.5e-2, up to 9 | 88-100% |

The tangled attempt (Newton near-missing, converging in about every other
sweep, err_abs drifting 2.4e-2 -> 5e-2) looks like bi_x t = 0.65 (drifted
1.25e-2 -> 2.7e-2 over 20 sweeps, then converged at sweep 41). The tangled
run may simply have needed more sweeps; at ~20 min per sweep on 152^3 that is
the cost problem. BD-Z's far misses after the snap are worse (and CG may
not recover from them -- the A/B decides), but they too have ended in
accepted increments.

### F8. The damage tolerance, not Newton's, sets the length of long increments (2026-11-08)

`13_small_rve/replay_tolerances.py`: for every attempt of >= 20 sweeps in
7 logs (13_small_rve uni/bi, tangled, 12_mini_eta pinned-eta uni/bi/tri), the
sweep at which it would have been accepted under looser tolerances (only the
stop sweep is replayed; the path after an earlier stop would differ):

| variant | sweeps spent | attempts accepted |
|---|---|---|
| as run: toler_st_abs 1e-3, toler_nr 1e-5 | 4726 | 101 |
| toler_nr 1e-4 | 4692 (-1%) | 101 |
| toler_st_abs 3e-3 | 3331 (-30%) | 106 |
| toler_st_abs 1e-2 | 2018 (-57%) | 107 |

Many 20-108-sweep increments would end at sweep 2-5 with 1e-2, i.e. err_abs
(max over voxels of one sweep's damage change) sits at a few 1e-3 for most of
those increments. The stalls (BD-Z t = 0.435, tangled t = 0.27, err_abs
1.2e-2 / 2.4e-2) stay unaccepted at 1e-2. Caveat: in a slowly contracting
fixed point a small step does not mean a small error (error ~ step / (1 - rate)),
so this needs an accuracy check, not just a sweep count.

**Accuracy A/B on uni_y** (`13_small_rve/compare_curves.py`; reference =
`small_uni_y` at 1e-3; all 0 rejections; 2026-11-08 09:40):

| toler_st_abs | sweeps vs 1e-3, same strain range | stress error / peak | work int sigma d eps | peak |
|---|---|---|---|---|
| 3e-3 (to 5%, done) | 257 vs 464 (1.8x fewer) | <= 0.17% | +0.07% | +0.04% |
| 1e-2 (to 5%, done) | 139 vs 464 (3.3x fewer) | 0.1-0.9%, one step +3.1% | +0.59% | +0.3% |

1e-2 is systematically slightly STIFF (damage accepted before it has caught
up = a lag, the same sign as eta's), and its one +3.1% step is the steepest
drop (eps 2.45%): the crack had not finished advancing; one step later it
agrees to 0.07%. So 1e-2 adds an error the size of eta's whole budget
(+0.5-0.9% on peak and work at the pinned eta, CLAUDE.md); 3e-3 is within
0.17% at 1.8x fewer sweeps. bi_x at 3e-3 (`small_bi_x_st3em3`, stopped by
user at 4.25%, 14:50): within 0.03% of peak, work +0.01%, but only 323 vs
363 sweeps to 4.25% (1.1x). Accurate on both cells; the saving is cell-dependent.

**BD-Z is damage-bound too** (2026-11-08 12:20). `small_bi_z_cg_ng` (CG, no
guard, 1e-3) spent 193+ sweeps (4.7 h) on t = 0.435 and was stopped just
before maxiter_st: Newton converged in 26 sweeps, missed narrowly
(r < 1e-4) in 68, badly in 99 -- but in 16 sweeps Newton converged AND
err_abs < 3e-3 (first: sweep 57), while err_abs never fell below 1e-3 with
Newton converged. At 3e-3 the increment would have ended at sweep 57.
`small_bi_z_gmres_ng` on t = 0.43 (dt 0.02): Newton converged or near-missed
in 84/108 sweeps, err_abs 6e-3..4e-2 -- purely damage-bound. So the
stiffness contrast (F2) costs Newton reliability, not the sweep count.
Replacement run: `small_bi_z_cg_st3em3` (CG, 3e-3), launched 12:20: the
dt 0.02 attempt at t = 0.43 ran 118+ sweeps (2 h) with err_abs 1.2-6e-2,
4-20x above 3e-3; `small_bi_z_gmres_ng` spent 184 sweeps (7 h) on the same
attempt, err_abs ~4.5e-2. Both stopped by user 14:50. The looser tolerance
does not get BD-Z past the post-snap increments. Next idea (not run):
one-pass staggered (toler_st_abs 1.0 = one sweep per increment) with a
small dt_max, checked against the uni_y reference first.

Side note on toler_nr: r = ||div sigma|| / ||sigma|| is NOT dimensionless.
xi = 2 pi k / L in 1/mm, so r scales with 1/L and with resolution (on the
0.103 mm 60^3 cell xi runs 61..1830 /mm: r = 1e-5 is a traction imbalance of
~2e-7..5e-9), and the mixed-BC rows enter with a factor sqrt(Nv). The same
toler_nr therefore means different things on the 60^3 and 152^3 cells.
Loosening it gains ~nothing on long increments (table above), so this is a
cleanup, not a fix.

### Open

- **Decide the model fix for F9** (options and reproducer numbers:
  `14_root_cause/README.md`, sections 4–6), then I2/I3 (constrained,
  increment-referenced damage bounds). After that, the per-jump sweep cost (F14)
  calls for an accelerator that can step back, or a coupled (u, d)
  Newton–Krylov solve. No new 60^3/152^3 solver experiments until the 24^3
  reproducer passes.
- eta in load-fraction units: make it strain-increment based — **after** the
  tangled run works (user decision). See `CLAUDE.md`.
- toler_st_abs: 3e-3 is accurate on uni_y and bi_x (F8); decide whether to pin it.
- BD-Z: one-pass staggered test (F8) -- proposed, not run.
- toler_nr: make the Newton criterion dimensionless (F8), then re-pin it.
- Tangled 152^3: rerun with the winning solver and enough wall time for slow
  increments (F7); decide immune vs breakable fibres first.
