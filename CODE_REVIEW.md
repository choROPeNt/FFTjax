# Code review: DP + phase-field fracture solver path (2026-11-12)

Issues and performance killers found by reading the code of the DP + AT2 path
(`problems.fracture`, `problems.mechanics`, `solvers.elliptic.*`,
`solvers.krylov.cg`, `materialmodels.phasefield.drucker_prager`,
`materialmodels.inelastic.*`, `materialmodels.assembly`), after going through
`FINDINGS.md` F1–F20 and `sim/sim/14_root_cause/` (README, HANDOFF, results A–E).

Each entry says how it was established: **measured** (script and log named) or
**by inspection** (read off the code, not measured). Tools, configs and logs:
`sim/sim/15_code_review/{tools,configs,runs,output}/` (not in git). The
solver-relevant results are also in `FINDINGS.md` (F21–F26).

Status: **fixed** (committed on `pff_dp_damage` 2026-11-13, tested), **setting** (no
code change needed, an existing config key), **open**, **ruled out**.

---

## Summary

| # | kind | what | status |
|---|---|---|---|
| R1 | **bug** | Mixed-BC CG preconditioner was indefinite (sign of the macroscopic block) | **fixed** |
| R2 | performance | Every staggered sweep recompiled the mechanics solve (F20); per-increment setup rebuilt | **fixed** |
| R3 | performance | Redundant, un-jitted local updates around every Newton solve | **fixed** |
| R5 | robustness | Newton never tested the iterate its last correction produced | **fixed** |
| R9 | robustness | `toler_lin 1e-7` + `maxiter_cg 500`: too strict where CG is easy, too few where it is hard | **setting** `toler_lin: 1e-3`, `maxiter_cg: 4000`; confirmed in full runs (T3), pinned in `CLAUDE.md` |
| R10 | robustness | Full-step Newton cycles across a kink (period 2), with accurate linear solves | **setting** `newton_line_search: 4`; confirmed in full runs (T3), pinned in `CLAUDE.md` |
| R4 | robustness | CG returns its last iterate, not its best | opt-in added (`linear_solver: cg_best`); measured: no gain |
| R11 | performance | CG matvec: complex FFTs of full 3x3 fields, unknown transformed twice per iteration | open; 2.15x measured |
| R6 | clarity | `scheme` is ignored for `formulation: displacement`, but written to the output as `rotated` | open |
| R7 | performance | Plastic return + tangent evaluated on fibre voxels and discarded | open, minor |
| R8 | clarity | `build_willot_freq` is not Willot's rotated scheme | open, not on the DP path |
| R12 | minor | Damage CG preconditioner mismatched at the Nyquist planes (`helm=False` in the reproducer) | open, minor |
| N1 | ruled out | Rotated / hexahedral gradient for the Newton solver | measured, not a fix |
| N2 | ruled out | Degradation-scaled preconditioner `g^-1/2 K0^-1 g^-1/2` | measured, not a fix |
| N3 | ruled out | Non-symmetric tangent as the reason CG is slow | measured: it was R1 |

**What it adds up to** (details in the sections below):

- 40^3 reproducer, production model, production settings: the pass criterion of
  `14_root_cause/README.md` section 6 is met (0 CG corrections with relres >= 1,
  was 383; Newton <= 10 iterations, was 27), in 677 s instead of 4,517 s.
- BD-Z at 60^3, the case that stalled at 2.15–2.56 % in every earlier variant,
  runs to 5 % in 2 h 32 min with the fixes alone (R1–R3, R5), production settings.
  It still has 97 sweeps with a failed Newton and 3 rejected attempts.
- Replays of its two critical increments show what those are (R9, R10) and that
  three existing config keys remove them. Two full BD-Z runs with them
  (`bi_z_fix2`, `bi_z_fix3`, read out 2026-11-13) reach 5 % without a single
  failed Newton solve in 2,397 sweeps; see section T3.
- With the mechanics robust, what limits a run is the staggered damage loop
  (I14, F8): replayed with accurate mechanics, the old stall increment does not
  converge in 200 sweeps, and the state production accepted at sweep 49 was not
  a converged one. In `bi_z_fix2` the five rejected attempts, every one of them
  200 sweeps with Newton converged, are 58 % of the sweeps and 74 % of the wall
  time. Section T3.
- The model is unchanged throughout: linear hardening, no plastic freeze. The
  effective-state runaway (F9, I1) is still there physically; it no longer breaks
  the mechanics solve.

---

## R1. Mixed-BC CG preconditioner was indefinite — bug, measured, fixed

**Where.** `problems/mechanics.py` (Newton correction), 
`solvers/elliptic/vector/displacement_based.py` (local and sharded bodies).
Present since the displacement-based solver was added (81f3034).

**What.** The bordered operator of the mixed strain/stress system is
`A = -E^H C E` (E maps the displacement fluctuation and the macroscopic strain
unknowns to strain): negative semidefinite on **both** blocks. CG on it is CG
on an SPD system only if the preconditioner M is positive definite. The
displacement block of M is `+K0^-1`, but the macroscopic block was
`P_extra^-1` with `P_extra = -Nv C0`: negative. M was indefinite. The comment at
that line stated the intent correctly ("keeping M positive definite against a
negative definite A"); the sign contradicted it.

Only the 1–5 macroscopic unknowns are affected, so CG still converged; its
inner product `r.Mr` changed sign along the way. Pure strain BC has no such
block and is unaffected. Every production fracture run uses `free_surfaces`
(5 stress-controlled components).

**Fix.** `solvers.elliptic.vector.mixed_bc.macro_block_preconditioner`
(`+Nv C0`, positive definite), used by all three call sites.

**Evidence, T0** (`tools/t0_pextra_sign.py`, `runs/t0_pextra_sign.log`; 24^3,
production card, stored states of `14_root_cause/runs`):

- `x^T A x`: displacement block -2.2e15, macroscopic block -1.2e9.
  `x^T M x`: displacement block +24, macroscopic block **-8.6e-9** (before).
- CG on the same Newton correction system, iterations to a relative residual:

| state | P_extra | to 1e-2 | to 1e-4 | to 1e-7 | relres at 500 its | steps with r.Mr < 0 |
|---|---|---|---|---|---|---|
| elastic, undamaged | -Nv C0 (before) | 14 | 30 | 50 | – | 6 |
| | +Nv C0 | 13 | 26 | 43 | – | 0 |
| t = 0.57, plastic, no capped voxel | -Nv C0 (before) | 208 | 455 | 826 | 4.2e-5 | 334 |
| | +Nv C0 | 86 | 162 | **275** | converged | 0 |
| t = 0.61, 64 capped voxels | -Nv C0 (before) | 612 | 2649 | > 3000 | **2.4e-2** | 1083 |
| | +Nv C0 | 242 | 643 | 1259 | **5.2e-4** | 0 |

The effect is small on an elastic composite (which is why no test caught it)
and large on a plastic tangent. With `maxiter_cg: 500` the production runs hit
the cap already *before* the first capped voxel (826 > 500).

**Evidence, T1** — see section T1. **Tests.**
`test_problems_mechanics_nonlinear.py` check 8 (definiteness of the solver's own
operator/preconditioner pair and of the helper; fails with the old sign while
checks 1–7 pass with it).

**Consequence for earlier findings.** F2 cause 1 / I6 ("CG cannot converge
against the d_cap contrast in 500 iterations"), F11 (kappa, asymmetry) and the
whole of F4 were measured with this preconditioner. See N3.

---

## R2. Per-sweep recompile and per-increment setup — performance, measured, fixed

**Where.** `problems/fracture.py` built a new `_nl` closure per sweep that
captured `d`; `problems/mechanics.py` defined its `@jit` functions inside
`solve_displacement_based_nonlinear`; `solve_fracture` rebuilt the material
closures, frequency grid, Gc field and (Lippmann-Schwinger path) the reference
Green operator for every increment attempt.

**What.** `jax.jit` caches per function object, so every sweep traced and
compiled the mechanics solve afresh with the damage field baked in as a
constant (F20: 3.5 s linear / 22 s linexp per sweep at 24^3).

A second consequence, met on 2026-11-12: compiling per sweep also creates new
cuFFT plans per sweep, so such a run can die with "Failed to make cuFFT batched
plan" whenever another process starts on the GPU, at any point of its life
(the pitfall of `14_root_cause/HANDOFF.md` section 5). Compiled once, a run is
exposed only during its first sweep.

**Fix.** `problems.mechanics.build_newton_kernels` / `NewtonKernels`: the
residual and the correction are compiled once per (constitutive callable, grid,
control mask, linear solver) and take everything that varies as arguments;
`solve_displacement_based_nonlinear(..., local_args=(d,), kernels=...)`.
`problems.fracture.prepare_fracture` / `FractureSetup` builds the per-run
objects once; `solve_fracture(..., setup=...)`; `solve_fracture_incremental`
uses it. Without `kernels` / `setup` both functions behave as before.
`scripts/simulation/solve_inelastic.py` reuses kernels per load case.

**Evidence.** Section T1 (24^3: 6.9 -> 2.6 s per pre-cap sweep from this alone).
Plain-plasticity example (`inelastic_drucker_prager_example.yaml`, 94x98x1, 41
load points, CPU): 186 s -> 136 s, identical Newton counts, stress identical to
7e-15. **Tests.** check 9 of `test_problems_mechanics_nonlinear.py` (no retrace
across solves; mismatched kernels rejected).

---

## R3. Redundant, un-jitted local updates around every Newton solve — performance, fixed

Per Newton solve the old code evaluated the return mapping at the final strain
up to four times: in the loop, eagerly with tangent for `sigma_final`, again
for `return_residual`, and eagerly once more in `_staggered_loop` after a
failed Newton; the driving force was evaluated eagerly every sweep.

**Fix.** The driver returns the stress, state and residual of the loop's own
last evaluation (`state_on_failure="trial"` hands the staggered loop the state
at the returned strain); the driving force is jitted once per run.
**Tests.** checks 4 and 11.

---

## R5. Newton never tested the iterate its last correction produced — fixed

The loop applied a correction in iteration `maxiter_nr` and returned
`converged = False` without testing the strain it returned. A solve whose last
correction converged was reported as failed and its sweep repeated. Now the
returned strain is evaluated (it had to be, for its stress) and tested.
**Tests.** check 10.

---

## R9. `toler_lin 1e-7` + `maxiter_cg 500` is the wrong way round — measured (T2), a setting

A fixed tight tolerance with a low cap makes CG run to 1e-7 where that is
cheap and unnecessary (24^3 before the cap: 275 iterations per Newton
iteration, against 86 for 1e-2 and 162 for 1e-4), and stops it at relres 0.3–2
where the system is hard (BD-Z after the snap: a solve to 1e-3 takes ~4.6 s
against 0.77 s for 500 iterations, i.e. roughly 3,000). Newton then takes 20–30
poor steps instead of 5 good ones, or fails.

Replay of BD-Z's old stall increment (t 0.42 -> 0.435, first 14 sweeps;
`tools/replay.py`, `runs/t2_*.log`), fixed code:

| linear solve | sweeps with failed Newton | Newton its (14 sweeps) | CG relres geo-mean / >= 1 | time |
|---|---|---|---|---|
| toler 1e-7, cap 500 (production) | 4 | 196 | 1.1e-1 / 53 | 267 s |
| same, `cg_best` (R4) | 5 | 199 | 1.3e-1 / 10 | 257 s |
| toler 1e-7, cap 2000 | **0** | 77 | 5.6e-5 / 0 | 265 s |
| **toler 1e-3, cap 4000** | **0** | **69** | 1e-3 / 0 | **148 s** |

The damage path is the same in all four (err_abs identical to four digits at
sweep 14). The inexact-Newton tolerance costs no Newton iterations before the
cap either (4–5 per sweep, as with 1e-7).

---

## R10. Full-step Newton cycles across a kink — measured (T2), a setting

In BD-Z's rejected attempt at t = 0.45 (dt 0.04), from sweep 12 on, the Newton
residual settles into a period-2 cycle 4.5 <-> 3.9 for all 30 iterations. The
linear solves are **not** the cause: with 2,000 CG iterations they reach relres
2e-3 and the cycle is identical. This is the "period-2 pattern" of
`14_root_cause/README.md` section 2.4. The start residual of such a sweep is
> 1 (the damage update moved the state far from equilibrium), and the first
steps *increase* it.

Replay of that attempt (first 14 sweeps):

| | sweeps with failed Newton | Newton its | time |
|---|---|---|---|
| cap 500 (production) | 7 (rejected: "Newton diverged") | 256 | 198 s |
| cap 2000 | 3 (the cycling ones) | 161 | 407 s |
| cap 500 + `newton_line_search: 4` | 7 (monotone now, but crawling on poor directions) | 257 | 203 s |
| **cap 2000 + `newton_line_search: 4`** | **0** | **105** | 237 s |

One step halving, in the second iteration of the sweep that cycled, is all the
line search does there. F3's verdict on the line search ("prevents the blow-up
but stalls the same way") was R1 and R9: it was fed corrections with relres
>= 1.

Not diagnosed: *which* kink. The candidate is the tension/compression switch of
the stress-level Amor split in capped voxels (bulk stiffness g K <-> K, a factor
1,600). Test: count, per Newton iteration, the voxels whose sign of p_eff or
whose plastic/elastic branch differs from the previous iterate.

---

## R4. CG returns its last iterate, not its best — opt-in added, measured: no gain

`solvers.krylov.cg.cg_solve_best` returns the smallest-residual iterate when
CG stops at its cap (`linear_solver: cg_best`; identical to `cg` when CG
converges; `test_solvers_krylov_cg_best.py`). CG's residual is spiky (it
exceeds |b| in 10–21 of the first 500 iterations on the stored states, peaks
of 3.7–3.9), so this bounds relres by 1. On the BD-Z stall increment it cut
the corrections with relres >= 1 from 53 to 10 and changed nothing for Newton
(R9 table): a correction with relres 0.9 is as useless as one with 2. The
remedy for a capped CG is more iterations.

---

## R11. The CG matvec does about twice the work it needs — performance, measured, open

`tools/bench_matvec.py` (`runs/bench_matvec_60.log`, 60^3, GPU): one CG
iteration as in `src/` (operator + preconditioner) 8.2 ms. The same work with
real FFTs, strain and stress as 6 Mandel components instead of 9, and the
displacement unknown kept in Fourier space between preconditioner and operator
(no ifft/fft pair): 3.8 ms, **2.15x**. Tangent memory 133 -> 59 MiB at 60^3
(2.3 -> 1.0 GB at 152^3). After the cap, CG iterations are essentially the whole
run time, so this is the largest remaining factor that needs no change of
method. It needs a CG with its inner products taken in Fourier space
(Parseval, half-spectrum weights) and care with the DC-bin and Nyquist
conventions; not started.

---

## R6. `scheme` is ignored for `formulation: displacement` — clarity, open

`scheme: rotated` only reaches the Lippmann-Schwinger and Fourier-Galerkin
operators. The displacement path (the only one a DP material can use)
differentiates with the Fourier symbol `i xi`, yet the driver writes
`scheme = rotated` into the output file's attributes. See N1: the rotated
gradient is not better here, so this is a labelling issue.

## R7. Plastic return evaluated on every voxel, then masked — performance, minor, open

`assemble_inelastic_pff_local_update` evaluates each material's
`stress_and_tangent_field` on the whole grid and selects by phase. ~6 ms per
200k voxels per call against seconds of CG per Newton iteration.

## R8. `build_willot_freq` is not Willot's rotated scheme — clarity, by inspection, open

`operators/green.py`: `xi_eff_j = (2/h_j) sin(xi_j h_j / 2)`. Willot (2015) has
an additional factor `prod_{l != j} cos(xi_l h_l / 2)`. Not on the DP path; not
checked numerically.

## R12. Damage CG preconditioner mismatched at the Nyquist planes — minor, measured, open

`solve_damage_helmholtz_cg_het` zeroes the Nyquist component of the gradient in
the operator but keeps the full `|xi|^2` in the preconditioner, so modes on
those planes are preconditioned with up to 160x the wrong stiffness
(`tools/t0_damage_cg.py`). With the reproducer's one-voxel seeded slab the
damage solve then misses `toler_helm` in 300 iterations (`helm=False` on every
sweep before the cap, 12–50 per run). Production 60^3 runs start from d = 0 and
show 0–3 `helm=False` per run, so this affects the reproducer, not the results.

---

## T1. Reproducer results (`tools/repro.py`, `tools/log_stats.sh`)

Same reproducer and settings as `14_root_cause` (production card, linear
hardening, `toler_lin 1e-7`, `maxiter_cg 500`).

**24^3**, three runs sharing the GPU at the same time (`runs/r24_*.log`):

| | original code | restructured, old sign (R2, R3, R5) | all fixes (+ R1) |
|---|---|---|---|
| time per sweep, before the cap / cap increment | 6.9 / 8.2 s | 2.6 / 4.9 s | **1.1 / 2.9 s** |
| CG meeting 1e-7 before the cap | 33 of 147 | 33 of 147 | **147 of 147** |
| cap increment: CG relres geo-mean, >= 1 | 4e-2, 3 | 4e-2, 3 | **8e-5, 0** |
| Newton its per sweep, cap / after (mean, max) | 6.9, 10 / 9.0, 10 | 6.9, 10 / 9.0, 10 | **4.5, 5 / 4.7, 6** |
| sweeps at the cap / to 5 % | 33 / (176) | 33 / 176 | 33 / 176 |
| sigma_zz at the cap, 5 % | 10.909 / (2.840) | 10.909 / 2.840 | 10.909 / 2.840 |
| wall to 5 % | stopped at 3.25 % | 772 s | **460 s** |

The middle column reproduces the original run to every printed digit: the
restructuring changes no result. The sweep count does not move (I14 stands).

The non-plastic fracture path is unchanged too (16^3, 8 increments, new against
original code): Lippmann-Schwinger bit-identical (damage, stress, sweeps);
displacement + free surfaces identical sweeps, damage to 3e-9, stress to 7e-8
(the preconditioner changes CG's iterates within its tolerance).

**40^3** (`runs/n40_new.log` against `14_root_cause/runs/n40_base.log`):

| | 2026-11-08 | fixed |
|---|---|---|
| CG corrections with relres >= 1 | 383 of 3,415 | **0** of 1,343 |
| Newton its, cap increment / after (mean, max) | 14.0, 27 / 11.5, 16 | **6.2, 10 / 5.2, 7** |
| CG meeting 1e-7 before the cap | 28 of 104 | 104 of 104 |
| sweeps to 5 %, rejected | 348, 0 | 344, 0 |
| sigma_zz at 4.25 % / 5 % | 3.671 / 2.203 | 3.671 / 2.579 |
| wall | 4,517 s (GPU shared by ~5) | **677 s** (shared by 2) |

The two 40^3 runs agree to every printed digit through 4.25 % (22 increments:
sigma_zz, sweeps, n_cap). In the last four increments the sweep counts differ
and sigma_zz separates by up to 0.38 MPa (2.3 % of the peak). Both converged
every Newton solve to `toler_nr`; the accepted state of the production
formulation is path-dependent at `toler_st_abs 1e-3` (F17), so this says the
late tail is sensitive, not which run is right.

Pass criterion of `14_root_cause/README.md` section 6 (no relres >= 1, Newton
<= 10 at 40^3): **met, with the production model.**

---

## T3. BD-Z at 60^3 (`configs/bi_z_fix*.yaml`)

`13_small_rve/configs/small_bi_z.yaml` unchanged except output, jobname,
`checkpoint: true`.

**Fixes only, production settings** (`runs/bi_z_fix.log`, `output/bi_z_fix.*`):

| | earlier runs (F1, F6) | 2026-11-12 |
|---|---|---|
| reached | 2.10 % (baseline), 2.47 % (GMRES), 2.56 % (guard) | **5.00 %** |
| wall | 1.1 h / 7.7 h / 7.7 h, unfinished | **2 h 32 min** |
| the old stall increment t = 0.435 | 193+ sweeps, 4.7 h, not accepted | 49 sweeps, 14.5 min |
| rejected attempts | 2 / 6 / 14 | 3 |
| peak | 25.24 MPa at 2.10 % | 25.24 MPa at 2.10 % |

Not clean: 97 of 761 sweeps with a failed Newton, CG relres geo-mean 0.5 after
the snap with 12–29 % of corrections at relres >= 1 (R9), two attempts rejected
for diverging Newton (R10), and one attempt of 200 sweeps (40 min) in which
Newton converged every sweep and the damage did not (I14).

**What accurate mechanics exposes: the stall increment does not converge in 200 sweeps.**
Full replay of the old stall increment (t 0.42 -> 0.435) with `maxiter_cg: 2000`
(`runs/t2_inc13_cg2000_full.log`, 87 min on a shared GPU): Newton converges in
all 200 sweeps (948 iterations, at most 8), and the damage does not:

| sweep | 14 | 48 | 56–59 | 80 | 100 | 120 | 140–150 | 170 | 200 |
|---|---|---|---|---|---|---|---|---|---|
| err_abs | 4.1e-2 | 4.6e-3 | **1.025e-3 … 1.009e-3** | 1.7e-3 | 2.5e-3 | 9.8e-3 | 3.6e-2 | 3.2e-2 | 8.4e-3 |

It misses `toler_st_abs: 1e-3` by 1 % at sweep 59, then a second growth burst
starts; at sweep 200 the cell has 679 capped voxels and sigma_zz 20.16 MPa. The
production-settings run (`bi_z_fix`) accepted this increment at sweep 49, on a
dip to 9.4e-4 that its noisier mechanics happened to produce, with 232 capped
voxels and 22.06 MPa. So:

- That accepted state was not a converged one: the crack was still running at
  fixed strain. This is F8's caveat (a small sweep-to-sweep change is not a
  small error) on the increment that matters most.
- At 1e-3 the increment is decided in the third digit; at 3e-3 it would be
  accepted at sweep 56 in the accurate replay.
- A robust Newton removes the accidental early exits ("Newton diverged") and
  the accidental acceptances alike. What is left is the staggered loop
  resolving a running crack sweep by sweep (I14), and 200 sweeps are not
  enough for it here.

**Full runs with the settings** (started 2026-11-12, read out 2026-11-13; both
reached 5.00 %):

| run | settings on top of the fixes | config, log |
|---|---|---|
| `bi_z_fix2` (22:22–07:33) | `toler_lin: 1e-3`, `maxiter_cg: 4000`, `newton_line_search: 4` | `configs/bi_z_fix2.yaml`, `runs/bi_z_fix2.log` |
| `bi_z_fix3` (22:52–05:03) | the same + `toler_st_abs: 3e-3` | `configs/bi_z_fix3.yaml`, `runs/bi_z_fix3.log` |

| | `bi_z_fix` | `bi_z_fix2` | `bi_z_fix3` |
|---|---|---|---|
| sweeps with a failed Newton | 97 of 761 | **0** of 1,727 | **0** of 670 |
| CG corrections with relres >= 1 | 1,532 of 10,785 | **0** of 4,994 | **0** of 2,301 |
| Newton its per sweep, mean / max | 15.0 / 30 | 3.9 / 11 | 4.4 / 10 |
| accepted increments, sweeps in them | 29, 528 | 32, 727 | 27, 470 |
| rejected attempts | 3: 2 Newton diverged, 1 at 200 sweeps | 5, all at 200 sweeps | 1, at 200 sweeps |
| wall time in rejected attempts | 47 min of 2 h 32 | 6 h 49 of 9 h 11 | 2 h 10 of 6 h 12 |
| time per sweep for t > 0.53, GPU alone | 9.8 s | 9.8 s | (GPU shared throughout) |

The wall times are not comparable: `fix2` and `fix3` shared the GPU with each
other until 05:03 and with a replay until 23:49. Alone, a sweep of `fix2` costs
what a sweep of `bi_z_fix` did, with the mechanics converged in it.

1. **The mechanics failures are gone** (R9, R10 confirmed). No failed Newton
   solve and no CG correction with relres >= 1 in either run, rejected attempts
   included. The three settings are pinned in `CLAUDE.md`.
2. **Up to the peak nothing changes.** `fix2` matches `bi_z_fix` in every
   printed digit through increment 12 (t = 0.42: sigma_zz, sweeps), `fix3`
   within 0.013 MPa.
3. **Every remaining rejection is the staggered loop at its 200-sweep cap with
   Newton converged in every sweep** (I14). `fix2`:

   | attempt (t, dt) | 0.45, 0.04 | 0.43, 0.02 | 0.435, 0.015 | 0.4556, 0.0169 | 0.4788, 0.0190 |
   |---|---|---|---|---|---|
   | smallest err_abs, at sweep | 2.3e-3, 193 | 2.5e-3, 131 | 1.1e-3, 56 | 3.1e-3, 95 | 7.3e-3, 150 |
   | wall | 2 h 10 | 1 h 41 | 1 h 31 | 49 min | 39 min |

   The third is the increment of the full replay above and behaves as it did
   there. The snap increments that were accepted had dt 0.0075–0.0142
   (10–76 sweeps), plus one of dt 0.0214 that took 152. The last two rejections
   are the stepper growing dt by 1.5 after an accepted snap increment.
   `fix3`'s one rejection is the first attempt (t = 0.45, dt 0.04); at 3e-3 the
   dt 0.02 attempt was accepted after 138 sweeps.
4. **The snap interval of the curve depends on the step sequence.** sigma_zz in
   MPa, at the strain in brackets where the runs have no common point:

   | eps_zz | `bi_z_fix` | `bi_z_fix2` | `bi_z_fix3` |
   |---|---|---|---|
   | peak | 25.24 (2.10 %) | 25.24 (2.10 %) | 25.21 (2.05 %; no point at 2.10 %) |
   | ~2.15–2.19 % | 22.06 (2.175) | 22.94 (2.194) | 18.78 (2.150) |
   | ~2.30 % | 17.45 (2.316) | 22.32 (2.299) | 16.54 (2.300) |
   | ~2.50 % | 16.29 (2.442) | 9.92 (2.525) | 8.04 (2.500) |
   | ~2.9 % | 7.29 (2.832) | 7.73 (2.885) | 7.01 (2.900) |
   | ~3.5 % | 5.89 (3.432) | 6.00 (3.485) | 5.75 (3.500) |
   | 5.00 % | 4.72 | 4.77 | 4.76 |

   The runs agree at the peak and within 0.05 MPa (0.2 % of the peak) at 5 %,
   and differ by up to 5.8 MPa at equal strain inside the snap. `fix2` and
   `fix3` took different steps through it (dt 0.0075–0.021 against 0.02–0.04),
   so the pair does not separate the effect of `toler_st_abs` from that of the
   step size and cannot decide between 1e-3 and 3e-3. That needs both
   tolerances on the same fixed steps, restarted from the same checkpoint. Until
   then the curve between 2.1 and 2.7 % is not a converged result in any of the
   three.

Caveat on the pair: `fix2` was started at 22:22, before the last edit of
`src/problems/mechanics.py` that evening (22:30), `fix3` after it (22:52). What
that edit changed was not recorded.

---

## Ruled out (negative results)

### N1. Rotated / hexahedral gradient in the Newton solver — measured, not a fix

Idea: spectral gradients couple badly to near-void voxels; a compact
finite-difference gradient (trilinear hexahedron with one integration point =
Willot's scheme) might make CG insensitive to the d_cap contrast.
`tools/t0_gradient_scheme.py`, `tools/t0_precond.py`
(`runs/t0_precond_pexfix.log`), corrected preconditioner:

| state | gradient | its to 1e-2 | to 1e-4 | to 1e-7 |
|---|---|---|---|---|
| t = 0.57, no capped voxel | Fourier | 86 | 162 | 275 |
| | hexahedral | 49 | 114 | 215 |
| t = 0.61, 64 capped | Fourier | 242 | 643 | 1259 |
| | hexahedral | **679** | 2962 | > 3000 |

Slightly better before the cap, clearly worse after it. On a toy with a sharp
planar band (elastic, g = 6.3e-4 or 1e-6) both converge in 105–415 iterations:
a sharp band is not the hard case.

### N2. Preconditioner scaled by the degradation — measured, not a fix

`M = S K0^-1 S`, `S = g(d)^-1/2` (and two variants built from `tr(C_tan)`),
`tools/t0_precond.py`. Worse in every variant: capped state, 1925 iterations to
1e-2 against 242. Same conclusion as F4's Fourier-Jacobi attempt, from a
different construction.

### N3. Non-symmetric tangent as the reason CG is slow — measured, it was R1

With the old preconditioner, replacing the tangent by its symmetric part halved
the CG iterations (826 -> 363 before the cap). With the preconditioner
corrected the two are the same: 275 vs 280 (before the cap), 1259 vs 1234
(capped) (`tools/t0_decompose.py`, `runs/t0_decompose_pexfix.log`). The 13–17 %
asymmetry does not slow CG on these states. What remains at the capped state is
the damage field: elastic tangent with the real d, 956 iterations to 1e-7;
plasticity alone (d = 0), 245; plastic freeze, 1115.

---

## Pitfalls met during this review

- A running job holds an HDF5 lock on its output: `h5py.File(..., "r")` fails.
  The writer flushes per increment, so `cp` the live file and read the copy
  (`tools/replay.py` does not do this for you).
- Do not edit a bash script while it runs (bash reads it incrementally):
  overwriting `t2_batch.sh` mid-run cost one replay log.
- Starting any GPU process can kill a run of the *old* code at any time (R2).
- T0 comparisons that copy a solver's operator copy its bugs too: the first
  decomposition (N3) measured the preconditioner sign, not the tangent.
