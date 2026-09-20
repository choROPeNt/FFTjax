# CLAUDE.md

Project guidance for Claude Code sessions in this repo. Keep entries here
short and factual; the long-form reasoning lives in the per-study handoffs
under `sim/sim/*/HANDOFF.md`.

## Running things

- Scripts and tests are run **from the repo root** with `.venv/bin/python`
  (they do `sys.path.insert(0, "src")`).
- Tests are plain scripts, not pytest: `.venv/bin/python test/<name>.py`.
- Long GPU jobs must be launched detached — `setsid nohup ... < /dev/null &`.
  A Claude session ending has twice killed every GPU job it had started.

## Phase-field fracture: the eta viscosity

**eta is a MODELLING parameter in load-fraction units, not a free solver
knob. Its numerical value only means something together with the config's
target strain.**

eta enters the damage solve only as `eta/dt`
(`src/solvers/elliptic/scalar.py:111`), and `dt` there is the **load fraction**
`t` in `[0, 1]` — not strain, and not physical time. Two configs with
different `eps_bar` at `t = 1` therefore need different eta for the same
physics.

Two quantities must be preserved when porting eta between configs:

| invariant | expression | why |
|---|---|---|
| stabilization | `eta / dt` | sets the staggered map's gain; what a cutback raises |
| damage lag, in strain | `eta / (Gc/l0) * eps_target` | the rate-dependence eta introduces |

They scale differently, so **a single eta satisfies both only if the configs
use the same strain per step** (`dt_max * eps_target`). Keep new configs at
**0.2% strain per step** and the conversion stays a single factor.

**Pinned values** (Gc/l0 = 4.0 N/mm², 0.2% strain per step):

| config | eps_target | dt_max | eta | eta/dt | lag |
|---|---|---|---|---|---|
| `fracture_mini_eta_sweep.yaml` | 10% | 0.02 | **0.025** | 1.25 | 0.0625% |
| `fracture_uni_transverse_pp.yaml` | 2% | 0.10 | **0.125** | 1.25 | 0.0625% |

Pinned by the 10-run sweep in `sim/sim/12_mini_eta/` (uni/bi/tri × eta,
2026-09-19). The error is linear in eta; at 0.025 mini units it is +0.5 to
+0.9% on peak stress and dissipated work against the eta → 0 limit, versus
+2 to +3.6% at 4× that eta. Full evidence and method: section 00 of
`sim/sim/12_mini_eta/HANDOFF.md`.

**Two traps:**

1. **Changing a config's target strain silently changes the model.** Raise
   `eps_bar` and leave eta alone, and the damage lag scales with it while
   `eta/dt` shifts — a different constitutive response, a probably-less-stable
   run, and nothing in the log to say so. Rescale eta and `dt_max` together.
2. **`eta: 1.0e-6` in an older config means eta is inert, not "small".** It
   does not regularize anything; an unstable softening increment diverges at
   any dt. Any new fracture run needs the pinned value for its target strain.

The durable fix is to divide by the strain increment rather than the
load-fraction increment, so eta transfers 1:1 between configs. That is a model
change — decide it before tuning further production values.

## Phase-field fracture: other settings that are not free

- **`maxiter_st: 200`**, not 40. At 40 the rejected attempts took 58-76% of
  wall time; at 20 the runs entered a cutback death spiral (a crack running at
  fixed strain is re-run from scratch after every rejection).
- **`d_cap: 0.975`** is a conditioning regularization and a deliberate
  modelling change. Quote it alongside any result — a crack never fully opens
  and dissipated energy is slightly under-counted.
- **`Gc` is tied to `l0`.** Changing `l0` requires rescaling `Gc` with it;
  what controls AT2 damage growth is the ratio `Gc/l0`. See the header of
  `configs/simulation/fracture_tangled_pp.yaml`.
- **Shear-free (uniaxial stress) BC** needs off-diagonal `control` entries set
  to 1, e.g. `[[0,1,1],[1,1,1],[1,1,1]]` for X loading. Holding shear *strain*
  at zero instead locks an inclined crack band and props up the softening
  tail. This path depends on `sm2sv` returning the adjoint of `sv2sm`
  (`src/problems/mechanics.py:449`) — without it the bordered CG operator is
  asymmetric whenever shear is stress-controlled.

## Git

- No commit attribution: no `Co-Authored-By` or "Generated with Claude Code"
  lines on commits or PRs. Enforced in `settings.json`.
- `sim/` is a symlink to a Windows drive and is **not in git**.
