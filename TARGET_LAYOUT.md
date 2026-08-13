# Target `src/` Layout

Working roadmap for the next restructuring pass. A plain layout reference, not a build log —
detailed history (what broke, what got verified, when) lives in project memory and git log.

- ⚙️ = not done yet, but needed for the mechanical (elasticity) problem — phase-1 priority.
- ✅ = actually done on `refactor/target-layout` (verified against the repo, not aspirational).
- No mark = planned, not yet prioritized.
- "TO DISCUSS" = needs a decision before implementation.

```
src/
├── operators/
│   ├── __init__.py
│   ├── base.py                         ✅ LinearOperator ABC: __call__, __matmul__/compose, .T (adjoint)
│   ├── differential.py                 # ∇, ∇· — rank-aware (scalar grad vs. sym grad for elasticity)
│   ├── green.py                        ✅ GreenOperatorBasic, GreenOperatorWillot — swappable discretization;
│   │                                   #   DFT/DCT/DST periodic vs. non-periodic BC — TO DISCUSS
│   ├── projection.py                   ✅ Gamma0Operator — Γ0 fixed-point op (FFT ↔ green_op ↔ IFFT);
│   │                                   #   no separate ∇ operator, Γ0 is fused into green.py's G via n_hat
│   ├── boundary.py                     # BCType, FaceBC, BoundaryConditions — periodic/Dirichlet/Neumann/mixed
│   └── fft_utils.py                    # rfftn/irfftn wrappers, raw k-vector construction — not needed yet
│
├── solvers/
│   ├── __init__.py
│   ├── base.py                         # shared convergence criteria, iteration bookkeeping
│   ├── krylov/
│   │   └── cg.py                       ✅ operator-agnostic (P)CG driver, shared by every elliptic PDE-type
│   ├── elliptic/
│   │   ├── scalar.py                   # thermal, diffusion (Fickian), phase-field φ-subproblem
│   │   └── vector/
│   │       ├── base.py                 ✅ ElasticitySolver ABC (.solve()), ElasticitySolution
│   │       │                           #   (NamedTuple, not dataclass — pytree-safe for jit/grad)
│   │       ├── lippmann_schwinger.py   ✅ solve_lippmann_schwinger + LippmannSchwingerSolver;
│   │       │                           #   strain-based, periodic; DCT/DST non-periodic — TO DISCUSS
│   │       └── displacement_based.py   # direct u-solve via ∇/∇·, mixed BC via boundary.py, penalty enforcement
│   ├── parabolic/
│   │   └── reaction_diffusion.py       # Allen-Cahn / Cahn-Hilliard time-stepping over elliptic.scalar
│   ├── coupling/
│   │   └── staggered.py                # generic staggered driver (mechanics ↔ phase-field, thermal ↔ diffusion, ...)
│   └── adjoint/
│       └── implicit_diff.py            # custom_vjp via implicit function theorem — differentiable solve() for free
│
├── materialmodels/
│   ├── base.py                         ✅ ConstitutiveModel ABC (thin — just enough to hold C)
│   ├── assembly.py                     ✅ assemble_C_field, describe_materials — hard/sharp-interface
│   │                                   #   per-voxel field assembly, generic over any ConstitutiveModel
│   ├── elastic/
│   │   ├── isotropic.py                ✅ LinearElasticIsotropic on ConstitutiveModel
│   │   ├── orthotropic.py
│   │   ├── transversely_isotropic.py
│   │   └── anisotropic.py
│   ├── inelastic/
│   │   ├── plasticity_j2.py
│   │   └── damage.py
│   ├── thermal/
│   │   ├── isotropic.py
│   │   └── orthotropic.py
│   ├── diffusion/
│   │   ├── fickian.py                  # linear Fick's law, isotropic/orthotropic D
│   │   └── nonlinear.py                # concentration-dependent D(c) — needs Newton-outer/CG-inner, not drop-in linear
│   ├── phasefield/
│   │   ├── degradation.py              # g(φ) AT1/AT2, spatially varying Gc
│   │   └── regularization.py           # ℓ-dependent gradient-energy term
│   ├── averaging.py                    ⚙️ VoxelAveraging ABC: at minimum ArithmeticAveraging (needed at interfaces)
│   ├── tensors.py                      ⚙️ Voigt/Mandel ↔ tensor (4th-order C, 2nd-order ε/σ), rotation, symmetry checks
│   └── utils.py                        # phase-fraction computation, raw geometry measurement (true utils)
│
├── problems/                           # thin wiring layer: pick strategies, build C(x), average, solve, unpack
│   ├── mechanics.py                    ✅ solve_mechanics: formulation="lippmann_schwinger" done, "displacement"
│   │                                   #   raises NotImplementedError; reference-medium averaging is a plain
│   │                                   #   Lame-parameter mean, placeholder for materialmodels/averaging.py
│   ├── thermal.py
│   ├── diffusion.py
│   └── fracture.py                     # staggered mechanics + phase-field
│
├── generation/                         # RVE/RSA/Matérn cluster generation (existing name kept over "microstructure/")
│
├── post/                               # post-processing: field-space derived quantities only, no I/O
│   └── fields.py                       ✅ field_to_grid, von_mises, compute_displacement
│
├── utils/
│   ├── io/
│   │   ├── xdmf_writer.py              ⚙️ IncrementalWriter, currently in post/io.py — this is its target home
│   │   │                               #   (a move, not new code); project-wide standard for XDMF/HDF5 output
│   │   └── checkpoint.py               # solver-state restart
│   └── logging.py
│
└── tests/
    ├── operators/
    │   └── boundary/
    ├── solvers/
    │   ├── elliptic/vector/            ⚙️ validate against known analytical case (e.g. two-phase laminate bound)
    │   └── adjoint/                    # jax.grad vs. finite-difference
    ├── materialmodels/
    │   ├── elastic/  inelastic/  thermal/  diffusion/  phasefield/
    │   └── tensors/                    ⚙️ Voigt round-trip tests
    └── utils/io/
```
