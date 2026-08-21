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
├── solvers/                            # numerics only: operate on prepared arrays/operators (C_field, green_op,
│   │                                   #   xi_flat, ...) — never import materialmodels/ or know about phase/
│   │                                   #   materials/"which degradation law". Multi-physics coupling (staggered
│   │                                   #   loops etc.) lives in problems/ instead, one driver per problem file —
│   │                                   #   see problems/fracture.py's module docstring for the reasoning
│   ├── __init__.py
│   ├── base.py                         # shared convergence criteria, iteration bookkeeping
│   ├── krylov/
│   │   └── cg.py                       ✅ operator-agnostic (P)CG driver, shared by every elliptic PDE-type
│   ├── elliptic/
│   │   ├── scalar.py                   ✅ solve_damage_helmholtz_cg — AT2 phase-field damage sub-problem, incl.
│   │   │                               #   per-voxel k_res scaling; thermal, diffusion (Fickian) not built yet
│   │   └── vector/
│   │       ├── base.py                 ✅ ElasticitySolver ABC (.solve()), ElasticitySolution
│   │       │                           #   (NamedTuple, not dataclass — pytree-safe for jit/grad);
│   │       │                           #   eps_bar field (default None) carries mixed-BC output
│   │       ├── lippmann_schwinger.py   ✅ solve_lippmann_schwinger + LippmannSchwingerSolver;
│   │       │                           #   strain-based, periodic; DCT/DST non-periodic — TO DISCUSS
│   │       └── displacement_based.py   ✅ solve_displacement_based + DisplacementBasedSolver — direct
│   │                                   #   u-solve via inline FFT ∇/∇· (Nyquist-safe), true heterogeneous
│   │                                   #   tangent (no reference medium); mixed macroscopic strain/stress
│   │                                   #   BC via a static (3,3) control mask, not operators/boundary.py
│   │                                   #   (that's for physical-domain face BCs — a different concept;
│   │                                   #   this is periodic-domain homogenization throughout)
│   ├── parabolic/
│   │   └── reaction_diffusion.py       # Allen-Cahn / Cahn-Hilliard time-stepping over elliptic.scalar
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
│   │   ├── transversely_isotropic.py   ✅ TransverseIsotropicFibre (5-constant, reference fibre
│   │   │                               #   axis Z); stiffness_tensor_rotated(fiber_dir) via
│   │   │                               #   tensors.py; assemble_C_field_oriented (per-voxel
│   │   │                               #   orientation field, vmapped rotation) not built yet
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
│   │   ├── degradation.py              ✅ degradation_at2 g(d), degrade_stiffness_field, k_res_field (per-phase
│   │   │                               #   residual stiffness, incl. k_res=1 damage-immune); AT1, spatially
│   │   │                               #   varying Gc not built yet
│   │   ├── driving_force.py            ✅ lame_field, strain_energy_amor_split, update_history_hybrid (hybrid
│   │   │                               #   irreversibility, Steinke & Kaliske 2019); Miehe/spectral split not
│   │   │                               #   built yet
│   │   └── regularization.py           # ℓ-dependent gradient-energy term
│   └── tensors.py                      ✅ voigt_to_tensor4/tensor4_to_voigt (4th-order C only —
│                                       #   2nd-order ε/σ Voigt/Mandel is post.fields.to_voigt/
│                                       #   from_voigt, doesn't belong here), rotation_from_direction/
│                                       #   rotate_tensor4, is_major_symmetric/is_minor_symmetric
│
├── problems/                           # thin wiring layer: pick strategies, build C(x), average, solve, unpack —
│   │                                   #   AND, per fracture.py, owns any staggered/multi-physics loop for that
│   │                                   #   problem (solvers/ stays numerics-only, see its note above)
│   ├── mechanics.py                    ✅ solve_mechanics: both formulation="lippmann_schwinger" (reference-
│   │                                   #   medium, plain Lame-parameter mean — placeholder for
│   │                                   #   materialmodels/averaging.py) and "displacement" (true heterogeneous
│   │                                   #   tangent, mixed strain/stress BC via control) done
│   ├── thermal.py
│   ├── diffusion.py
│   └── fracture.py                     ✅ solve_fracture: staggered mechanics<->AT2 phase-field, one call per
│                                       #   time increment; both formulations (mirrors mechanics.py, incl.
│                                       #   control for mixed BC); the staggered loop itself lives here, not
│                                       #   in solvers/
│
├── generation/                         # RVE/RSA/Matérn cluster generation (existing name kept over "microstructure/")
│
├── preprocessing/
│   └── averaging.py                    ✅ VoxelAveraging ABC, ArithmeticAveraging — smooths C at interface
│                                        #   voxels using a partial volume_fraction (vs. assembly.py's hard phase index)
│
├── post/                               # post-processing: field-space derived quantities only, no I/O
│   └── fields.py                       ✅ field_to_grid, von_mises, compute_displacement, to_voigt, from_voigt
│
├── utils/
│   ├── io/
│   │   ├── xdmf_writer.py              ✅ IncrementalWriter — moved from post/io.py (post/io.py deleted);
│   │   │                               #   project-wide standard for XDMF/HDF5 output
│   │   ├── reader.py                   ✅ read_xdmf, read_vtu, read_npz, SimulationReader —
│   │   │                               #   merged from io_read.py + io_texgen.py (both deleted)
│   │   └── checkpoint.py               # solver-state restart
│   ├── logging.py
│   ├── precision.py                    ✅ float64/GPU-memory setup — import first in every entry-point module
│   └── geometry.py                     ✅ phase_fraction (arbitrary N classes, jnp.bincount-based)
│
└── tests/
    ├── operators/
    │   └── boundary/
    ├── solvers/
    │   ├── elliptic/vector/            ⚙️ validate against known analytical case (e.g. two-phase laminate bound)
    │   └── adjoint/                    # jax.grad vs. finite-difference
    ├── materialmodels/
    │   ├── elastic/  inelastic/  thermal/  diffusion/  phasefield/
    │   └── tensors/                    ✅ Voigt round-trip tests -- actually
    │                                   #   test/test_materialmodels_tensors.py (flat, matching
    │                                   #   this repo's real test/ convention, not this nested sketch)
    └── utils/io/
```
