# Target `src/` Layout

Working roadmap for the next restructuring pass.

- ⚙️ = not done yet, but needed for the mechanical (elasticity) problem — phase-1 priority.
- ✅ = actually done on `refactor/target-layout` (verified against the repo, not aspirational).
- No mark = planned, not yet prioritized.
- "TO DISCUSS" = needs a decision before implementation.

```
src/
├── operators/
│   ├── __init__.py
│   ├── base.py                             ✅ LinearOperator ABC: __call__, __matmul__/compose, .T (adjoint)
│   ├── differential.py                     # ∇, ∇· — rank-aware (scalar grad vs. sym grad for elasticity)
│   ├── green.py                            ✅ GreenOperatorBasic, GreenOperatorWillot — swappable discretization;
│   │                                       #   transform type (DFT/DCT/DST) sets periodic vs. non-periodic BC — TO DISCUSS
│   ├── projection.py                       ✅ Γ0 fixed-point operator — real-space FFT↔green_op↔IFFT wrapper;
│   │                                       #   no separate ∇ LinearOperator: Γ0 is degree-0 homogeneous in ξ,
│   │                                       #   so G0∘∇ is already fully fused into green.py's G via n_hat
│   ├── boundary.py                         # BCType, FaceBC, BoundaryConditions — periodic/Dirichlet/Neumann/mixed, masks + values
│   └── fft_utils.py                        # util: rfftn/irfftn wrappers, raw k-vector construction (no physics, no scheme choice)
│                                            #   not needed yet — build_freq_grid (green.py) and Gamma0Operator's
│                                            #   inline fftn/ifftn (projection.py) already cover current use
│
├── solvers/
│   ├── __init__.py
│   ├── base.py                             # shared convergence criteria, iteration bookkeeping
│   ├── krylov/
│   │   └── cg.py                           ✅ operator-agnostic (P)CG driver, shared by every elliptic PDE-type
│   ├── elliptic/
│   │   ├── scalar.py                       # thermal, diffusion (Fickian), phase-field φ-subproblem
│   │   └── vector/
│   │       ├── base.py                     ✅ ElasticitySolver ABC (.solve()), ElasticitySolution —
│   │       │                              #   NamedTuple not dataclass (pytree-safe for jit/grad, matches
│   │       │                              #   solvers.types.SolveState's own convention/rationale)
│   │       ├── lippmann_schwinger.py       ✅ solve_lippmann_schwinger + LippmannSchwingerSolver(ElasticitySolver);
│   │       │                              #   problems/mechanics.py builds a LippmannSchwingerSolver and calls
│   │       │                              #   .solve() (not the bare function) so the ABC is actually exercised by
│   │       │                              #   its one real caller, not just its own tests -- checked and fixed
│   │       │                              #   2026-08-13 after confirming nothing used the class before.
│   │       │                              #   strain-based, periodic, built on GreenOperatorBasic/Willot +
│   │       │                              #   Gamma0Operator + krylov/cg.py; not yet jit-compiled at this level
│   │       │                              #   (LinearOperator isn't a registered pytree yet) — extend for
│   │       │                              #   DCT/DST non-periodic — TO DISCUSS. solvers/mechanical/strain_nw_cg.py
│   │       │                              #   (the function this replaces) is deleted -- all pure-elasticity
│   │       │                              #   callers migrated + verified (identical output). KNOWN BREAKAGE:
│   │       │                              #   scripts/simulation/pff_nw_cg_strain.py, benchmark/single_notch_plate/
│   │       │                              #   pff_{tension,shear}.py, examples/pff_damage.py still import the
│   │       │                              #   deleted module (ImportError) -- explicitly deferred, not migrated;
│   │       │                              #   the staggered mechanics+damage loop needs solvers/coupling/
│   │       │                              #   staggered.py + elliptic/scalar.py first, not just a solver swap.
│   │       │                              #   solvers/mechanical/ and solvers/damage/ (the WHOLE packages --
│   │       │                              #   displacement_nw_cg.py, displacement_nw_plastic.py, pff_damage.py,
│   │       │                              #   anderson.py) are ALSO deleted, explicit user choice, with NO
│   │       │                              #   replacement anywhere yet. Additional KNOWN BREAKAGE from that:
│   │       │                              #   test/test_displacement_nw_cg.py, test/test_helmholtz.py (both
│   │       │                              #   were passing before), examples/lin_elastic_mixed_bc.py,
│   │       │                              #   notebooks/lin-elastic_mixed-BC.ipynb, notebooks/j2-plasticity.ipynb,
│   │       │                              #   scripts/simulation/elastic_nw_cg_disp_mixed.py,
│   │       │                              #   scripts/simulation/pff_nw_cg_disp_mixed.py
│   │       └── displacement_based.py       # direct u-solve via ∇/∇·, mixed BC via boundary.py, penalty enforcement
│   ├── parabolic/
│   │   └── reaction_diffusion.py           # Allen-Cahn / Cahn-Hilliard time-stepping over elliptic.scalar
│   ├── coupling/
│   │   └── staggered.py                    # generic staggered driver (mechanics ↔ phase-field, thermal ↔ diffusion, ...)
│   └── adjoint/
│       └── implicit_diff.py                # custom_vjp via implicit function theorem — differentiable solve() for free
│
├── materialmodels/
│   ├── base.py                             ✅ ConstitutiveModel ABC (thin — just enough to hold C)
│   ├── assembly.py                         ✅ assemble_C_field(materials, phase) -- hard/sharp-interface
│   │                                       #   per-voxel field, generic over any ConstitutiveModel; built
│   │                                       #   fresh (not ported) once problems/mechanics.py needed it.
│   │                                       #   No smooth/oriented assembler yet (those stay deferred).
│   ├── elastic/
│   │   ├── isotropic.py                    ✅ LinearElasticIsotropic on ConstitutiveModel. mat_models/ (the
│   │   │                                  #   whole package -- elastic.py incl. TransverseIsotropicFibre +
│   │   │                                  #   assemble_C_field_oriented/_smooth, plastic.py, micromechanics.py)
│   │   │                                  #   is DELETED, explicit user choice, "build from scratch not port" --
│   │   │                                  #   see CLAUDE.md. materialmodels/elastic/ is isotropic-only for now;
│   │   │                                  #   TransverseIsotropicFibre has NO new-layout home yet.
│   │   │                                  #   src/problems/mechanics.py FIXED (assemble_C_field rebuilt, verified
│   │   │                                  #   end-to-end with a synthetic two-phase problem). Still KNOWN BROKEN
│   │   │                                  #   (need TransverseIsotropicFibre and/or smooth/oriented assembly,
│   │   │                                  #   not yet rebuilt): scripts/simulation/elastic_nw_cg_strain.py,
│   │   │                                  #   test/test_transverse_isotropic_orientation.py. Trivially fixable
│   │   │                                  #   now but NOT yet touched (stayed scoped to what was asked):
│   │   │                                  #   test/test_problems_mechanics.py (only needs LinearElasticIsotropic's
│   │   │                                  #   import path swapped). test/test_materialmodels_elastic_isotropic.py
│   │   │                                  #   and test/test_elliptic_vector_lippmann_schwinger.py's composite-RVE
│   │   │                                  #   check both compared against mat_models as ground truth -- that
│   │   │                                  #   premise is gone with the module; need repurposing, not a swap.
│   │   │                                  #   notebooks/lin-elastic_strain.ipynb (outputs still display fine,
│   │   │                                  #   breaks on next run)
│   │   ├── orthotropic.py
│   │   ├── transversely_isotropic.py       # NOT built -- was about to be a verbatim port, explicitly rejected;
│   │   │                                  #   next attempt should derive fresh, not copy mat_models/elastic.py
│   │   └── anisotropic.py
│   ├── inelastic/
│   │   ├── plasticity_j2.py
│   │   └── damage.py
│   ├── thermal/
│   │   ├── isotropic.py
│   │   └── orthotropic.py
│   ├── diffusion/
│   │   ├── fickian.py                      # linear Fick's law, isotropic/orthotropic D
│   │   └── nonlinear.py                    # concentration-dependent D(c) — needs Newton-outer/CG-inner, not drop-in linear
│   ├── phasefield/
│   │   ├── degradation.py                  # g(φ) AT1/AT2, spatially varying Gc
│   │   └── regularization.py               # ℓ-dependent gradient-energy term
│   ├── averaging.py                        ⚙️ VoxelAveraging ABC: at minimum ArithmeticAveraging (needed at interfaces)
│   ├── tensors.py                          ⚙️ Voigt/Mandel ↔ tensor (4th-order C, 2nd-order ε/σ), rotation, symmetry checks
│   └── utils.py                            # phase-fraction computation, raw geometry measurement (true utils)
│
├── problems/                               # thin wiring layer: pick strategies, build C(x), average, solve, unpack
│   ├── mechanics.py                        ✅ solve_mechanics: formulation="lippmann_schwinger" done,
│   │                                       #   "displacement" raises NotImplementedError (no solver yet);
│   │                                       #   reference-medium averaging is a plain Lame-parameter mean,
│   │                                       #   placeholder for materialmodels/averaging.py
│   ├── thermal.py
│   ├── diffusion.py
│   └── fracture.py                         # staggered mechanics + phase-field
│
├── microstructure/                         # RVE/RSA/Matérn cluster generation (existing) — only if generating own RVEs
│
├── utils/
│   ├── io/
│   │   ├── xdmf_writer.py                  # incremental HDF5 + growing .xdmf temporal index, streaming per-step
│   │   └── checkpoint.py                   # solver-state restart
│   └── logging.py
│
└── tests/
    ├── operators/
    │   └── boundary/
    ├── solvers/
    │   ├── elliptic/vector/                ⚙️ validate against known analytical case (e.g. two-phase laminate bound)
    │   └── adjoint/                        # jax.grad vs. finite-difference
    ├── materialmodels/
    │   ├── elastic/  
    │   ├── inelastic/  
    │   ├── thermal/  
    │   ├── diffusion/  
    │   └── phasefield/
    │       └── tensors/                        ⚙️ Voigt round-trip tests
    └── utils/io/
```
