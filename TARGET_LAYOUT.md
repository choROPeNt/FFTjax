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
│   └── fft_utils.py                        ⚙️ util: rfftn/irfftn wrappers, raw k-vector construction (no physics, no scheme choice)
│
├── solvers/
│   ├── __init__.py
│   ├── base.py                             # shared convergence criteria, iteration bookkeeping
│   ├── krylov/
│   │   └── cg.py                           ⚙️ operator-agnostic (P)CG driver, shared by every elliptic PDE-type
│   ├── elliptic/
│   │   ├── scalar.py                       # thermal, diffusion (Fickian), phase-field φ-subproblem
│   │   └── vector/
│   │       ├── base.py                     ⚙️ ElasticitySolver ABC, ElasticitySolution dataclass
│   │       ├── lippmann_schwinger.py       ⚙️ strain-based, periodic (extend for DCT/DST non-periodic — TO DISCUSS)
│   │       └── displacement_based.py       # direct u-solve via ∇/∇·, mixed BC via boundary.py, penalty enforcement
│   ├── parabolic/
│   │   └── reaction_diffusion.py           # Allen-Cahn / Cahn-Hilliard time-stepping over elliptic.scalar
│   ├── coupling/
│   │   └── staggered.py                    # generic staggered driver (mechanics ↔ phase-field, thermal ↔ diffusion, ...)
│   └── adjoint/
│       └── implicit_diff.py                # custom_vjp via implicit function theorem — differentiable solve() for free
│
├── materialmodels/
│   ├── base.py                             ⚙️ ConstitutiveModel ABC (thin — just enough to hold C)
│   ├── elastic/
│   │   ├── isotropic.py                    ⚙️ start with one symmetry class
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
│   ├── mechanics.py                        ⚙️ formulation="lippmann_schwinger" | "displacement"
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
