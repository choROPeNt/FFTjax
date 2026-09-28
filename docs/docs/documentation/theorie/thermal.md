# Thermal Conduction

`solvers.elliptic.scalar.solve_thermal_conduction` solves steady-state heat conduction on a
periodic voxel grid — the scalar analogue of the [displacement-based elastic
solve](mechanical#three-formulations-one-problem), mirrored line-for-line rather than re-derived:
the unknown rank drops from a displacement vector to a scalar temperature, so a symmetrized strain
becomes a plain gradient, and the `(3,3)` strain-BC `control`/rank-4 stiffness become a `(3,)`
gradient-BC `control`/rank-2 conductivity. `problems/thermal.py` is the thin wiring layer, mirroring
`problems/mechanics.py`.

## Governing equation

The temperature field decomposes as $T(\mathbf{x}) = \overline{\nabla T}\cdot\mathbf{x} +
T'(\mathbf{x})$, with $T'$ periodic (zero mean — a gauge choice; only $\nabla T'$ is physically
meaningful, same reasoning `post.fields.compute_displacement` handles on the elasticity side).
Fourier's law and equilibrium (no heat source) give

$$
\mathbf{q} = -\mathbb{K}(\mathbf{x})\cdot\nabla T, \qquad
\nabla\!\cdot\!\big(\mathbb{K}(\mathbf{x})\cdot\nabla T'\big) = -\nabla\!\cdot\!\big(\mathbb{K}(\mathbf{x})\cdot\overline{\nabla T}\big)
$$

solved for the periodic fluctuation $T'$ by CG, with the true (possibly heterogeneous)
conductivity field $\mathbb{K}(\mathbf{x})$ applied directly — there is no reference-medium
Lippmann-Schwinger analogue for this equation in this project (or in the FFTMAD reference
implementation's own diffusion module): a heterogeneous $\nabla\!\cdot\!(\mathbb{K}\nabla\,\cdot)$
problem is treated as a direct CG solve either way, so `problems.mechanics`'s `formulation` switch
has no counterpart here.

## Mixed gradient/flux boundary conditions

A macroscopic temperature gradient can be prescribed on some directions and a macroscopic heat
flux on others, via a `(3,)` `control` mask (1 = flux-controlled) — the same DC-bin embedding
trick the displacement-based elastic solver uses for mixed strain/stress BC (see [Mechanical
Solvers](mechanical#mixed-strainstress-boundary-conditions)): the macroscopic-gradient correction
on flux-controlled directions is folded directly into the zero-frequency (DC) Fourier mode of the
gradient field, solved jointly with the temperature fluctuation through one bordered CG system.
The frequency-domain preconditioner is a plain scalar reciprocal per frequency,
$(\boldsymbol\xi\cdot\mathbb{K}_0\cdot\boldsymbol\xi)^{-1}$, rather than a per-frequency
$3\times3$ matrix inverse — the unknown here has one component, not three. As with the
displacement-based solver, the gradient/divergence are odd powers of $\boldsymbol\xi$ and need
`nyquist_safe_xi` (see [Frequency-Space Operators](operators#the-nyquist-gotcha)).

## Conductivity models

`materialmodels/thermal/` implements `ConductivityModel`'s `conductivity_tensor()`, the rank-2
analogue of `ConstitutiveModel`:

- **`ThermalConductivityIsotropic`** — a single conductivity constant $k$, $\mathbb{K} =
  k\,\mathbf{I}$.
- **`ThermalConductivityTransverseIsotropic`** — a fiber-reinforced material with only **2**
  independent constants ($k_L$ along the fiber, $k_T$ transverse), versus the elastic case's 5: a
  rank-2 tensor's transverse plane is isotropic with nothing left to independently constrain once
  $k_T$ is fixed — there is no conductivity analogue of a Poisson-ratio-style coupling term. Mirrors
  `TransverseIsotropic`'s `fiber_dir` rotation machinery (one direction, or a per-voxel orientation
  field), via a plain rank-2 similarity transform $\mathbb{K}' = R\,\mathbb{K}\,R^\top$ rather than
  the rank-4 rotation the elastic case needs.

## Solution fields

`ThermalSolution` carries `grad_T`, `flux = -K:grad_T`, and the periodic fluctuation `T_prime` —
**not** the absolute temperature field, which also needs the macroscopic linear part
$\overline{\nabla T}\cdot\mathbf{x}$; reconstructing it is a gauge-free choice (an additive
constant is arbitrary) not yet built here, mirroring why `post.fields.compute_displacement` exists
as a separate step on the elasticity side.
