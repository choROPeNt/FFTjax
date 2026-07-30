# 📖 Theorie

FFTjax solves variational partial differential equations on voxelized domains using a spectral Fourier-based approach.

## Spectral solution of PDEs

Consider a generic equilibrium problem in small-strain elasticity:


$$
\nabla \cdot \sigma(\mathbf{x}) = 0
$$

with constitutive relation

$$
\sigma = \mathbb{C} : \varepsilon, 
\qquad 
\varepsilon = \frac{1}{2}(\nabla u + \nabla u^\top)
$$

In the spectral approach, fields are transformed into Fourier space:

$$
\hat{f}(\mathbf{k}) = \mathcal{F}[f(\mathbf{x})]
$$

Spatial derivatives become multiplications:

$$
\nabla \rightarrow i\mathbf{k}
$$



which converts differential operators into algebraic expressions. The equilibrium equation is then solved iteratively in Fourier space using a Green operator formulation, enabling efficient convolution-based updates.

The computational complexity scales as $\mathcal{O}(N \log N)$ due to the use of Fast Fourier Transforms.

---

## Mechanical solver formulations

FFTjax provides two Newton-CG solvers for the mechanical equilibrium problem (`solvers/mechanical/`), differing in which field is the CG unknown.

**Strain-based (`strain_nw_cg.py`, Vondřejc et al. 2014 / Lucarini & Segurado 2018).**
The unknown is the periodic strain fluctuation $\Delta\varepsilon$. Equilibrium is enforced through a fixed reference-medium Green's operator $\hat{\Gamma}_0(\boldsymbol{\xi})$ (isotropic, "standard" or Willot "rotated" discretisation):

$$
A(\Delta\varepsilon) = \mathcal{F}^{-1}\!\big[\hat{\Gamma}_0 : \mathcal{F}[\mathbb{C}:\Delta\varepsilon]\big], \qquad
b = -\mathcal{F}^{-1}\!\big[\hat{\Gamma}_0 : \mathcal{F}[\mathbb{C}:\varepsilon_0 - \sigma_{\text{goal}}]\big]
$$

$\hat{\Gamma}_0$ only ever appears in *even* powers of $\boldsymbol{\xi}$ (e.g. $\xi_i\xi_j$), which is what makes the CG operator provably symmetric positive-definite for arbitrarily heterogeneous $\mathbb{C}(\mathbf{x})$, and is why it needs no extra preconditioning beyond the reference medium itself.

**Displacement-based (`displacement_nw_cg.py`).**
The unknown is instead the periodic displacement fluctuation $\hat{u}$ itself, with the strain built directly from its symmetric gradient in Fourier space, $\hat{\varepsilon} = \tfrac{1}{2}(i\boldsymbol{\xi}\otimes\hat{u} + \text{sym.})$, and the true (possibly heterogeneous) tangent $\mathbb{C}(\mathbf{x})$ applied with **no reference-medium approximation**. Because $\boldsymbol{\xi}$ now appears to an *odd* power, the Nyquist frequency component must be zeroed on even grid dimensions to preserve the Hermitian symmetry a real-input FFT requires — an FFT-Galerkin-specific correction not needed by the Green's-operator formulation. Convergence for realistic (high-contrast) composites additionally needs a frequency-domain preconditioner $(\boldsymbol{\xi}\cdot\mathbb{C}_0\cdot\boldsymbol{\xi})^{-1}$ built from a reference stiffness $\mathbb{C}_0$.

**Mixed strain/stress boundary conditions.**
Both formulations can prescribe macroscopic strain on some tensor components and macroscopic stress on others (e.g. a free-lateral-surface uniaxial-stress test), selected via a `(3,3)` `control` mask. The two schemes solve this differently:

- The *displacement-based* solver embeds the stress-controlled average-strain correction directly as an extra unknown in the zero-frequency mode of the strain field, coupled to the fluctuation field through the same symmetric CG operator — valid for arbitrary (including heterogeneous) materials.
- The *strain-based* solver can instead reclaim the Green's operator's own (otherwise always-zero) DC block for this purpose (`dstrain_nw_cg_mixed`), which is cheaper but only symmetric — and therefore only a valid one-shot CG solve — for **homogeneous** materials; for heterogeneous materials the mixed-BC CG operator loses symmetry and the displacement-based solver must be used instead.

---

## Variational formulation

The solver is based on an energy minimization principle. For elasticity and phase-field fracture, the total energy functional reads

$$
\Pi(u, d) =
\int_\Omega g(d)\,\psi_e(\varepsilon(u)) \, d\Omega
+
\int_\Omega G_c \left(
\frac{d^2}{2\ell} + \frac{\ell}{2} |\nabla d|^2
\right) d\Omega
$$

where

- $u$ is the displacement field  
- $d$ is the phase-field variable  
- $g(d)$ is the degradation function  
- $\mathcal{G}_c$ is the fracture toughness  
- $\ell$ is the length scale parameter  

---

## Staggered solution scheme

A variational staggered scheme is employed:

1. **Elastic step**  
   Minimize $\Pi(u, d^{n})$ with respect to $u$
   → Linear/nonlinear equilibrium solved via spectral operator.

$$
A_u \, u = b_u ,
$$

where $A_u$ represents the spectral stiffness operator.
The system is solved iteratively using the Conjugate Gradient (CG) method, exploiting the matrix-free application of $A_u$ in Fourier space.

2. **Phase-field step**  
   Minimize $\Pi(u^{n+1}, d)$ with respect to  $d$  
   → Helmholtz-type equation solved in Fourier space.
$$
A_d \, d = b_d ,
$$
which is likewise solved using CG with spectral evaluation of the differential operators.

The two fields are updated alternately until convergence of the coupled system.

---

## Image-based discretization

The computational domain is defined on a regular voxel grid derived from segmented experimental data. Material heterogeneity is directly assigned per voxel, avoiding geometric idealization and enabling direct microstructure-to-simulation coupling.

Periodic boundary conditions are naturally satisfied within the spectral framework.

---

This formulation enables:

- Efficient voxel-scale simulations
- Direct use of image-based microstructures
- Differentiability for inverse parameter identification
- GPU acceleration via JAX