FFTjax solves variational partial differential equations on voxelized domains using a spectral Fourier-based


### Spectral solution of PDEs

Consider a generic equilibrium problem in small-strain elasticity:


\[
\nabla \cdot \sigma(\mathbf{x}) = 0
\]

with constitutive relation

\[
\sigma = \mathbb{C} : \varepsilon, 
\qquad 
\varepsilon = \frac{1}{2}(\nabla u + \nabla u^\top)
\]

In the spectral approach, fields are transformed into Fourier space:

\[
\hat{f}(\mathbf{k}) = \mathcal{F}[f(\mathbf{x})]
\]

Spatial derivatives become multiplications:

\[
\nabla \rightarrow i\mathbf{k}
\]



which converts differential operators into algebraic expressions. The equilibrium equation is then solved iteratively in Fourier space using a Green operator formulation, enabling efficient convolution-based updates.

The computational complexity scales as \( \mathcal{O}(N \log N) \) due to the use of Fast Fourier Transforms.

---

### Variational formulation

The solver is based on an energy minimization principle. For elasticity and phase-field fracture, the total energy functional reads

\[
\Pi(u, d) =
\int_\Omega g(d)\,\psi_e(\varepsilon(u)) \, d\Omega
+
\int_\Omega G_c \left(
\frac{d^2}{2\ell} + \frac{\ell}{2} |\nabla d|^2
\right) d\Omega
\]

where

- \( u \) is the displacement field  
- \( d \) is the phase-field variable  
- \( g(d) \) is the degradation function  
- \( \mathcal{G}_c \) is the fracture toughness  
- \( \ell \) is the length scale parameter  

---

### Staggered solution scheme

A variational staggered scheme is employed:

1. **Elastic step**  
   Minimize \( \Pi(u, d^{n}) \) with respect to $u$
   → Linear/nonlinear equilibrium solved via spectral operator.

\[
A_u \, u = b_u ,
\]

where \( A_u \) represents the spectral stiffness operator.
The system is solved iteratively using the Conjugate Gradient (CG) method, exploiting the matrix-free application of \( A_u \) in Fourier space.

2. **Phase-field step**  
   Minimize \( \Pi(u^{n+1}, d) \) with respect to  $d$  
   → Helmholtz-type equation solved in Fourier space.
\[
A_d \, d = b_d ,
\]
which is likewise solved using CG with spectral evaluation of the differential operators.

The two fields are updated alternately until convergence of the coupled system.

---

### Image-based discretization

The computational domain is defined on a regular voxel grid derived from segmented experimental data. Material heterogeneity is directly assigned per voxel, avoiding geometric idealization and enabling direct microstructure-to-simulation coupling.

Periodic boundary conditions are naturally satisfied within the spectral framework.

---

This formulation enables:

- Efficient voxel-scale simulations
- Direct use of image-based microstructures
- Differentiability for inverse parameter identification
- GPU acceleration via JAX