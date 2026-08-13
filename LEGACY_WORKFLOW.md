# Legacy Linear-Elastic FFT Workflow (pre-refactor reference)

This document freezes a description of FFTjax's original linear-elastic solve pipeline, as it
existed prior to the `refactor/target-layout` restructuring (see `TARGET_LAYOUT.md`). It exists so
the method can be cited/described in a paper independent of ongoing source-layout changes — the
numerics described here are unchanged by the refactor (the new class-based implementation in
`solvers/elliptic/vector/lippmann_schwinger.py` was verified bit-identical to it before it was
retired), only the code organization differs.

## 1. Problem statement

For a periodic representative volume element (RVE) $\Omega$ with heterogeneous stiffness
$\mathbb{C}(\mathbf{x})$, the mechanical equilibrium problem under a prescribed macroscopic strain
$\bar{\varepsilon}$ is

$$
\nabla \cdot \sigma(\mathbf{x}) = 0, \qquad
\sigma = \mathbb{C}(\mathbf{x}):\varepsilon, \qquad
\varepsilon = \tfrac{1}{2}\big(\nabla u + \nabla u^\top\big), \qquad
\langle \varepsilon(\mathbf{x}) \rangle_\Omega = \bar{\varepsilon}.
$$

## 2. Lippmann-Schwinger reformulation

Following Moulinec & Suquet (1994, 1998), introduce a homogeneous reference medium
$\mathbb{C}_0$ (typically the average of the constituents' stiffnesses) and its periodic Green's
operator $\Gamma_0$. The equilibrium problem is equivalent to the fixed-point (Lippmann-Schwinger)
integral equation for the strain field:

$$
\varepsilon(\mathbf{x}) + \Gamma_0 * \big[(\mathbb{C}(\mathbf{x}) - \mathbb{C}_0):\varepsilon(\mathbf{x})\big] = \bar{\varepsilon}.
$$

$\Gamma_0$ is evaluated in Fourier space in closed form for an isotropic reference medium (DBFFT /
FFTMAD convention):

$$
\hat{K}_{ik}(\xi) = \frac{\delta_{ik} - c\, \hat{n}_i \hat{n}_k}{\mu_0}, \qquad
c = \frac{\lambda_0 + \mu_0}{\lambda_0 + 2\mu_0}, \qquad
\hat{\Gamma}_{0,ijkl}(\xi) = \tfrac{1}{4}\big(\hat{K}_{ik}\hat{n}_j\hat{n}_l + \text{3 sym. terms}\big),
$$

where $\hat{n} = \xi / |\xi|$ and $\hat{\Gamma}_0(0) = 0$ (the mean strain is prescribed
separately, not solved for). This closed form is exactly degree-0 homogeneous in $\xi$ — it
depends only on direction, not magnitude, because it is the fused double-symmetric-gradient of the
reference medium's Green's function; the $1/|\xi|^2$ singularity of the Green's function is
canceled by the $|\xi|^2$ from the two gradients.

Two discretizations of $\xi$ are used:

- **`standard`** — the continuous DFT frequencies (equivalent to Moulinec & Suquet's original
  discretization; prone to Gibbs oscillations and a 45° anisotropy bias at high stiffness
  contrast).
- **`rotated`** (Willot 2015) — effective frequencies
  $\xi_j^{\text{eff}} = (2/h_j)\sin(\xi_j h_j / 2)$, equivalent to linear hexahedral finite
  elements with reduced integration (Schneider et al. 2016). Removes the anisotropy bias and
  converges markedly better under high stiffness contrast — used throughout this project's
  examples.

## 3. Krylov (CG) reduction

Trigonometric collocation (Zeman et al. 2010; Vondrejc et al. 2012, electrostatics) turns the
integral equation above into a linear system, solved by conjugate gradients rather than the
original fixed-point "basic scheme" — extended to elasticity by Vondrejc et al. (2014) and further
by Lucarini & Segurado (2018), which is the formulation this workflow implements.

Splitting $\varepsilon = \varepsilon_0 + \Delta\varepsilon$ with $\varepsilon_0 = \bar\varepsilon$
uniform, and using $\Gamma_0 * (\mathbb{C}_0 : \Delta\varepsilon) = \Delta\varepsilon$ for any
zero-mean field together with $\hat\Gamma_0(0) = 0$, the $\mathbb{C}_0$-dependence cancels exactly,
leaving a linear system for the correction alone:

$$
\underbrace{\Gamma_0 * (\mathbb{C} : \Delta\varepsilon)}_{A(\Delta\varepsilon)}
\;=\;
\underbrace{-\,\Gamma_0 * (\mathbb{C} : \varepsilon_0 - \sigma_{\text{goal}})}_{b},
$$

i.e. $A(v) = \mathrm{iFFT}\big(\hat\Gamma_0 : \mathrm{FFT}(\mathbb{C}:v)\big)$, solved by CG against
$b = -\mathrm{iFFT}\big(\hat\Gamma_0 : \mathrm{FFT}(\mathbb{C}:\varepsilon_0 - \sigma_{\text{goal}})\big)$
to give $\Delta\varepsilon$, then $\varepsilon = \varepsilon_0 + \Delta\varepsilon$ and
$\sigma = \mathbb{C}:\varepsilon$. $A$ is symmetric and exactly singular at the zero (DC) frequency
by construction ($\hat\Gamma_0(0)=0$); starting the CG iteration at $x_0 = 0$ never explores that
null direction, so the correction's mean is exactly zero and the macroscopic strain stays pinned
at $\bar\varepsilon$.

## 4. Algorithm summary

1. **Geometry.** Generate or import a phase-labeled voxel grid $(n_x, n_y, n_z)$ over domain
   $L = (L_x, L_y, L_z)$.
2. **Materials.** Assign each phase a constitutive model exposing a 4th-order stiffness tensor
   $\mathbb{C}$; assemble the per-voxel field $\mathbb{C}(\mathbf{x})$ by phase index.
3. **Frequency grid.** Build the angular-frequency grid $\xi = 2\pi k / L$ from the voxel counts
   and domain size.
4. **Reference medium.** Choose $\mathbb{C}_0$ (e.g. the phase-average Lamé parameters
   $\lambda_0, \mu_0$).
5. **Green's operator.** Evaluate $\hat\Gamma_0(\xi)$ pointwise (standard or rotated scheme).
6. **CG solve.** Solve $A(\Delta\varepsilon) = b$ as defined above for the prescribed
   $\bar\varepsilon$ (and optional target stress $\sigma_{\text{goal}}$ for mixed strain/stress
   components).
7. **Post-process.** Recover $\varepsilon$, $\sigma$, the displacement field (by integrating
   $\varepsilon$ in Fourier space), and any derived quantities (von Mises stress, effective
   moduli, ...).

## 5. Reference implementation

The three functions below are the literal legacy implementation (last present at commit `98ff5db~1`
in this repository's history, function bodies unchanged since; retrieve with
`git show 98ff5db~1:src/mat_models/elastic.py` etc. for the complete file including docstrings and
the transversely-isotropic and assembly-helper variants omitted here for brevity).

**Material stiffness** (`mat_models/elastic.py`):

```python
class LinearElasticIsotropic:
    def __init__(self, E: float, nu: float, name: str = ""):
        self.E, self.nu, self.name = float(E), float(nu), name
        self.lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        self.mu  = E / (2.0 * (1.0 + nu))

    def stiffness_tensor(self) -> jnp.ndarray:
        d = jnp.eye(3)
        return (self.lam * jnp.einsum('ij,kl->ijkl', d, d)
                + self.mu * (jnp.einsum('ik,jl->ijkl', d, d)
                             + jnp.einsum('il,jk->ijkl', d, d)))


def assemble_C_field(materials: list, phase: jnp.ndarray) -> jnp.ndarray:
    C_stack = jnp.stack([m.stiffness_tensor() for m in materials], axis=-1)  # (3,3,3,3,n_mats)
    return C_stack[..., phase]  # (3, 3, 3, 3, Nv)
```

**Green's operator** (`operators/green.py`, unchanged by the refactor):

```python
def build_freq_grid(n: tuple[int, ...], L: tuple[float, ...]) -> jnp.ndarray:
    freqs = [jnp.fft.fftfreq(ni, d=Li / ni) * 2.0 * jnp.pi for ni, Li in zip(n, L)]
    grids = jnp.meshgrid(*freqs, indexing='ij')
    return jnp.stack([g.ravel() for g in grids])  # (ndim, Nv)


def build_willot_freq(xi_flat: jnp.ndarray, dx: tuple[float, ...]) -> jnp.ndarray:
    dx_arr = jnp.array(dx)
    return (2.0 / dx_arr[:, None]) * jnp.sin(xi_flat * dx_arr[:, None] / 2.0)


def build_green_operator(xi_flat, lam0, mu0, scheme='standard', dx=None) -> jnp.ndarray:
    xi = build_willot_freq(xi_flat, dx) if scheme == 'rotated' else xi_flat
    xi_sq = jnp.sum(xi ** 2, axis=0)
    safe  = xi_sq > 0
    n_hat = xi / jnp.sqrt(jnp.where(safe, xi_sq, 1.0))[None, :]

    c    = (lam0 + mu0) / (lam0 + 2.0 * mu0)
    d    = jnp.eye(3)
    Kinv = (d[:, :, None] - c * jnp.einsum('iN,jN->ijN', n_hat, n_hat)) / mu0
    nn   = jnp.einsum('iN,jN->ijN', n_hat, n_hat)
    Gamma = 0.25 * (jnp.einsum('ikN,jlN->ijklN', Kinv, nn)
                     + jnp.einsum('ilN,jkN->ijklN', Kinv, nn)
                     + jnp.einsum('jkN,ilN->ijklN', Kinv, nn)
                     + jnp.einsum('jlN,ikN->ijklN', Kinv, nn))
    return jnp.where(safe[None, None, None, None, :], Gamma, 0.0)
```

**Newton-CG solve** (`solvers/mechanical/strain_nw_cg.py`):

```python
def dstrain_nw_cg(n_i, C_field, G_glob, eps_bar, stress_goal=None, toler_lin=1e-4, maxiter=1000):
    Nv = prod(n_i)

    def fft_(x):
        s = x.shape
        return jnp.fft.fftn(x.reshape(s[:-1] + n_i), axes=(-3, -2, -1)).reshape(s)

    def ifft_(x):
        s = x.shape
        return jnp.fft.ifftn(x.reshape(s[:-1] + n_i), axes=(-3, -2, -1)).real.reshape(s)

    def A_op(v_flat):
        v   = v_flat.reshape(3, 3, Nv)
        Cv  = jnp.einsum("ijklm,klm->ijm", C_field, v)
        GCv = jnp.einsum("ijklm,klm->ijm", G_glob, fft_(Cv))
        return ifft_(GCv).reshape(-1)

    sg     = jnp.zeros((3, 3, Nv)) if stress_goal is None else stress_goal
    eps0   = jnp.ones((3, 3, Nv)) * eps_bar[:, :, None]
    sigma0 = jnp.einsum("ijklm,klm->ijm", C_field, eps0)
    bb     = -ifft_(jnp.einsum("ijklm,klm->ijm", G_glob, fft_(sigma0 - sg))).reshape(-1)

    x0 = jnp.zeros_like(bb)
    delta_flat, converged = cg_solve(A_op, bb, x0, toler_lin, maxiter)

    delta = delta_flat.reshape(3, 3, Nv)
    eps   = eps0 + delta
    sigma = jnp.einsum("ijklm,klm->ijm", C_field, eps)
    return eps, sigma, delta, converged
```

`cg_solve` is a thin wrapper around `jax.scipy.sparse.linalg.cg` (early-exit via
`jax.lax.while_loop`) that additionally reports convergence, since JAX's own `info` return was an
unimplemented placeholder at the time.

## 6. References

- Moulinec, H., Suquet, P. (1994). *A fast numerical method for computing the linear and nonlinear
  mechanical properties of composites.* C. R. Acad. Sci. Paris II, 318, 1417–1423.
- Moulinec, H., Suquet, P. (1998). *A numerical method for computing the overall response of
  nonlinear composites with complex microstructure.* Comput. Methods Appl. Mech. Engrg., 157,
  69–94.
- Zeman, J., Vondřejc, J., Novák, J., Marek, I. (2010). *Accelerating a FFT-based solver for
  numerical homogenization of periodic media by conjugate gradients.* J. Comput. Phys., 229,
  8065–8071.
- Vondřejc, J., Zeman, J., Marek, I. (2012). *Guaranteed upper-lower bounds on homogenized
  properties by FFT-based Galerkin method.* (electrostatics formulation).
- Vondřejc, J., Zeman, J., Marek, I. (2014). *An FFT-based Galerkin method for homogenization of
  periodic media.* Comput. Math. Appl., 68, 156–173.
- Willot, F. (2015). *Fourier-based schemes for computing the mechanical response of composites
  with accurate local fields.* C. R. Mécanique, 343, 232–245.
- Schneider, M., Ospald, F., Kabel, M. (2016). *Computational homogenization of elasticity on a
  staggered grid.* Int. J. Numer. Methods Engrg., 105, 693–720.
- Lucarini, S., Segurado, J. (2018). *An algorithm for stress and mixed control in Galerkin-based
  FFT homogenization.* Int. J. Numer. Methods Engrg., 119, 797–805.
- Schneider, M., Kästner, M. (2024) — this project's benchmark/validation reference,
  doi:10.1111/ffe.14553.

## 7. Status on `refactor/target-layout`

Superseded, not deleted-without-trace: `solvers/elliptic/vector/lippmann_schwinger.py`'s
`solve_lippmann_schwinger`/`LippmannSchwingerSolver` implement the identical algorithm above on a
`LinearOperator`-composed stack (`GreenOperatorBasic`/`GreenOperatorWillot` +
`Gamma0Operator`), verified bit-identical to `dstrain_nw_cg` before the latter was retired. See
`TARGET_LAYOUT.md` for the current module layout and what remains unbuilt (displacement-based
formulation, mixed BC, phase-field damage).
