# Inverse Calibration — Recovering the Matrix Modulus

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/choROPeNt/FFTjax/blob/main/notebooks/lin-elastic_inverse-calibration.ipynb)

Given a composite's macroscopic (homogenized) stress-strain response, can we recover an unknown
constituent property by matching FFTjax's forward FFT-homogenization model to that curve? This
example calibrates a glass-fiber/epoxy RVE's matrix Young's modulus $E_m$ against a synthetic
shear stress-strain curve, treating everything else (fiber modulus, both Poisson's ratios, the
microstructure itself) as known.

The forward model — solve the FFT elastic equilibrium problem at a given $E_m$, read off the
homogenized shear stress — is an ordinary JAX function, so the natural approach is **gradient-based
optimization**: differentiate the misfit between simulated and "observed" stress through the whole
FFT solve via `jax.jacfwd`, then step $E_m$ downhill with `optax`.

## Why forward-mode (`jax.jacfwd`), not `jax.grad`

`solvers.krylov.cg.cg_solve` (the production CG solve every solver in this project uses) is built
on `jax.lax.while_loop` for early-exit speed. Reverse-mode autodiff (`jax.grad`) has no rule for
`while_loop` and silently gives a **wrong** gradient here — verified in
[`notebooks/archiv/lin-elastic_gradient-check.ipynb`](https://github.com/choROPeNt/FFTjax/blob/main/notebooks/archiv/lin-elastic_gradient-check.ipynb),
off by 10 orders of magnitude on this exact class of operator. That notebook's fix
(`solvers.krylov.cg.cg_solve_scan`, a `lax.scan`-based CG with a well-defined reverse-mode rule) is
one option, but not needed here: **forward-mode** (`jax.jvp`/`jax.jacfwd`) differentiates through
`while_loop` correctly regardless. Since we're calibrating a single scalar ($E_m$), forward-mode is
also the cheaper tool anyway (cost scales with input count, not output count) — so this example
uses the fast production `cg_solve` directly, no special CG variant needed.

```python
import jax
import jax.numpy as jnp
import numpy as np
import optax

from generation.rve import make_random_composite_rve
from operators.green import build_freq_grid, build_green_operator
from solvers.krylov.cg import cg_solve

# Randomly-packed RVE, phi~0.55 -- small on purpose: the calibration loop below
# calls the forward solver many times (once per strain level, per optimizer
# step, per forward-mode probe).
phase_np, n, L, phi_act, centres = make_random_composite_rve(
    phi=0.55, r_fiber=0.0035, dx=0.0006, size_in_r=10, nz=1, K=15, seed=67,
)
Nv = int(np.prod(n))
dx_vox = tuple(Li / ni for Li, ni in zip(L, n))
phase = jnp.array(phase_np.reshape(-1))
xi_flat = build_freq_grid(n, L)

nu_matrix, nu_fiber = 0.35, 0.20
E_fiber = 70.0e3   # MPa, known/fixed -- only E_matrix is unknown

I2 = jnp.eye(3)
def lame(E, nu):
    return E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)), E / (2.0 * (1.0 + nu))
def stiffness(lam, mu):
    return (lam * jnp.einsum('ij,kl->ijkl', I2, I2)
            + mu * (jnp.einsum('ik,jl->ijkl', I2, I2) + jnp.einsum('il,jk->ijkl', I2, I2)))
def fft_(x):
    s = x.shape
    return jnp.fft.fftn(x.reshape(s[:-1] + n), axes=(-3, -2, -1)).reshape(s)
def ifft_(x):
    s = x.shape
    return jnp.fft.ifftn(x.reshape(s[:-1] + n), axes=(-3, -2, -1)).real.reshape(s)

# solve_elastic_diff rebuilds C_field, the reference-medium Green's operator,
# and the Lippmann-Schwinger operator from scratch for a given E_matrix --
# everything downstream is plain differentiable jax.numpy, so jax.jacfwd
# traces all the way through.
def solve_elastic_diff(E_matrix, gamma, maxiter=100, toler_lin=1e-6):
    lam_m, mu_m = lame(E_matrix, nu_matrix)
    lam_f, mu_f = lame(E_fiber, nu_fiber)
    C_matrix = stiffness(lam_m, mu_m)
    C_fiber  = stiffness(lam_f, mu_f)
    C_field = ((1 - phase)[None, None, None, None, :] * C_matrix[..., None]
               + phase[None, None, None, None, :] * C_fiber[..., None])

    lam0 = 0.5 * (lam_m + lam_f)
    mu0  = 0.5 * (mu_m + mu_f)
    G_glob = build_green_operator(xi_flat, lam0, mu0, scheme="rotated", dx=dx_vox)
    eps_bar = jnp.array([[0., gamma / 2, 0.], [gamma / 2, 0., 0.], [0., 0., 0.]])

    def A_op(v_flat):
        v = v_flat.reshape(3, 3, Nv)
        Cv  = jnp.einsum("ijklm,klm->ijm", C_field, v)
        GCv = jnp.einsum("ijklm,klm->ijm", G_glob, fft_(Cv))
        return ifft_(GCv).reshape(-1)

    eps0   = jnp.ones((3, 3, Nv)) * eps_bar[:, :, None]
    sigma0 = jnp.einsum("ijklm,klm->ijm", C_field, eps0)
    bb = -ifft_(jnp.einsum("ijklm,klm->ijm", G_glob, fft_(sigma0))).reshape(-1)
    x0 = jnp.zeros_like(bb)

    delta_flat, converged = cg_solve(A_op, bb, x0, toler_lin, maxiter)
    eps = eps0 + delta_flat.reshape(3, 3, Nv)
    sigma = jnp.einsum("ijklm,klm->ijm", C_field, eps)
    return jnp.mean(sigma[0, 1]), converged

def tau_xy(E_matrix, gamma):
    tau, _ = solve_elastic_diff(E_matrix, gamma)
    return tau

# Synthetic "observed" stress-strain curve from a known ground truth --
# calibration below pretends not to know E_matrix_true.
E_matrix_true = 3.5e3
gammas_obs = jnp.array([2.0e-3, 4.0e-3, 6.0e-3, 8.0e-3, 1.0e-2])
tau_obs, converged_obs = jax.vmap(lambda g: solve_elastic_diff(E_matrix_true, g))(gammas_obs)

# Least-squares misfit, differentiated with jax.jacfwd through the vmapped
# batch of forward solves; optax.adam with a decaying learning rate takes it
# from there. Starting ~43% off the true value, so this is a real optimization.
def loss(E_matrix):
    tau_pred, _ = jax.vmap(lambda g: solve_elastic_diff(E_matrix, g))(gammas_obs)
    return jnp.sum((tau_pred - tau_obs) ** 2)

loss_and_grad = jax.jit(lambda E: (loss(E), jax.jacfwd(loss)(E)))

E_m = jnp.array(5.0e3)
schedule = optax.exponential_decay(init_value=60.0, transition_steps=15, decay_rate=0.7)
optimizer = optax.adam(learning_rate=schedule)
opt_state = optimizer.init(E_m)

for step in range(70):
    loss_val, grad_val = loss_and_grad(E_m)
    updates, opt_state = optimizer.update(grad_val, opt_state, E_m)
    E_m = optax.apply_updates(E_m, updates)

E_matrix_calibrated = float(E_m)
```

```text title="Output"
grid n : (75, 78, 1)
total voxels Nv: 5850
fiber volume fraction (actual): 0.5508

tau_xy at E_matrix=3500 MPa: 22.620570
d(tau_xy)/d(E_matrix)  jax.jvp            : 4.886294e-03
d(tau_xy)/d(E_matrix)  central finite diff : 4.886295e-03
relative difference: 7.888e-08  (expect ~1e-6 to 1e-10, FD is only 1st-order accurate)
PASSED -- forward-mode gradient matches finite difference.

step   0   E_matrix =  4940.00 MPa   loss = 4.1057e+02
step  10   E_matrix =  4421.22 MPa   loss = 1.7869e+02
step  20   E_matrix =  4047.88 MPa   loss = 6.6496e+01
step  30   E_matrix =  3804.57 MPa   loss = 2.1322e+01
step  40   E_matrix =  3661.67 MPa   loss = 6.1383e+00
step  50   E_matrix =  3585.12 MPa   loss = 1.7114e+00
step  60   E_matrix =  3546.71 MPa   loss = 5.1080e-01
step  69   E_matrix =  3529.31 MPa   loss = 1.9788e-01

true E_matrix       : 3500.00 MPa
calibrated E_matrix : 3529.31 MPa
relative error      : +0.8375%

held-out gamma = 0.012:
  tau (true E_matrix)       : 54.2894 MPa
  tau (calibrated E_matrix) : 54.6326 MPa
  relative error            : +0.6322%
```

![Inverse calibration RVE](/img/lin_elastic_inverse_calibration_rve.png)

![Inverse calibration convergence and fit](/img/lin_elastic_inverse_calibration_fit.png)

Recovered `E_matrix` to within 0.84% of the true value from an initial guess ~43% off, with a
smooth, monotonic loss curve — and the calibrated model still matches within 0.63% at a strain
level (`gamma = 0.012`) never used during fitting, a real generalization check, not just
interpolation between the fitted points.

:::note[Reproducing]
This page's code, output, and plots are generated by
[`notebooks/lin-elastic_inverse-calibration.ipynb`](https://github.com/choROPeNt/FFTjax/blob/main/notebooks/lin-elastic_inverse-calibration.ipynb)
(linked via the Colab badge above) — there is no standalone `examples/` script for this case,
since it would just be a redundant duplicate of the notebook.

If the notebook changes, re-run it and update the pasted output/images above — there's no
build-time execution here, since Docusaurus can't run Python.
:::

## Next steps

- Calibrate more than one parameter at once (e.g. matrix `E` and `nu` together) — `jax.jacfwd`'s
  cost grows with the number of *inputs*, so this stays cheap for a handful of parameters before
  reverse-mode (via `cg_solve_scan`) becomes the better tool.
- Calibrate the nonlinear J2 plasticity parameters (`sigma_y0`, `H`) from a hysteresis curve like
  [In-Elastic Solve](./inelastic-j2.md)'s — same mechanism, just a nonlinear forward model.
- Add synthetic noise to the "observed" curve and check calibration robustness — this notebook's
  curve is noise-free by construction, so recovery is limited only by optimizer convergence, not
  measurement uncertainty.
- For an expensive or noisy forward model where computing gradients at all is impractical, see
  `notebooks/archiv/01_gp_bo_latent-uncertainty_1d.ipynb`'s Gaussian-process/Bayesian-optimization
  approach (`gpjax`) instead — a black-box surrogate with built-in uncertainty on the calibrated
  parameter, at the cost of needing many more forward solves than a gradient-based method.
