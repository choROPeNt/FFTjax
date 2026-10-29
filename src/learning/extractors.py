"""Structure-property extractors: (phase, materials, n, L) -> one scalar property via one FFT solve."""

import jax.numpy as jnp

from problems.mechanics import solve_mechanics


def effective_modulus(
    phase: jnp.ndarray,
    materials: list,
    n: tuple,
    L: tuple,
    component: tuple[int, int] = (0, 0),
    eps0: float = 1.0e-3,
    toler_lin: float = 1e-6,
    maxiter: int = 200,
) -> tuple[float, dict[int, float], bool]:
    """
    Engineering modulus E_ii and Poisson's ratios from one uniaxial test:
    eps_ii = eps0, the other normal stresses zero (free surfaces), unlike
    the stiffer pure-strain coefficient.

    Parameters
    ----------
    phase, materials, n, L : as solve_mechanics
    component : (i, i); shear is not supported
    eps0      : applied strain
    toler_lin, maxiter : CG tolerance and cap

    Returns
    -------
    E         : sigma_ii / eps_ii
    nu        : {j: -eps_jj / eps_ii} for the other directions
    converged : not checked here, so a sweep doesn't stop mid-loop
    """
    i, j = component
    if i != j:
        raise ValueError(f"component must be a diagonal (i, i) index, got {component}")

    control = tuple(
        tuple(1 if (r == c and r != i) else 0 for c in range(3))
        for r in range(3)
    )
    stress_goal = jnp.zeros((3, 3))
    eps_bar = jnp.zeros((3, 3)).at[i, i].set(eps0)

    results = solve_mechanics(
        n, L, phase, materials, eps_bar,
        formulation="displacement", control=control, stress_goal=stress_goal,
        toler_lin=toler_lin, maxiter=maxiter,
    )
    sol = results[0].solution

    eps_ii = float(sol.eps_bar[i, i])
    E = float(jnp.mean(sol.sigma[i, i])) / eps_ii
    nu = {d: float(-sol.eps_bar[d, d] / eps_ii) for d in range(3) if d != i}
    return E, nu, bool(sol.converged)
