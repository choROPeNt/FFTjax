"""
Property extractors: (phase, materials, n, L) -> a scalar structure-property
value, via one in-process FFT solve. Structure-property sweeps and
surrogate-fitting workflows (e.g. notebooks/structure-property_phi-sweep.ipynb)
call one of these instead of each hand-rolling its own boundary-condition
and homogenization arithmetic -- see learning.surrogates.GPSurrogate for the
matching surrogate-fitting half.
"""

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
    Effective engineering modulus E_ii and the associated transverse
    Poisson's ratios, from one mixed-BC displacement solve: direction `i`
    strain-controlled to `eps0`, every other diagonal direction stress-
    controlled to zero (free lateral surfaces) -- a real uniaxial tensile
    test, not the stiffer constrained coefficient a pure-strain BC would
    give. Same convention as notebooks/lin-elastic_mixed-BC.ipynb.

    Parameters
    ----------
    phase, materials, n, L : see problems.mechanics.solve_mechanics
    component : (i, i) diagonal index to load -- only i == j is supported;
                a shear/off-diagonal probe has no free-surface Poisson's-
                ratio pair to report, use solve_mechanics directly for that
    eps0      : probe strain magnitude on the loaded direction
    toler_lin, maxiter : mechanical CG tolerance / iteration cap

    Returns
    -------
    E         : float -- sigma_ii / eps_ii at the loaded direction
    nu        : dict[int, float] -- {j: -eps_jj / eps_ii} for every other
                diagonal j (e.g. {1: nu_12, 2: nu_13} when component=(0, 0))
    converged : bool -- the mechanical CG solve's own convergence flag;
                NOT checked here, since a one-shot property extractor used
                inside a sweep shouldn't raise mid-loop -- callers that care
                (e.g. a surrogate fit) should filter on it themselves
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
