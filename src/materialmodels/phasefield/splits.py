"""
Crack-driving-force strain-energy splits for phase-field fracture -- one
formula per split, defined once here so every consumer stays in sync:
materialmodels.phasefield.driving_force's field-level API (for materials
degraded uniformly via materialmodels.phasefield.degradation.
degrade_stiffness_field) and materialmodels.phasefield.isotropic.
PhaseFieldIsotropic's per-voxel autodiff path (which differentiates a split
directly to get its tangent) both call the same single-voxel function
rather than each carrying their own copy of the formula.

Only Amor et al. (2009)'s volumetric-deviatoric split is implemented -- it's
what the single-notch-plate benchmark (benchmark/benchmark_1/
pff_single_notch.py) uses. Add a Miehe et al. (2010) spectral split here,
alongside it, if a future problem needs one -- it separates tension/
compression by strain eigenvalue rather than only by the sign of the trace,
which matters for stress states Amor's split can't distinguish (e.g. biaxial
tension-compression), at the cost of an eigendecomposition per voxel instead
of a closed form.
"""

import jax.numpy as jnp


def amor_split(
    eps: jnp.ndarray,
    lam: float | jnp.ndarray,
    mu: float | jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Volumetric-deviatoric energy split (Amor et al. 2009), one voxel:
    (3,3), scalar, scalar -> (psi_pos, psi_neg).

        K       = lam + 2*mu/3                     (3-D bulk modulus)
        psi_pos = K/2 * <tr(eps)>_+^2  +  mu * eps_dev : eps_dev
        psi_neg = K/2 * <tr(eps)>_-^2

    where eps_dev = eps - (tr(eps)/3)*I is the deviatoric strain. All
    deviatoric energy is assigned to psi_pos -- a crack shouldn't heal or
    grow under compression, so only the volumetric part distinguishes
    tension from compression. psi_pos + psi_neg reproduces the ordinary
    (undegraded) elastic energy exactly.

    Symmetrizes eps first -- required whenever this is differentiated
    directly (jax.grad/jax.jacfwd, as PhaseFieldIsotropic.stress_and_tangent
    does): JAX otherwise treats eps_ij/eps_ji as independent entries, same
    gotcha as materialmodels.inelastic.plasticity_j2.J2Plasticity.stress.
    Harmless (and a no-op) when eps is already symmetric, as it always is
    when this is called through driving_force.strain_energy_amor_split on
    an already-symmetric solver strain field.

    Parameters
    ----------
    eps : (3, 3)     local strain, undegraded
    lam : float      Lame lambda
    mu  : float      shear modulus

    Returns
    -------
    psi_pos, psi_neg : scalar, scalar
    """
    eps = 0.5 * (eps + eps.T)
    K = lam + 2.0 * mu / 3.0
    tr_eps = jnp.trace(eps)
    eps_dev = eps - (tr_eps / 3.0) * jnp.eye(3)
    psi_pos = 0.5 * K * jnp.maximum(tr_eps, 0.0) ** 2 + mu * jnp.sum(eps_dev ** 2)
    psi_neg = 0.5 * K * jnp.minimum(tr_eps, 0.0) ** 2
    return psi_pos, psi_neg
