"""
Crack driving force for phase-field fracture: the tensile part ψ⁺ of the
elastic strain energy density (Amor et al. 2009 volumetric-deviatoric
split), and the irreversibility law that turns ψ⁺ into a monotone history
variable H.

Only Amor is implemented -- it's what the two single-notch-plate benchmarks
(pff_tension.py, pff_shear.py) use; add a Miehe/spectral split here if a
future problem needs it.
"""

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)

from collections.abc import Sequence

import jax
import jax.numpy as jnp

from materialmodels.base import ConstitutiveModel


def lame_field(materials: Sequence[ConstitutiveModel], phase: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Per-voxel Lamé constants gathered from each material's ``.lam``/``.mu``
    attributes by phase index -- same hard (sharp-interface) assembly
    pattern as ``materialmodels.assembly.assemble_C_field``, but for the
    scalar (λ, μ) pair the Amor split needs instead of the full C tensor.
    Only valid where every material in ``materials`` is isotropic (exposes
    ``.lam``/``.mu``).

    Parameters
    ----------
    materials : list of ConstitutiveModel with .lam/.mu, indexed by phase
    phase     : (Nv,) int   phase index per voxel

    Returns
    -------
    lam_vox, mu_vox : (Nv,), (Nv,)
    """
    lam_stack = jnp.array([m.lam for m in materials])
    mu_stack = jnp.array([m.mu for m in materials])
    return lam_stack[phase], mu_stack[phase]


def strain_energy_amor_split(
    eps_loc: jnp.ndarray,
    lam_vox: jnp.ndarray,
    mu_vox: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Per-voxel volumetric-deviatoric energy split (Amor et al. 2009).

    Decomposes  ψ_e = ψ⁺ + ψ⁻  by separating volumetric tension from
    compression; all deviatoric energy is assigned to the tensile part:

        K   = λ + 2μ/3          (3-D bulk modulus)
        ψ⁺  = K/2 ⟨tr ε⟩₊²  +  μ ε_dev : ε_dev
        ψ⁻  = K/2 ⟨tr ε⟩₋²

    where  ε_dev = ε − (tr ε / 3) I  is the deviatoric strain.
    ψ⁺ + ψ⁻ = ψ_e exactly. Only ψ⁺ drives crack growth -- a crack shouldn't
    heal or grow under compression.

    Parameters
    ----------
    eps_loc : (3, 3, Nv)   local strain, undegraded
    lam_vox : (Nv,)        Lamé λ per voxel -- use ``lame_field``
    mu_vox  : (Nv,)        shear modulus μ per voxel

    Returns
    -------
    psi_pos, psi_neg : (Nv,), (Nv,)
    """
    def _one(args):
        eps, lam, mu = args
        K = lam + 2.0 * mu / 3.0
        tr_eps = jnp.trace(eps)
        eps_dev = eps - (tr_eps / 3.0) * jnp.eye(3)
        dev_norm_sq = jnp.sum(eps_dev ** 2)
        psi_pos = 0.5 * K * jnp.maximum(tr_eps, 0.0) ** 2 + mu * dev_norm_sq
        psi_neg = 0.5 * K * jnp.minimum(tr_eps, 0.0) ** 2
        return psi_pos, psi_neg

    eps_vox = eps_loc.transpose(2, 0, 1)   # (Nv, 3, 3)
    psi_pos, psi_neg = jax.vmap(_one)((eps_vox, lam_vox, mu_vox))
    return psi_pos, psi_neg


def update_history_hybrid(
    H_prev: jnp.ndarray,
    psi_pos: jnp.ndarray,
    d_prev: jnp.ndarray,
    d_thres: float = 0.95,
) -> jnp.ndarray:
    """
    Hybrid damage-/crack-like irreversibility (Steinke & Kaliske 2019, as
    adopted by Schneider & Kästner 2025, doi:10.1111/ffe.14553).

    Pure damage-like irreversibility (H = max(H_prev, ψ⁺) everywhere) locks
    in the history field as soon as any damage nucleates, which over-widens
    the diffuse process zone. The hybrid formulation only enforces the
    monotone history lock once a point has effectively cracked (d ≥
    d_thres); below the threshold the driving force is left unrestricted so
    the pre-crack process zone can still relax:

        H = ψ⁺                     if d_prev < d_thres   (unrestricted)
        H = max(H_prev, ψ⁺)        if d_prev ≥ d_thres    (irreversible)

    Parameters
    ----------
    H_prev  : (Nv,)   history variable from the previous (staggered or
                       time) iteration
    psi_pos : (Nv,)   current tensile strain energy density ψ⁺
    d_prev  : (Nv,)   damage field used to gate the switch
    d_thres : float   phase-field threshold above which irreversibility is
                       enforced (default 0.95)

    Returns
    -------
    H : (Nv,)
    """
    return jnp.where(d_prev >= d_thres, jnp.maximum(H_prev, psi_pos), psi_pos)
