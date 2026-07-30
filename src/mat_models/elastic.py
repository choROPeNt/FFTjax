"""
Linear elastic material models.

Voigt index convention (Abaqus order):
    0 = 11,  1 = 22,  2 = 33,  3 = 12,  4 = 13,  5 = 23

All Voigt representations use *tensor* shear components (ε₁₂, not γ₁₂ = 2ε₁₂).
This is consistent with `post.io.to_voigt` and with the σ = C:ε einsum in the
FFT solver.

    σ_I = Σ_J  C_IJ  ε_J        (tensor convention throughout)

Consequence: the shear–shear block of C_voigt is **2μ** (not μ), because
    σ₁₂ = C₁₂₁₂ ε₁₂ + C₁₂₂₁ ε₂₁ = μ ε₁₂ + μ ε₂₁ = 2μ ε₁₂

For Abaqus UMAT compatibility (engineering shear γ = 2ε), call
``stiffness_voigt(engineering=True)`` which gives μ on the shear diagonal
and expects γ-convention strains.
"""

import jax
import jax.numpy as jnp
import numpy as np


# Voigt index pairs — same order as post.io._VOIGT_IJ
_VOIGT_IJ = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))


# ------------------------------------------------------------------
# Private low-level helpers
# ------------------------------------------------------------------

def _voigt_eng_to_C4(C_eng: np.ndarray) -> np.ndarray:
    """
    Convert a 6×6 engineering-Voigt stiffness matrix to a (3,3,3,3)
    4th-order stiffness tensor.

    Engineering Voigt: σ_I = Σ_J C_eng_IJ γ_J,  γ_J = 2ε_J for shear.

    The mapping  C4_ijkl = C_eng_IJ  holds exactly: the factor of 2 in
    γ = 2ε cancels the factor of 2 that comes from summing both (k,l) and
    (l,k) in the full double contraction σ_ij = Σ_kl C_ijkl ε_kl.
    """
    C4 = np.zeros((3, 3, 3, 3))
    for I, (i, j) in enumerate(_VOIGT_IJ):
        for J, (k, l) in enumerate(_VOIGT_IJ):
            v = C_eng[I, J]
            C4[i, j, k, l] = v
            C4[j, i, k, l] = v   # minor symmetry
            C4[i, j, l, k] = v   # minor symmetry
            C4[j, i, l, k] = v   # major + minor
    return C4


def _rotation_from_fiber_dir(d: jnp.ndarray) -> jnp.ndarray:
    """
    Build a (3, 3) rotation matrix R such that  R @ e_z = d,
    where the reference fibre axis is e_z = [0, 0, 1].

    The columns of R are the new (e₁′, e₂′, d) basis expressed in the
    global frame.  This function is fully vmappable over batches of d.
    """
    d   = d / jnp.linalg.norm(d)
    ref = jnp.where(jnp.abs(d[0]) < 0.9,
                    jnp.array([1., 0., 0.]),
                    jnp.array([0., 1., 0.]))
    e1  = ref - jnp.dot(ref, d) * d
    e1  = e1 / jnp.linalg.norm(e1)
    e2  = jnp.cross(d, e1)
    return jnp.stack([e1, e2, d], axis=1)   # columns: e₁′ | e₂′ | d


def _rotate_C4(R: jnp.ndarray, C_ref: jnp.ndarray) -> jnp.ndarray:
    """
    Rotate 4th-order stiffness tensor from a local frame to the global frame.

        C_global_ijkl = R_ia R_jb R_kc R_ld  C_ref_abcd

    R maps the local frame to the global frame (R @ e_local = e_global).
    """
    return jnp.einsum('ia,jb,kc,ld,abcd->ijkl', R, R, R, R, C_ref)


# ------------------------------------------------------------------
# Material models
# ------------------------------------------------------------------

class LinearElasticIsotropic:
    """
    Isotropic linear elastic constitutive model.

    Parameters
    ----------
    E    : float  Young's modulus (any consistent unit, e.g. MPa)
    nu   : float  Poisson's ratio  (−1 < ν < 0.5)
    name : str    Optional label for display.
    """

    def __init__(self, E: float, nu: float, name: str = ""):
        self.E    = float(E)
        self.nu   = float(nu)
        self.name = name
        self.lam  = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        self.mu   = E / (2.0 * (1.0 + nu))

    # ------------------------------------------------------------------
    # Stiffness representations
    # ------------------------------------------------------------------

    def stiffness_tensor(self) -> jnp.ndarray:
        """
        Full 4th-order stiffness tensor C_ijkl, shape (3, 3, 3, 3).

        Used directly by the FFT solver::

            sigma = jnp.einsum('ijkl,kl->ij', C, eps)
        """
        d = jnp.eye(3)
        return (self.lam * jnp.einsum('ij,kl->ijkl', d, d)
                + self.mu * (jnp.einsum('ik,jl->ijkl', d, d)
                             + jnp.einsum('il,jk->ijkl', d, d)))

    def stiffness_voigt(self, engineering: bool = False) -> jnp.ndarray:
        """
        6×6 Voigt stiffness matrix C_IJ, shape (6, 6).

        Parameters
        ----------
        engineering : bool
            False (default) — tensor shear convention (ε₁₂, shear block = 2μ),
            compatible with ``post.io.to_voigt`` and the FFT solver output.
            True — engineering / Abaqus-UMAT convention (γ₁₂ = 2ε₁₂, shear block = μ).
        """
        lam, mu = self.lam, self.mu
        shear_factor = mu if engineering else 2.0 * mu
        C = np.zeros((6, 6))
        C[:3, :3] = lam
        C[0, 0] += 2 * mu;  C[1, 1] += 2 * mu;  C[2, 2] += 2 * mu
        C[3, 3] = shear_factor
        C[4, 4] = shear_factor
        C[5, 5] = shear_factor
        return jnp.array(C)

    def stress_voigt(self, eps_voigt: jnp.ndarray, engineering: bool = False) -> jnp.ndarray:
        """Compute Voigt stress from Voigt strain (..., 6) → (..., 6)."""
        return eps_voigt @ self.stiffness_voigt(engineering=engineering).T

    def stress_field(self, eps: jnp.ndarray) -> jnp.ndarray:
        """Compute stress from full-tensor strain field (3,3,Nv) → (3,3,Nv)."""
        return jnp.einsum('ijkl,klm->ijm', self.stiffness_tensor(), eps)

    @property
    def bulk_modulus(self) -> float:
        return self.E / (3.0 * (1.0 - 2.0 * self.nu))

    @property
    def shear_modulus(self) -> float:
        return self.mu

    def __repr__(self) -> str:
        tag = f" ({self.name})" if self.name else ""
        return (f"LinearElasticIsotropic{tag}: "
                f"E={self.E:.3g}, nu={self.nu:.3g}, "
                f"lam={self.lam:.3g}, mu={self.mu:.3g}")


class TransverseIsotropicFibre:
    """
    Transversely isotropic (fibre-reinforced) material.

    The *reference* fibre axis is local  **Z = [0, 0, 1]**.
    Call ``stiffness_tensor_rotated(fiber_dir)`` to obtain the stiffness
    for an arbitrary fibre direction in the global frame.

    5 independent elastic constants (engineering / test-data notation):

    E_L   : Young's modulus along the fibre (longitudinal)
    E_T   : Young's modulus perpendicular to the fibre (transverse)
    G_LT  : Shear modulus in planes that contain the fibre axis
    nu_LT : Poisson's ratio  (–ε_T / ε_L when loaded uniaxially along L)
    nu_TT : Poisson's ratio in the transverse isotropic plane

    Derived:
    G_TT  = E_T / (2(1 + ν_TT))    transverse–transverse shear modulus
    nu_TL = nu_LT · E_T / E_L      reciprocal Poisson ratio (symmetry of S)

    Compliance matrix (engineering Voigt, Abaqus order, fibre along Z = index 2):

        S = diag(1/E_T, 1/E_T, 1/E_L, 1/G_TT, 1/G_LT, 1/G_LT)
            + off-diag coupling:
              S_01 = S_10 = –ν_TT/E_T
              S_02 = S_20 = S_12 = S_21 = –ν_LT/E_L  (= –ν_TL/E_T by symmetry)
    """

    def __init__(
        self,
        E_L:   float,
        E_T:   float,
        G_LT:  float,
        nu_LT: float,
        nu_TT: float,
        name:  str = "",
    ):
        self.E_L   = float(E_L)
        self.E_T   = float(E_T)
        self.G_LT  = float(G_LT)
        self.nu_LT = float(nu_LT)
        self.nu_TT = float(nu_TT)
        self.name  = name
        self.G_TT  = float(E_T / (2.0 * (1.0 + nu_TT)))
        self.nu_TL = float(nu_LT * E_T / E_L)   # S symmetry: ν_LT/E_L = ν_TL/E_T

    # ------------------------------------------------------------------
    # Stiffness representations
    # ------------------------------------------------------------------

    def _compliance_engineering(self) -> np.ndarray:
        """6×6 engineering compliance S (fibre along Z = Voigt index 2)."""
        S = np.zeros((6, 6))
        S[0, 0] = S[1, 1] = 1.0 / self.E_T
        S[2, 2]            = 1.0 / self.E_L
        S[0, 1] = S[1, 0] = -self.nu_TT / self.E_T
        S[0, 2] = S[2, 0] = S[1, 2] = S[2, 1] = -self.nu_LT / self.E_L
        S[3, 3]            = 1.0 / self.G_TT    # 12  transverse shear
        S[4, 4] = S[5, 5] = 1.0 / self.G_LT    # 13, 23  longitudinal shear
        return S

    def stiffness_tensor(self) -> jnp.ndarray:
        """
        (3, 3, 3, 3) 4th-order stiffness in the reference frame
        (fibre along Z = [0, 0, 1]).
        """
        C_eng = np.linalg.inv(self._compliance_engineering())
        return jnp.array(_voigt_eng_to_C4(C_eng))

    def stiffness_tensor_rotated(self, fiber_dir: jnp.ndarray) -> jnp.ndarray:
        """
        (3, 3, 3, 3) stiffness rotated so that the fibre axis aligns with
        ``fiber_dir`` in the global frame.

            C_global_ijkl = R_ia R_jb R_kc R_ld  C_ref_abcd

        where  R  is the rotation matrix that maps the local Z to ``fiber_dir``.

        Parameters
        ----------
        fiber_dir : (3,) unit vector (global frame)
        """
        R     = _rotation_from_fiber_dir(jnp.asarray(fiber_dir, float))
        C_ref = self.stiffness_tensor()
        return _rotate_C4(R, C_ref)

    def stiffness_voigt(self, engineering: bool = False) -> jnp.ndarray:
        """
        6×6 Voigt stiffness in the reference frame (fibre along Z).

        engineering=False (default) → tensor shear convention (compatible
        with ``post.io.to_voigt`` and the FFT solver).
        engineering=True            → Abaqus / UMAT γ convention.
        """
        C_eng = np.linalg.inv(self._compliance_engineering())
        if engineering:
            return jnp.array(C_eng)
        C_tens = C_eng.copy()
        C_tens[:, 3:] *= 2.0   # γ → ε: multiply shear columns by 2
        return jnp.array(C_tens)

    def stress_voigt(self, eps_voigt: jnp.ndarray, engineering: bool = False) -> jnp.ndarray:
        """Compute Voigt stress from Voigt strain (..., 6) → (..., 6)."""
        return eps_voigt @ self.stiffness_voigt(engineering=engineering).T

    def __repr__(self) -> str:
        tag = f" ({self.name})" if self.name else ""
        return (f"TransverseIsotropicFibre{tag}: "
                f"E_L={self.E_L:.3g}  E_T={self.E_T:.3g}  G_LT={self.G_LT:.3g}  "
                f"G_TT={self.G_TT:.3g}  nu_LT={self.nu_LT:.3g}  nu_TT={self.nu_TT:.3g}")


# ------------------------------------------------------------------
# Assembly helpers
# ------------------------------------------------------------------

def assemble_C_field(
    materials: list,
    phase: jnp.ndarray,
) -> jnp.ndarray:
    """
    Build a per-voxel stiffness field from a list of materials and phase indices.

    Parameters
    ----------
    materials : list  Each must implement ``stiffness_tensor() → (3,3,3,3)``.
    phase     : (Nv,) int   phase index per voxel (0-based).

    Returns
    -------
    C_field : (3, 3, 3, 3, Nv)
    """
    C_stack = jnp.stack(
        [m.stiffness_tensor() for m in materials], axis=-1
    )  # (3, 3, 3, 3, n_mats)
    return C_stack[..., phase]  # (3, 3, 3, 3, Nv)


def assemble_C_field_oriented(
    materials:    list,
    phase:        jnp.ndarray,
    orientations: jnp.ndarray,
) -> jnp.ndarray:
    """
    Build a per-voxel stiffness field with per-voxel fibre orientations.

    For materials that implement ``stiffness_tensor_rotated(fiber_dir)``,
    the stiffness is rotated voxel-by-voxel using the supplied orientation
    vectors via ``jax.vmap``.  Isotropic / orientation-free materials fall
    back to their fixed ``stiffness_tensor()``.

    Parameters
    ----------
    materials    : list of material objects
    phase        : (Nv,) int    phase index per voxel
    orientations : (3, Nv)      unit vectors giving the local fibre direction
                                per voxel in the global frame.
                                Ignored for phases whose material does not
                                implement ``stiffness_tensor_rotated``.

    Returns
    -------
    C_field : (3, 3, 3, 3, Nv)

    Example
    -------
    >>> # Fibre along Z everywhere (default):
    >>> Nv = int(np.prod(n))
    >>> orientations = jnp.zeros((3, Nv)).at[2, :].set(1.0)
    >>> C_field = assemble_C_field_oriented(materials, phase, orientations)
    """
    Nv      = int(phase.shape[0])
    C_field = jnp.zeros((3, 3, 3, 3, Nv))

    for mat_idx, mat in enumerate(materials):
        mask  = phase == mat_idx                                      # (Nv,)
        mask5 = jnp.broadcast_to(mask[None, None, None, None, :],
                                  (3, 3, 3, 3, Nv))

        if hasattr(mat, 'stiffness_tensor_rotated'):
            # ── anisotropic / oriented phase ─────────────────────────────────
            # Compute per-voxel rotation matrices and rotate C, all via vmap.
            C_ref  = mat.stiffness_tensor()                           # (3,3,3,3)
            R_all  = jax.vmap(_rotation_from_fiber_dir)(             # (Nv,3,3)
                orientations.T                                        # (Nv,3)
            )
            C_rot  = jax.vmap(lambda R: _rotate_C4(R, C_ref))(R_all)# (Nv,3,3,3,3)
            C_rot  = jnp.moveaxis(C_rot, 0, -1)                      # (3,3,3,3,Nv)
            C_field = jnp.where(mask5, C_rot, C_field)

        else:
            # ── isotropic / orientation-free phase ────────────────────────────
            C_iso   = mat.stiffness_tensor()                          # (3,3,3,3)
            C_iso_b = jnp.broadcast_to(C_iso[..., None], (3, 3, 3, 3, Nv))
            C_field = jnp.where(mask5, C_iso_b, C_field)

    return C_field


def assemble_C_field_smooth(
    materials:       list,
    orientations:    jnp.ndarray,
    volume_fraction: jnp.ndarray,
) -> jnp.ndarray:
    """
    Smooth Voigt blending of matrix (phase 0) and yarn (phase 1) stiffness
    using a per-voxel volume fraction φ ∈ [0, 1]:

        C(x) = φ(x) · C_yarn(n(x))  +  (1 − φ(x)) · C_matrix

    ``orientations`` must cover ALL voxels (including interface matrix voxels)
    with the nearest yarn tangent direction; zero vectors are replaced with
    a safe default before rotation (their contribution is negligible where φ≈0).

    Parameters
    ----------
    materials       : [matrix_mat, yarn_mat]  (index 0 = matrix, 1 = yarn)
    orientations    : (3, Nv)  nearest yarn tangent per voxel
    volume_fraction : (Nv,)    smooth φ in [0, 1]

    Returns
    -------
    C_field : (3, 3, 3, 3, Nv)
    """
    Nv = int(volume_fraction.shape[0])

    # Replace zero-magnitude orientations with e_z to avoid NaN in rotation;
    # phi≈0 for those voxels so their yarn contribution is negligible.
    mag       = jnp.linalg.norm(orientations, axis=0)          # (Nv,)
    safe_ori  = jnp.where(mag[None, :] > 1e-8,
                          orientations,
                          jnp.broadcast_to(jnp.array([[0.], [0.], [1.]]),
                                           (3, Nv)))            # (3, Nv)

    # ── yarn stiffness — rotated per voxel ───────────────────────────────────
    yarn_mat = materials[1]
    if hasattr(yarn_mat, 'stiffness_tensor_rotated'):
        C_ref  = yarn_mat.stiffness_tensor()
        R_all  = jax.vmap(_rotation_from_fiber_dir)(safe_ori.T)  # (Nv, 3, 3)
        C_yarn = jax.vmap(lambda R: _rotate_C4(R, C_ref))(R_all) # (Nv, 3,3,3,3)
        C_yarn = jnp.moveaxis(C_yarn, 0, -1)                      # (3,3,3,3,Nv)
    else:
        C_yarn = jnp.broadcast_to(yarn_mat.stiffness_tensor()[..., None],
                                   (3, 3, 3, 3, Nv))

    # ── matrix stiffness — broadcast ─────────────────────────────────────────
    C_mat = jnp.broadcast_to(materials[0].stiffness_tensor()[..., None],
                              (3, 3, 3, 3, Nv))

    # ── smooth Voigt blend ────────────────────────────────────────────────────
    phi = volume_fraction[None, None, None, None, :]
    return phi * C_yarn + (1.0 - phi) * C_mat


# ------------------------------------------------------------------
# Strain energy helpers
# ------------------------------------------------------------------

def strain_energy_density(
    eps_loc: jnp.ndarray,
    C_field: jnp.ndarray,
) -> jnp.ndarray:
    """
    Total elastic strain energy density per voxel.

        ψ_e = ½ ε : C : ε

    Parameters
    ----------
    eps_loc : (3, 3, Nv)
    C_field : (3, 3, 3, 3, Nv)

    Returns
    -------
    psi : (Nv,)
    """
    sigma = jnp.einsum("ijklm,klm->ijm", C_field, eps_loc)
    return 0.5 * jnp.einsum("ijm,ijm->m", sigma, eps_loc)


def lame_from_C_field(
    C_field: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Extract per-voxel Lamé constants from an isotropic stiffness field.

        λ = C_1122,   μ = C_1212

    Only valid for isotropic phases.

    Parameters
    ----------
    C_field : (3, 3, 3, 3, Nv)

    Returns
    -------
    lam_vox : (Nv,)
    mu_vox  : (Nv,)
    """
    return C_field[0, 0, 1, 1, :], C_field[0, 1, 0, 1, :]


def strain_energy_miehe_split(
    eps_loc:  jnp.ndarray,
    lam_vox:  jnp.ndarray,
    mu_vox:   jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Per-voxel spectral energy split (Miehe et al. 2010).

    Decomposes  ψ_e = ψ⁺ + ψ⁻  based on the sign of the principal strains:

        ψ⁺ = λ/2 ⟨tr ε⟩₊²  +  μ Σᵢ ⟨εᵢ⟩₊²
        ψ⁻ = λ/2 ⟨tr ε⟩₋²  +  μ Σᵢ ⟨εᵢ⟩₋²

    ⟨x⟩₊ = max(x, 0),  ⟨x⟩₋ = min(x, 0).
    Only ψ⁺ (tensile) drives crack growth.

    Parameters
    ----------
    eps_loc : (3, 3, Nv)
    lam_vox : (Nv,)   Lamé λ per voxel — use ``lame_from_C_field``
    mu_vox  : (Nv,)   shear modulus μ per voxel

    Returns
    -------
    psi_pos : (Nv,)
    psi_neg : (Nv,)
    """
    def _one(args):
        eps, lam, mu = args
        eigvals, _ = jnp.linalg.eigh(eps)
        pos = jnp.maximum(eigvals, 0.0)
        neg = jnp.minimum(eigvals, 0.0)
        psi_pos = 0.5 * lam * jnp.sum(pos) ** 2 + mu * jnp.sum(pos ** 2)
        psi_neg = 0.5 * lam * jnp.sum(neg) ** 2 + mu * jnp.sum(neg ** 2)
        return psi_pos, psi_neg

    eps_vox = eps_loc.transpose(2, 0, 1)    # (Nv, 3, 3)
    psi_pos, psi_neg = jax.vmap(_one)((eps_vox, lam_vox, mu_vox))
    return psi_pos, psi_neg


def strain_energy_amor_split(
    eps_loc:  jnp.ndarray,
    lam_vox:  jnp.ndarray,
    mu_vox:   jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Per-voxel volumetric-deviatoric energy split (Amor et al. 2009).

    Decomposes  ψ_e = ψ⁺ + ψ⁻  by separating volumetric tension from
    compression; all deviatoric energy is assigned to the tensile part:

        K   = λ + 2μ/3          (3-D bulk modulus)
        ψ⁺  = K/2 ⟨tr ε⟩₊²  +  μ ε_dev : ε_dev
        ψ⁻  = K/2 ⟨tr ε⟩₋²

    where  ε_dev = ε − (tr ε / 3) I  is the deviatoric strain.
    ψ⁺ + ψ⁻ = ψ_e exactly.

    Parameters
    ----------
    eps_loc : (3, 3, Nv)
    lam_vox : (Nv,)   Lamé λ per voxel — use ``lame_from_C_field``
    mu_vox  : (Nv,)   shear modulus μ per voxel

    Returns
    -------
    psi_pos : (Nv,)
    psi_neg : (Nv,)
    """
    def _one(args):
        eps, lam, mu = args
        K      = lam + 2.0 * mu / 3.0
        tr_eps = jnp.trace(eps)
        eps_dev     = eps - (tr_eps / 3.0) * jnp.eye(3)
        dev_norm_sq = jnp.sum(eps_dev ** 2)          # ε_dev : ε_dev
        psi_pos = 0.5 * K * jnp.maximum(tr_eps, 0.0) ** 2 + mu * dev_norm_sq
        psi_neg = 0.5 * K * jnp.minimum(tr_eps, 0.0) ** 2
        return psi_pos, psi_neg

    eps_vox = eps_loc.transpose(2, 0, 1)    # (Nv, 3, 3)
    psi_pos, psi_neg = jax.vmap(_one)((eps_vox, lam_vox, mu_vox))
    return psi_pos, psi_neg


# backward-compatible alias
strain_energy_spectral_split = strain_energy_miehe_split
