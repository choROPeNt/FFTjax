"""
4th-order stiffness utilities: Voigt conversion, rotation, symmetry checks.

Voigt conversion is numpy (once per material); rotation is jax.numpy (per voxel).
2nd-order Voigt lives in post.fields.
"""

import jax.numpy as jnp
import numpy as np

_VOIGT_IJ = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))


def voigt_to_tensor4(C_voigt: np.ndarray, engineering: bool = False) -> np.ndarray:
    """
    6x6 Voigt matrix (order 11, 22, 33, 12, 13, 23) -> (3, 3, 3, 3) tensor.

    Parameters
    ----------
    C_voigt     : (6, 6)
    engineering : False: tensor shear strains (isotropic shear block 2 mu);
                  True: engineering shear gamma = 2 eps (Abaqus, block mu)

    Returns
    -------
    C4 : (3, 3, 3, 3), major- and minor-symmetric
    """
    C_voigt = np.asarray(C_voigt, dtype=float)
    if not engineering:
        C_voigt = C_voigt.copy()
        C_voigt[:, 3:] /= 2.0

    C4 = np.zeros((3, 3, 3, 3))
    for I, (i, j) in enumerate(_VOIGT_IJ):
        for J, (k, l) in enumerate(_VOIGT_IJ):
            v = C_voigt[I, J]
            C4[i, j, k, l] = v
            C4[j, i, k, l] = v
            C4[i, j, l, k] = v
            C4[j, i, l, k] = v
    return C4


def tensor4_to_voigt(C4: np.ndarray, engineering: bool = False) -> np.ndarray:
    """Inverse of ``voigt_to_tensor4`` for the same ``engineering`` flag."""
    C4 = np.asarray(C4, dtype=float)
    C_voigt = np.array([[C4[i, j, k, l] for (k, l) in _VOIGT_IJ] for (i, j) in _VOIGT_IJ])
    if not engineering:
        C_voigt[:, 3:] *= 2.0
    return C_voigt


def rotation_from_direction(d: jnp.ndarray) -> jnp.ndarray:
    """
    Rotation R with ``R @ e_z = d``; columns are (e1', e2', d).

    A zero ``d`` gives the identity, not NaN: zero marks voxels outside the
    oriented phase, and one NaN there would poison the grid-wide average of
    the reference medium before phase masking removes it.
    """
    norm = jnp.linalg.norm(d)
    degenerate = norm < 1e-12
    d = jnp.where(degenerate, jnp.array([0., 0., 1.]), d / jnp.where(degenerate, 1.0, norm))
    ref = jnp.where(jnp.abs(d[0]) < 0.9, jnp.array([1., 0., 0.]), jnp.array([0., 1., 0.]))
    e1 = ref - jnp.dot(ref, d) * d
    e1 = e1 / jnp.linalg.norm(e1)
    e2 = jnp.cross(d, e1)
    return jnp.stack([e1, e2, d], axis=1)


def rotate_tensor4(R: jnp.ndarray, C4: jnp.ndarray) -> jnp.ndarray:
    """``C_ijkl = R_ia R_jb R_kc R_ld C_abcd`` (local to global frame)."""
    return jnp.einsum('ia,jb,kc,ld,abcd->ijkl', R, R, R, R, C4)


def rotation_field_from_directions(d_field: jnp.ndarray) -> jnp.ndarray:
    """
    Per-voxel ``rotation_from_direction``: (3, Nv) -> (3, 3, Nv).

    Written with the voxel axis trailing instead of ``vmap``: vmap batches the
    leading axis and needs a final full-size transpose, an extra multi-GiB
    buffer that ran out of GPU memory at tens of millions of voxels.
    """
    norm = jnp.linalg.norm(d_field, axis=0)
    degenerate = norm < 1e-12
    z_hat = jnp.array([0.0, 0.0, 1.0])[:, None]
    safe_norm = jnp.where(degenerate, 1.0, norm)
    d = jnp.where(degenerate[None, :], z_hat, d_field / safe_norm[None, :])

    x_hat = jnp.array([1.0, 0.0, 0.0])[:, None]
    y_hat = jnp.array([0.0, 1.0, 0.0])[:, None]
    ref = jnp.where(jnp.abs(d[0:1, :]) < 0.9, x_hat, y_hat)

    e1 = ref - jnp.sum(ref * d, axis=0, keepdims=True) * d
    e1 = e1 / jnp.linalg.norm(e1, axis=0, keepdims=True)
    e2 = jnp.cross(d, e1, axis=0)
    return jnp.stack([e1, e2, d], axis=1)


def rotate_tensor4_field(R_field: jnp.ndarray, C4: jnp.ndarray) -> jnp.ndarray:
    """One local ``C4`` rotated per voxel: (3, 3, Nv) x (3, 3, 3, 3) -> (3, 3, 3, 3, Nv)."""
    return jnp.einsum('iav,jbv,kcv,ldv,abcd->ijklv',
                       R_field, R_field, R_field, R_field, C4)


def isotropic_equivalent_lame(C4: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Isotropic projection ``(lam0, mu0)`` of a stiffness tensor; exact for isotropic C4.

    K = C_iijj / 9 and mu0 = (C_ijij - 3K) / 10 are linear in C4, so this
    commutes with averaging over materials.
    """
    K = jnp.einsum('iijj->', C4) / 9.0
    mu0 = (jnp.einsum('ijij->', C4) - 3.0 * K) / 10.0
    lam0 = K - 2.0 * mu0 / 3.0
    return lam0, mu0


def is_major_symmetric(C4: np.ndarray, atol: float = 1e-10) -> bool:
    """``C_ijkl == C_klij``."""
    C4 = np.asarray(C4)
    return bool(np.allclose(C4, np.transpose(C4, (2, 3, 0, 1)), atol=atol))


def is_minor_symmetric(C4: np.ndarray, atol: float = 1e-10) -> bool:
    """``C_ijkl == C_jikl == C_ijlk``."""
    C4 = np.asarray(C4)
    return bool(
        np.allclose(C4, np.transpose(C4, (1, 0, 2, 3)), atol=atol)
        and np.allclose(C4, np.transpose(C4, (0, 1, 3, 2)), atol=atol)
    )
