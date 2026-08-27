"""
4th-order stiffness tensor utilities: Voigt/engineering conversion, rotation,
symmetry checks. Generalizes what used to be private, single-purpose helpers
duplicated inside each anisotropic material model (e.g. the old
``TransverseIsotropicFibre``'s own ``_voigt_eng_to_C4``/``_rotate_C4``) into
shared functions any ``ConstitutiveModel`` can build on.

2nd-order (strain/stress) Voigt/Mandel conversion already lives in
``post.fields.to_voigt``/``from_voigt`` -- this module is 4th-order only, so
the two don't overlap.

Voigt/engineering conversion is plain numpy (built once per material at
setup time, not per-voxel); rotation is jax.numpy (applied per-voxel via
``vmap`` over an orientation field, so must be jit-compatible).
"""

import jax.numpy as jnp
import numpy as np

# Voigt index pairs -- Abaqus convention: [11, 22, 33, 12, 13, 23], same
# order as post.fields._VOIGT_IJ
_VOIGT_IJ = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))


def voigt_to_tensor4(C_voigt: np.ndarray, engineering: bool = False) -> np.ndarray:
    """
    Convert a 6x6 Voigt stiffness matrix to a (3,3,3,3) 4th-order tensor.

    Parameters
    ----------
    C_voigt     : (6, 6)  Voigt stiffness matrix
    engineering : bool    False (default) -- C_voigt uses the tensor shear
                  convention (strain input epsilon, shear block = 2*mu for
                  an isotropic material). True -- engineering/Abaqus-UMAT
                  convention (strain input gamma = 2*epsilon, shear block
                  = mu). Only the strain side of the map differs between
                  conventions; stress Voigt is the same either way, which
                  is why only the shear *columns* (J = 3, 4, 5) are rescaled.

    Returns
    -------
    C4 : (3, 3, 3, 3), major- and minor-symmetric
    """
    C_voigt = np.asarray(C_voigt, dtype=float)
    if not engineering:
        C_voigt = C_voigt.copy()
        C_voigt[:, 3:] /= 2.0   # undo the gamma = 2*epsilon column scaling

    C4 = np.zeros((3, 3, 3, 3))
    for I, (i, j) in enumerate(_VOIGT_IJ):
        for J, (k, l) in enumerate(_VOIGT_IJ):
            v = C_voigt[I, J]
            C4[i, j, k, l] = v
            C4[j, i, k, l] = v   # minor symmetry
            C4[i, j, l, k] = v   # minor symmetry
            C4[j, i, l, k] = v   # major + minor
    return C4


def tensor4_to_voigt(C4: np.ndarray, engineering: bool = False) -> np.ndarray:
    """
    Convert a (3,3,3,3) 4th-order tensor to a 6x6 Voigt stiffness matrix.
    Inverse of ``voigt_to_tensor4`` -- round-trips exactly for the same
    ``engineering`` flag.

    Parameters
    ----------
    C4          : (3, 3, 3, 3), major- and minor-symmetric
    engineering : bool  see ``voigt_to_tensor4``

    Returns
    -------
    C_voigt : (6, 6)
    """
    C4 = np.asarray(C4, dtype=float)
    C_voigt = np.array([[C4[i, j, k, l] for (k, l) in _VOIGT_IJ] for (i, j) in _VOIGT_IJ])
    if not engineering:
        C_voigt[:, 3:] *= 2.0
    return C_voigt


def rotation_from_direction(d: jnp.ndarray) -> jnp.ndarray:
    """
    Build a (3, 3) rotation matrix R such that R @ e_z = d, where the
    reference axis is e_z = [0, 0, 1]. Fully vmap-able over a per-voxel
    direction field (e.g. a spatially varying fibre orientation).

    Parameters
    ----------
    d : (3,)  target direction (not required to be pre-normalized)

    Returns
    -------
    R : (3, 3)  columns are the new (e1', e2', d) basis in the global frame
    """
    d = d / jnp.linalg.norm(d)
    ref = jnp.where(jnp.abs(d[0]) < 0.9, jnp.array([1., 0., 0.]), jnp.array([0., 1., 0.]))
    e1 = ref - jnp.dot(ref, d) * d
    e1 = e1 / jnp.linalg.norm(e1)
    e2 = jnp.cross(d, e1)
    return jnp.stack([e1, e2, d], axis=1)


def rotate_tensor4(R: jnp.ndarray, C4: jnp.ndarray) -> jnp.ndarray:
    """
    Rotate a 4th-order stiffness tensor from a local frame to the global
    frame: C_global_ijkl = R_ia R_jb R_kc R_ld C_local_abcd.

    Parameters
    ----------
    R  : (3, 3)  rotation matrix mapping local frame to global frame
    C4 : (3, 3, 3, 3)  stiffness tensor in the local frame

    Returns
    -------
    C4_rotated : (3, 3, 3, 3)
    """
    return jnp.einsum('ia,jb,kc,ld,abcd->ijkl', R, R, R, R, C4)


def isotropic_equivalent_lame(C4: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Voigt-average isotropic Lame parameters (lam0, mu0) of a general (3,3,3,3)
    stiffness tensor -- exact for an isotropic C4 (recovers its own lam, mu),
    an isotropization for an anisotropic one (e.g. a rotated
    TransverseIsotropic). Both K = C_iijj/9 (bulk modulus) and
    mu0 = (C_ijij - 3K)/10 are linear in C4, so this commutes with averaging
    stiffness tensors across materials -- e.g. mean-over-materials then
    isotropize gives the same lam0/mu0 as isotropize-then-mean.
    """
    K = jnp.einsum('iijj->', C4) / 9.0
    mu0 = (jnp.einsum('ijij->', C4) - 3.0 * K) / 10.0
    lam0 = K - 2.0 * mu0 / 3.0
    return lam0, mu0


def is_major_symmetric(C4: np.ndarray, atol: float = 1e-10) -> bool:
    """C_ijkl == C_klij."""
    C4 = np.asarray(C4)
    return bool(np.allclose(C4, np.transpose(C4, (2, 3, 0, 1)), atol=atol))


def is_minor_symmetric(C4: np.ndarray, atol: float = 1e-10) -> bool:
    """C_ijkl == C_jikl == C_ijlk."""
    C4 = np.asarray(C4)
    return bool(
        np.allclose(C4, np.transpose(C4, (1, 0, 2, 3)), atol=atol)
        and np.allclose(C4, np.transpose(C4, (0, 1, 3, 2)), atol=atol)
    )
