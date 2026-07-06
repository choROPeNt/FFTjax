"""
Tensor utilities for FFTjax
Provides identity tensors and permutation tensor in JAX form.

Author: FFTjax
"""

import jax
import jax.numpy as jnp


# ---------------------------------------
# Tensor contractions
# ---------------------------------------

@jax.jit
def ddot42(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    """
    Double contraction  C_ijm = A_ijklm B_klm.

    Shapes: (3,3,3,3,Nv) × (3,3,Nv) → (3,3,Nv)
    Used for  σ = C : ε  and  Γ : σ  in the FFT solver.
    """
    return jnp.einsum('ijklm,klm->ijm', A, B)


@jax.jit
def odot11(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    """
    Dyadic product  C_ijm = A_im B_jm.

    Shapes: (3,Nv) × (3,Nv) → (3,3,Nv)
    Used to build the (unsymmetrized) Fourier-space displacement gradient
    ``odot11(u_hat, iξ)`` in the displacement-based FFT solver.
    """
    return jnp.einsum('im,jm->ijm', A, B)


@jax.jit
def dot21(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    """
    Contraction  c_im = A_ijm B_jm.

    Shapes: (3,3,Nv) × (3,Nv) → (3,Nv)
    Used to take the Fourier-space divergence of a stress field,
    ``dot21(σ_hat, iξ)``, in the displacement-based FFT solver.
    """
    return jnp.einsum('ijm,jm->im', A, B)


# ---------------------------------------
# 3D Identity Tensors
# ---------------------------------------

@jax.jit
def identity_tensors_3d(dtype=jnp.float32):
    """
    Returns:
        I2   : 2nd-order identity (3x3)
        I4   : 4th-order identity δik δjl
        I4t  : transpose identity δil δjk
        I4s  : symmetric projector
        IxI  : outer product I ⊗ I
        I4d  : deviatoric symmetric projector
    """
    I2 = jnp.eye(3, dtype=dtype)

    I4  = jnp.einsum("ik,jl->ijkl", I2, I2)
    I4t = jnp.einsum("il,jk->ijkl", I2, I2)
    I4s = 0.5 * (I4 + I4t)

    IxI = jnp.einsum("ij,kl->ijkl", I2, I2)
    I4d = I4s - (1.0 / 3.0) * IxI

    return I2, I4, I4t, I4s, IxI, I4d


# ---------------------------------------
# 2D Identity Tensors
# ---------------------------------------

@jax.jit
def identity_tensors_2d(dtype=jnp.float32):
    """
    Returns:
        I2   : 2x2 identity
        I4   : 4th-order identity
        I4t  : transpose identity
        I4s  : symmetric projector
    """
    I2 = jnp.eye(2, dtype=dtype)

    I4  = jnp.einsum("ik,jl->ijkl", I2, I2)
    I4t = jnp.einsum("il,jk->ijkl", I2, I2)
    I4s = 0.5 * (I4 + I4t)

    return I2, I4, I4t, I4s


# ---------------------------------------
# Levi-Civita permutation tensor (3D)
# ---------------------------------------

@jax.jit
def levi_civita_3d(dtype=jnp.float32):
    """
    Returns:
        ε_ijk permutation tensor
    """
    eps = jnp.zeros((3, 3, 3), dtype=dtype)

    eps = eps.at[0, 1, 2].set( 1)
    eps = eps.at[1, 2, 0].set( 1)
    eps = eps.at[2, 0, 1].set( 1)
    eps = eps.at[0, 2, 1].set(-1)
    eps = eps.at[2, 1, 0].set(-1)
    eps = eps.at[1, 0, 2].set(-1)

    return eps