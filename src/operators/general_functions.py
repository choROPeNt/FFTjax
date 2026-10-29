"""Tensor contractions, identity tensors and the permutation tensor."""

import jax
import jax.numpy as jnp


@jax.jit
def ddot42(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    """``C_ijm = A_ijklm B_klm``: (3,3,3,3,Nv) x (3,3,Nv) -> (3,3,Nv), e.g. sigma = C : eps."""
    return jnp.einsum('ijklm,klm->ijm', A, B)


@jax.jit
def odot11(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    """``C_ijm = A_im B_jm``: (3,Nv) x (3,Nv) -> (3,3,Nv), e.g. grad u_hat = u_hat (x) i xi."""
    return jnp.einsum('im,jm->ijm', A, B)


@jax.jit
def dot21(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    """``c_im = A_ijm B_jm``: (3,3,Nv) x (3,Nv) -> (3,Nv), e.g. div sigma_hat."""
    return jnp.einsum('ijm,jm->im', A, B)


@jax.jit
def identity_tensors_3d(dtype=jnp.float32):
    """3-D identities: ``I2, I4 (d_ik d_jl), I4t (d_il d_jk), I4s, IxI, I4d`` (deviatoric)."""
    I2 = jnp.eye(3, dtype=dtype)

    I4  = jnp.einsum("ik,jl->ijkl", I2, I2)
    I4t = jnp.einsum("il,jk->ijkl", I2, I2)
    I4s = 0.5 * (I4 + I4t)

    IxI = jnp.einsum("ij,kl->ijkl", I2, I2)
    I4d = I4s - (1.0 / 3.0) * IxI

    return I2, I4, I4t, I4s, IxI, I4d


@jax.jit
def identity_tensors_2d(dtype=jnp.float32):
    """2-D identities: ``I2, I4, I4t, I4s``."""
    I2 = jnp.eye(2, dtype=dtype)

    I4  = jnp.einsum("ik,jl->ijkl", I2, I2)
    I4t = jnp.einsum("il,jk->ijkl", I2, I2)
    I4s = 0.5 * (I4 + I4t)

    return I2, I4, I4t, I4s


@jax.jit
def levi_civita_3d(dtype=jnp.float32):
    """Permutation tensor ``eps_ijk``."""
    eps = jnp.zeros((3, 3, 3), dtype=dtype)

    eps = eps.at[0, 1, 2].set( 1)
    eps = eps.at[1, 2, 0].set( 1)
    eps = eps.at[2, 0, 1].set( 1)
    eps = eps.at[0, 2, 1].set(-1)
    eps = eps.at[2, 1, 0].set(-1)
    eps = eps.at[1, 0, 2].set(-1)

    return eps
