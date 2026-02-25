import os
os.environ["JAX_ENABLE_X64"] = "1"


import jax
import jax.numpy as jnp
import numpy as np



def set_freq_jax(ndim: int, n: tuple[int, ...], L: tuple[float, ...]):
    # n and L are STATIC python values here
    if ndim == 2:
        nx, ny = n
        Lx, Ly = L

        kx = nx * jnp.fft.fftfreq(nx, d=Lx)
        ky = ny * jnp.fft.fftfreq(ny, d=Ly)[: ny // 2 + 1]

        mg = jnp.array(jnp.meshgrid(kx, ky, indexing="ij"))  # (2, nx, nyh)
        return (2.0j * jnp.pi) * mg.reshape(2, nx * (ny // 2 + 1))

    elif ndim == 3:
        nx, ny, nz = n
        Lx, Ly, Lz = L

        kx = jnp.fft.fftfreq(nx, d=Lx)
        ky = jnp.fft.fftfreq(ny, d=Ly)
        kz = jnp.fft.fftfreq(nz, d=Lz)[: nz // 2 + 1]

        mg = jnp.array(jnp.meshgrid(kx, ky, kz, indexing="ij"))  # (3, nx, ny, nzh)
        # match FFTMAD scaling by n[i]
        nvec = jnp.array([nx, ny, nz], dtype=mg.dtype)[:, None]
        return (2.0j * jnp.pi) * (nvec * mg.reshape(3, nx * ny * (nz // 2 + 1)))

    else:
        raise ValueError("ndim must be 2 or 3")


# ✅ mark ALL shape-relevant args as static IMPORTANT for JIT compilation and performance
set_freq_jax_jit = jax.jit(set_freq_jax, static_argnames=("ndim", "n", "L"))



## Old NumPy version for benchmarking
def set_freq_np(ndim, n, L):
    n = np.asarray(n)
    L = np.asarray(L)

    if ndim == 2:
        kk_glob = 2.0j * np.pi * np.array(
            np.meshgrid(
                n[0] * np.fft.fftfreq(n[0], L[0]),
                n[1] * np.fft.fftfreq(n[1], L[1])[0 : n[1] // 2 + 1],
                indexing="ij",
            )
        ).reshape(2, n[0] * (n[1] // 2 + 1))
    elif ndim == 3:
        kk_glob = 2.0j * np.pi * np.einsum(
            "i,ix->ix",
            n,
            np.array(
                np.meshgrid(
                    np.fft.fftfreq(n[0], L[0]),
                    np.fft.fftfreq(n[1], L[1]),
                    np.fft.fftfreq(n[2], L[2])[0 : n[2] // 2 + 1],
                    indexing="ij",
                )
            ).reshape(3, n[0] * n[1] * (n[2] // 2 + 1)),
        )
    else:
        raise ValueError("ndim must be 2 or 3")
    return kk_glob