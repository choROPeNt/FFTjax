"""Abstract linear-operator interface (Green's operators and their compositions)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import jax.numpy as jnp


class LinearOperator(ABC):
    """
    Linear map on per-voxel tensor fields (real or Fourier space).

    ``A @ B`` composes lazily; ``.T`` is the adjoint.
    """

    @abstractmethod
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        ...

    @property
    def T(self) -> "LinearOperator":
        # No self-adjoint default: a forgotten override (e.g. grad, adjoint -div)
        # must fail loudly, not return a wrong adjoint.
        raise NotImplementedError(
            f"{type(self).__name__} does not define its adjoint (.T)"
        )

    def __matmul__(self, other: "LinearOperator") -> "LinearOperator":
        return _Composed(self, other)


class _Composed(LinearOperator):
    """``(A @ B)(x) = A(B(x))``; adjoint ``B.T @ A.T``."""

    def __init__(self, a: LinearOperator, b: LinearOperator):
        self._a = a
        self._b = b

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self._a(self._b(x))

    @property
    def T(self) -> LinearOperator:
        return self._b.T @ self._a.T
