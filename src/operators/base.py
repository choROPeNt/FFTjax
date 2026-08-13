"""
Abstract linear-operator interface for operators/ (Green's operators,
differential operators, and their compositions).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import jax.numpy as jnp


class LinearOperator(ABC):
    """
    A linear map acting on per-voxel tensor fields (real or Fourier space).

    Concrete operators implement ``__call__``. Composition via ``@`` builds a
    new ``LinearOperator`` without evaluating either side, and ``.T`` returns
    the adjoint — so e.g. ``Gamma0 = G0 @ grad`` is itself a ``LinearOperator``
    that can be further composed or transposed.
    """

    @abstractmethod
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        ...

    @property
    def T(self) -> "LinearOperator":
        """Adjoint operator. Self-adjoint by default — override where A != A.T."""
        return self

    def __matmul__(self, other: "LinearOperator") -> "LinearOperator":
        return _Composed(self, other)


class _Composed(LinearOperator):
    """``(A @ B)(x) = A(B(x))``; adjoint is ``B.T @ A.T``."""

    def __init__(self, a: LinearOperator, b: LinearOperator):
        self._a = a
        self._b = b

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self._a(self._b(x))

    @property
    def T(self) -> LinearOperator:
        return self._b.T @ self._a.T
