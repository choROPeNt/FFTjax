"""Abstract material-model interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

import jax.numpy as jnp


class ConstitutiveModel(ABC):
    """
    Stress-strain law as the FFT solver needs it: ``sigma = C : eps``.

    ``elastic_stiffness_tensor()`` is the constant elastic tensor. It is the
    tangent of a linear model, but only the virgin tensor of a nonlinear one
    (e.g. for the reference medium); those provide their own state-dependent
    ``stiffness_tensor``.
    """

    @abstractmethod
    def elastic_stiffness_tensor(self) -> jnp.ndarray:
        """(3, 3, 3, 3) elastic stiffness ``C_ijkl``."""
        ...


class ConductivityModel(ABC):
    """Fourier's law as the FFT solver needs it: ``q = -K grad T`` (rank 2, so not a ConstitutiveModel)."""

    @abstractmethod
    def conductivity_tensor(self) -> jnp.ndarray:
        """(3, 3) conductivity ``K_ij``."""
        ...


@runtime_checkable
class PhaseFieldMaterial(Protocol):
    """Phase-field attributes (``k_res``, ``Gc``) on top of the elastic tensor; structural type."""

    k_res: float
    Gc:    float | None

    def elastic_stiffness_tensor(self) -> jnp.ndarray: ...
