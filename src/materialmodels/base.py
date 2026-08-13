"""Abstract constitutive-model interface for materialmodels/."""

from __future__ import annotations

from abc import ABC, abstractmethod

import jax.numpy as jnp


class ConstitutiveModel(ABC):
    """
    A material's stress-strain law, reduced to what the FFT solver needs: a
    4th-order stiffness tensor C_ijkl relating stress and strain,
    sigma = C : eps. Deliberately thin -- symmetry class, parametrization,
    and any derived quantities (moduli, Voigt form, ...) are up to each
    concrete model.
    """

    @abstractmethod
    def stiffness_tensor(self) -> jnp.ndarray:
        """(3, 3, 3, 3) stiffness tensor C_ijkl."""
        ...
