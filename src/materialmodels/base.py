"""Abstract constitutive-model interface for materialmodels/."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

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


@runtime_checkable
class PhaseFieldMaterial(Protocol):
    """
    What materialmodels.phasefield.degradation's k_res_field/Gc_field (and
    any other phase-field-side gathering code) need beyond plain
    ConstitutiveModel -- structural, not a ConstitutiveModel subclass,
    since not every constitutive model is phase-field-capable and
    ConstitutiveModel is deliberately kept thin. LinearElasticIsotropic and
    TransverseIsotropic both satisfy this already (k_res, Gc are
    constructor kwargs on both), no inheritance change needed.
    """

    k_res: float
    Gc:    float | None

    def stiffness_tensor(self) -> jnp.ndarray: ...
