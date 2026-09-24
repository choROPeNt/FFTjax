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

    ``elastic_stiffness_tensor()`` is the one every model must provide: a
    constant, state-independent reference tensor -- for a linear model
    (``LinearElasticIsotropic``, ``TransverseIsotropic``) this *is* the
    tangent; for a nonlinear one (``J2Plasticity``, ``DruckerPrager``,
    ``PhaseFieldIsotropic``) it's the undamaged/virgin elastic tensor used
    e.g. as the FFT solver's reference medium (``operators.green.
    build_reference_green_operator``), never the true state-dependent
    tangent -- see each such model's own ``stiffness_tensor(eps, ...)``
    for that.
    """

    @abstractmethod
    def elastic_stiffness_tensor(self) -> jnp.ndarray:
        """(3, 3, 3, 3) constant elastic stiffness tensor C_ijkl."""
        ...


class ConductivityModel(ABC):
    """
    A material's flux-gradient law, reduced to what the FFT solver needs: a
    2nd-order conductivity tensor K_ij relating flux and gradient via
    Fourier's law, q = -K : grad(T). Deliberately thin, mirroring
    ConstitutiveModel -- symmetry class, parametrization, and any derived
    quantities are up to each concrete model. Not a ConstitutiveModel
    subclass: a conductivity tensor is rank 2, not rank 4, so there is
    nothing to share beyond the pattern (see materialmodels.assembly.
    assemble_K_field, materialmodels.thermal.isotropic.
    ThermalConductivityIsotropic).
    """

    @abstractmethod
    def conductivity_tensor(self) -> jnp.ndarray:
        """(3, 3) conductivity tensor K_ij."""
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

    def elastic_stiffness_tensor(self) -> jnp.ndarray: ...
